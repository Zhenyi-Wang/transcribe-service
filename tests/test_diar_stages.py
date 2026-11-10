"""分离管线五段阶段事件测试（spec §5「并行分离」行，Task 6）。

五段契约（diarize_stage 槽位可见序列）：decoding → segmenting → embedding →
clustering → reconstructing。发射锚点：
- decoding：子进程 _main 解码起点（diarization/worker.py）；
- segmenting / embedding：pyannote hook 批进度首事件（completed=0）起点锚定；
- clustering：嵌入阶段末尾的管线缝隙补发（评审 R2A：聚类无现成钩子）；
- reconstructing：自研 BoundedSpeakerDiarization.reconstruct 起点直发。
父进程 DiarizationWorker.run 收 {"event": "stage"} 后经 stage_cb 透传；
transcribe.py 全链（process_transcription → _run_diarization → _diarize_samples
→ worker.run，diarize_only / diarize_merge_subtitle 同链）逐层透传。
阶段回调契约与 progress_cb 一致：不得抛出异常；各层兜底吞异常，发射失败
绝不中断分离本身。
"""
import asyncio
import importlib
import subprocess
import sys
import types

import numpy as np
import pytest
from unittest.mock import MagicMock

import transcribe as T
from backends.base import TranscribeResult
from diarization.manager import DiarizationManager, SpeakerTurn


# ========== 父进程侧：worker.run 透传 stage 事件（真实 Pipe + 桩子进程） ==========

@pytest.fixture
def worker_module(monkeypatch):
    module = importlib.import_module("diarization.worker")
    popen = subprocess.Popen

    def spawn_stub(command, **kwargs):
        # 与 test_diarization_worker 同款：保留真实 Pipe/进程组/回收，只换推理体。
        command = [sys.executable, "-m", "tests.diarization_worker_stub", command[3]]
        return popen(command, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", spawn_stub)
    monkeypatch.setattr(module, "_manager_settings", lambda: {})
    return module


@pytest.mark.asyncio
async def test_parent_forwards_stage_events_in_order(tmp_path, worker_module):
    """桩子进程依序发 decoded/stage×2/result → stage_cb 只收到 stage 序列。"""
    worker = worker_module.DiarizationWorker(memory_mb=256, max_jobs=10)
    stages = []
    try:
        turns = await worker.run(str(tmp_path / "stages"), timeout=5, stage_cb=stages.append)
        assert stages == ["segmenting", "embedding"], "stage 事件按序透传，decoded 等信息事件不算 stage"
        assert [(t.start, t.end, t.speaker) for t in turns] == [(0.0, 4.0, 0), (4.0, 8.0, 1)]
    finally:
        worker.unload()


@pytest.mark.asyncio
async def test_parent_stage_cb_exception_does_not_break_diarization(tmp_path, worker_module):
    """stage_cb 抛异常必须被吞掉，分离照常完成（进度回调不得影响主流程）。"""
    worker = worker_module.DiarizationWorker(memory_mb=256, max_jobs=10)

    def bad_cb(name):
        raise RuntimeError("stage 回调不应中断分离")

    try:
        turns = await worker.run(str(tmp_path / "stages"), timeout=5, stage_cb=bad_cb)
        assert [(t.start, t.end, t.speaker) for t in turns] == [(0.0, 4.0, 0), (4.0, 8.0, 1)]
    finally:
        worker.unload()


@pytest.mark.asyncio
async def test_parent_default_stage_cb_none_unchanged(tmp_path, worker_module):
    """不传 stage_cb（遗留行为）照常返回 turns。"""
    worker = worker_module.DiarizationWorker(memory_mb=256, max_jobs=10)
    try:
        turns = await worker.run(str(tmp_path / "stages"), timeout=5)
        assert len(turns) == 2
    finally:
        worker.unload()


# ========== manager 侧：diarize() 四段发射（decoding 由子进程 _main 负责） ==========

class _FakeAnnotation:
    def __init__(self, raw):
        self._raw = raw

    def itertracks(self, yield_label=False):
        assert yield_label
        for start, end, label in self._raw:
            yield types.SimpleNamespace(start=start, end=end), None, label


class _FakePipeline:
    """复刻 pyannote SpeakerDiarization.apply 的 hook 事件序列（3.3.2 源码锚定）。

    - ("segmentation", completed=0/total=4)：分割批进度首事件（起点）
    - ("segmentation", completed=2)：批中进度（不得发射）
    - ("segmentation", artefact)：阶段末尾（speaker_counting 前后均无映射）
    - ("embeddings", completed=0)：嵌入起点
    - ("embeddings", 阶段末尾 artefact)：嵌入结束 → 聚类开始的管线缝隙
    - reconstruct：真实子类在此直发 reconstructing——同款读暂存的发射闭包
      self._diarize_stage_emit（manager.diarize 注入），本桩据此模拟。
    """

    def __call__(self, audio, hook=None, **kwargs):
        hook("segmentation", None, total=4, completed=0, file=None)
        hook("segmentation", None, total=4, completed=2, file=None)
        hook("segmentation", object(), file=None)
        hook("speaker_counting", object(), file=None)
        hook("embeddings", None, total=2, completed=0, file=None)
        hook("embeddings", None, total=2, completed=2, file=None)
        hook("embeddings", object(), file=None)
        emit = getattr(self, "_diarize_stage_emit", None)
        if emit is not None:
            emit("reconstructing")
        hook("discrete_diarization", object(), file=None)
        return _FakeAnnotation([(0.0, 4.0, "SPEAKER_00"), (4.0, 8.0, "SPEAKER_01")])


def _make_manager():
    m = DiarizationManager(
        backend="pyannote-hybrid", cluster_threshold=0.7046, min_cluster_size=12,
        num_speakers=-1, embedding_model="~/models/diarization/fake.onnx", hf_token="")
    m._load = lambda: None  # 不加载真实管线
    m._pipeline = _FakePipeline()
    return m


def test_manager_emits_four_pipeline_stages_in_order():
    """diarize(stage_cb=...) 依次收到 segmenting/embedding/clustering/reconstructing。

    精确序列同时验证：批中进度事件（completed=2）与未映射阶段
    （speaker_counting / discrete_diarization）不发射。
    """
    m = _make_manager()
    stages = []
    turns = m.diarize(np.zeros(16000, dtype=np.float32), stage_cb=stages.append)
    assert stages == ["segmenting", "embedding", "clustering", "reconstructing"]
    assert [(t.speaker, t.start, t.end) for t in turns] == [(0, 0.0, 4.0), (1, 4.0, 8.0)]
    assert m._pipeline._diarize_stage_emit is None, "发射闭包用毕即清，不留旧回调跨任务引用"


def test_manager_default_no_stage_cb_is_noop():
    """不传 stage_cb（遗留行为）：管线照常出结果，无发射副作用。"""
    m = _make_manager()
    turns = m.diarize(np.zeros(16000, dtype=np.float32))
    assert [(t.speaker, t.start, t.end) for t in turns] == [(0, 0.0, 4.0), (1, 4.0, 8.0)]
    assert m._pipeline._diarize_stage_emit is None


def test_manager_stage_cb_exception_does_not_break_diarize():
    """stage_cb 抛异常被 manager 层兜底吞掉，分离照常完成。"""
    m = _make_manager()

    def bad_cb(name):
        raise RuntimeError("stage 回调不应中断分离")

    turns = m.diarize(np.zeros(16000, dtype=np.float32), stage_cb=bad_cb)
    assert len(turns) == 2


# ========== transcribe.py 全链透传：process_transcription → _run_diarization
# → _diarize_samples → worker.run（monkeypatch 最深层断言调用形状） ==========

class _FakeBackend:
    name = "fake"
    device = "cpu"

    def transcribe(self, path, lang=None, context=None, progress_cb=None, request_id=None):
        chars = "甲乙甲乙二人对话内容持续输出一直到十六字整"
        return TranscribeResult(
            text=chars,
            language="zh",
            timestamps=[{"text": c, "start": i * 0.5, "end": i * 0.5 + 0.4}
                        for i, c in enumerate(chars)],
            performance={"rtf": 0.1},
        )


def _make_service():
    svc = T.TranscriptionService(MagicMock())
    svc.model_manager.load_model_if_needed.return_value = _FakeBackend()
    return svc


def _enable_diarization(monkeypatch):
    monkeypatch.setattr(type(T.config), "diarization_enabled",
                        property(lambda self: True))


def _install_capturing_diarize(monkeypatch, sink):
    """把 T._diarize_samples 换成捕获 stage_cb 的桩：记录回调对象并模拟一次发射。"""
    async def infer(path, timeout, stage_cb=None):
        sink["stage_cb"] = stage_cb
        sink["path"] = path
        sink["timeout"] = timeout
        if stage_cb is not None:
            stage_cb("decoding")
        return [SpeakerTurn(0.0, 4.0, 0), SpeakerTurn(4.0, 8.0, 1)]
    monkeypatch.setattr(T, "_diarize_samples", infer)


@pytest.mark.asyncio
async def test_chain_process_transcription_reaches_worker(tmp_path, monkeypatch):
    """process_transcription(stage_cb=...) 的回调对象原样到达最深层并可用。"""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 8.0)
    _enable_diarization(monkeypatch)
    sink = {}
    _install_capturing_diarize(monkeypatch, sink)
    stages = []
    cb = stages.append  # 绑定方法每次求值生成新对象，先捕获再传，保证身份可比
    svc = _make_service()
    resp = await svc.process_transcription(str(wav), "a.wav", no_cache=True, diarize=True,
                                           stage_cb=cb)
    assert resp["status"] == "success"
    assert sink["stage_cb"] is cb, "同一回调对象逐层到达 _diarize_samples"
    assert sink["path"] == str(wav) and sink["timeout"] > 0
    assert stages == ["decoding"], "最深层的模拟发射经全链回到调用方"


@pytest.mark.asyncio
async def test_chain_default_stage_cb_is_none(tmp_path, monkeypatch):
    """遗留调用（不传 stage_cb）：最深层收到 None，行为不变。"""
    wav = tmp_path / "b.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 8.0)
    _enable_diarization(monkeypatch)
    sink = {}
    _install_capturing_diarize(monkeypatch, sink)
    svc = _make_service()
    resp = await svc.process_transcription(str(wav), "b.wav", no_cache=True, diarize=True)
    assert resp["status"] == "success"
    assert sink["stage_cb"] is None


@pytest.mark.asyncio
async def test_chain_diarize_only_passes_stage_cb(tmp_path, monkeypatch):
    wav = tmp_path / "c.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 8.0)
    _enable_diarization(monkeypatch)
    sink = {}
    _install_capturing_diarize(monkeypatch, sink)
    stages = []
    cb = stages.append
    resp = await T.diarize_only(str(wav), "vid", stage_cb=cb)
    assert resp["status"] == "success"
    assert sink["stage_cb"] is cb
    assert stages == ["decoding"]


@pytest.mark.asyncio
async def test_chain_diarize_merge_subtitle_passes_stage_cb(tmp_path, monkeypatch):
    wav = tmp_path / "d.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 8.0)
    _enable_diarization(monkeypatch)
    sink = {}
    _install_capturing_diarize(monkeypatch, sink)
    stages = []
    cb = stages.append
    body = [{"text": "甲", "start": 0.0, "end": 4.0}, {"text": "乙", "start": 4.0, "end": 8.0}]
    resp = await T.diarize_merge_subtitle(body, str(wav), "vid", stage_cb=cb)
    assert resp["status"] == "success"
    assert sink["stage_cb"] is cb
    assert stages == ["decoding"]
