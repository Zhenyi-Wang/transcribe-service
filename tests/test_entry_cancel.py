"""entry 控制句柄与取消传播测试（spec §4.3 / §6，Task 2）。

覆盖：
1. 取消探针命中 → JobCancelledError 上抛且部分结果绝不入缓存；
2. 探针① 位置在 await diar_task 之前（经既有 except 链回收分离子任务）；
3. 取消传播顺序：thread_future 完成前不释放模型计数/音频租约；
4. ASR 基线缓存命中路径（_annotate_cached_asr）的入口探针；
5. _run_diarization 分离忙退避（wait_deadline，deadline=0 遗留行为不变）；
6. cancel_computation 语义（flag 置位 / asr_started 门控 cancel_request / 键不存在 None）；
7. 遗留先创建 + job 后加入命中同一 entry（build_inflight_key）；
8. JobCancelledError 派生 BaseException，穿透外层 except Exception 不转 error dict；
9. 探针②：await diar_task 期间到达的取消在字幕生成/缓存写入前中止。
"""
import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import transcribe as T
from backends.base import TranscribeResult
from cache_manager import CacheManager
from diarization.manager import SpeakerTurn


class StubBackend:
    """可控桩后端：记录取消信号与线程事件，可阻塞模拟长转录。"""

    name = "stub"
    device = "cpu"

    def __init__(self, sleep=0.0):
        self.sleep = sleep
        self.cancel_requests = []
        self.events = []           # asr 线程内追加，用于事件顺序断言
        self.last_request_id = None

    def transcribe(self, path, lang=None, context=None, progress_cb=None, request_id=None):
        self.last_request_id = request_id
        self.events.append("asr_thread_start")
        if self.sleep:
            time.sleep(self.sleep)
        self.events.append("asr_thread_end")
        return TranscribeResult(
            text="甲说乙说", language="zh",
            timestamps=[{"text": c, "start": i * 2.0, "end": i * 2.0 + 1.0}
                        for i, c in enumerate("甲说乙说")],
            performance={},
        )

    def cancel_request(self, request_id):
        self.cancel_requests.append(request_id)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(type(T.config), "diarization_enabled", property(lambda self: True))
    monkeypatch.setattr(type(T.config), "cache_dir", property(lambda self: str(tmp_path / "cache")))
    monkeypatch.setattr(type(T.config), "cache_enabled", property(lambda self: True))
    monkeypatch.setattr(T, "cache_manager", CacheManager())
    monkeypatch.setattr(T, "get_audio_duration", lambda path: 8.0)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake-audio")
    backend = StubBackend()
    manager = MagicMock()
    manager.load_model_if_needed.return_value = backend
    service = T.TranscriptionService(manager)
    return service, backend, str(audio)


def _wait_inflight(service, timeout=2.0):
    """等 owner 协程把 entry 注册进 _inflight，返回 entry。"""
    async def wait():
        for _ in range(int(timeout * 100)):
            if service._inflight:
                return next(iter(service._inflight.values()))
            await asyncio.sleep(0.01)
        raise AssertionError("entry 未注册")
    return wait()


@pytest.mark.asyncio
async def test_probe_aborts_before_cache(env, monkeypatch):
    """探针恒命中 + 桩 backend 正常返回 → JobCancelledError 且缓存零写入。

    传 file_path_for_cache 使 cache_args 非空（no_cache 会绕开缓存链路，断言空转）：
    未取消时该调用必写基线缓存，故 saves==[] 是有内容的断言。
    """
    service, backend, path = env
    saves = []
    monkeypatch.setattr(T.cache_manager, "save_transcript_to_cache",
                        lambda *a, **k: saves.append(k))
    with pytest.raises(T.JobCancelledError):
        await asyncio.wait_for(
            service.process_transcription(path, file_path_for_cache=path,
                                          cancelled=lambda: True), 5)
    assert saves == [], "部分结果不得入缓存"
    assert backend.events == ["asr_thread_start", "asr_thread_end"]


@pytest.mark.asyncio
async def test_probe_position_before_diar_await(env, monkeypatch):
    """diarize=True：探针① 在 await diar_task 之前命中 → 既有 except 链取消分离子任务。"""
    service, backend, path = env
    cleaned = asyncio.Event()

    async def infer(audio_path, timeout, stage_cb=None):
        try:
            await asyncio.Event().wait()  # 挂起：若探针位于 await 之后，流程将卡死于此
        finally:
            cleaned.set()

    monkeypatch.setattr(T, "_diarize_samples", infer)
    with pytest.raises(T.JobCancelledError):
        await asyncio.wait_for(
            service.process_transcription(path, no_cache=True, diarize=True,
                                          cancelled=lambda: True), 5)
    assert cleaned.is_set(), "探针中止必须经既有 except 链取消分离子任务"


@pytest.mark.asyncio
async def test_thread_future_join_before_release(env, monkeypatch, tmp_path):
    """取消传播顺序：模型计数与音频租约的释放在 thread_future 完成（线程退出）之后。"""
    service, backend, _ = env
    upload_dir = tmp_path / "tmp"
    upload_dir.mkdir()
    upload = upload_dir / "upload.wav"
    upload.write_bytes(b"fake-audio")
    monkeypatch.chdir(tmp_path)

    events = []
    backend.events = events        # asr 线程直接写入共享事件列表
    backend.sleep = 0.3

    real_release = service.model_manager.release

    def rec_release():
        events.append("model_release")
        return real_release()

    service.model_manager.release = rec_release

    real_unlink = Path.unlink

    def rec_unlink(self, missing_ok=False):
        if self.name.startswith(".job-"):
            events.append("lease_release")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", rec_unlink)

    owner = asyncio.create_task(service.process_transcription(str(upload), no_cache=True))
    entry = await _wait_inflight(service)
    for _ in range(200):
        if entry["asr_started"] and entry["thread_future"] is not None:
            break
        await asyncio.sleep(0.01)
    assert entry["asr_started"] and entry["thread_future"] is not None

    assert service.cancel_computation(entry["inflight_key"]) is entry
    assert backend.cancel_requests == [entry["request_id"]]
    # 取消方 join：thread_future 在线程自然跑完（0.3s）后才完成
    entry["thread_future"].result(timeout=2)

    with pytest.raises(T.JobCancelledError):
        await asyncio.wait_for(owner, 5)

    assert events[:2] == ["asr_thread_start", "asr_thread_end"]
    assert events.index("asr_thread_end") < events.index("model_release") \
        < events.index("lease_release"), f"释放顺序错误: {events}"


@pytest.mark.asyncio
async def test_annotate_probe(env, monkeypatch):
    """ASR 基线缓存命中路径：入口探针命中 → JobCancelledError 且不进入分离。"""
    service, backend, path = env
    diar_calls = []

    async def must_not_run(audio_path, timeout, stage_cb=None):
        diar_calls.append(audio_path)
        return [SpeakerTurn(0.0, 8.0, 0)]

    monkeypatch.setattr(T, "_diarize_samples", must_not_run)
    # 先落 ASR 基线缓存（diarize=False 正常转录一次）
    await service.process_transcription(path, file_path_for_cache=path, diarize=False)
    assert T.cache_manager.get_cached_transcript(file_path=path, diarize=False, include_asr=True)
    with pytest.raises(T.JobCancelledError):
        await asyncio.wait_for(
            service.process_transcription(path, file_path_for_cache=path, diarize=True,
                                          cancelled=lambda: True), 5)
    assert diar_calls == [], "缓存命中取消不得进入分离"


@pytest.mark.asyncio
async def test_busy_wait_deadline(monkeypatch):
    """分离忙退避：deadline=0 首忙即降级（遗留行为）；deadline=65s 每 30s 重试至成功。"""
    from diarization.worker import DiarizationBusyError
    attempts = []
    sleeps = []
    real_sleep = asyncio.sleep

    async def busy_twice(audio_path, timeout, stage_cb=None):
        attempts.append(audio_path)
        if len(attempts) <= 2:
            raise DiarizationBusyError("busy")
        return [SpeakerTurn(0.0, 8.0, 0)]

    async def rec_sleep(d):
        sleeps.append(d)
        await real_sleep(0)

    monkeypatch.setattr(T, "_diarize_samples", busy_twice)
    monkeypatch.setattr(T.asyncio, "sleep", rec_sleep)

    turns, error, _ = await T._run_diarization("x.wav", 8.0, wait_deadline=0.0)
    assert turns is None and error["busy"] is True
    assert len(attempts) == 1 and 30.0 not in sleeps, "deadline=0 不得重试"

    attempts.clear()
    deadline = time.monotonic() + 65.0
    turns, error, _ = await T._run_diarization("x.wav", 8.0, wait_deadline=deadline)
    assert error is None and turns is not None
    assert len(attempts) == 3
    assert sleeps.count(30.0) == 2, f"两次 busy 应各退避 30s: {sleeps}"


@pytest.mark.asyncio
async def test_cancel_computation(env, monkeypatch):
    """cancel_computation：flag 置位；asr_started 门控 cancel_request；键不存在返回 None。"""
    service, backend, path = env

    key = service.build_inflight_key(path, None, None, None, None, None, True, True)
    assert service.cancel_computation(key) is None  # 键不存在

    # 未开始 ASR：挡在时长探测（to_thread）处，entry 已存在但 asr_started=False
    gate = threading.Event()
    diar_cleaned = asyncio.Event()

    def gated_duration(p):
        gate.wait(2)
        return 8.0

    async def infer(audio_path, timeout, stage_cb=None):
        try:
            await asyncio.Event().wait()
        finally:
            diar_cleaned.set()

    monkeypatch.setattr(T, "get_audio_duration", gated_duration)
    monkeypatch.setattr(T, "_diarize_samples", infer)
    owner = asyncio.create_task(service.process_transcription(path, no_cache=True, diarize=True))
    entry = await _wait_inflight(service)
    assert entry["asr_started"] is False
    assert service.cancel_computation(key) is entry
    assert entry["cancel_flag"].is_set()
    assert backend.cancel_requests == [], "未开始 ASR 不得调 cancel_request"
    gate.set()
    with pytest.raises(T.JobCancelledError):
        await asyncio.wait_for(owner, 5)
    assert diar_cleaned.is_set()
    assert service._inflight == {}

    # 已开始 ASR：cancel_request 被调（带 entry.request_id）
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 8.0)
    started, release = threading.Event(), threading.Event()

    def blocking(path_, lang=None, context=None, progress_cb=None, request_id=None):
        started.set()
        release.wait(2)
        return TranscribeResult(text="甲说", language="zh",
                                timestamps=[{"text": "甲", "start": 0.0, "end": 1.0},
                                            {"text": "说", "start": 1.0, "end": 2.0}])

    backend.transcribe = blocking
    owner2 = asyncio.create_task(service.process_transcription(path, no_cache=True))
    await asyncio.to_thread(started.wait, 2)  # ASR 线程已运行 → asr_started=True
    entry2 = next(iter(service._inflight.values()))
    assert entry2["asr_started"] is True and entry2["thread_future"] is not None
    assert service.cancel_computation(entry2["inflight_key"]) is entry2
    assert backend.cancel_requests == [entry2["request_id"]]
    release.set()
    with pytest.raises(T.JobCancelledError):
        await asyncio.wait_for(owner2, 5)


@pytest.mark.asyncio
async def test_cancel_computation_signal_gate(env, monkeypatch):
    """fixwave F1：signal=False 只置 entry.cancel_flag、跳过 backend.cancel_request
    （legacy 引擎的取消端点是全局语义，scoped HTTP 会误杀他人转录）；
    默认 signal=True 行为不变。"""
    service, backend, path = env
    started, release = threading.Event(), threading.Event()

    def blocking(path_, lang=None, context=None, progress_cb=None, request_id=None):
        started.set()
        release.wait(5)
        return TranscribeResult(text="甲说", language="zh",
                                timestamps=[{"text": "甲", "start": 0.0, "end": 1.0},
                                            {"text": "说", "start": 1.0, "end": 2.0}])

    backend.transcribe = blocking
    owner = asyncio.create_task(service.process_transcription(path, no_cache=True))
    await asyncio.to_thread(started.wait, 5)
    entry = next(iter(service._inflight.values()))
    assert entry["asr_started"] is True and entry["thread_future"] is not None

    # signal=False：置标志、不发 backend 信号
    assert service.cancel_computation(entry["inflight_key"], signal=False) is entry
    assert entry["cancel_flag"].is_set()
    assert backend.cancel_requests == [], "signal=False 不得调 cancel_request（不发 scoped HTTP）"

    # 默认 signal=True：归属化信号照发（既有语义）
    assert service.cancel_computation(entry["inflight_key"]) is entry
    assert backend.cancel_requests == [entry["request_id"]]
    release.set()
    with pytest.raises(T.JobCancelledError):
        await asyncio.wait_for(owner, 5)


@pytest.mark.asyncio
async def test_shared_entry_cross_caller(env, monkeypatch):
    """遗留先创建 + job 后加入：build_inflight_key 同实参命中同一 entry。"""
    service, backend, path = env
    started, release = threading.Event(), threading.Event()

    def blocking(path_, lang=None, context=None, progress_cb=None, request_id=None):
        started.set()
        release.wait(2)
        return TranscribeResult(text="甲说", language="zh",
                                timestamps=[{"text": "甲", "start": 0.0, "end": 1.0},
                                            {"text": "说", "start": 1.0, "end": 2.0}])

    backend.transcribe = blocking
    owner = asyncio.create_task(service.process_transcription(path, no_cache=True))
    await asyncio.to_thread(started.wait, 2)
    # job 侧：以与调用 process_transcription 完全相同的实参组装键
    key = service.build_inflight_key(path, None, None, None, None, None, False, True)
    entry = service.cancel_computation(key)
    assert entry is not None
    assert entry is service._inflight.get(key)
    assert entry["cancel_flag"].is_set()
    assert backend.cancel_requests == [entry["request_id"]]
    release.set()
    with pytest.raises(T.JobCancelledError):
        await asyncio.wait_for(owner, 5)


@pytest.mark.asyncio
async def test_jobcancelled_passthrough(env, monkeypatch):
    """JobCancelledError 派生 BaseException，穿透外层 except Exception 不转 error dict。"""
    assert issubclass(T.JobCancelledError, BaseException)
    assert not issubclass(T.JobCancelledError, Exception)
    service, backend, path = env
    saves = []
    monkeypatch.setattr(T.cache_manager, "save_transcript_to_cache",
                        lambda *a, **k: saves.append(k))
    with pytest.raises(T.JobCancelledError):
        await asyncio.wait_for(
            service.process_transcription(path, file_path_for_cache=path,
                                          cancelled=lambda: True), 5)
    assert saves == []
    assert service._inflight == {}


@pytest.mark.asyncio
async def test_probe2_catches_cancel_during_diar_await(env, monkeypatch):
    """探针②：取消到达于 await diar_task 期间 → 字幕生成/缓存写入前中止。

    定序消除竞速：等 thread_future.done()（ASR 已返回、探针① 已过）+ 多次 sleep(0)
    让 owner 协程确定挂在 await diar_task，之后才置位取消——命中点必然是探针②；
    infer 桩记录「正常完成 vs 被取消」两种路径，断言走的是正常完成（证明到达探针②
    的是带 result 的路径，而非经 except 链回收分离的探针① 路径）。
    """
    service, backend, path = env
    release = asyncio.Event()
    outcome = {}

    async def infer(audio_path, timeout, stage_cb=None):
        try:
            await release.wait()
            outcome["diar"] = "completed"
            return [SpeakerTurn(0.0, 8.0, 0), SpeakerTurn(2.0, 8.0, 1)]
        except asyncio.CancelledError:
            outcome["diar"] = "cancelled"  # 探针① 路径的标记：出现即定序失败
            raise

    monkeypatch.setattr(T, "_diarize_samples", infer)
    saves = []
    monkeypatch.setattr(T.cache_manager, "save_transcript_to_cache",
                        lambda *a, **k: saves.append(k))
    owner = asyncio.create_task(service.process_transcription(
        path, file_path_for_cache=path, diarize=True))
    entry = await _wait_inflight(service)
    for _ in range(200):
        if entry["thread_future"] is not None and entry["thread_future"].done():
            break
        await asyncio.sleep(0.01)
    assert entry["thread_future"] is not None and entry["thread_future"].done()
    # ASR 已返回 → owner 推进经过探针①，确定挂在 await diar_task（探针② 前夜）
    for _ in range(10):
        await asyncio.sleep(0)
    service.cancel_computation(entry["inflight_key"])
    release.set()
    with pytest.raises(T.JobCancelledError):
        await asyncio.wait_for(owner, 5)
    assert outcome["diar"] == "completed", "取消必须命中于分离正常完成之后（探针②），而非经 except 链回收分离"
    assert saves == [], "await diar_task 期间到达的取消也必须中止于缓存写入之前"
