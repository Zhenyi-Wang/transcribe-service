"""asr() 进度回调 progress_cb 与取消标记测试。

不加载真实模型：__new__ 跳过 __init__，手工注入最小依赖（encoder.encode /
_build_prompt_embd / _safe_decode 假实现），让 chunk 循环真实跑通。

验证接口契约：
1. progress_cb(done, total) 在每个 chunk 完成边界收到顺序值 (1,2)、(2,2)；
2. 取消信号已置位时提前终止（done < total），返回结果
   performance["cancelled"] is True；
3. transcribe() 将 progress_cb 透传给 asr()。
"""
import threading
import types

import numpy as np

from qwen_asr_gguf.inference.asr import QwenASREngine
from qwen_asr_gguf.inference.schema import DecodeResult

SR = 16000
N_EMBD = 4


def make_engine():
    """构造不加载真实模型的引擎实例，三个推理依赖全部为假实现。"""
    eng = QwenASREngine.__new__(QwenASREngine)
    eng.aligner = None
    eng.verbose = False
    eng._cancel_event = None

    # encoder 桩：返回固定 embedding 与耗时
    eng.encoder = types.SimpleNamespace()
    eng.encoder.encode = lambda chunk: (np.zeros((10, N_EMBD), dtype=np.float32), 0.001)

    # prompt 构建桩：直接透传（守卫逻辑不在本测试范围）
    eng._build_prompt_embd = lambda audio_embd, prefix_text, context, language: audio_embd

    # 解码桩：每个 chunk 返回固定文本与零统计
    eng._safe_decode = lambda full_embd, prefix_text, rollback_num, is_last_chunk, temperature: DecodeResult(text="测试")
    return eng


def two_chunk_audio():
    """恰好 2 个 chunk（chunk_size_sec=1）的假音频，无短尾段。"""
    return np.zeros(2 * SR, dtype=np.float32)


def test_progress_cb_reports_each_chunk():
    """progress_cb 在每轮 chunk 完成处收到 (done, total) 顺序值。"""
    eng = make_engine()
    calls = []
    result = eng.asr(
        audio=two_chunk_audio(),
        context="",
        language="Chinese",
        chunk_size_sec=1,
        progress_cb=lambda done, total: calls.append((done, total)),
    )
    assert calls == [(1, 2), (2, 2)]
    # 两个 chunk 都真实走完，文本拼接正确
    assert result.text == "测试测试"


def test_cancel_presets_flag_and_stops_early():
    """取消信号预置位：首个 chunk 边界即终止，performance 标记 cancelled。"""
    eng = make_engine()
    ev = threading.Event()
    ev.set()
    eng.set_cancel_event(ev)
    calls = []
    result = eng.asr(
        audio=two_chunk_audio(),
        context="",
        language="Chinese",
        chunk_size_sec=1,
        progress_cb=lambda done, total: calls.append((done, total)),
    )
    # 提前终止：没有任何 chunk 完成（done=0 < total=2）
    assert calls == []
    assert result.performance["cancelled"] is True


def test_cancel_midway_reports_completed_chunks():
    """第一个 chunk 完成后置位取消：已完成进度保留，第二个 chunk 被跳过。"""
    eng = make_engine()
    ev = threading.Event()
    eng.set_cancel_event(ev)
    calls = []

    def cb(done, total):
        calls.append((done, total))
        ev.set()  # 首个 chunk 完成后置位 → 下一边界取消

    result = eng.asr(
        audio=two_chunk_audio(),
        context="",
        language="Chinese",
        chunk_size_sec=1,
        progress_cb=cb,
    )
    assert calls == [(1, 2)]
    assert result.performance["cancelled"] is True


def test_transcribe_passes_progress_cb_through(monkeypatch):
    """transcribe() 将 progress_cb（含默认 None）透传给 asr()。"""
    # transcribe 先经 load_audio 读文件，桩掉返回假音频
    monkeypatch.setattr(
        "qwen_asr_gguf.inference.audio.load_audio",
        lambda *a, **k: np.zeros(1, dtype=np.float32),
    )
    eng = make_engine()
    eng.config = types.SimpleNamespace(chunk_size=1, memory_num=2)
    captured = {}

    def fake_asr(**kwargs):
        captured.update(kwargs)
        return "sentinel"

    eng.asr = fake_asr

    cb = lambda done, total: None
    ret = eng.transcribe("fake.wav", language="Chinese", context="", progress_cb=cb)
    assert ret == "sentinel"
    assert captured["progress_cb"] is cb

    # 不传 progress_cb：默认 None 也应透传（调用方无回调时向后兼容）
    eng.transcribe("fake.wav", language="Chinese", context="")
    assert captured["progress_cb"] is None
