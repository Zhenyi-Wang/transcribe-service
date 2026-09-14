"""clamp_asr_context 钳制公式测试（与 qwen_asr_gguf 默认定容公式同源）。"""
from transcribe import clamp_asr_context


def test_default_budget_is_976():
    """默认配置（chunk_size=40, memory_num=1）：n_ubatch=2048, audio=1040 → 976。"""
    assert clamp_asr_context("x" * 5000) == "x" * 976


def test_short_context_unchanged():
    assert clamp_asr_context("你好世界") == "你好世界"


def test_none_and_empty_return_none():
    assert clamp_asr_context(None) is None
    assert clamp_asr_context("") is None
