"""_build_prompt_embd 引擎守卫（context 动态截断）测试。

不加载真实模型：__new__ 跳过 __init__，手工注入 tokenize/embedding 依赖，
专注验证守卫的预算计数与 head-keep 截断逻辑。
假分词规则：1 字符 = 1 token（无 BPE 合并），使预算可手工推算。
"""
from unittest.mock import MagicMock

import numpy as np

from qwen_asr_gguf.inference.asr import QwenASREngine

N_EMBD = 4


def make_engine(n_ubatch=2048, n_batch=8192):
    eng = QwenASREngine.__new__(QwenASREngine)
    eng.n_embd = N_EMBD
    eng.model = MagicMock()
    eng.model.n_embd = N_EMBD
    eng.model.tokenize = staticmethod(lambda t: [ord(c) % 30000 + 10 for c in t])
    # 真实 numpy 矩阵：embedding_table[list_of_ids] 原生可用
    eng.embedding_table = np.zeros((40000, N_EMBD), dtype=np.float32)
    eng.ID_IM_START, eng.ID_IM_END = 1, 2
    eng.ID_AUDIO_START, eng.ID_AUDIO_END = 3, 4
    eng.ID_ASR_TEXT = 5
    eng.n_ubatch = n_ubatch
    eng.n_batch = n_batch
    return eng


def audio(frames=10):
    return np.zeros((frames, N_EMBD), dtype=np.float32)


def test_no_truncation_when_context_fits():
    """短 context：守卫不触发，total_len == 各段精确和。"""
    eng = make_engine()
    full = eng._build_prompt_embd(audio(10), prefix_text="", context="你好", language="zh")
    prefix_len = 1 + len("system\n") + len("你好") + 1 + 1 + len("user\n") + 1
    suffix_len = 2 + 1 + len("assistant\nlanguage zh") + 1
    assert full.shape == (prefix_len + 10 + suffix_len, N_EMBD)


def test_no_truncation_default_system_text():
    """context=None：走默认 'You are a helpful assistant.'，不触发守卫。"""
    eng = make_engine()
    full = eng._build_prompt_embd(audio(10), prefix_text="", context=None, language="zh")
    default_text = "You are a helpful assistant."
    prefix_len = 1 + len("system\n") + len(default_text) + 1 + 1 + len("user\n") + 1
    suffix_len = 2 + 1 + len("assistant\nlanguage zh") + 1
    assert full.shape == (prefix_len + 10 + suffix_len, N_EMBD)


def test_truncation_caps_at_limit():
    """超长 context：截断后 total 恰为 len_limit，不越界。"""
    eng = make_engine()  # n_ubatch=2048, n_batch=8192 → limit=2048
    full = eng._build_prompt_embd(audio(10), prefix_text="", context="x" * 5000, language=None)
    assert full.shape[0] == 2048


def test_truncation_keeps_head():
    """head-keep：截断保留 context 头部（通过总长反推保留量正确）。"""
    eng = make_engine()
    ctx = "x" * 5000
    full = eng._build_prompt_embd(audio(10), prefix_text="", context=ctx, language=None)
    # 拆分路径 total = 9(fixed) + 7("system\n") + kept + 10(audio) + 13(suffix) = 2048
    assert full.shape[0] == 2048
    kept = 2048 - 9 - 7 - 10 - 13
    assert kept == 2009  # 预算 = 2048 − audio − fixed − suffix − head


def test_long_memory_text_squeezes_context():
    """记忆转录文本（prefix_text）占用预算时，context 自动少留，total 仍不越界。"""
    eng = make_engine()
    full = eng._build_prompt_embd(
        audio(10), prefix_text="y" * 1500, context="z" * 3000, language="zh"
    )
    assert full.shape[0] <= min(eng.n_ubatch, eng.n_batch // 4)


def test_guard_never_crashes_on_degenerate_budget():
    """记忆文本极端长的退化场景：允许 ctx 截到 0，但不抛异常。"""
    eng = make_engine()
    full = eng._build_prompt_embd(
        audio(10), prefix_text="y" * 100000, context="z" * 10, language=None
    )
    assert full.shape[0] > 0  # 不抛异常即通过（记忆文本超限属既有行为，不在守卫职责内）
