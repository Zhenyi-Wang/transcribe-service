"""_merge_char_timestamps 的回归测试。

背景：讲章批量转录出现过两类内容丢失，最终都死在本函数的失配/耗尽 continue 上：
1. 音乐复读导致 aligner 截断，timestamps 覆盖少于 text；
2. text 夹杂英文（ts_chars 含空格、clean_seg 无空格）导致 find 失配、错位累积。
修复目标：时间戳对不上可以降级为估算时间，但文本段一个都不能丢。
"""
import re

from transcribe import _merge_char_timestamps


def _chars(text, start=0.0, step=0.3):
    """把文本逐字变成字级 timestamps（模拟 aligner 输出）。"""
    return [
        {"text": ch, "start": round(start + i * step, 2), "end": round(start + (i + 1) * step, 2)}
        for i, ch in enumerate(text)
    ]


def _body_text(body):
    return "".join(seg["content"] for seg in body)


def _assert_monotonic(body):
    """时间戳单调不减（from 不早于上一段 from，不倒退）。"""
    prev_from = None
    for seg in body:
        if prev_from is not None:
            assert seg["from"] >= prev_from, f"时间倒退: {seg}"
        prev_from = seg["from"]


def test_normal_alignment_keeps_all_text():
    """正常场景：text 与 timestamps 完全一致，全部保留且时间精确。"""
    text = "今天我们学习圣经。阿门。"
    ts = _chars(re.sub(r'[^\w]', '', text))
    body = _merge_char_timestamps(text, ts)
    assert _body_text(body) == text
    assert body[0]["from"] == 0.0
    _assert_monotonic(body)


def test_timestamps_exhausted_keeps_tail_text():
    """核心回归：timestamps 只覆盖前半段（复读/截断），后半段文本不能丢。

    旧实现：ts_offset >= len(ts_chars) 后 continue，尾部段落全部静默丢弃。
    新要求：尾部段落用上一段 to + 估算时长兜底，文本全量保留。
    """
    head = "耶和华是我的牧者。"          # 9 字
    tail = "我必不致缺乏。他使我躺卧在青草地上。领我在可安歇的水边。"  # 24 字
    text = head + tail
    ts = _chars(re.sub(r'[^\w]', '', head))  # 只有前半段的时间戳

    body = _merge_char_timestamps(text, ts)

    # 文本一个字都不能少
    assert _body_text(body) == text
    # 有时间戳的前半段用真实时间
    assert body[0]["from"] == 0.0
    # 兜底段时间不倒退、不早于上一段
    _assert_monotonic(body)
    # 兜底段时间要向前推进（to > from）
    for seg in body:
        assert seg["to"] > seg["from"], f"零长段: {seg}"


def test_mismatch_recovery_keeps_all_text():
    """find 失配（英文空格错位）场景：兜底推进后文本仍然全量保留。"""
    head = "约翰福音第三章十六节说，"
    en = "For God so loved the world that he gave his only begotten Son,"
    tail = "这就是爱。"
    text = head + en + tail
    # timestamps 只按中文逐字生成（英文部分完全没有对齐结果）
    ts = _chars(re.sub(r'[^\w]', '', head))

    body = _merge_char_timestamps(text, ts)
    assert _body_text(body) == text
    _assert_monotonic(body)


def test_empty_segments_and_zero_len():
    """纯标点段（清洗后 0 字）并入前一段，不独立成段、不丢字。"""
    text = "阿门。！！！"
    ts = _chars("阿门")
    body = _merge_char_timestamps(text, ts)
    assert _body_text(body) == "阿门。！！！"
    assert len(body) == 1
