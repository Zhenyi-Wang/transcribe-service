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


def test_fallback_clamped_to_audio_duration():
    """兜底估算段钳制到音频总时长（2026-09 祷告会事故）。

    场景：音乐复读导致时间戳耗尽，兜底按 0.3s/字推进，曾一路超出音频
    实际时长 45s（尾部碎片时间戳到 8878.44s，音频只有 8832.98s）。
    要求：传 audio_duration 后兜底段 to 不超过音频末尾；已越过末尾的段
    钉在音频末尾（文本不丢）。
    """
    head = "耶和华是我的牧者。"          # 9 字，有真实时间戳
    tail = "你们的喜乐，你们的荣耀，" * 20  # 120 字，无时间戳走兜底
    text = head + tail
    ts = _chars(re.sub(r'[^\w]', '', head))

    body = _merge_char_timestamps(text, ts, audio_duration=13.0)

    # 文本一个不丢
    assert _body_text(body) == text
    # 所有段时间不超音频末尾
    for seg in body:
        assert seg["to"] <= 13.0 + 1e-9, f"超界段: {seg}"
    # 越过末尾的兜底段钉在末尾（from == to == 13.0）
    pinned = [seg for seg in body if seg["from"] >= 13.0]
    assert pinned, "应存在钉在末尾的兜底段"
    for seg in pinned:
        assert seg["from"] == seg["to"] == 13.0


def test_fallback_without_duration_unchanged():
    """不传 audio_duration 时行为与旧版一致（向后兼容）：兜底自由推进。"""
    head = "耶和华是我的牧者。"
    tail = "你们的喜乐，你们的荣耀，" * 20
    text = head + tail
    ts = _chars(re.sub(r'[^\w]', '', head))

    body = _merge_char_timestamps(text, ts)
    assert _body_text(body) == text
    _assert_monotonic(body)
    # 旧版无钳制：兜底段应向前推进且末段 to 明显大于前段（自由推进特征）
    assert body[-1]["to"] > body[0]["to"]
