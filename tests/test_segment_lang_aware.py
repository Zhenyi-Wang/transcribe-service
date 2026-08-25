"""语言感知分段回归测试（2026-08 印地语转录事故）。

背景：Qwen3-ForcedAligner 对天城文输出碎片级时间戳（基字符/组合符号，avg_len<=2），
generate_subtitle_segments_from_timestamps 误判为"字级"走 _merge_char_timestamps，
其分句硬编码中文标点（。！？，、；：），印地语无标点时分句失效且无强制拆分兜底，
导致整段 1502 字符输出为 1 条字幕（from=0.4 to=101.04），笔记只剩一个 [00:00]。

修复目标：非 CJK 语言的碎片级时间戳先按空格重组为词级再分段；
_segment_by_punctuation 认印地语句末标点 ।॥；
_merge_char_timestamps 对无标点超长段强制拆分。
样本取自 BV172846DEJT 实际 asr-engine 响应（94 秒印地语音频）。
"""

from transcribe import (
    _merge_char_timestamps,
    _segment_by_punctuation,
    generate_subtitle_segments_from_timestamps,
)

# 真实碎片样本（word, start, end）：天城文按基字符+组合符号对齐，尾空格标记词边界
_HINDI_FRAGMENTS = [
    ["म", 0.4, 0.48], ["ैं ", 0.48, 0.48], ["वह", 0.48, 0.8], ["ाँ ", 0.8, 0.8],
    ["स", 0.8, 0.8], ["े ", 0.8, 0.8], ["आ", 0.8, 0.96], [" ", 0.96, 0.96],
    ["गए", 0.96, 0.96], [" ", 0.96, 0.96], ["थ", 1.12, 1.28], ["े ", 1.28, 1.28],
    ["आप", 1.28, 1.76], [" ", 1.76, 1.76], ["सभ", 1.76, 1.76], ["ी ", 1.76, 1.76],
    ["क", 1.76, 1.76], ["ो ", 1.76, 1.76], ["believe", 1.76, 1.92], [" ", 1.92, 1.92],
    ["लिए", 2, 2.24], [" ", 2.24, 2.24], ["एप्प", 2.24, 2.24], [" ", 2.24, 2.24],
    ["पर", 2.24, 2.48], [" ", 2.48, 2.48], ["त", 2.8, 2.8], ["ो ", 2.8, 2.8],
    ["म", 2.88, 3.2], ["ैं ", 3.2, 3.2], ["भ", 3.2, 3.2], ["ी ", 3.2, 3.2],
    ["एक", 3.2, 3.52], [" ", 3.52, 3.52], ["YouTube", 3.52, 3.92], [" ", 3.92, 3.92],
    ["चैनल", 3.92, 4.24], [" ", 4.24, 4.24], ["थ", 4.24, 4.4], ["ा ", 4.4, 4.4],
]


def _fragments_to_timestamps(fragments):
    return [{"text": w, "start": s, "end": e} for w, s, e in fragments]


def _fragments_text(fragments):
    """碎片拼接并压缩连续空格，得到与 timestamps 对应的转录文本。"""
    import re

    return re.sub(r" {2,}", " ", "".join(w for w, _, _ in fragments))


def _repeat_fragments(fragments, times, step):
    """重复样本构造长输入：时间戳按块递增，模拟长音频。"""
    out = []
    for i in range(times):
        offset = i * step
        out.extend([[w, s + offset, e + offset] for w, s, e in fragments])
    return out


def _body_text(body):
    return "".join(seg["content"] for seg in body)


def _assert_monotonic(body):
    prev_from = None
    for seg in body:
        if prev_from is not None:
            assert seg["from"] >= prev_from, f"时间倒退: {seg}"
        prev_from = seg["from"]


def test_hindi_fragments_not_single_segment():
    """核心回归：印地语碎片级时间戳不得输出为单条超长字幕。

    旧实现：avg_len<=2 误入 _merge_char_timestamps，无标点不拆分 → 1 条。
    新要求：按空格重组词级后有强制拆分兜底，输出多段、文本全保留。
    """
    fragments = _repeat_fragments(_HINDI_FRAGMENTS, 8, 30.0)
    text = _fragments_text(fragments)
    timestamps = _fragments_to_timestamps(fragments)

    body = generate_subtitle_segments_from_timestamps(text, timestamps, "hi")

    assert len(body) > 3, f"仍为单条/过少分段: {len(body)} 条"
    # 文本全保留（去空白后比对，分段拼接可能重组空格）
    import re

    assert re.sub(r"\s", "", _body_text(body)) == re.sub(r"\s", "", text)
    _assert_monotonic(body)
    for seg in body:
        assert seg["to"] >= seg["from"], f"零长段: {seg}"
        # 单段不得超长（max_len=20，强制拆分上限 60）
        assert len(seg["content"]) <= 60 + 5, f"超长段: len={len(seg['content'])}"


def test_danda_is_sentence_end():
    """印地语句末标点 । 触发分段（_segment_by_punctuation 此前只认 .!?）。"""
    words = [
        {"text": "मैं ", "start": 0.0, "end": 0.5},
        {"text": "आया ", "start": 0.5, "end": 1.0},
        {"text": "।", "start": 1.0, "end": 1.1},
        {"text": "धन्यवाद ", "start": 1.2, "end": 1.8},
        {"text": "।", "start": 1.8, "end": 1.9},
    ]
    text = "मैं आया। धन्यवाद।"
    body = _segment_by_punctuation(words, text)
    assert len(body) == 2
    assert body[0]["to"] == 1.1


def test_merge_char_forces_split_without_punct():
    """_merge_char_timestamps 无标点超长段强制拆分（防御其他无空格语言）。"""
    text = "这是一段没有任何标点的超长中文文本" * 10  # 160 字，无标点
    ts = [{"text": ch, "start": i * 0.3, "end": (i + 1) * 0.3} for i, ch in enumerate(text)]
    body = _merge_char_timestamps(text, ts)
    assert len(body) > 1, "无标点超长段未强制拆分"
    assert _body_text(body) == text
    _assert_monotonic(body)
    for seg in body:
        assert len(seg["content"]) <= 60 + 5, f"超长段: len={len(seg['content'])}"


def test_cjk_char_level_still_merges():
    """回归保护：中文字级时间戳仍走 _merge_char_timestamps 正常分句。"""
    text = "今天我们学习圣经。明天我们一起祷告。后天我们赞美主。"
    ts = [{"text": ch, "start": i * 0.3, "end": (i + 1) * 0.3} for i, ch in enumerate(text) if ch not in "。"]
    body = generate_subtitle_segments_from_timestamps(text, ts, "zh")
    assert _body_text(body) == text
    # 三句均 >= min_len(5) 字，各自独立成段
    assert len(body) == 3
