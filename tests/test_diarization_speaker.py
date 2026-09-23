from transcribe import _assign_speakers_to_timestamps, _merge_char_timestamps
from diarization.manager import SpeakerTurn


TS = lambda text, s, e: {"text": text, "start": s, "end": e}
TURNS = [SpeakerTurn(0.0, 10.0, 0), SpeakerTurn(10.0, 20.0, 1)]


def test_assign_by_overlap_majority():
    """字主体在 turn0、尾巴漂进 turn1 → 重叠面积最大者胜出"""
    ts = [TS("我", 9.0, 10.8)]
    out = _assign_speakers_to_timestamps(ts, TURNS)
    assert out[0]["speaker"] == 0


def test_assign_in_gap_nearest_turn():
    """完全落间隙的字（ASR 边界漂移）→ 归时间最近 turn"""
    turns = [SpeakerTurn(0.0, 9.0, 0), SpeakerTurn(11.0, 20.0, 1)]
    out = _assign_speakers_to_timestamps([TS("啊", 9.8, 9.9)], turns)
    assert out[0]["speaker"] == 0  # 中点 9.85 距 turn0 尾 0.85 < 距 turn1 头 1.15


def test_assign_out_of_range_nearest():
    ts = [TS("字", 25.0, 25.5)]
    out = _assign_speakers_to_timestamps(ts, TURNS)
    assert out[0]["speaker"] == 1


def test_original_dicts_not_mutated():
    ts = [TS("我", 1.0, 2.0)]
    out = _assign_speakers_to_timestamps(ts, TURNS)
    assert "speaker" not in ts[0] and out[0]["speaker"] == 0


def _char_ts(chars_spk):
    """[(char, speaker), ...] → 字级 timestamps（间隔 0.5s，带 speaker 键）"""
    return [
        {"text": c, "start": i * 0.5, "end": i * 0.5 + 0.4, "speaker": spk}
        for i, (c, spk) in enumerate(chars_spk)
    ]


def test_merge_splits_at_speaker_change_without_punct():
    """无标点长句跨说话人 → 段内边界拆分，两段各自单 speaker"""
    text = "今天天气不错我们出去玩吧好吧那就这样决定"  # 20 字，A 前 10 字 + B 后 10 字，无标点
    ts = _char_ts([(c, 0 if i < 10 else 1) for i, c in enumerate(text)])
    body = _merge_char_timestamps(text, ts)
    assert len(body) == 2
    assert body[0]["speaker"] == 0 and body[1]["speaker"] == 1
    assert body[0]["content"] == "今天天气不错我们出去" and body[1]["content"] == "玩吧好吧那就这样决定"
    # 对齐不变式：两段内容拼接（去标点后）== 原 clean 文本
    import re
    assert "".join(re.sub(r"[^\w]", "", b["content"]) for b in body) == text


def test_merge_fallback_segment_inherits_previous_speaker():
    """估算兜底段（时间戳耗尽）继承前一段 speaker——句号切两段，时间戳只覆盖第一段"""
    text = "前半句是甲说的。后半句超出时间戳范围走估算兜底。"  # 句号保证第一段独立成段
    ts = _char_ts([(c, 0) for c in "前半句是甲说的"])  # 时间戳只覆盖第一段（7 字）
    body = _merge_char_timestamps(text, ts)
    assert body[0].get("speaker") == 0
    assert len(body) >= 2
    assert all(seg.get("speaker") == 0 for seg in body[1:])  # 兜底段继承 last_speaker


def test_merge_baseline_unchanged_without_speaker_keys():
    """无 speaker 键的输入 → 输出不含 speaker 键（回归红线）"""
    text = "你好。今天天气怎么样？挺好的，谢谢。"
    ts_plain = [{"text": c, "start": i * 0.3, "end": i * 0.3 + 0.25}
                for i, c in enumerate(text.replace("。", "").replace("？", "").replace("，", ""))]
    import copy
    ts_copy = copy.deepcopy(ts_plain)
    body = _merge_char_timestamps(text, ts_plain)
    assert all("speaker" not in seg for seg in body)
    assert ts_plain == ts_copy  # 输入未被改动


def test_regroup_preserves_speaker():
    from transcribe import _regroup_fragments_by_space
    frags = [
        {"text": "नम", "start": 0.0, "end": 0.3, "speaker": 0},
        {"text": "स्ते ", "start": 0.3, "end": 0.6, "speaker": 0},
        {"text": "हाल ", "start": 5.0, "end": 5.4, "speaker": 1},
    ]
    words = _regroup_fragments_by_space(frags)
    assert [w.get("speaker") for w in words] == [0, 1]


def test_regroup_cross_speaker_word_takes_major_overlap():
    from transcribe import _regroup_fragments_by_space
    frags = [
        {"text": "ab", "start": 0.0, "end": 0.8, "speaker": 0},
        {"text": "cd ", "start": 0.8, "end": 1.0, "speaker": 1},
    ]
    words = _regroup_fragments_by_space(frags)
    assert words[0]["speaker"] == 0  # 0.0-1.0 词与 turn0 碎片重叠 0.8 > turn1 碎片 0.2


def test_segment_by_punct_flushes_at_speaker_change():
    from transcribe import _segment_by_punctuation
    ts = [
        {"text": "hello", "start": 0.0, "end": 0.5, "speaker": 0},
        {"text": "world", "start": 0.5, "end": 1.0, "speaker": 0},
        {"text": "bonjour", "start": 2.0, "end": 2.5, "speaker": 1},
    ]
    body = _segment_by_punctuation(ts, "hello world bonjour")
    assert len(body) == 2
    assert body[0]["speaker"] == 0 and body[1]["speaker"] == 1


def test_posthoc_align_threshold():
    from transcribe import _posthoc_align_speakers
    turns = [SpeakerTurn(0.0, 5.0, 0), SpeakerTurn(5.0, 10.0, 1)]
    body = [
        {"from": 1.0, "to": 4.0, "sid": 1, "location": 2, "content": "a", "music": 0},   # 全在 turn0
        {"from": 9.5, "to": 11.0, "sid": 2, "location": 2, "content": "b", "music": 0},  # 重叠 0.5 < 时长1.5 的一半 → -1
        {"from": 8.0, "to": 9.0, "sid": 3, "location": 2, "content": "c", "music": 0},   # 全在 turn1
    ]
    out = _posthoc_align_speakers(body, turns)
    assert out[0]["speaker"] == 0
    assert out[1]["speaker"] == -1  # 大范围落在 turns 覆盖外，最大重叠 0.5s < 1.5s*50%
    assert out[2]["speaker"] == 1


def test_dispatch_word_level_two_speakers():
    from transcribe import generate_subtitle_segments_from_timestamps
    text = "甲说这里是第一句话乙说这里是第二句话"  # 18 字无标点，前 8 甲后 10 乙
    ts = [{"text": c, "start": i * 0.5, "end": i * 0.5 + 0.4,
           "speaker": 0 if i < 8 else 1} for i, c in enumerate(text)]
    turns = [SpeakerTurn(0.0, 4.0, 0), SpeakerTurn(4.0, 9.0, 1)]
    body = generate_subtitle_segments_from_timestamps(text, ts, "zh", turns=turns)
    assert [seg["speaker"] for seg in body] == [0, 1]


def test_dispatch_single_speaker_degrades_to_plain():
    """单 turn（单人）→ body 不含 speaker 键"""
    from transcribe import generate_subtitle_segments_from_timestamps
    text = "只有一个人在说话的一段话没有标点符号"
    ts = [{"text": c, "start": i * 0.3, "end": i * 0.3 + 0.25, "speaker": 0}
          for i, c in enumerate(text)]
    turns = [SpeakerTurn(0.0, len(text) * 0.3 + 1.0, 0)]
    body = generate_subtitle_segments_from_timestamps(text, ts, "zh", turns=turns)
    assert all("speaker" not in seg for seg in body)


def test_dispatch_turns_none_keeps_baseline():
    from transcribe import generate_subtitle_segments_from_timestamps
    text = "你好。世界。"
    ts = [{"text": c, "start": i * 0.3, "end": i * 0.3 + 0.25}
          for i, c in enumerate("你好世界")]
    body = generate_subtitle_segments_from_timestamps(text, ts, "zh")
    assert all("speaker" not in seg for seg in body)


def test_dispatch_funasr_posthoc():
    """句级 item（funasr）→ post-hoc 对齐"""
    from transcribe import generate_subtitle_segments_from_timestamps
    ts = [{"text": "甲说的一句话", "start": 0.0, "end": 3.0, "speaker": 0},
          {"text": "乙说的一句话", "start": 5.0, "end": 8.0, "speaker": 1}]
    text = "甲说的一句话，乙说的一句话。"
    turns = [SpeakerTurn(0.0, 4.0, 0), SpeakerTurn(4.0, 10.0, 1)]
    body = generate_subtitle_segments_from_timestamps(text, ts, "zh", turns=turns)
    assert [seg["speaker"] for seg in body] == [0, 1]


def test_aggregate_speakers_excludes_negative():
    from transcribe import _aggregate_speakers
    body = [
        {"from": 0.0, "to": 2.0, "speaker": 0},
        {"from": 2.0, "to": 5.0, "speaker": 1},
        {"from": 5.0, "to": 6.0, "speaker": -1},
        {"from": 6.0, "to": 7.0},  # 无键
    ]
    assert _aggregate_speakers(body) == [
        {"id": 0, "duration": 2.0, "segments": 1},
        {"id": 1, "duration": 3.0, "segments": 1},
    ]


def test_merge_split_absorbs_tiny_speaker_drift():
    """句中短暂 speaker 漂移（2 字 1.0s）不拆分：整段保留，标重叠主导 speaker"""
    text = "前面是一很长的话中途两个字漂移后面继续说完整个句子内容"  # 27 字无标点
    spk_seq = [0] * 12 + [1] * 2 + [0] * 13  # 中间 2 字漂移
    ts = _char_ts([(c, spk) for c, spk in zip(text, spk_seq)])
    body = _merge_char_timestamps(text, ts)
    assert len(body) == 1  # 漂移被吸收，不拆
    assert body[0]["speaker"] == 0  # 主导 speaker
    assert body[0]["content"] == text


def test_merge_split_keeps_both_sides_when_large_enough():
    """两侧都足够长（≥4 字且 ≥1s）的变化点仍然拆分"""
    text = "甲说话说了很长的一段内容乙接话也非常长啊这对话真长啊"
    spk_seq = [0] * 14 + [1] * 12
    ts = _char_ts([(c, spk) for c, spk in zip(text, spk_seq)])
    body = _merge_char_timestamps(text, ts)
    assert len(body) == 2
    assert body[0]["speaker"] == 0 and body[1]["speaker"] == 1


def test_merge_split_no_zero_duration_segments():
    """拆分子段内部时间全部塌缩（start==end）时补最小时长，杜绝零时长字幕"""
    text = "甲说话说了很长的一段内容乙接话也非常长啊这对话真长啊"
    spk_seq = [0] * 14 + [1] * 12
    ts = _char_ts([(c, spk) for c, spk in zip(text, spk_seq)])
    # 第二子段（项 14-25）时间全部塌缩到 7.0
    for i in range(14, 26):
        ts[i]["start"] = ts[i]["end"] = 7.0
    body = _merge_char_timestamps(text, ts)
    assert len(body) == 2
    for seg in body:
        assert seg["to"] > seg["from"], f"zero-duration segment: {seg}"
