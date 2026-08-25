"""印地语实际音频的端到端分段集成测试。

音频：test/hindi_BV172846DEJT.m4s（BV172846DEJT，94 秒印地语，2026-08 转录事故样本）。
链路：asr-engine /v1/audio/transcriptions (verbose_json, word) → generate_subtitle_segments_from_timestamps。

ASR 为 LLM 推理，两次输出不完全一致，断言用结构性约束（段数/文本保留/时间单调）
而非精确值。依赖本机 asr-engine（127.0.0.1:31090），不可达时跳过。
"""
import re
from pathlib import Path

import pytest

from transcribe import generate_subtitle_segments_from_timestamps

AUDIO = Path(__file__).parent.parent / "test" / "hindi_BV172846DEJT.m4s"
ASR_ENGINE_URL = "http://127.0.0.1:31090"


def _asr_engine_available() -> bool:
    try:
        import httpx

        httpx.get(f"{ASR_ENGINE_URL}/health", timeout=2)
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def hindi_result():
    if not AUDIO.exists():
        pytest.skip(f"测试音频不存在: {AUDIO}")
    if not _asr_engine_available():
        pytest.skip("asr-engine 不可达（tmux asr 未运行）")

    import httpx

    with open(AUDIO, "rb") as f:
        resp = httpx.post(
            f"{ASR_ENGINE_URL}/v1/audio/transcriptions",
            files={"file": (AUDIO.name, f)},
            data={
                "model": "qwen3-asr-q4k",
                "response_format": "verbose_json",
                "timestamp_granularities": "word",
            },
            timeout=300,
        )
    resp.raise_for_status()
    payload = resp.json()
    timestamps = [
        {"text": w["word"], "start": float(w["start"]), "end": float(w["end"])}
        for w in payload.get("words") or []
    ]
    return payload["text"], timestamps


def test_hindi_audio_segments_not_single(hindi_result):
    """实际音频端到端：不得再输出单条覆盖全片的字幕。"""
    text, timestamps = hindi_result
    body = generate_subtitle_segments_from_timestamps(text, timestamps, "hi")

    assert len(body) > 5, f"分段过少: {len(body)} 条"
    got = re.sub(r"\s", "", "".join(seg["content"] for seg in body))
    assert got == re.sub(r"\s", "", text), "分段文本与转录文本不一致（丢字/错位）"

    prev_from = None
    # 强制拆分在整词边界触发（total_len >= max_len*3 时 flush），
    # 段长上限 = max_len*3 + 单个词的长度余量
    from config import config as _config

    seg_limit = _config.max_segment_length * 3 + 15
    for seg in body:
        assert seg["to"] >= seg["from"], f"零长段: {seg}"
        assert len(seg["content"]) <= seg_limit, f"超长段: len={len(seg['content'])}"
        if prev_from is not None:
            assert seg["from"] >= prev_from, f"时间倒退: {seg}"
        prev_from = seg["from"]


def test_hindi_audio_timestamps_cover_duration(hindi_result):
    """首段接近 0 开始，末段结束时间接近音频时长（分段时间戳真实可用）。"""
    text, timestamps = hindi_result
    body = generate_subtitle_segments_from_timestamps(text, timestamps, "hi")

    assert body[0]["from"] < 2.0, f"首段时间戳异常: {body[0]['from']}"
    last_to = body[-1]["to"]
    # aligner 末碎片 end 允许轻微超界，但必须在量级上接近真实时长（~94s）
    assert 85.0 < last_to < 120.0, f"末段时间戳异常: {last_to}"
    # 分段应铺开时间轴：中间段起止不得全部挤在同一时刻
    distinct_starts = {seg["from"] for seg in body}
    assert len(distinct_starts) > len(body) // 2, "段起始时间未铺开，疑似时间戳未生效"
