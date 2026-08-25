# coding=utf-8
import numpy as np
from typing import List, Optional

from .schema import ForcedAlignItem

SUPPORTED_LANGUAGES: List[str] = [
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian"
]

def normalize_language_name(language: str) -> str:
    """
    将语言名称归一化为 Qwen3-ASR 使用的标准格式：
    首字母大写，其余小写（例如 'cHINese' -> 'Chinese'）。
    """
    if language is None:
        raise ValueError("language is None")
    s = str(language).strip()
    if not s:
        raise ValueError("language is empty")
    return s[:1].upper() + s[1:].lower()

def validate_language(language: str) -> None:
    """
    验证语言是否在支持列表中。
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}")


def clamp_to_audio_end(items: List, audio_end_sec: float) -> List:
    """把超界的起止时间 clamp 到本块音频真实末尾。

    TS 帧索引是对齐 LLM 的概率预测（argmax），尾部注意力塌缩、末块 ASR 在
    np.pad 音频上跑导致时间尺度偏移，末词 end 可超出音频实际时长数秒
    （2026-08：94.25s 音频出现 101.04/98.72）；fix_timestamps 只修单调性
    无上界。ForcedAlignItem 为 frozen dataclass，超界项重建而非原地修改。
    """
    out: List[ForcedAlignItem] = []
    for it in items:
        if it.end_time <= audio_end_sec and it.start_time <= audio_end_sec:
            out.append(it)
        else:
            # min 保序：clamp 后 start <= end 依然成立
            out.append(ForcedAlignItem(
                text=it.text,
                start_time=min(it.start_time, audio_end_sec),
                end_time=min(it.end_time, audio_end_sec),
            ))
    return out

