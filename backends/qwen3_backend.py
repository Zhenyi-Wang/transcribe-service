"""Qwen3-ASR Transformers 后端实现"""
import gc
import time
import torch
from typing import Optional, List, Dict, Any

from .base import ASRBackend, TranscribeResult


class Qwen3Backend(ASRBackend):
    """Qwen3-ASR Transformers 后端"""

    def __init__(self, config):
        self.config = config
        self._model = None
        self._device = "cpu"

    @property
    def name(self) -> str:
        return "qwen3-asr"

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        """加载 Qwen3-ASR 模型"""
        from qwen_asr import Qwen3ASRModel

        # 优先尝试 GPU
        target_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        dtype = torch.bfloat16 if self.config.qwen3_dtype == "bfloat16" else torch.float16

        # 构建时间戳对齐模型参数
        forced_aligner_kwargs = None
        if self.config.qwen3_forced_aligner:
            forced_aligner_kwargs = dict(
                dtype=dtype,
                device_map=target_device,
            )

        try:
            self._model = Qwen3ASRModel.from_pretrained(
                self.config.qwen3_asr_model,
                dtype=dtype,
                device_map=target_device,
                max_new_tokens=self.config.qwen3_max_new_tokens,
                attn_implementation="flash_attention_2",
                forced_aligner=self.config.qwen3_forced_aligner if self.config.qwen3_forced_aligner else None,
                forced_aligner_kwargs=forced_aligner_kwargs,
            )
            self._device = target_device

        except Exception as e:
            # 显存不足时切回 CPU
            if "out of memory" in str(e).lower() and target_device.startswith("cuda"):
                torch.cuda.empty_cache()

                forced_aligner_kwargs = None
                if self.config.qwen3_forced_aligner:
                    forced_aligner_kwargs = dict(
                        dtype=dtype,
                        device_map="cpu",
                    )

                self._model = Qwen3ASRModel.from_pretrained(
                    self.config.qwen3_asr_model,
                    dtype=dtype,
                    device_map="cpu",
                    max_new_tokens=self.config.qwen3_max_new_tokens,
                    attn_implementation="flash_attention_2",
                    forced_aligner=self.config.qwen3_forced_aligner if self.config.qwen3_forced_aligner else None,
                    forced_aligner_kwargs=forced_aligner_kwargs,
                )
                self._device = "cpu"
            else:
                raise e

    def unload(self) -> None:
        """释放模型资源"""
        if self._model:
            del self._model
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def transcribe(self, audio_file: str, language: str = None) -> TranscribeResult:
        """执行转录"""
        if not self._model:
            raise RuntimeError("模型未加载，请先调用 load()")

        t_start = time.time()
        res = self._model.transcribe(
            audio=audio_file,
            language=language,
            return_time_stamps=bool(self.config.qwen3_forced_aligner),
        )
        t_total = time.time() - t_start

        # res 是 list[ASRTranscription]，取第一个结果
        asr_result = res[0] if res and len(res) > 0 else None

        # 提取文本
        text = asr_result.text if asr_result else ""

        # 提取语言
        detected_lang = "zh"
        if asr_result and hasattr(asr_result, "language"):
            from transcribe import LANG_MAP
            detected_lang = LANG_MAP.get(asr_result.language, "zh")

        # 提取时间戳（从 time_stamps 属性转换为统一格式）
        timestamps = None
        if asr_result and hasattr(asr_result, "time_stamps") and asr_result.time_stamps:
            timestamps = []
            for ts in asr_result.time_stamps:
                start = ts.start_time if hasattr(ts, "start_time") else ts[1]
                end = ts.end_time if hasattr(ts, "end_time") else ts[2]
                timestamps.append({
                    "text": ts.text if hasattr(ts, "text") else "",
                    "start": start,
                    "end": end,
                })

        # 计算 RTF
        audio_duration = 0
        if asr_result and hasattr(asr_result, "audio_duration"):
            audio_duration = asr_result.audio_duration
        rtf = t_total / audio_duration if audio_duration > 0 else 0

        return TranscribeResult(
            text=text,
            language=detected_lang,
            timestamps=timestamps,
            performance={
                "rtf": rtf,
                "time": t_total,
                "audio_duration": audio_duration,
            }
        )
