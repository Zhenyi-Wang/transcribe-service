"""GGUF 后端实现"""
import os
import time
from typing import Optional, List, Dict, Any

from .base import ASRBackend, TranscribeResult


class GGUFBackend(ASRBackend):
    """GGUF + llama.cpp 后端"""

    def __init__(self, config):
        self.config = config
        self._engine = None
        self._device = "cpu"

    @property
    def name(self) -> str:
        return "gguf"

    @property
    def device(self) -> str:
        return self._device

    def _get_model_filename(self) -> str:
        """根据配置获取 ASR 模型文件名"""
        precision = self.config.gguf_asr_precision  # f16, q8_0, q4_k, q4_k_m
        return f"qwen3_asr_llm.{precision}.gguf"

    def load(self) -> None:
        """加载 GGUF 模型"""
        from qwen_asr_gguf.inference import QwenASREngine, ASREngineConfig, AlignerConfig

        model_dir = self.config.gguf_model_dir
        asr_filename = self._get_model_filename()

        cfg = ASREngineConfig(
            model_dir=model_dir,
            llm_fn=asr_filename,
            encoder_frontend_fn="qwen3_asr_encoder_frontend.int4.onnx",
            encoder_backend_fn="qwen3_asr_encoder_backend.int4.onnx",
            onnx_provider='CUDA' if self.config.gguf_use_cuda else 'CPU',
            llm_use_gpu=self.config.gguf_use_cuda,
            enable_aligner=True,
            align_config=AlignerConfig(
                model_dir=model_dir,
                llm_fn="qwen3_aligner_llm.q4_k.gguf",  # Aligner 只有 Q4_K 可用
                encoder_frontend_fn="qwen3_aligner_encoder_frontend.int4.onnx",
                encoder_backend_fn="qwen3_aligner_encoder_backend.int4.onnx",
                onnx_provider='CUDA' if self.config.gguf_use_cuda else 'CPU',
                llm_use_gpu=self.config.gguf_use_cuda,
            ),
            verbose=False,
        )
        self._engine = QwenASREngine(cfg)
        self._device = "cuda" if self.config.gguf_use_cuda else "cpu"

    def unload(self) -> None:
        """释放模型资源"""
        if self._engine:
            self._engine.shutdown()
            self._engine = None

    def transcribe(self, audio_file: str, language: str = None) -> TranscribeResult:
        """执行转录"""
        if not self._engine:
            raise RuntimeError("模型未加载，请先调用 load()")

        t_start = time.time()
        result = self._engine.transcribe(audio_file, language=language or "Chinese")
        t_total = time.time() - t_start

        # 转换时间戳格式
        timestamps = None
        if result.alignment and result.alignment.items:
            timestamps = [
                {"text": item.text, "start": item.start_time, "end": item.end_time}
                for item in result.alignment.items
            ]

        # 从 performance 计算 RTF
        audio_duration = result.performance.get("audio_duration", 0) if result.performance else 0
        rtf = t_total / audio_duration if audio_duration > 0 else 0

        return TranscribeResult(
            text=result.text,
            language=language or "zh",
            timestamps=timestamps,
            performance={
                "rtf": rtf,
                "time": t_total,
                "audio_duration": audio_duration,
                **(result.performance or {})
            }
        )
