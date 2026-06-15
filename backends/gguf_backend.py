"""GGUF 后端实现"""
import os
import sys
import time
import io
import threading
import contextlib
from typing import Optional, List, Dict, Any

from .base import ASRBackend, TranscribeResult


class GGUFBackend(ASRBackend):
    """GGUF + llama.cpp 后端"""

    def __init__(self, config):
        self.config = config
        self._engine = None
        self._device = "cpu"
        self._transcribe_lock = threading.Lock()

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
        """执行转录（互斥，防止并发访问 llama.cpp 导致段错误）"""
        if not self._engine:
            raise RuntimeError("模型未加载，请先调用 load()")

        if not self._transcribe_lock.acquire(blocking=True, timeout=3600):
            raise RuntimeError("转录排队超时（前一个任务占用超过 3600 秒）")
        # 诊断日志：记录进入互斥区。用 stderr（redirect_stdout 只吞 stdout，不吞 stderr），
        # 与 GGML_ABORT backtrace 同流，方便看并发/请求边界与崩溃的时序关系。
        print(f"[DIAG-GGUF-ENTER] file={audio_file} pid={os.getpid()} lock_acquired=ok",
              file=sys.stderr, flush=True)
        try:
            t_start = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                result = self._engine.transcribe(audio_file, language=language)
            t_total = time.time() - t_start
        finally:
            self._transcribe_lock.release()

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

        # 检测语言
        detected_lang = "zh"
        if result.language:
            from transcribe import LANG_MAP
            detected_lang = LANG_MAP.get(result.language, "zh")

        return TranscribeResult(
            text=result.text,
            language=detected_lang,
            timestamps=timestamps,
            performance={
                "rtf": rtf,
                "time": t_total,
                "audio_duration": audio_duration,
                **(result.performance or {})
            }
        )
