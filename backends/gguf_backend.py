"""GGUF 后端实现"""
import os
import sys
import time
import io
import threading
import contextlib
from typing import Optional, List, Dict, Any

from .base import ASRBackend, TranscribeResult

# 进程内 pending 取消集合参数（spec §6 引擎侧第 3/4 条 gguf 同构）：
# cancel_request 在计算尚未开始（未持 transcribe_lock）时先登记 request_id；
# transcribe 持锁挂接共享库取消事件时命中即置位并移除。条目带时间戳，
# TTL 过期清理 + 容量上限逐最旧，均在登记时（校验时）pruning。
_PENDING_CANCEL_TTL_SEC = 300.0
_PENDING_CANCEL_MAX = 64


class GGUFBackend(ASRBackend):
    """GGUF + llama.cpp 后端"""

    def __init__(self, config, clock=None):
        self.config = config
        self._engine = None
        self._device = "cpu"
        self._transcribe_lock = threading.Lock()
        self._clock = clock or time.monotonic
        self._pending_cancels: Dict[str, float] = {}  # request_id -> 登记时刻（clock 值）
        self._pending_lock = threading.Lock()
        # 当前活跃挂接（transcribe 持 transcribe_lock 期间）：request_id -> 取消事件。
        # 使 cancel_request 对已在转录中的计算可直达置位（spec §6 引擎侧第 3 条
        # 「活跃归属匹配 → 置位」的 gguf 同构）。
        self._active_cancels: Dict[str, threading.Event] = {}

    def cancel_request(self, request_id: str) -> None:
        """归属化取消（spec §6 引擎侧第 3 条 gguf 同构）：活跃归属命中即置位，否则入 pending 集合。

        - 计算已在转录中（transcribe 持 transcribe_lock 期间挂接的活跃事件）：
          直接置位，共享库 chunk 边界下一轮检查即中止（performance["cancelled"]=True）。
        - 计算尚未开始（排队等 transcribe_lock）：登记进 pending 集合（带时间戳，
          TTL/容量 pruning），挂接时命中即置位并移除。
        """
        if not request_id:
            return
        now = self._clock()
        with self._pending_lock:
            active = self._active_cancels.get(request_id)
            if active is not None:
                active.set()          # 活跃归属命中 → 置位，不入集合
                return
            self._prune_pending(now)
            self._pending_cancels[request_id] = now
            while len(self._pending_cancels) > _PENDING_CANCEL_MAX:
                oldest = min(self._pending_cancels, key=self._pending_cancels.get)
                del self._pending_cancels[oldest]

    def _prune_pending(self, now: float) -> None:
        """TTL 过期清理（须持 _pending_lock 调用）。"""
        expired = [rid for rid, ts in self._pending_cancels.items()
                   if now - ts >= _PENDING_CANCEL_TTL_SEC]
        for rid in expired:
            del self._pending_cancels[rid]

    def probe_capability(self) -> str:
        """gguf 恒 progress+scoped：本地 chunk 回调给真实 percent，归属取消进程内闭环。"""
        return "progress+scoped"

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

    def transcribe(self, audio_file: str, language: str = None, context: str = None,
                   progress_cb=None, request_id: str = None) -> TranscribeResult:
        """执行转录（互斥，防止并发访问 llama.cpp 导致段错误）

        progress_cb 透传共享库 QwenASREngine.transcribe（chunk 完成边界回调，
        spec §5 transcribing 行）；request_id 用于归属化取消：持 transcribe_lock
        后挂接共享库取消事件，pending 集合命中即置位并移除（spec §6 引擎侧第
        2/3 条 gguf 同构）。
        """
        if not self._engine:
            raise RuntimeError("模型未加载，请先调用 load()")

        if not self._transcribe_lock.acquire(blocking=True, timeout=3600):
            raise RuntimeError("转录排队超时（前一个任务占用超过 3600 秒）")
        # 诊断日志：记录进入互斥区。用 stderr（redirect_stdout 只吞 stdout，不吞 stderr），
        # 与 GGML_ABORT backtrace 同流，方便看并发/请求边界与崩溃的时序关系。
        print(f"[DIAG-GGUF-ENTER] file={audio_file} pid={os.getpid()} lock_acquired=ok",
              file=sys.stderr, flush=True)
        # 局部引用（评审 Important 修复）：unload() 不持本锁，排队窗口内可能把
        # self._engine 置 None。全程使用局部引用 + 收尾 AttributeError 守卫，
        # 保证任何失败路径都先清事件、后释放锁——锁永不因解引用 None 泄漏成
        # 3600s 排队死锁。清事件必须在 release() 之前：先释放会让本 finally 的
        # set_cancel_event(None) 覆盖下一次转录刚装上的取消事件。
        engine = self._engine
        if engine is None:
            self._transcribe_lock.release()
            raise RuntimeError("模型未加载（排队期间被卸载），请重试")
        try:
            # 挂接 per-call 取消事件（锁内；spec §6 引擎侧第 2 条同构）：
            # pending 集合命中 → 挂接即置位（消费即移除）；并登记活跃归属，
            # 使转录中的 cancel_request 可直达置位。
            cancel_event = threading.Event()
            with self._pending_lock:
                self._prune_pending(self._clock())
                if request_id and self._pending_cancels.pop(request_id, None) is not None:
                    cancel_event.set()
                if request_id:
                    self._active_cancels[request_id] = cancel_event
            engine.set_cancel_event(cancel_event)

            t_start = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                result = engine.transcribe(audio_file, language=language,
                                           context=context, progress_cb=progress_cb)
            t_total = time.time() - t_start
        finally:
            # 释放前清除挂接：活跃归属移除 + 共享库恢复不可取消（防事件泄漏到下一次调用）。
            # AttributeError 守卫：engine 对象被并发 unload 破坏等极端情况下保底释放锁。
            with self._pending_lock:
                if request_id:
                    self._active_cancels.pop(request_id, None)
            try:
                engine.set_cancel_event(None)
            except AttributeError:
                pass
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
