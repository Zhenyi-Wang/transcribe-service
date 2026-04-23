"""FunASR 后端实现"""
import gc
import time
from typing import Optional, List, Dict, Any

from .base import ASRBackend, TranscribeResult


class FunASRBackend(ASRBackend):
    """FunASR 后端（原版实现）"""

    def __init__(self, config):
        self.config = config
        self._model = None
        self._device = "cpu"

    @property
    def name(self) -> str:
        return "funasr"

    @property
    def device(self) -> str:
        return self._device

    def _build_model_kwargs(self, device: str) -> dict:
        """构建模型参数"""
        model_kwargs = {
            "model": self.config.funasr_model,
            "vad_model": self.config.funasr_vad_model,
            "punc_model": self.config.funasr_punc_model,
            "device": device,
            "disable_update": True,
            "trust_remote_code": True,
        }

        # 如果启用时间戳，添加相应参数
        if self.config.enable_timestamp:
            model_kwargs["sentence_timestamp"] = True

        return model_kwargs

    def load(self) -> None:
        """加载 FunASR 模型"""
        from funasr import AutoModel

        # 优先尝试 GPU
        target_device = "cuda" if self._is_cuda_available() else "cpu"

        try:
            model_kwargs = self._build_model_kwargs(target_device)
            self._model = AutoModel(**model_kwargs)
            self._device = target_device

        except Exception as e:
            # 显存不足时切回 CPU
            if "out of memory" in str(e).lower() and target_device == "cuda":
                import torch
                torch.cuda.empty_cache()

                model_kwargs = self._build_model_kwargs("cpu")
                self._model = AutoModel(**model_kwargs)
                self._device = "cpu"
            else:
                raise e

    def _is_cuda_available(self) -> bool:
        """检查 CUDA 是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def unload(self) -> None:
        """释放模型资源"""
        if self._model:
            del self._model
            self._model = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def transcribe(self, audio_file: str, language: str = None) -> TranscribeResult:
        """执行转录"""
        if not self._model:
            raise RuntimeError("模型未加载，请先调用 load()")

        t_start = time.time()
        res = self._model.generate(
            input=audio_file,
            batch_size_s=self.config.batch_size_s,
            disable_pbar=True,
        )
        t_total = time.time() - t_start

        # 提取文本
        text = res[0].get("text", "") if res and len(res) > 0 else ""

        # 提取时间戳
        timestamps = None
        if res and len(res) > 0 and "sentence_info" in res[0]:
            timestamps = []
            for seg in res[0]["sentence_info"]:
                if isinstance(seg, dict) and "text" in seg:
                    timestamps.append({
                        "text": seg["text"],
                        "start": seg.get("start", 0) / 1000,  # ms -> s
                        "end": seg.get("end", 0) / 1000,
                    })

        # 检测语言
        detected_lang = self._detect_language(text)

        # 估算 RTF（FunASR 不直接提供音频时长）
        rtf = 0.0  # 需要外部计算

        return TranscribeResult(
            text=text,
            language=detected_lang,
            timestamps=timestamps,
            performance={
                "rtf": rtf,
                "time": t_total,
            }
        )

    def _detect_language(self, text: str) -> str:
        """从文本检测语言"""
        import re

        if not text:
            return "zh"

        # 统计中文字符比例
        chinese_chars = len(re.findall(r'[一-鿿]', text))
        total_chars = len(re.sub(r'\s', '', text))

        if total_chars == 0:
            return "zh"

        chinese_ratio = chinese_chars / total_chars

        if chinese_ratio > self.config.chinese_ratio_threshold:
            return "zh"
        elif re.match(r'^[a-zA-Z\s\d\W]+$', text):
            return "en"
        elif re.search(r'[぀-ゟ゠-ヿ]', text):
            return "ja"
        elif re.search(r'[가-힯]', text):
            return "ko"
        else:
            return "zh"
