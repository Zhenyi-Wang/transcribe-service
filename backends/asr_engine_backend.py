"""ASREngineClientBackend：把 transcribe-service 的转录转发到 asr-engine（HTTP）。

实现 ASRBackend 接口（base.py）。transcribe() 调 asr-engine 的 OpenAI 端点，
把 verbose_json 转成 base.TranscribeResult（含 timestamps [{text,start,end}]）。
load/unload 空操作——client 不占本地 GPU，模型由 asr-engine 管理。
"""
import json
import logging
import os
from typing import Optional

import httpx

from backends.base import ASRBackend, TranscribeResult

_logger = logging.getLogger(__name__)

# OpenAI language 全名(小写) → transcribe-service 短代码（与 transcribe.py LANG_MAP 一致）
_LANG_MAP = {
    "chinese": "zh", "english": "en", "japanese": "ja", "korean": "ko",
    "french": "fr", "german": "de", "spanish": "es", "portuguese": "pt",
    "russian": "ru", "arabic": "ar", "thai": "th", "vietnamese": "vi",
    "indonesian": "id", "italian": "it", "cantonese": "yue", "turkish": "tr",
    "hindi": "hi", "malay": "ms",
}


class ASREngineClientBackend(ASRBackend):
    def __init__(self, config, timeout: float = 3600.0):
        # 与其它 backend 一致接收 config（工厂 create() 用 backend_class(config) 统一构造，零特殊分支）。
        # url/token 从 config 读。
        self._url = config.asr_engine_url.rstrip("/")
        self._token = getattr(config, "asr_engine_token", "") or ""
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "asr-engine"

    @property
    def device(self) -> str:
        return "remote"

    def load(self) -> None:
        pass  # 模型由 asr-engine 管理

    def unload(self) -> None:
        pass

    def transcribe(self, audio_file: str, language: str = None) -> TranscribeResult:
        # 流式上传：直接把打开的文件对象传给 httpx，避免 f.read() 全量读入内存。
        # try 包裹整个 with open + httpx.post，确保 f 在 post 完成前一直存活（with 作用域内）。
        try:
            with open(audio_file, "rb") as f:
                files = {"file": (os.path.basename(audio_file), f)}
                data = {
                    "model": "qwen3-asr-q4k",
                    "response_format": "verbose_json",
                    "timestamp_granularities": "word",
                }
                headers = {}
                if self._token:
                    headers["Authorization"] = f"Bearer {self._token}"

                resp = httpx.post(
                    f"{self._url}/v1/audio/transcriptions",
                    files=files, data=data, headers=headers,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                payload = resp.json()
        except httpx.ConnectError as e:
            raise RuntimeError(f"asr-engine 不可达 ({self._url}): {e}") from e
        except httpx.ReadTimeout as e:
            raise RuntimeError(f"asr-engine 转录超时 ({self._timeout}s): {e}") from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"asr-engine 返回 {e.response.status_code}: {e.response.text[:200]}"
            ) from e
        except httpx.HTTPError as e:
            # 兜底：ReadError/RemoteProtocolError 等其它 httpx 网络/协议异常
            raise RuntimeError(f"asr-engine 请求失败 ({self._url}): {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"asr-engine 返回非 JSON 响应: {e}") from e

        lang_full = (payload.get("language") or "").lower()
        short = _LANG_MAP.get(lang_full)
        if lang_full and not short:
            _logger.warning(
                "未知语言 %r 未在 _LANG_MAP 中，回退到 'zh'（与 gguf_backend LANG_MAP 行为一致）；"
                "将来应提取共享 LANG_MAP 消除偏差",
                lang_full,
            )
            short = "zh"
        language = short or "zh"

        timestamps = [
            {"text": w["word"], "start": float(w["start"]), "end": float(w["end"])}
            for w in payload.get("words") or []
        ]
        return TranscribeResult(
            text=payload.get("text", ""),
            language=language,
            timestamps=timestamps,
            performance={"audio_duration": payload.get("duration"), "processing_time": payload.get("processing_time")},
        )
