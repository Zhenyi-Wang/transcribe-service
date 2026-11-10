"""ASREngineClientBackend：把 transcribe-service 的转录转发到 asr-engine（HTTP）。

实现 ASRBackend 接口（base.py）。transcribe() 调 asr-engine 的 OpenAI 端点，
把 verbose_json 转成 base.TranscribeResult（含 timestamps [{text,start,end}]）。
load/unload 空操作——client 不占本地 GPU，模型由 asr-engine 管理。
"""
import json
import logging
import os
import threading
import time
from typing import Optional

import httpx

from backends.base import ASRBackend, TranscribeResult

_logger = logging.getLogger(__name__)

# asr-engine 崩溃（GGML abort 等 C 层错误）后由 run.sh 自动重启，窗口约 30s。
# ConnectError（进程不在监听）时等待重试可跨过窗口，把"任务失败"降级为"任务延迟"。
# 只重试连接拒绝类错误：ReadTimeout 可能请求已在处理，重试会重复转录。
_CONNECT_RETRY_WAIT_SEC = 20.0
_CONNECT_MAX_RETRIES = 2

# 能力探测（spec §6 引擎侧第 3 条）：GET /v1/progress 404 → 负缓存 10min 后重探；
# 200 → 正缓存持续（进程内）。瞬时错误不缓存，下次重探。
_CAPABILITY_NEG_TTL_SEC = 600.0
_CAPABILITY_PROBE_TIMEOUT_SEC = 5.0

# OpenAI language 全名(小写) → transcribe-service 短代码（与 transcribe.py LANG_MAP 一致）
_LANG_MAP = {
    "chinese": "zh", "english": "en", "japanese": "ja", "korean": "ko",
    "french": "fr", "german": "de", "spanish": "es", "portuguese": "pt",
    "russian": "ru", "arabic": "ar", "thai": "th", "vietnamese": "vi",
    "indonesian": "id", "italian": "it", "cantonese": "yue", "turkish": "tr",
    "hindi": "hi", "malay": "ms",
}


class UpstreamPausedError(RuntimeError):
    """asr-engine 暂停(503 + paused body)。需原样穿透到 server 端点转 503,不得转 error dict。"""

    def __init__(self, message: str, resume_at: str = None, paused_at: str = None):
        super().__init__(message)
        self.resume_at = resume_at
        self.paused_at = paused_at


class ASREngineClientBackend(ASRBackend):
    def __init__(self, config, timeout: float = 3600.0):
        # 与其它 backend 一致接收 config（工厂 create() 用 backend_class(config) 统一构造，零特殊分支）。
        # url/token 从 config 读。
        self._url = config.asr_engine_url.rstrip("/")
        self._token = getattr(config, "asr_engine_token", "") or ""
        self._timeout = timeout
        # capability 探测状态（probe_capability 持锁读写；探测罕见，锁内做短 HTTP 可接受）
        self._cap_lock = threading.Lock()
        self._cap_positive = False          # 探测到 /v1/progress → 进程内持续
        self._cap_negative_until = None     # 404 负缓存到期时刻（time.monotonic() 值）

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

    def transcribe(self, audio_file: str, language: str = None, context: str = None,
                   progress_cb=None, request_id: str = None) -> TranscribeResult:
        """转发转录到 asr-engine。

        progress_cb 接受但不使用（spec §5：asr-engine 进度经采样器轮询
        GET /v1/progress，本地回调不承载）；request_id 有则随 form 上传，
        供远端归属化取消与进度匹配（spec §6 引擎侧第 3 条）。
        """
        # 流式上传：直接把打开的文件对象传给 httpx，避免 f.read() 全量读入内存。
        # try 包裹整个 with open + httpx.post，确保 f 在 post 完成前一直存活（with 作用域内）。
        payload = None
        for attempt in range(_CONNECT_MAX_RETRIES + 1):
            try:
                with open(audio_file, "rb") as f:
                    files = {"file": (os.path.basename(audio_file), f)}
                    data = {
                        "model": "qwen3-asr-q4k",
                        "response_format": "verbose_json",
                        "timestamp_granularities": "word",
                    }
                    if context:
                        data["context"] = context
                    if request_id:
                        data["request_id"] = request_id
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
                break
            except httpx.ConnectError as e:
                if attempt < _CONNECT_MAX_RETRIES:
                    _logger.warning(
                        "asr-engine 不可达 (%s)，%.0fs 后重试（%d/%d；疑似崩溃自动重启中）",
                        self._url, _CONNECT_RETRY_WAIT_SEC, attempt + 1, _CONNECT_MAX_RETRIES,
                    )
                    time.sleep(_CONNECT_RETRY_WAIT_SEC)
                    continue
                raise RuntimeError(f"asr-engine 不可达 ({self._url}): {e}") from e
            except httpx.ReadTimeout as e:
                raise RuntimeError(f"asr-engine 转录超时 ({self._timeout}s): {e}") from e
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503:
                    # asr-engine 暂停信号（扁平 body：paused/resume_at/paused_at 顶层），
                    # 原样穿透到 server 端点转 503 paused，不得转 RuntimeError error dict
                    try:
                        body = e.response.json()
                    except Exception:
                        body = {}
                    if isinstance(body, dict) and body.get("paused") is True:
                        raise UpstreamPausedError(
                            body.get("detail") or "ASR 引擎暂停中",
                            resume_at=body.get("resume_at"), paused_at=body.get("paused_at"))
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
            performance={
                "audio_duration": payload.get("duration"),
                "processing_time": payload.get("processing_time"),
                # 远端引擎被归属化取消时顶层 "cancelled": true（引擎侧 format.py 已透出）；
                # 消费方（队列取消探针）据此把该计算按已取消处理（spec §6 引擎侧第 6 条）
                "cancelled": payload.get("cancelled") is True,
            },
        )

    def cancel_request(self, request_id: str) -> None:
        """归属化取消（spec §6 引擎侧第 3 条）：POST /admin/cancel，scoped json body。

        best-effort：传输失败或远端尚未部署该端点（404 等）记日志不抛——取消
        失败的降级（标志+join）由消费方处理。legacy 态本方法照常发送（本后端
        只经 probe_capability 如实报告能力，不自行吞请求）。
        """
        if not request_id:
            return
        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            resp = httpx.post(f"{self._url}/admin/cancel", json={"request_id": request_id},
                              headers=headers, timeout=10.0)
            if resp.status_code >= 400:
                _logger.warning("asr-engine 取消请求返回 %s（best-effort 忽略）", resp.status_code)
        except httpx.HTTPError as e:
            _logger.warning("asr-engine 取消请求失败（best-effort 忽略）: %s", e)

    def probe_capability(self) -> str:
        """探测远端引擎能力（spec §6 引擎侧第 3 条）：'progress+scoped' | 'legacy'。

        首次 GET /v1/progress（带 token）：200 → 正缓存（进程内持续）；
        404 → 负缓存 10min 后重探；其它（连接失败/5xx 等）→ legacy 且不缓存，
        下次调用再探。legacy 态由消费方（JobManager）关 percent、取消降级
        「标志+join」——本后端只如实报告。
        """
        if self._cap_positive:
            return "progress+scoped"
        with self._cap_lock:
            if self._cap_positive:
                return "progress+scoped"
            now = time.monotonic()
            if self._cap_negative_until is not None and now < self._cap_negative_until:
                return "legacy"
            headers = {}
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            try:
                resp = httpx.get(f"{self._url}/v1/progress", headers=headers,
                                 timeout=_CAPABILITY_PROBE_TIMEOUT_SEC)
            except httpx.HTTPError as e:
                _logger.warning("asr-engine 能力探测失败（不缓存，下次重探）: %s", e)
                return "legacy"
            if resp.status_code == 200:
                self._cap_positive = True
                self._cap_negative_until = None
                return "progress+scoped"
            if resp.status_code == 404:
                self._cap_negative_until = now + _CAPABILITY_NEG_TTL_SEC
                return "legacy"
            _logger.warning("asr-engine 能力探测异常状态 %s（不缓存，下次重探）", resp.status_code)
            return "legacy"
