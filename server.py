import os
import time
import math
import asyncio
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Literal, Optional, Union

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware

from config import config
from downloaders import BilibiliDownloader
from downloaders import DouyinDownloader
from downloaders.base import DownloadCancelled
from transcribe import TranscriptionService, diarize_only, diarize_merge_subtitle
from backends.asr_engine_backend import UpstreamPausedError
from logger_config import setup_logger
from cache_manager import cache_manager
from pause_manager import PauseManager, MAX_PAUSE_HOURS
from diarization import worker as diarization_mgr
from jobs import JobManager, QueueFullError
import webdav_archive
from pydantic import BaseModel, Field

# 设置 HuggingFace 缓存目录和日志
os.environ['HF_HOME'] = str(Path.home() / ".cache/huggingface")
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# 使用统一的logger配置
logger = setup_logger('server')


class ModelManager:
    """模型管理器 - 使用 BackendFactory 创建后端"""

    def __init__(self):
        self._backend = None
        self.lock = threading.Lock()
        self.last_active_time = 0
        self._in_use = 0

    def load_model_if_needed(self):
        """按需加载模型"""
        self.last_active_time = time.time()

        if self._backend is None:
            with self.lock:
                if self._backend is None:
                    from backends import BackendFactory
                    self._backend = BackendFactory.create(config)
                    logger.info(f"正在加载后端: {self._backend.name}...")
                    try:
                        self._backend.load()
                        logger.info(f"后端加载成功！设备: {self._backend.device}")
                    except Exception:
                        self._backend = None
                        raise

        return self._backend

    def unload_model(self):
        """释放模型资源"""
        with self.lock:
            if self._backend is not None:
                logger.info(f"闲置超时，释放后端资源: {self._backend.name}...")
                self._backend.unload()
                self._backend = None

    def acquire(self):
        """标记模型正在使用"""
        self._in_use += 1

    def release(self):
        """标记模型使用完毕"""
        self._in_use = max(0, self._in_use - 1)

    @property
    def in_use(self):
        """是否有转录任务正在使用模型"""
        return self._in_use > 0

    @property
    def backend(self):
        """获取当前后端名称"""
        if self._backend:
            return self._backend.name
        return config.backend_name

_ALLOWED_AUDIO_EXTS = {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wma'}


def _safe_audio_ext(filename: str) -> str:
    """扩展名白名单（与 generate_safe_filename 同源）；缺失/非法 → .tmp"""
    ext = Path(filename or "").suffix.lower()
    return ext if ext in _ALLOWED_AUDIO_EXTS else '.tmp'


def generate_safe_filename(filename: str) -> str:
    """生成安全的临时文件名"""
    if not filename:
        filename = "audio"

    ext = _safe_audio_ext(filename)

    # 使用UUID + 时间戳确保唯一性
    unique_id = str(uuid.uuid4())[:8]
    timestamp = int(time.time())

    return f"temp_{timestamp}_{unique_id}{ext}"

def get_temp_dir():
    """获取并创建临时目录"""
    temp_dir = Path("tmp")
    temp_dir.mkdir(exist_ok=True)
    return temp_dir


def cleanup_stale_tmp_files(max_age_hours: int = 24, protected: Optional[set] = None):
    """清理超过指定时间的 tmp 文件（jobs 队列活跃感知，spec §6 终态清理）。

    protected：JobManager 活跃 upload 登记路径集——命中即跳过；其 `.job-*`
    硬链接别名（_audio_lease 派生，继承旧 mtime）经 inode 匹配同步豁免（评审 R3）。
    未登记的超龄 `.job-*` 残留（原文件已消亡的孤儿别名）照常按 24h 兜底清理
    （评审 R2A 修正措辞：无条件跳过与 24h 兜底矛盾，故按 inode 归属豁免）。
    """
    temp_dir = Path("tmp")
    if not temp_dir.exists():
        return
    cutoff = time.time() - max_age_hours * 3600
    protected_abs = {os.path.abspath(str(p)) for p in protected} if protected else set()
    protected_inodes = set()
    for p in protected_abs:
        try:
            st = os.stat(p)
        except OSError:
            continue
        protected_inodes.add((st.st_dev, st.st_ino))
    for f in temp_dir.iterdir():
        if not f.is_file():
            continue
        if os.path.abspath(str(f)) in protected_abs:
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        if (st.st_dev, st.st_ino) in protected_inodes:
            continue  # 活跃 upload 的硬链接别名（原路径已豁免，别名继承旧 mtime）
        if st.st_mtime < cutoff:
            try:
                f.unlink()
                logger.info(f"清理过期临时文件: {f.name}")
            except Exception as e:
                logger.warning(f"清理临时文件失败 {f.name}: {e}")

manager = ModelManager()
downloader = BilibiliDownloader()
downloader_douyin = DouyinDownloader()
transcription_service = TranscriptionService(manager)

# ================= 暂停管理单例 =================
pause_manager = PauseManager(notify_url=config.pause_notify_url,
                             notify_token=config.pause_notify_token)
PAUSED_REJECT_PATHS = {"/transcribe", "/transcribe_url", "/transcribe_file", "/transcribe_douyin",
                       "/jobs", "/jobs/upload"}


def _fmt_resume_at(resume_at):
    if resume_at is None:
        return "未知"
    return f"{resume_at:%H:%M}" if resume_at.date() == datetime.now().date() else f"{resume_at:%m-%d %H:%M}"


# ================= 任务队列（spec §6 执行模型：装配 + 真实执行体工厂） =================

job_manager = JobManager(config)

# asr-engine 暂停态非阻塞快照（worker 取件门消费；后台周期刷新）
_asr_paused_snapshot = {"paused": False, "checked_at": 0.0}
UPSTREAM_PAUSE_REFRESH_SECONDS = 60.0
QUERY_BATCH_LIMIT = 500

# httpx AsyncClient 惰性共享（采样器 ~1s 一发；shutdown 关闭）
_progress_http_client = None

# startup 拉起的队列辅助后台任务（shutdown 取消）
_jobs_bg_tasks = []


def _upstream_pause_snapshot() -> bool:
    """`_upstream_pause_probe` 注入实现：读缓存快照，绝不阻塞事件循环。

    `_query_asr_paused` 是 3s 同步 get，直接绑定会卡死 worker 取件门与自冷
    轮询（jobs.py 注入点契约）；真实查询由 `_upstream_pause_refresh_loop`
    经 to_thread 周期刷新（60s 间隔，启动即首查）。"""
    return bool(_asr_paused_snapshot["paused"])


async def _upstream_pause_refresh_loop():
    while True:
        try:
            paused = await asyncio.to_thread(_query_asr_paused)
            _asr_paused_snapshot["paused"] = bool(paused)
            _asr_paused_snapshot["checked_at"] = time.time()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"asr-engine 暂停态快照刷新失败(保留旧值): {e}")
        await asyncio.sleep(UPSTREAM_PAUSE_REFRESH_SECONDS)


async def _poll_asr_engine_progress(job, request_id) -> Optional[dict]:
    """`_progress_poll` 注入实现：asr-engine GET /v1/progress（带 timeout + token）。

    任何失败（网络/非 200/解析）返回 None——采样器本轮忽略（spec §5：
    inactive/不匹配/失败 → 本轮忽略）；鉴权用 config.asr_engine_token。"""
    global _progress_http_client
    try:
        import httpx
        if _progress_http_client is None:
            _progress_http_client = httpx.AsyncClient(timeout=2.0)
        headers = None
        token = config.asr_engine_token
        if token:
            headers = {"Authorization": f"Bearer {token}"}
        resp = await _progress_http_client.get(
            f"{config.asr_engine_url.rstrip('/')}/v1/progress", headers=headers)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if isinstance(data, dict) else None
    except Exception:
        return None


async def _close_progress_client():
    global _progress_http_client
    if _progress_http_client is not None:
        try:
            await _progress_http_client.aclose()
        except Exception:
            pass
        _progress_http_client = None


async def _warm_capability_once():
    """`_capability` 预热：启动后台探测一次 backend capability（Task 5 正/负缓存），
    首个任务即得真实能力；backend 未加载时静默跳过（懒加载路径由 _capability
    调用点兜底）。"""
    try:
        backend = manager._backend
        if backend is not None:
            await asyncio.to_thread(backend.probe_capability)
    except Exception as e:
        logger.info(f"capability 预热跳过(不影响功能): {e}")


def _unlink_quiet(path) -> None:
    """尽力删除（OSError 静默）——上传暂存失败/被拒提交的清理。"""
    try:
        os.unlink(path)
    except OSError:
        pass


def _job_error_dict(message: str) -> dict:
    """与现有端点错误响应同构的错误 dict（JobManager 终态映射：status==error → job error）。"""
    return {"status": "error", "message": message,
            "type": config.subtitle_config["type"],
            "version": config.subtitle_config["version"],
            "body": [], "rtf": 0.0}


def _delete_tmp_artifact(path) -> None:
    """队列执行体产物清理：仅删 tmp 下的文件；缓存目录文件保留（与遗留端点一致）。"""
    if not path or not os.path.exists(path):
        return
    if path.startswith("tmp/") or "/tmp/" in path:
        try:
            os.remove(path)
            logger.info(f"队列任务临时文件已删除: {path}")
        except OSError as e:
            logger.warning(f"队列任务临时文件删除失败 {path}: {e}")


def _cleanup_part_files(video_id: str, save_dir: str) -> None:
    """DownloadCancelled 后清理半截 .part 文件（spec §6：命中 → 清理 tmp → cancelled）。"""
    try:
        for f in Path(save_dir).glob(f"*{video_id}*.part*"):
            try:
                f.unlink()
                logger.info(f"已清理未完成下载: {f.name}")
            except OSError:
                pass
    except Exception as e:
        logger.warning(f".part 清理失败(忽略): {e}")


async def _exec_bilibili(job, sinks, req):
    """B 站执行体：下载(to_thread + 进度/取消) → diarize_only 变体 / process_transcription。

    下载段进度经 sinks["progress"] 写转录槽（downloading 段 percent 为 B 站真实值）；
    成功后 note_download_time → JobManager 终态映射注入 timing.download；
    DownloadCancelled → 清 .part → 上抛（终态 cancelled 由 JobManager 映射）。"""
    video_id = req.get("bvid") or ""
    cookie = req.get("cookie") or ""
    page = int(req.get("page") or 1)
    context = req.get("context")
    diarize = bool(req.get("diarize"))
    no_cache = bool(req.get("no_cache"))
    save_dir = str(get_temp_dir())

    sinks["phase"]("downloading")
    download_start = time.time()
    try:
        success, dl = await asyncio.to_thread(
            downloader.download_bilibili_audio, video_id, cookie, save_dir, page,
            progress_cb=sinks["progress"], cancel_flag=sinks["cancel_event"])
    except DownloadCancelled:
        _cleanup_part_files(video_id, save_dir)
        raise
    download_time = time.time() - download_start
    sinks["note_download_time"](download_time)

    if not success:
        return _job_error_dict(f"音频下载失败: {dl}")

    temp_filename = dl["file_path"]
    try:
        if sinks["cancelled"]():     # 下载完成后取消检查（sinks 契约）
            raise DownloadCancelled(f"下载完成后已取消: {video_id}")

        if req.get("diarize_only"):
            # diarize_only 无 transcribing 段：downloading → formatting → done，
            # percent 恒 null（spec §4.2）
            sinks["phase"]("formatting")
            if req.get("body"):
                return await diarize_merge_subtitle(
                    req.get("body"), temp_filename, video_id, download_time,
                    wait_deadline=sinks["diarize_wait_deadline"], stage_cb=sinks["stage"])
            return await diarize_only(
                temp_filename, video_id, download_time,
                wait_deadline=sinks["diarize_wait_deadline"], stage_cb=sinks["stage"])

        display_name = f"Bilibili_{video_id}_p{page}" if page > 1 else f"Bilibili_{video_id}"
        cache_video_id = f"{video_id}_p{page}" if page > 1 else video_id
        audio_url = dl.get("audio_url")
        audio_id = dl.get("audio_id")
        sinks["phase"]("transcribing")
        # inflight_key 与下方 process_transcription 完全同参构建（取消联动依赖）
        key = transcription_service.build_inflight_key(
            temp_filename, audio_url=audio_url, bvid=cache_video_id, audio_id=audio_id,
            context=context, diarize=diarize, no_cache=no_cache)
        sinks["set_inflight_key"](key)
        return await transcription_service.process_transcription(
            temp_filename, display_name, audio_url, cache_video_id, audio_id, no_cache,
            context=context, diarize=diarize,
            progress_cb=sinks["progress"], cancelled=sinks["cancelled"],
            diarize_wait_deadline=sinks["diarize_wait_deadline"], stage_cb=sinks["stage"],
            phase_cb=_phase_sink_cb(sinks))
    finally:
        _delete_tmp_artifact(temp_filename)


def _phase_sink_cb(sinks):
    """process_transcription 的 phase_cb 装配（fixwave F4，spec §5）：
    formatting → percent 直接 100；其余 phase 不带 percent。
    （diarize_only 变体不经此：其 percent 恒 null，直接 sinks["phase"]("formatting")。）"""
    def _cb(name):
        sinks["phase"](name, 100 if name == "formatting" else None)
    return _cb


async def _exec_webdav(job, sinks, req):
    """网盘执行体：路径解析 + inbox→processed 回退 → process_transcription
    → 成功归档（取消/失败不归档，spec §6）。

    归档与取消竞态（fixwave F3）：归档前查 job 取消标志（sinks["cancelled"] 探针），
    已取消 → 跳过归档；归档一旦开始即为提交点（job.commit_done）——取消在其后
    到达时 done 胜出取消（JobManager 按提交点落态，result 与已归档文件一致）。
    """
    path = req.get("path") or ""
    context = req.get("context")
    diarize = bool(req.get("diarize"))
    no_cache = bool(req.get("no_cache"))
    webdav_base = config.get('webdav.base_path', '/mnt/webdav')
    relative_path = path.lstrip('/')
    full_file_path = os.path.join(webdav_base, relative_path)

    logger.info(f"队列网盘文件转录: {path} -> {full_file_path}")
    if not os.path.exists(full_file_path):
        if relative_path.startswith('inbox/'):
            fallback_path = 'processed/' + relative_path[len('inbox/'):]
            fallback_full_path = os.path.join(webdav_base, fallback_path)
            logger.info(f"inbox 文件不存在，尝试 processed 回退: {fallback_full_path}")
            if os.path.exists(fallback_full_path):
                relative_path = fallback_path
                full_file_path = fallback_full_path
    if not os.path.exists(full_file_path):
        return _job_error_dict(f"文件不存在: {path}")
    if not os.path.isfile(full_file_path):
        return _job_error_dict(f"路径不是文件: {path}")
    if not os.access(full_file_path, os.R_OK):
        return _job_error_dict(f"文件不可读: {path}")

    sinks["phase"]("transcribing")
    try:
        key = transcription_service.build_inflight_key(
            full_file_path, file_path_for_cache=full_file_path,
            context=context, diarize=diarize, no_cache=no_cache)
        sinks["set_inflight_key"](key)
        result = await transcription_service.process_transcription(
            full_file_path, path, audio_url=None, bvid=None, audio_id=None,
            no_cache=no_cache, file_path_for_cache=full_file_path,
            context=context, diarize=diarize,
            progress_cb=sinks["progress"], cancelled=sinks["cancelled"],
            diarize_wait_deadline=sinks["diarize_wait_deadline"], stage_cb=sinks["stage"],
            phase_cb=_phase_sink_cb(sinks))

        # 归档仅在成功路径执行：取消（JobCancelledError/CancelledError 上抛穿透）
        # 与失败（status==error）都不归档（评审 Review Focus #2）；归档开始前
        # 查取消标志（fixwave F3），置位即跳过
        if isinstance(result, dict) and result.get("status") != "error":
            if sinks["cancelled"]():
                logger.info(f"任务已取消，跳过归档: {relative_path}")
            else:
                job.commit_done = True   # 提交点：归档开始后 done 胜出取消（F3）
                await asyncio.to_thread(webdav_archive.move_to_processed,
                                        webdav_base, relative_path)
        return result
    except UpstreamPausedError:
        raise   # 暂停信号穿透（JobManager 映射 error + paused/resume_at/paused_at）
    except Exception as e:
        logger.error(f"队列网盘文件转录失败: {e}")
        return _job_error_dict(str(e))


async def _exec_douyin(job, sinks, req):
    """抖音执行体：to_thread 下载（ffmpeg 无逐块取消点 → 标志 + join 自然结束）
    → process_transcription。"""
    aweme_id = req.get("aweme_id") or ""
    context = req.get("context")
    diarize = bool(req.get("diarize"))
    no_cache = bool(req.get("no_cache"))

    sinks["phase"]("downloading")
    download_start = time.time()
    success, dl = await asyncio.to_thread(
        downloader_douyin.download_douyin_audio, aweme_id,
        save_dir=str(get_temp_dir()))
    download_time = time.time() - download_start
    sinks["note_download_time"](download_time)

    if not success:
        return _job_error_dict(f"音频下载失败: {dl}")

    temp_filename = dl["file_path"]
    try:
        if sinks["cancelled"]():
            raise DownloadCancelled(f"下载完成后已取消: {aweme_id}")
        audio_url = dl.get("audio_url")
        audio_id = dl.get("audio_id")
        sinks["phase"]("transcribing")
        key = transcription_service.build_inflight_key(
            temp_filename, audio_url=audio_url, bvid=aweme_id, audio_id=audio_id,
            context=context, diarize=diarize, no_cache=no_cache)
        sinks["set_inflight_key"](key)
        return await transcription_service.process_transcription(
            temp_filename, f"Douyin_{aweme_id}", audio_url, aweme_id, audio_id, no_cache,
            context=context, diarize=diarize,
            progress_cb=sinks["progress"], cancelled=sinks["cancelled"],
            diarize_wait_deadline=sinks["diarize_wait_deadline"], stage_cb=sinks["stage"],
            phase_cb=_phase_sink_cb(sinks))
    finally:
        _delete_tmp_artifact(temp_filename)


async def _exec_upload(job, sinks, req):
    """上传执行体：内部键路径（tmp/jobfile_*）→ process_transcription；
    文件删除归 JobManager 终态清理（_mark_finished），此处不删。"""
    path = job.upload_path
    context = req.get("context")
    diarize = bool(req.get("diarize"))
    sinks["phase"]("transcribing")
    # upload 无稳定 identity：identity = abspath(内部键)（唯一 tmp 名，天然不与他人合并）
    key = transcription_service.build_inflight_key(
        path, context=context, diarize=diarize, no_cache=False)
    sinks["set_inflight_key"](key)
    return await transcription_service.process_transcription(
        path, req.get("original_filename") or Path(path).name,
        diarize=diarize, context=context,
        progress_cb=sinks["progress"], cancelled=sinks["cancelled"],
        diarize_wait_deadline=sinks["diarize_wait_deadline"], stage_cb=sinks["stage"],
        phase_cb=_phase_sink_cb(sinks))


async def _make_executor(job, sinks):
    """JobManager._executor_factory 注入（spec §6 执行模型末段）：按 source 分派。"""
    req = job.request or {}
    # 回填 backend name（采样器归属依赖：asr-engine → GET /v1/progress 远端采样；
    # 懒加载窗口取 config.backend_name，与其后真实加载的后端一致）
    sinks["set_backend_name"](manager.backend)
    if job.source == "bilibili":
        return await _exec_bilibili(job, sinks, req)
    if job.source == "webdav":
        return await _exec_webdav(job, sinks, req)
    if job.source == "douyin":
        return await _exec_douyin(job, sinks, req)
    if job.source == "upload":
        return await _exec_upload(job, sinks, req)
    raise ValueError(f"未知任务来源: {job.source}")


def _setup_job_manager(jm=None):
    """JobManager 六注入点装配（幂等；模块导入与测试隔离共同入口）。

    注入函数以闭包引用 server 模块全局（运行时解析）——monkeypatch
    pause_manager/manager/downloader 等后无需重装配。
    直接反向 import server 会成环，故 jobs.py 侧全部走注入点（评审 R5A）。"""
    mgr = jm if jm is not None else job_manager
    mgr._executor_factory = _make_executor
    mgr._cancel_computation = transcription_service.cancel_computation
    mgr._entry_lookup = transcription_service._inflight.get
    mgr._backend_lookup = lambda: manager._backend
    mgr._pause_gate = lambda: pause_manager.is_paused()
    mgr._upstream_pause_probe = _upstream_pause_snapshot
    mgr._progress_poll = _poll_asr_engine_progress
    return mgr


_setup_job_manager()


def _queue_full_response(e: QueueFullError) -> JSONResponse:
    """429：Retry-After 响应头与 body 字段双写（spec §4.1，评审 R1A fold）。"""
    return JSONResponse(status_code=429,
                        content={"detail": e.detail, "retry_after": e.retry_after},
                        headers={"Retry-After": str(e.retry_after)})


def _parse_iso_dt(value):
    """ISO 时间字符串 → aware datetime(缺时区按本地补);空/非法返回 None"""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt.astimezone() if dt.tzinfo is None else dt


def _paused_503_response(resume_at=None, paused_at=None):
    """暂停 503 统一构造器(中间件拦截与三端点 UpstreamPausedError 分支共用,响应体四键扁平)。

    resume_at/paused_at 优先取 UpstreamPausedError 携带的 asr body 值(端点路径传入)——
    asr 与 transcribe 暂停窗口同步,提前恢复后 transcribe 态已清空,asr body 的值才能
    正确支持 noteflow 迟到事件判定;缺失回退 transcribe 自身暂停态,仍缺失则 resume_at
    取当前时间(窄 TOCTOU:检查与构造之间被 /resume)、paused_at 置 None。
    Retry-After 按 transcribe 自身暂停态剩余秒数估算;无暂停态则按 resume_at 与当前
    时间差;仍不可得(已过期/非法)则省略该 header,不误导客户端等待一个过期时刻。
    """
    resume_dt = _parse_iso_dt(resume_at)
    if resume_dt is None:
        resume_dt = pause_manager.resume_at()
    paused_dt = _parse_iso_dt(paused_at)
    if paused_dt is None:
        paused_dt = pause_manager.paused_at()
    now = datetime.now().astimezone()
    if resume_dt is None:  # 窄 TOCTOU:检查与构造之间被 /resume
        resume_dt = now
    headers = {}
    remaining = pause_manager.remaining_seconds()
    if remaining > 0:  # transcribe 自身暂停态优先
        headers["Retry-After"] = str(max(1, math.ceil(remaining)))
    elif resume_dt > now:  # 走到此分支时 resume_dt 必为异常携带值(aware),不会混入本地 naive
        headers["Retry-After"] = str(max(1, math.ceil((resume_dt - now).total_seconds())))
    return JSONResponse(
        status_code=503,
        content={"detail": f"服务暂停中，预计 {_fmt_resume_at(resume_dt)} 恢复",
                 "paused": True,
                 "resume_at": resume_dt.astimezone().isoformat(),
                 "paused_at": paused_dt.astimezone().isoformat() if paused_dt else None},
        headers=headers or None)


def _pause_asr_engine(hours: float) -> str:
    """转发暂停到 asr-engine；容错(其 idle 卸载兜底)。校验业务码:200 且 ok=true 才算 paused"""
    import httpx
    try:
        resp = httpx.post(f"{config.asr_engine_url.rstrip('/')}/admin/pause",
                          json={"hours": hours}, timeout=5.0)
        if resp.status_code != 200:
            return f"http_{resp.status_code}"
        body = resp.json()  # 解析失败/非 dict 异常落入下方 except → "failed"
        if body.get("ok") is True:
            return "paused"
        return f"rejected_{body}"  # 200 但业务码拒绝(如暂停被上游校验驳回)
    except Exception as e:
        logger.warning(f"转发暂停到 asr-engine 失败(容错): {e}")
        return "failed"


def _resume_asr_engine() -> str:
    """转发恢复到 asr-engine(手动提前恢复必须同步，否则 asr 挂到原期限)；容错"""
    import httpx
    try:
        resp = httpx.post(f"{config.asr_engine_url.rstrip('/')}/admin/resume", timeout=5.0)
        return "resumed" if resp.status_code == 200 else f"http_{resp.status_code}"
    except Exception as e:
        logger.warning(f"转发恢复到 asr-engine 失败(容错，其到点自恢复兜底): {e}")
        return "failed"


def _release_gpu_for_pause(gen: int) -> dict:
    """后台释放链(仅本地卸载)：世代令牌防交错；绝不在请求线程同步调用(会等推理锁)。

    asr-engine 的暂停转发不在此链中——/pause 端点同步调用 _pause_asr_engine(快速 HTTP)，
    不受本地 in_use/推理锁阻塞，也不会因等锁延迟或延长上游暂停窗口。
    """
    result = {"backend": "not_loaded", "diarization": "not_loaded"}
    if not pause_manager.is_paused():
        return {"skipped": "resumed"}
    pause_manager.mark_unload_attempted()
    if manager._backend is not None and not manager.in_use:
        try:
            manager.unload_model()
            result["backend"] = "released"
        except Exception as e:
            logger.warning(f"暂停卸载 backend 失败: {e}")
            result["backend"] = "failed"
    if pause_manager.generation != gen:
        return result  # 等锁期间已 resume，中止(双检之一;等锁后的二次校验在 unload_global 锁内)
    try:
        result["diarization"] = ("released" if diarization_mgr.unload_global(
            should_abort=lambda: pause_manager.generation != gen) else "not_loaded")
    except Exception as e:
        logger.warning(f"暂停卸载 diarization 失败: {e}")
        result["diarization"] = "failed"
    logger.info(f"暂停本地释放链完成: {result}")
    return result

# 定义请求模型
class BilibiliTranscribeRequest(BaseModel):
    bvid: str
    cookie: str
    no_cache: bool = False
    page: int = 1
    context: Optional[str] = None  # ASR 偏置文本（标题/UP主/简介），可选
    diarize: bool = False  # 说话人分离（显式传入才启用）
    diarize_only: bool = False  # 仅说话人分离（不跑 ASR，返回说话人时间轴）
    body: Optional[List[dict]] = None  # diarize_only=true 时携带官方字幕 body → 分离+标注拼接模式

    class Config:
        populate_by_name = True


class WebdavTranscribeRequest(BaseModel):
    path: str
    no_cache: bool = False
    context: Optional[str] = None  # ASR 偏置文本（分类/作者/主题），可选
    diarize: bool = False  # 说话人分离（显式传入才启用）

    class Config:
        populate_by_name = True


class DouyinTranscribeRequest(BaseModel):
    aweme_id: str = Field(..., pattern=r"^\d{15,20}$")  # 19位数字ID，字符串防精度丢失
    no_cache: bool = False
    context: Optional[str] = None  # ASR 偏置文本（标题/作者），可选
    diarize: bool = False

# ================= 后台保活线程 =================
def monitor_loop():
    while True:
        time.sleep(config.check_interval)
        try:
            if pause_manager.should_unload():
                # 补做本地卸载(等 diarization 推理锁属排空语义)；asr 转发已在 /pause 同步完成
                _release_gpu_for_pause(pause_manager.generation)
            elif (not pause_manager.is_paused() and manager._backend is not None
                  and not manager.in_use):
                if time.time() - manager.last_active_time > config.idle_timeout:
                    manager.unload_model()
        except Exception as e:
            logger.warning(f"monitor_loop 异常(忽略继续): {e}")

bg_thread = threading.Thread(target=monitor_loop, daemon=True)
bg_thread.start()

# ================= API 接口 =================
app = FastAPI()

# 启动时清理过期缓存 + 预加载模型
@app.on_event("startup")
async def startup_event():
    """应用启动时的事件处理"""
    logger.info("服务启动中...")
    cache_manager.cleanup_expired_cache()
    # 任务队列拉起（worker/采样器/清扫，幂等）+ 辅助后台任务
    await job_manager.start()
    _jobs_bg_tasks.append(asyncio.create_task(_upstream_pause_refresh_loop()))
    _jobs_bg_tasks.append(asyncio.create_task(_warm_capability_once()))
    if pause_manager.is_paused():
        logger.info(f"启动时处于暂停状态(至 {_fmt_resume_at(pause_manager.resume_at())})，跳过预加载")
        return
    logger.info("预加载模型...")
    try:
        manager.load_model_if_needed()
        logger.info("预加载完成，服务器已就绪！")
    except Exception as e:
        logger.warning(f"预加载失败 - {e}")
        logger.info("服务器将继续启动，将在首次请求时重试加载模型")

@app.on_event("shutdown")
async def shutdown_event():
    # 先收尾队列（running 任务走取消语义 join 完成后，才释放分离/模型资源）
    await job_manager.stop()
    for t in _jobs_bg_tasks:
        t.cancel()
    for t in _jobs_bg_tasks:
        try:
            await t
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning("队列辅助后台任务停止异常", exc_info=True)
    _jobs_bg_tasks.clear()
    await _close_progress_client()
    await asyncio.to_thread(diarization_mgr.unload_global)


# Token验证中间件
@app.middleware("http")
async def token_validation_middleware(request: Request, call_next):
    # 如果配置了token，则进行验证
    if config.api_token:
        # 获取Authorization头
        authorization = request.headers.get("Authorization")

        if not authorization:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing Authorization header"},
                headers={"WWW-Authenticate": "Bearer"}
            )

        # 验证Bearer token格式
        if not authorization.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid authorization format. Expected: Bearer <token>"},
                headers={"WWW-Authenticate": "Bearer"}
            )

        # 提取token
        token = authorization.split(" ", 1)[1]

        # 验证token是否匹配
        if token != config.api_token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid token"},
                headers={"WWW-Authenticate": "Bearer"}
            )

    # 暂停拦截(token 校验通过后、处理前；鉴权优先于 503)
    if (request.url.path in PAUSED_REJECT_PATHS and request.method == "POST"
            and pause_manager.is_paused()):
        return _paused_503_response()

    # 继续处理请求
    response = await call_next(request)
    return response

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), diarize: bool = Form(False)):
    """上传音频文件转录接口"""
    # 存临时文件
    temp_dir = get_temp_dir()
    temp_filename = temp_dir / generate_safe_filename(file.filename)

    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 使用转录服务处理
        result = await transcription_service.process_transcription(str(temp_filename), file.filename, diarize=diarize)

        return result
    except UpstreamPausedError as e:
        return _paused_503_response(resume_at=e.resume_at, paused_at=e.paused_at)
    finally:
        # 确保清理临时文件
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except Exception as e:
            logger.warning(f"警告：临时文件删除失败 {temp_filename}: {e}")
        cleanup_stale_tmp_files(protected=job_manager.active_upload_paths())

@app.post("/transcribe_url")
async def transcribe_bilibili_audio(request: BilibiliTranscribeRequest):
    """转录B站音频接口"""
    temp_filename = None
    try:
        # 1. 下载音频文件到tmp目录
        logger.info(f"开始下载B站音频: bvid={request.bvid}")

        download_start = time.time()
        success, result = downloader.download_bilibili_audio(
            request.bvid,
            request.cookie,
            save_dir=str(get_temp_dir()),
            page=request.page
        )
        download_time = time.time() - download_start

        if not success:
            return {
                "status": "error",
                "message": f"音频下载失败: {result}",
                "type": config.subtitle_config["type"],
                "version": config.subtitle_config["version"],
                "body": [],
                "rtf": 0.0,
                "timing": {"download": round(download_time, 3)}
            }

        temp_filename = result["file_path"]  # 从字典中获取文件路径
        audio_url = result["audio_url"]  # 获取音频URL
        audio_id = result.get("audio_id")  # 获取音频ID（可选）
        logger.info(f"音频下载完成: {temp_filename}")

        # 仅说话人分离：不进 ASR。携带 body → 分离+标注拼接模式（官方字幕场景），
        # 否则返回说话人时间轴
        if request.diarize_only:
            if request.body:
                return await diarize_merge_subtitle(request.body, temp_filename, request.bvid, download_time)
            return await diarize_only(temp_filename, request.bvid, download_time)

        # 2. 使用转录服务处理
        # 使用更友好的文件名用于日志显示，page 信息编码到 bvid 中确保缓存键唯一
        display_name = f"Bilibili_{request.bvid}_p{request.page}" if request.page > 1 else f"Bilibili_{request.bvid}"
        cache_bvid = f"{request.bvid}_p{request.page}" if request.page > 1 else request.bvid
        result = await transcription_service.process_transcription(temp_filename, display_name, audio_url, cache_bvid, audio_id, request.no_cache, context=request.context, diarize=request.diarize)

        # 注入下载耗时到 timing
        if "timing" in result:
            result["timing"]["download"] = round(download_time, 3)

        return result

    except UpstreamPausedError as e:
        return _paused_503_response(resume_at=e.resume_at, paused_at=e.paused_at)

    finally:
        # 确保清理临时文件（只清理tmp目录下的文件，不清理cache目录）
        if temp_filename and os.path.exists(temp_filename):
            # 只有当文件在tmp目录下时才删除
            if temp_filename.startswith("tmp/") or "/tmp/" in temp_filename:
                try:
                    os.remove(temp_filename)
                    logger.info(f"临时文件已删除: {temp_filename}")
                except Exception as e:
                    logger.warning(f"警告：临时文件删除失败 {temp_filename}: {e}")
            else:
                logger.info(f"缓存文件保留: {temp_filename}")
        cleanup_stale_tmp_files(protected=job_manager.active_upload_paths())


@app.post("/transcribe_file")
async def transcribe_webdav_file(request: WebdavTranscribeRequest):
    """转录网盘文件接口"""
    # 拼接完整文件路径
    webdav_base = config.get('webdav.base_path', '/mnt/webdav')
    # 清理路径：移除开头的/，确保拼接正确
    relative_path = request.path.lstrip('/')
    full_file_path = os.path.join(webdav_base, relative_path)

    logger.info(f"网盘文件转录: {request.path} -> {full_file_path}")

    # 检查文件是否存在，不存在时尝试 inbox -> processed 回退
    if not os.path.exists(full_file_path):
        if relative_path.startswith('inbox/'):
            fallback_path = 'processed/' + relative_path[len('inbox/'):]
            fallback_full_path = os.path.join(webdav_base, fallback_path)
            logger.info(f"inbox 文件不存在，尝试 processed 回退: {fallback_full_path}")
            if os.path.exists(fallback_full_path):
                relative_path = fallback_path
                full_file_path = fallback_full_path
        if not os.path.exists(full_file_path):
            return {
                "status": "error",
                "message": f"文件不存在: {request.path}",
                "type": config.subtitle_config["type"],
                "version": config.subtitle_config["version"],
                "body": [],
                "rtf": 0.0
            }

    # 检查是否是文件
    if not os.path.isfile(full_file_path):
        return {
            "status": "error",
            "message": f"路径不是文件: {request.path}",
            "type": config.subtitle_config["type"],
            "version": config.subtitle_config["version"],
            "body": [],
            "rtf": 0.0
        }

    # 检查文件是否有读取权限
    if not os.access(full_file_path, os.R_OK):
        return {
            "status": "error",
            "message": f"文件不可读: {request.path}",
            "type": config.subtitle_config["type"],
            "version": config.subtitle_config["version"],
            "body": [],
            "rtf": 0.0
        }

    try:
        try:
            # 使用转录服务处理，传入完整文件路径作为标识用于缓存
            result = await transcription_service.process_transcription(
                full_file_path,
                request.path,  # 使用原始相对路径作为显示名
                audio_url=None,
                bvid=None,
                audio_id=None,
                no_cache=request.no_cache,
                file_path_for_cache=full_file_path,  # 传入完整路径用于缓存
                context=request.context,
                diarize=request.diarize
            )

            # 转录成功后归档 inbox -> processed（2026-11-02 改造，接替 mryk24「创建任务即移动」；
            # 幂等跳过/防覆盖/伴移在 move_to_processed 内处理，失败静默记日志不影响转录返回）
            if isinstance(result, dict) and result.get("status") != "error":
                webdav_archive.move_to_processed(webdav_base, relative_path)

            return result

        except UpstreamPausedError:
            raise  # 暂停信号穿透到统一包装转 503,不得被 except Exception 转 error dict
        except Exception as e:
            logger.error(f"网盘文件转录失败: {e}")
            return {
                "status": "error",
                "message": str(e),
                "type": config.subtitle_config["type"],
                "version": config.subtitle_config["version"],
                "body": [],
                "rtf": 0.0
            }
    except UpstreamPausedError as e:
        return _paused_503_response(resume_at=e.resume_at, paused_at=e.paused_at)


@app.post("/transcribe_douyin")
async def transcribe_douyin_audio(request: DouyinTranscribeRequest):
    """转录抖音视频音频接口（下载+ffmpeg提音频在本服务侧完成）"""
    temp_filename = None
    try:
        logger.info(f"开始下载抖音音频: aweme_id={request.aweme_id}")
        download_start = time.time()
        # 同步阻塞下载（ffmpeg 拉流最长 20 分钟）移入线程池，避免卡死事件循环（/pause、/status 全停摆）
        success, result = await asyncio.to_thread(
            downloader_douyin.download_douyin_audio,
            request.aweme_id, save_dir=str(get_temp_dir()))
        download_time = time.time() - download_start

        if not success:
            return {
                "status": "error",
                "message": f"音频下载失败: {result}",
                "type": config.subtitle_config["type"],
                "version": config.subtitle_config["version"],
                "body": [],
                "rtf": 0.0,
                "timing": {"download": round(download_time, 3)},
            }

        temp_filename = result["file_path"]
        audio_url = result["audio_url"]
        display_name = f"Douyin_{request.aweme_id}"
        result_out = await transcription_service.process_transcription(
            temp_filename, display_name, audio_url,
            request.aweme_id,  # bvid 位作转录缓存键
            result.get("audio_id"), request.no_cache,
            context=request.context, diarize=request.diarize)
        if "timing" in result_out:
            result_out["timing"]["download"] = round(download_time, 3)
        return result_out

    except UpstreamPausedError as e:
        return _paused_503_response(resume_at=e.resume_at, paused_at=e.paused_at)
    finally:
        if temp_filename and os.path.exists(temp_filename):
            if temp_filename.startswith("tmp/") or "/tmp/" in temp_filename:
                try:
                    os.remove(temp_filename)
                except Exception as e:
                    logger.warning(f"临时文件删除失败: {temp_filename}: {e}")
            else:
                logger.info(f"缓存文件保留: {temp_filename}")
        cleanup_stale_tmp_files(protected=job_manager.active_upload_paths())


# ================= /jobs 异步任务端点（spec §4 API 契约） =================

class JobBilibiliSubmit(BaseModel):
    """字段校验与 BilibiliTranscribeRequest 同构 + 判别标签 + 可选 job_id"""
    source: Literal["bilibili"]
    bvid: str
    cookie: str
    page: int = 1
    context: Optional[str] = None  # ASR 偏置文本，可选
    diarize: bool = False
    diarize_only: bool = False
    body: Optional[List[dict]] = None  # diarize_only=true 时的官方字幕 body
    no_cache: bool = False
    job_id: Optional[str] = None  # 客户端幂等键（服务端不透明，格式不合法 → 422）


class JobWebdavSubmit(BaseModel):
    source: Literal["webdav"]
    path: str
    context: Optional[str] = None
    diarize: bool = False
    no_cache: bool = False
    job_id: Optional[str] = None


class JobDouyinSubmit(BaseModel):
    source: Literal["douyin"]
    aweme_id: str = Field(..., pattern=r"^\d{15,20}$")  # 与 DouyinTranscribeRequest 同构
    context: Optional[str] = None
    diarize: bool = False
    no_cache: bool = False
    job_id: Optional[str] = None


JobSubmitRequest = Annotated[Union[JobBilibiliSubmit, JobWebdavSubmit, JobDouyinSubmit],
                             Field(discriminator="source")]


class JobQueryRequest(BaseModel):
    ids: List[str]


@app.post("/jobs")
async def submit_job(request: JobSubmitRequest):
    """统一异步任务入口（spec §4.1）：202 新任务 / 200 重放+dedup /
    429 队列满（Retry-After 头+body 双写）/ 422 校验失败。
    暂停中由中间件 503（PAUSED_REJECT_PATHS 含 /jobs）。"""
    job_id = request.job_id
    params = request.model_dump(exclude={"source", "job_id"})
    try:
        out = await job_manager.submit(request.source, params, job_id=job_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except QueueFullError as e:
        return _queue_full_response(e)
    return JSONResponse(status_code=(200 if out.get("dedup") else 202), content=out)


@app.post("/jobs/upload")
async def submit_upload_job(file: UploadFile = File(...),
                            diarize: bool = Form(False),
                            context: Optional[str] = Form(None),
                            job_id: Optional[str] = Form(None)):
    """上传转录任务（multipart，spec §4.1）：文件落盘内部键 tmp/jobfile_{uuid}{ext}
    （job_id 永不进文件名；ext 沿用 generate_safe_filename 白名单）。
    同 id 重放：200 + dedup，新上传文件立即删除。"""
    temp_dir = get_temp_dir()
    internal_path = temp_dir / f"jobfile_{uuid.uuid4().hex}{_safe_audio_ext(file.filename)}"
    try:
        with open(internal_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception:
        _unlink_quiet(internal_path)
        raise

    params = {"diarize": diarize, "context": context or None,
              "original_filename": file.filename}
    try:
        out = await job_manager.submit("upload", params, job_id=job_id,
                                       upload_tmp_path=str(internal_path))
    except ValueError as e:
        _unlink_quiet(internal_path)   # 提交被拒：文件无人接管，就地删除
        raise HTTPException(status_code=422, detail=str(e))
    except QueueFullError as e:
        _unlink_quiet(internal_path)
        return _queue_full_response(e)
    return JSONResponse(status_code=(200 if out.get("dedup") else 202), content=out)


@app.post("/jobs/query")
async def query_jobs(request: JobQueryRequest):
    """批量查询（spec §4.2）：单次上限 500，静默截断——超出部分按 missing 返回
    （评审 R5A 定死语义，与终态驱逐同语义：missing = attempt+1 重提）。"""
    ids = request.ids or []
    batch, overflow = ids[:QUERY_BATCH_LIMIT], ids[QUERY_BATCH_LIMIT:]
    out = job_manager.query(batch)
    out["missing"] = list(out["missing"]) + list(overflow)
    return out


@app.get("/jobs")
async def list_jobs(status: Optional[str] = None):
    """任务列表（调试/对账用，spec §4.2）：?status= 过滤，倒序，cap 只作用于终态。"""
    return job_manager.list_jobs(status)


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """单任务查询（spec §4.2 完整 12 键形状）；未知 id → 404。"""
    view = job_manager.get(job_id)
    if view is None:
        raise HTTPException(status_code=404,
                            detail="job not found (可能因服务重启丢失)")
    return view


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """取消任务（spec §4.3「已接受」语义）：返回请求后即时状态；
    终态幂等返回当前态；未知 id → 404（与 GET 一致，评审 R4A 勘误）。"""
    out = await job_manager.cancel(job_id)
    if out is None:
        raise HTTPException(status_code=404,
                            detail="job not found (可能因服务重启丢失)")
    return out


class PauseRequest(BaseModel):
    hours: float = Field(gt=0, le=MAX_PAUSE_HOURS, description="暂停时长（小时，支持小数）")


@app.post("/pause")
async def pause_service(request: PauseRequest):
    """暂停 N 小时：立即拒新请求；同步转发 asr 暂停(快速 HTTP)；本地释放链后台执行"""
    resume_at = pause_manager.pause(request.hours)  # generation 在此 +1
    gen = pause_manager.generation                  # 捕获新世代供释放链校验
    logger.info(f"服务暂停 {request.hours}h，至 {_fmt_resume_at(resume_at)}")
    asr_status = await asyncio.to_thread(_pause_asr_engine, request.hours)
    asyncio.get_running_loop().run_in_executor(None, _release_gpu_for_pause, gen)
    return {"paused": True,
            "resume_at": resume_at.astimezone().isoformat(),
            "resume_at_display": _fmt_resume_at(resume_at),
            "hours": request.hours,
            "asr_engine": asr_status,
            "local_release": "background(在途任务跑完后完成，通常 ≤90s，长视频说话人分离可能更久；最终状态可查 /status)"}


@app.post("/resume")
async def resume_service():
    """提前恢复：清暂停态+世代自增使旧释放链中止+转发 asr-engine 恢复+立即通知 noteflow 唤醒"""
    was_paused = pause_manager.resume()
    asr_status = await asyncio.to_thread(_resume_asr_engine) if was_paused else "not_paused"
    logger.info(f"收到恢复请求(此前{'处于' if was_paused else '不在'}暂停状态，asr: {asr_status})")
    return {"paused": False, "was_paused": was_paused, "asr_engine": asr_status}


def _query_asr_paused() -> bool:
    """查询 asr-engine 暂停态；不可达时报告 False(其进程不在=未暂停)"""
    try:
        import httpx
        return httpx.get(f"{config.asr_engine_url.rstrip('/')}/health", timeout=3.0).json().get("paused", False)
    except Exception as e:
        logger.warning(f"查询 asr-engine 暂停态失败(视为未暂停): {e}")
        return False


@app.get("/status")
async def service_status():
    return {**pause_manager.status(),
            "backend": manager.backend,
            "model_loaded": manager._backend is not None,
            "in_use": manager.in_use,
            "diarization_loaded": diarization_mgr.is_loaded(),
            "asr_engine_paused": await asyncio.to_thread(_query_asr_paused)}


if __name__ == "__main__":
    import uvicorn

    # 预加载模型，避免第一次请求延迟
    logger.info("启动时预加载模型...")
    logger.info("注意：第一次运行时仍需要从 HuggingFace 下载模型，请耐心等待...")
    if pause_manager.is_paused():
        logger.info(f"启动时处于暂停状态(至 {_fmt_resume_at(pause_manager.resume_at())})，跳过预加载")
    else:
        try:
            manager.load_model_if_needed()
            logger.info("预加载完成，服务器已就绪！")
        except Exception as e:
            logger.warning(f"警告：预加载失败 - {e}")
            logger.info("服务器将继续启动，将在首次请求时重试加载模型")

    # 从配置获取API配置
    api_config = config.api_config
    reload = api_config.get("reload", False)
    logger.info(f"启动服务器 http://{api_config['host']}:{api_config['port']}" + (" (自动重载已启用)" if reload else ""))
    uvicorn.run(app, host=api_config["host"], port=api_config["port"], reload=reload)
    