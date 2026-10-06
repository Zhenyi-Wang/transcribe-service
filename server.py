import os
import time
import math
import asyncio
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware

from config import config
from downloaders import BilibiliDownloader
from downloaders import DouyinDownloader
from transcribe import TranscriptionService, diarize_only, diarize_merge_subtitle
from backends.asr_engine_backend import UpstreamPausedError
from logger_config import setup_logger
from cache_manager import cache_manager
from pause_manager import PauseManager, MAX_PAUSE_HOURS
from diarization import manager as diarization_mgr
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

def generate_safe_filename(filename: str) -> str:
    """生成安全的临时文件名"""
    if not filename:
        filename = "audio"

    # 获取文件扩展名
    ext = Path(filename).suffix.lower()
    if not ext:
        ext = ".tmp"  # 默认扩展名

    # 限制扩展名到常见音频格式
    allowed_exts = {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
    if ext not in allowed_exts:
        ext = '.tmp'

    # 使用UUID + 时间戳确保唯一性
    unique_id = str(uuid.uuid4())[:8]
    timestamp = int(time.time())

    return f"temp_{timestamp}_{unique_id}{ext}"

def get_temp_dir():
    """获取并创建临时目录"""
    temp_dir = Path("tmp")
    temp_dir.mkdir(exist_ok=True)
    return temp_dir


def cleanup_stale_tmp_files(max_age_hours: int = 24):
    """清理超过指定时间的 tmp 文件"""
    temp_dir = Path("tmp")
    if not temp_dir.exists():
        return
    cutoff = time.time() - max_age_hours * 3600
    for f in temp_dir.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
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
PAUSED_REJECT_PATHS = {"/transcribe", "/transcribe_url", "/transcribe_file", "/transcribe_douyin"}


def _fmt_resume_at(resume_at):
    if resume_at is None:
        return "未知"
    return f"{resume_at:%H:%M}" if resume_at.date() == datetime.now().date() else f"{resume_at:%m-%d %H:%M}"


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
        cleanup_stale_tmp_files()

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
        cleanup_stale_tmp_files()


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
        success, result = downloader_douyin.download_douyin_audio(
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
        cleanup_stale_tmp_files()


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
    