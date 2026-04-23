import os
import time
import shutil
import threading
import gc
import torch
import json
import re
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware

from config import config
from downloaders import BilibiliDownloader
from transcribe import TranscriptionService
from logger_config import setup_logger
from cache_manager import cache_manager
from pydantic import BaseModel

# 设置 HuggingFace 缓存目录和日志
os.environ['HF_HOME'] = str(Path.home() / ".cache/huggingface")
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# 使用统一的logger配置
logger = setup_logger('server')

class ModelManager:
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()
        self.last_active_time = 0
        self.device = "cpu"

    def load_model_if_needed(self):
        self.last_active_time = time.time()

        if self.model is None:
            with self.lock:
                if self.model is None:
                    # 优先尝试 GPU
                    target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    logger.info(f"正在加载模型 (Target: {target_device})...")
                    logger.info("如果是第一次运行，正在自动从 HuggingFace 下载模型，请耐心等待...")

                    try:
                        from qwen_asr import Qwen3ASRModel

                        dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16

                        # 构建模型参数
                        forced_aligner_kwargs = None
                        if config.forced_aligner:
                            forced_aligner_kwargs = dict(
                                dtype=dtype,
                                device_map=target_device,
                            )
                            logger.info("已启用时间戳对齐模型")

                        self.model = Qwen3ASRModel.from_pretrained(
                            config.asr_model,
                            dtype=dtype,
                            device_map=target_device,
                            max_new_tokens=config.max_new_tokens,
                            forced_aligner=config.forced_aligner if config.forced_aligner else None,
                            forced_aligner_kwargs=forced_aligner_kwargs,
                        )
                        self.device = target_device
                        logger.info(f"模型加载成功！运行在: {self.device}")

                    except Exception as e:
                        # 如果是显存炸了(OOM)，切回 CPU 重试
                        if "out of memory" in str(e).lower() and target_device.startswith("cuda"):
                            logger.warning("显存不足，正在切换回 CPU 模式...")
                            torch.cuda.empty_cache()

                            from qwen_asr import Qwen3ASRModel

                            dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16

                            forced_aligner_kwargs = None
                            if config.forced_aligner:
                                forced_aligner_kwargs = dict(
                                    dtype=dtype,
                                    device_map="cpu",
                                )

                            self.model = Qwen3ASRModel.from_pretrained(
                                config.asr_model,
                                dtype=dtype,
                                device_map="cpu",
                                max_new_tokens=config.max_new_tokens,
                                forced_aligner=config.forced_aligner if config.forced_aligner else None,
                                forced_aligner_kwargs=forced_aligner_kwargs,
                            )
                            self.device = "cpu"
                            logger.info("CPU 模式加载成功。")
                        else:
                            # 其他错误直接抛出
                            raise e
        return self.model

    def unload_model(self):
        with self.lock:
            if self.model is not None:
                logger.info("闲置超时，释放模型资源...")
                del self.model
                self.model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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

manager = ModelManager()
downloader = BilibiliDownloader()
transcription_service = TranscriptionService(manager)

# 定义请求模型
class BilibiliTranscribeRequest(BaseModel):
    bvid: str
    cookie: str
    no_cache: bool = False

    class Config:
        populate_by_name = True


class WebdavTranscribeRequest(BaseModel):
    path: str
    no_cache: bool = False

    class Config:
        populate_by_name = True

# ================= 后台保活线程 =================
def monitor_loop():
    while True:
        time.sleep(config.check_interval)
        if manager.model is not None:
            if time.time() - manager.last_active_time > config.idle_timeout:
                manager.unload_model()

bg_thread = threading.Thread(target=monitor_loop, daemon=True)
bg_thread.start()

# ================= API 接口 =================
app = FastAPI()

# 启动时清理过期缓存
@app.on_event("startup")
async def startup_event():
    """应用启动时的事件处理"""
    logger.info("服务启动中...")
    cache_manager.cleanup_expired_cache()
    logger.info("服务启动完成")

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

    # 继续处理请求
    response = await call_next(request)
    return response

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """上传音频文件转录接口"""
    # 存临时文件
    temp_dir = get_temp_dir()
    temp_filename = temp_dir / generate_safe_filename(file.filename)

    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 使用转录服务处理
        result = await transcription_service.process_transcription(str(temp_filename), file.filename)

        return result
    finally:
        # 确保清理临时文件
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except Exception as e:
            logger.warning(f"警告：临时文件删除失败 {temp_filename}: {e}")

@app.post("/transcribe_url")
async def transcribe_bilibili_audio(request: BilibiliTranscribeRequest):
    """转录B站音频接口"""
    temp_filename = None
    try:
        # 1. 下载音频文件到tmp目录
        logger.info(f"开始下载B站音频: bvid={request.bvid}")

        success, result = downloader.download_bilibili_audio(
            request.bvid,
            request.cookie,
            save_dir=str(get_temp_dir())
        )

        if not success:
            return {
                "status": "error",
                "message": f"音频下载失败: {result}",
                "type": config.subtitle_config["type"],
                "version": config.subtitle_config["version"],
                "body": [],
                "rtf": 0.0
            }

        temp_filename = result["file_path"]  # 从字典中获取文件路径
        audio_url = result["audio_url"]  # 获取音频URL
        audio_id = result.get("audio_id")  # 获取音频ID（可选）
        logger.info(f"音频下载完成: {temp_filename}")

        # 2. 使用转录服务处理
        # 使用更友好的文件名用于日志显示
        display_name = f"Bilibili_{request.bvid}"
        result = await transcription_service.process_transcription(temp_filename, display_name, audio_url, request.bvid, audio_id, request.no_cache)

        return result

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


@app.post("/transcribe_file")
async def transcribe_webdav_file(request: WebdavTranscribeRequest):
    """转录网盘文件接口"""
    # 拼接完整文件路径
    webdav_base = config.get('webdav.base_path', '/mnt/webdav')
    # 清理路径：移除开头的/，确保拼接正确
    relative_path = request.path.lstrip('/')
    full_file_path = os.path.join(webdav_base, relative_path)

    logger.info(f"网盘文件转录: {request.path} -> {full_file_path}")

    # 检查文件是否存在
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
        # 使用转录服务处理，传入完整文件路径作为标识用于缓存
        result = await transcription_service.process_transcription(
            full_file_path,
            request.path,  # 使用原始相对路径作为显示名
            audio_url=None,
            bvid=None,
            audio_id=None,
            no_cache=request.no_cache,
            file_path_for_cache=full_file_path  # 传入完整路径用于缓存
        )

        return result

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

if __name__ == "__main__":
    import uvicorn

    # 预加载模型，避免第一次请求延迟
    logger.info("启动时预加载模型...")
    logger.info("注意：第一次运行时仍需要从 HuggingFace 下载模型，请耐心等待...")
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
    