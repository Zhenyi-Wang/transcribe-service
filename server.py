import os
import time
import shutil
import threading
import gc
import torch
import json
import re
import uuid
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from funasr import AutoModel
from config import config
from downloader import BilibiliDownloader
from transcribe import TranscriptionService
from pydantic import BaseModel

# 减少FunASR的冗余日志输出
os.environ['MODELSCOPE_CACHE'] = str(Path.home() / ".cache/modelscope")
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'  # 禁用进度条

# 配置logger
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# 创建不缓冲的文件处理器
class UnbufferedFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.stream.flush()

file_handler = UnbufferedFileHandler(log_dir / "server.log", encoding='utf-8')
file_handler.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 设置格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 配置根日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 添加处理器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 防止日志重复
logger.propagate = False

class ModelManager:
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()
        self.last_active_time = 0
        self.device = "cpu"

    def _build_model_kwargs(self, device: str) -> dict:
        """构建模型参数，GPU和CPU共享

        Args:
            device: 设备类型 ('cuda' 或 'cpu')

        Returns:
            dict: 模型参数字典
        """
        model_kwargs = {
            "model": config.model_name,
            "vad_model": config.vad_model,
            "punc_model": config.punc_model,
            "device": device,
            "disable_update": config.disable_update  # 禁止每次都去检查更新，加快加载速度
        }

        # 如果启用时间戳，添加相应参数
        if config.enable_timestamp:
            # 使用句子级别时间戳，更适合字幕生成
            model_kwargs["sentence_timestamp"] = True
            logger.info("已启用句子级时间戳")

        return model_kwargs

    def load_model_if_needed(self):
        self.last_active_time = time.time()
        
        if self.model is None:
            with self.lock:
                if self.model is None:
                    # 优先尝试 GPU
                    target_device = "cuda" if torch.cuda.is_available() else "cpu"
                    logger.info(f"正在加载模型 (Target: {target_device})...")
                    logger.info("如果是第一次运行，正在自动从 ModelScope 下载模型，请耐心等待...")

                    try:
                        logger.info("注意：可能显示'Downloading Model'但实际上使用缓存...")

                        # 构建模型参数
                        model_kwargs = self._build_model_kwargs(target_device)

                        self.model = AutoModel(**model_kwargs)
                        self.device = target_device
                        logger.info(f"模型加载成功！运行在: {self.device}")
                        
                    except Exception as e:
                        # 如果是显存炸了(OOM)，切回 CPU 重试
                        if "out of memory" in str(e).lower() and target_device == "cuda":
                            logger.warning("显存不足，正在切换回 CPU 模式...")
                            torch.cuda.empty_cache()

                            # 构建CPU模式的模型参数
                            model_kwargs = self._build_model_kwargs("cpu")

                            self.model = AutoModel(**model_kwargs)
                            self.device = "cpu"
                            logger.info("CPU 模式加载成功。")
                        else:
                            # 其他错误（如下载失败）直接抛出
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
    cid: str
    cookie: str

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

        temp_filename = result  # result是文件路径
        logger.info(f"音频下载完成: {temp_filename}")

        # 2. 使用转录服务处理
        # 使用更友好的文件名用于日志显示
        display_name = f"Bilibili_{request.bvid}"
        result = await transcription_service.process_transcription(temp_filename, display_name)

        return result

    finally:
        # 确保清理临时文件
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
                logger.info(f"临时文件已删除: {temp_filename}")
            except Exception as e:
                logger.warning(f"警告：临时文件删除失败 {temp_filename}: {e}")

if __name__ == "__main__":
    import uvicorn

    # 预加载模型，避免第一次请求延迟
    logger.info("启动时预加载模型...")
    logger.info("注意：第一次运行时仍需要从 ModelScope 下载模型，请耐心等待...")
    try:
        manager.load_model_if_needed()
        logger.info("预加载完成，服务器已就绪！")
    except Exception as e:
        logger.warning(f"警告：预加载失败 - {e}")
        logger.info("服务器将继续启动，将在首次请求时重试加载模型")

    # 从配置获取API配置
    api_config = config.api_config
    logger.info(f"启动服务器 http://{api_config['host']}:{api_config['port']}")
    uvicorn.run(app, host=api_config["host"], port=api_config["port"])
    