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
from funasr import AutoModel
from config import config

# 减少FunASR的冗余日志输出
os.environ['MODELSCOPE_CACHE'] = '/home/zhenyi/.cache/modelscope'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'  # 禁用进度条

def detect_language_from_result(result):
    """从FunASR结果中提取语言信息"""
    try:
        if not result or len(result) == 0:
            return "zh"  # 默认中文

        # FunASR会在结果中包含语言信息
        # 通常在result[0]中可能包含language字段或通过文本内容判断
        text = result[0].get("text", "")

        # 简单的基于字符的语言检测
        if not text:
            return "zh"

        # 统计中文字符比例
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(re.sub(r'\s', '', text))

        if total_chars == 0:
            return "zh"

        chinese_ratio = chinese_chars / total_chars

        # 如果中文字符超过阈值，认为是中文
        if chinese_ratio > config.chinese_ratio_threshold:
            return "zh"
        # 检测英文
        elif re.match(r'^[a-zA-Z\s\d\W]+$', text):
            return "en"
        # 检测日文（包含平假名和片假名）
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return "ja"
        # 检测韩文
        elif re.search(r'[\uac00-\ud7af]', text):
            return "ko"
        else:
            return "zh"  # 默认中文

    except Exception as e:
        print(f"语言检测失败: {e}")
        return "zh"

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
                    target_device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"[{time.strftime('%H:%M:%S')}] 正在加载模型 (Target: {target_device})...")
                    print("如果是第一次运行，正在自动从 ModelScope 下载模型，请耐心等待...")
                    
                    try:
                        print(f"[{time.strftime('%H:%M:%S')}] 注意：可能显示'Downloading Model'但实际上使用缓存...")
                        self.model = AutoModel(
                            model=config.model_name,
                            vad_model=config.vad_model,
                            punc_model=config.punc_model,
                            device=target_device,
                            disable_update=config.disable_update # 禁止每次都去检查更新，加快加载速度
                        )
                        self.device = target_device
                        print(f"[{time.strftime('%H:%M:%S')}] 模型加载成功！运行在: {self.device}")
                        
                    except Exception as e:
                        # 如果是显存炸了(OOM)，切回 CPU 重试
                        if "out of memory" in str(e).lower() and target_device == "cuda":
                            print(f"⚠️ 显存不足，正在切换回 CPU 模式...")
                            torch.cuda.empty_cache()
                            self.model = AutoModel(
                                model=config.model_name,
                                vad_model=config.vad_model,
                                punc_model=config.punc_model,
                                device="cpu"
                            )
                            self.device = "cpu"
                            print(f"CPU 模式加载成功。")
                        else:
                            # 其他错误（如下载失败）直接抛出
                            raise e
        return self.model

    def unload_model(self):
        with self.lock:
            if self.model is not None:
                print(f"[{time.strftime('%H:%M:%S')}] 闲置超时，释放模型资源...")
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

# ================= 后台保活线程 =================
def monitor_loop():
    while True:
        time.sleep(config.check_interval)
        if manager.model is not None:
            if time.time() - manager.last_active_time > config.idle_timeout:
                manager.unload_model()

bg_thread = threading.Thread(target=monitor_loop, daemon=True)
bg_thread.start()

def split_text_into_segments(text, max_length=None):
    """将长文本分割成适合字幕显示的短句段落"""
    if max_length is None:
        max_length = config.max_segment_length

    if not text:
        return []

    # 按标点符号分割
    sentences = re.split(r'[，。！？；：、]', text)
    segments = []
    current_segment = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # 如果当前段落加上新句子不超过最大长度，合并
        if len(current_segment + sentence) <= max_length:
            current_segment += sentence + "，"
        else:
            # 保存当前段落并开始新的
            if current_segment.strip():
                segments.append(current_segment.strip())
            current_segment = sentence + "，"

    # 添加最后一个段落
    if current_segment.strip():
        segments.append(current_segment.strip())

    return segments

def generate_subtitle_segments(text):
    """生成带时间戳的字幕段落"""
    segments = split_text_into_segments(text)
    body = []

    for i, segment in enumerate(segments):
        start_time = i * config.duration_per_segment
        end_time = (i + 1) * config.duration_per_segment

        body.append({
            "from": round(start_time, 2),
            "to": round(end_time, 2),
            "sid": i + 1,
            "location": 2,
            "content": segment,
            "music": 0
        })

    return body

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
    try:
        # 1. 触发懒加载（如果是第一次，会下载模型，这里会卡很久）
        asr_model = manager.load_model_if_needed()
    except Exception as e:
        return {
            "status": "error",
            "message": f"Model load failed: {str(e)}",
            "type": config.subtitle_config["type"],
            "version": config.subtitle_config["version"]
        }

    # 2. 存临时文件（使用安全的文件名生成）
    temp_dir = get_temp_dir()
    temp_filename = temp_dir / generate_safe_filename(file.filename)
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        print(f"[{time.strftime('%H:%M:%S')}] 开始识别: {file.filename}")

        # 3. 推理 (FunASR 内部会自动调用 ffmpeg 读取音频)
        res = asr_model.generate(
            input=str(temp_filename),  # 转换为字符串路径
            batch_size_s=config.batch_size_s,
            disable_pbar=True
        )

        # 刷新活跃时间
        manager.last_active_time = time.time()

        # 获取转录文本
        transcript_text = res[0]["text"] if res else ""

        # 检测语言
        detected_lang = detect_language_from_result(res)
        print(f"[{time.strftime('%H:%M:%S')}] 检测到语言: {detected_lang}")

        # 生成字幕格式
        subtitle_body = generate_subtitle_segments(transcript_text)

        # 从配置获取字幕样式
        subtitle_config = config.subtitle_config

        return {
            "font_size": subtitle_config["font_size"],
            "font_color": subtitle_config["font_color"],
            "background_alpha": subtitle_config["background_alpha"],
            "background_color": subtitle_config["background_color"],
            "Stroke": subtitle_config["stroke"],
            "type": subtitle_config["type"],
            "lang": detected_lang,
            "version": subtitle_config["version"],
            "body": subtitle_body,
            "device_used": manager.device,
            "status": "success"
        }
    except Exception as e:
        if "out of memory" in str(e).lower():
            manager.unload_model() # 遇到错误赶紧释放资源
        return {
            "status": "error",
            "message": str(e),
            "type": config.subtitle_config["type"],
            "version": config.subtitle_config["version"],
            "body": []
        }
    finally:
        # 确保清理临时文件
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 警告：临时文件删除失败 {temp_filename}: {e}")

if __name__ == "__main__":
    import uvicorn

    # 预加载模型，避免第一次请求延迟
    print(f"[{time.strftime('%H:%M:%S')}] 启动时预加载模型...")
    print("注意：第一次运行时仍需要从 ModelScope 下载模型，请耐心等待...")
    try:
        manager.load_model_if_needed()
        print(f"[{time.strftime('%H:%M:%S')}] 预加载完成，服务器已就绪！")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] 警告：预加载失败 - {e}")
        print("服务器将继续启动，将在首次请求时重试加载模型")

    # 从配置获取API配置
    api_config = config.api_config
    print(f"[{time.strftime('%H:%M:%S')}] 启动服务器 http://{api_config['host']}:{api_config['port']}")
    uvicorn.run(app, host=api_config["host"], port=api_config["port"])
    