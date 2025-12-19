import time
import os
import subprocess
from pathlib import Path
from config import config
from logger_config import setup_logger

logger = setup_logger(__name__)

def get_audio_duration(file_path: str) -> float:
    """获取音频文件的时长（秒）

    Args:
        file_path: 音频文件路径

    Returns:
        float: 音频时长（秒），如果获取失败返回0.0
    """
    try:
        # 方法1：尝试使用ffprobe（ffmpeg工具）
        if os.system("which ffprobe > /dev/null 2>&1") == 0:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries',
                'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                if duration > 0:
                    return duration

        # 方法2：使用mutagen库（如果有安装）
        try:
            from mutagen import File
            audio_file = File(file_path)
            if audio_file is not None and hasattr(audio_file, 'info'):
                duration = audio_file.info.length
                if duration > 0:
                    return duration
        except ImportError:
            pass

        # 方法3：尝试使用torchaudio（如果有安装）
        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(file_path)
            duration = waveform.shape[1] / sample_rate
            if duration > 0:
                return duration
        except ImportError:
            pass

        # 方法4：对于WAV文件，使用wave模块
        if file_path.lower().endswith('.wav'):
            import wave
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / sample_rate
                if duration > 0:
                    return duration

        logger.warning(f"无法获取音频时长: {file_path}")
        return 0.0

    except Exception as e:
        logger.error(f"获取音频时长失败: {e}")
        return 0.0

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
        import re
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
        logger.error(f"语言检测失败: {e}")
        return "zh"

def split_text_into_segments(text, max_length=None):
    """将长文本分割成适合字幕显示的短句段落"""
    import re

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

def format_duration(seconds: float) -> str:
    """将秒数格式化为时分秒格式

    Args:
        seconds: 秒数

    Returns:
        str: 格式化后的时长字符串 (H:MM:SS 或 M:SS)
    """
    if seconds < 0:
        return "0:00"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"

def generate_subtitle_segments(text, asr_result=None):
    """生成带时间戳的字幕段落

    Args:
        text: 转录文本
        asr_result: ASR原始结果（包含时间戳信息）
    """
    import re

    # 如果启用时间戳且ASR结果包含时间戳信息
    if config.enable_timestamp and asr_result and len(asr_result) > 0:
        # 尝试从ASR结果中提取时间戳
        result = asr_result[0]
        body = []

        # FunASR sentence_timestamp=True 时返回 sentence_info
        if "sentence_info" in result:
            sentence_info = result["sentence_info"]
            if isinstance(sentence_info, list):
                for i, seg in enumerate(sentence_info):
                    if isinstance(seg, dict) and "text" in seg:
                        # 获取句子文本和时间戳
                        sentence_text = seg["text"]
                        start_ms = seg.get("start", 0)
                        end_ms = seg.get("end", start_ms)

                        # 转换毫秒为秒
                        start_time = float(start_ms) / 1000
                        end_time = float(end_ms) / 1000

                        if sentence_text.strip():
                            body.append({
                                "from": round(start_time, 2),
                                "to": round(end_time, 2),
                                "sid": i + 1,
                                "location": 2,
                                "content": sentence_text.strip(),
                                "music": 0
                            })

                if body:  # 如果成功提取了时间戳
                    return body

        # 2. 尝试其他可能的时间戳字段
        timestamp_fields = ["segments", "sentences", "words", "timestamp_detail", "time_stamps"]
        for field in timestamp_fields:
            if field in result:
                segments = result[field]
                if isinstance(segments, list):
                    for i, seg in enumerate(segments):
                        if isinstance(seg, dict):
                            start_time = seg.get("start", seg.get("begin", 0))
                            end_time = seg.get("end", seg.get("finish", start_time + config.duration_per_segment))
                            text_seg = seg.get("text", seg.get("word", seg.get("txt", "")))

                            if text_seg.strip():
                                body.append({
                                    "from": round(start_time, 2),
                                    "to": round(end_time, 2),
                                    "sid": i + 1,
                                    "location": 2,
                                    "content": text_seg.strip(),
                                    "music": 0
                                })

                    if body:
                        return body

    # 如果没有时间戳信息或禁用了时间戳，使用原始逻辑
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

class TranscriptionService:
    """转录服务类，封装所有转录相关逻辑"""

    def __init__(self, model_manager):
        self.model_manager = model_manager

    async def process_transcription(self, audio_file_path: str, original_filename: str = None):
        """
        处理音频转录的主函数

        Args:
            audio_file_path: 音频文件的路径
            original_filename: 原始文件名（用于日志）

        Returns:
            dict: 转录结果
        """
        try:
            # 1. 触发懒加载
            asr_model = self.model_manager.load_model_if_needed()
        except Exception as e:
            # 获取音频时长
            audio_duration = get_audio_duration(audio_file_path)
            return {
                "status": "error",
                "message": f"Model load failed: {str(e)}",
                "type": config.subtitle_config["type"],
                "version": config.subtitle_config["version"],
                "audio_duration": round(audio_duration, 2),
                "processing_time": 0.0,
                "rtf": 0.0
            }

        try:
            # 2. 获取音频时长
            filename_to_log = original_filename or audio_file_path
            audio_duration = get_audio_duration(audio_file_path)
            if audio_duration > 0:
                logger.info(f"音频时长: {audio_duration:.2f}秒")
            else:
                logger.warning("无法获取音频时长")

            logger.info(f"开始识别: {filename_to_log}")

            # 记录转录开始时间，计算纯粹的转换时间（处理时长）
            transcription_start_time = time.time()

            res = asr_model.generate(
                input=audio_file_path,
                batch_size_s=config.batch_size_s,
                disable_pbar=True
            )

            # 计算处理时长（纯粹的转换时间）
            processing_time = time.time() - transcription_start_time

            # 刷新活跃时间
            self.model_manager.last_active_time = time.time()

            # 获取转录文本
            transcript_text = res[0]["text"] if res else ""

            # 计算RTF比值（处理时长/音频时长）
            rtf_ratio = 0.0
            if audio_duration > 0:
                rtf_ratio = processing_time / audio_duration

            # 控制台输出关键指标
            logger.info(f"\n{'='*50}")
            logger.info(f"转录完成!")
            logger.info(f"{'='*50}")
            logger.info(f"音频时长:     {format_duration(audio_duration)} ({audio_duration:.2f}秒)")
            logger.info(f"处理时长:     {format_duration(processing_time)} ({processing_time:.2f}秒)")
            logger.info(f"RTF比值:      {rtf_ratio:.3f}")
            if rtf_ratio < 1:
                logger.info(f"状态:         实时处理 ✅ (RTF < 1)")
            else:
                logger.info(f"状态:         非实时处理 ⏱️ (RTF ≥ 1)")
            logger.info(f"{'='*50}\n")

            # 记录所有指标到日志
            logger.info(f"转录完成 - 处理时长: {processing_time:.2f}秒, RTF比值: {rtf_ratio:.3f}")

            # 如果启用了时间戳，打印关键信息
            if config.enable_timestamp and res and "sentence_info" in res[0]:
                logger.info(f"检测到 {len(res[0]['sentence_info'])} 个句子")

            # 检测语言
            detected_lang = detect_language_from_result(res)
            logger.info(f"检测到语言: {detected_lang}")

            # 生成字幕格式（传递ASR结果以获取真实时间戳）
            subtitle_body = generate_subtitle_segments(transcript_text, res)

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
                "device_used": self.model_manager.device,
                "audio_duration": round(audio_duration, 2),
                "processing_time": round(processing_time, 2),
                "rtf": round(rtf_ratio, 3),
                "status": "success"
            }

        except Exception as e:
            if "out of memory" in str(e).lower():
                self.model_manager.unload_model()  # 遇到错误赶紧释放资源

            # 获取音频时长（如果还没有获取）
            audio_duration = 0.0
            if 'audio_duration' in locals():
                audio_duration = locals()['audio_duration']
            else:
                audio_duration = get_audio_duration(audio_file_path)

            # 如果在转录过程中出错，尝试计算部分处理时间
            processing_time = 0.0
            rtf_ratio = 0.0
            if 'transcription_start_time' in locals():
                processing_time = time.time() - locals()['transcription_start_time']
                if audio_duration > 0:
                    rtf_ratio = processing_time / audio_duration

                # 控制台输出错误信息
                logger.error(f"\n{'='*50}")
                logger.error(f"转录失败!")
                logger.error(f"{'='*50}")
                logger.error(f"音频时长:     {format_duration(audio_duration)} ({audio_duration:.2f}秒)")
                logger.error(f"处理时长:     {format_duration(processing_time)} ({processing_time:.2f}秒)")
                logger.error(f"RTF比值:      {rtf_ratio:.3f}")
                logger.error(f"错误信息:     {str(e)}")
                logger.error(f"{'='*50}\n")

                logger.warning(f"转录过程中出错 - 部分处理时长: {processing_time:.2f}秒, RTF比值: {rtf_ratio:.3f}")

            logger.error(f"转录失败: {str(e)}")

            return {
                "status": "error",
                "message": str(e),
                "type": config.subtitle_config["type"],
                "version": config.subtitle_config["version"],
                "body": [],
                "audio_duration": round(audio_duration, 2),
                "processing_time": round(processing_time, 2),
                "rtf": round(rtf_ratio, 3)
            }