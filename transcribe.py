import time
import os
import subprocess
from pathlib import Path
from config import config
from logger_config import setup_logger
from cache_manager import cache_manager

logger = setup_logger(__name__)

# 语言映射：Qwen3-ASR 返回全名 → 项目使用的短代码
LANG_MAP = {
    "Chinese": "zh", "English": "en", "Japanese": "ja",
    "Korean": "ko", "French": "fr", "German": "de",
    "Spanish": "es", "Portuguese": "pt", "Russian": "ru",
    "Arabic": "ar", "Thai": "th", "Vietnamese": "vi",
    "Indonesian": "id", "Italian": "it", "Cantonese": "yue",
    "Turkish": "tr", "Hindi": "hi", "Malay": "ms",
}

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
        asr_result: Qwen3-ASR 结果对象（包含 time_stamps 属性）
    """
    import re

    body = []

    # 如果有 ASR 结果且包含时间戳
    if asr_result is not None:
        time_stamps = getattr(asr_result, 'time_stamps', None)

        if time_stamps and len(time_stamps) > 0:
            # time_stamps 是字/词级别列表，每个元素有 .text, .start_time, .end_time
            # 按标点符号分组为句子级时间戳

            current_sentence = []
            sentence_start = None

            for ts in time_stamps:
                word_text = ts.text if hasattr(ts, 'text') else str(ts[0])
                word_start = ts.start_time if hasattr(ts, 'start_time') else ts[1]
                word_end = ts.end_time if hasattr(ts, 'end_time') else ts[2]

                if sentence_start is None:
                    sentence_start = word_start

                current_sentence.append(word_text)

                # 检查是否是句子结束标点
                if word_text in ['，', '。', '！', '？', '；', '、'] or word_text.strip() == '':
                    if current_sentence:
                        sentence_text = ''.join(current_sentence).strip()
                        if sentence_text:
                            body.append({
                                "from": round(sentence_start, 2),
                                "to": round(word_end, 2),
                                "sid": len(body) + 1,
                                "location": 2,
                                "content": sentence_text,
                                "music": 0
                            })
                        current_sentence = []
                        sentence_start = None

            # 处理最后一个未结束的句子
            if current_sentence:
                sentence_text = ''.join(current_sentence).strip()
                if sentence_text and sentence_start is not None:
                    last_end = time_stamps[-1].end_time if hasattr(time_stamps[-1], 'end_time') else time_stamps[-1][2]
                    body.append({
                        "from": round(sentence_start, 2),
                        "to": round(last_end, 2),
                        "sid": len(body) + 1,
                        "location": 2,
                        "content": sentence_text,
                        "music": 0
                    })

            if body:
                return body

    # 回退路径：无时间戳时使用均匀分配
    segments = split_text_into_segments(text)
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

    async def process_transcription(self, audio_file_path: str, original_filename: str = None, audio_url: str = None, bvid: str = None, audio_id: str = None, no_cache: bool = False, file_path_for_cache: str = None):
        """
        处理音频转录的主函数

        Args:
            audio_file_path: 音频文件的路径
            original_filename: 原始文件名（用于日志）
            audio_url: 音频URL（用于缓存，可选）
            bvid: B站视频ID（用于缓存，可选）
            audio_id: 音频ID（用于缓存，可选）
            no_cache: 是否禁用缓存（默认False）
            file_path_for_cache: 文件路径（用于网盘文件缓存，可选）

        Returns:
            dict: 转录结果
        """
        # 检查转录缓存（除非禁用缓存）
        if not no_cache:
            # 优先使用文件路径作为缓存键（网盘文件）
            if file_path_for_cache:
                cached_result = cache_manager.get_cached_transcript(file_path=file_path_for_cache)
                if cached_result:
                    cached_result.pop('cached_at', None)
                    logger.info(f"使用缓存的转录结果，音频时长: {cached_result.get('audio_duration', 'unknown')}秒")
                    return cached_result
            elif audio_id and bvid:
                # 优先使用BVID+音频ID检查转录缓存
                cached_result = cache_manager.get_cached_transcript(None, bvid, audio_id)
                if cached_result:
                    cached_result.pop('cached_at', None)
                    logger.info(f"使用缓存的转录结果，音频时长: {cached_result.get('audio_duration', 'unknown')}秒")
                    return cached_result
            elif audio_url or bvid:
                # 兼容旧方式
                cached_result = cache_manager.get_cached_transcript(audio_url, bvid)
                if cached_result:
                    cached_result.pop('cached_at', None)
                    logger.info(f"使用缓存的转录结果，音频时长: {cached_result.get('audio_duration', 'unknown')}秒")
                    return cached_result

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

            # 根据配置决定是否启用时间戳
            return_time_stamps = bool(config.forced_aligner)

            res = asr_model.transcribe(
                audio=audio_file_path,
                language=None,  # 自动检测
                return_time_stamps=return_time_stamps,
            )

            # 调试：打印返回结果结构
            if res:
                logger.info(f"Qwen3-ASR 返回: language={res[0].language}, text 长度={len(res[0].text)}")
                if return_time_stamps and hasattr(res[0], 'time_stamps'):
                    logger.info(f"时间戳数量: {len(res[0].time_stamps) if res[0].time_stamps else 0}")

            # 计算处理时长（纯粹的转换时间）
            processing_time = time.time() - transcription_start_time

            # 刷新活跃时间
            self.model_manager.last_active_time = time.time()

            # 获取转录文本
            transcript_text = res[0].text if res else ""

            # 计算RTF比值
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

            # 检测语言（从 Qwen3-ASR 结果直接获取）
            detected_lang = LANG_MAP.get(res[0].language, "zh") if res else "zh"
            logger.info(f"检测到语言: {detected_lang}")

            # 生成字幕格式（传递 ASR 结果对象以获取时间戳）
            subtitle_body = generate_subtitle_segments(transcript_text, res[0] if res else None)

            # 从配置获取字幕样式
            subtitle_config = config.subtitle_config

            result = {
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

            # 保存到缓存
            if file_path_for_cache:
                # 使用文件路径保存缓存
                cache_manager.save_transcript_to_cache(file_path=file_path_for_cache, transcript_data=result)
            elif audio_id and bvid:
                # 优先使用BVID+音频ID保存
                cache_manager.save_transcript_to_cache(None, result, bvid, audio_id)
            elif audio_url or bvid:
                # 兼容旧方式
                cache_manager.save_transcript_to_cache(audio_url, result, bvid)

            return result

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
