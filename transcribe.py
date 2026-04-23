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
        text: 转录文本（含标点）
        asr_result: Qwen3-ASR 结果对象（包含 time_stamps 属性）

    策略：text 包含标点，time_stamps 不含标点（ForcedAligner 的 clean_token 会过滤）。
    用 text 的标点位置分句，再通过 token 计数映射到 time_stamps 的起止时间。
    """
    import re
    import unicodedata

    body = []

    if asr_result is not None:
        time_stamps = getattr(asr_result, 'time_stamps', None)

        if time_stamps and len(time_stamps) > 0:

            def _is_cjk(ch):
                code = ord(ch)
                return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF
                        or 0x20000 <= code <= 0x2A6DF)

            def _is_kept(ch):
                if ch == "'":
                    return True
                cat = unicodedata.category(ch)
                return cat.startswith("L") or cat.startswith("N")

            def _count_align_tokens(seg_text):
                """模拟 ForcedAligner 的分词逻辑，计算一段文本对应多少个 token"""
                count = 0
                in_word = False
                for ch in seg_text:
                    if _is_cjk(ch):
                        count += 1
                        in_word = False
                    elif _is_kept(ch):
                        if not in_word:
                            count += 1
                            in_word = True
                    else:
                        in_word = False
                return count

            def _get_ts_time(ts, field):
                if hasattr(ts, field):
                    return getattr(ts, field)
                idx = 1 if field == 'start_time' else 2
                return ts[idx]

            # 先按句末标点分句（保留标点），再对过长句子按逗号拆分
            raw_sentences = re.split(r'(?<=[。！？])', text)
            segments = []
            for sent in raw_sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(sent) <= config.max_segment_length:
                    segments.append(sent)
                else:
                    for part in re.split(r'(?<=[，、；])', sent):
                        part = part.strip()
                        if part:
                            segments.append(part)

            # 将每个文本段映射到 time_stamps 的起止时间
            ts_idx = 0
            for seg_text in segments:
                token_count = _count_align_tokens(seg_text)
                if token_count == 0 or ts_idx >= len(time_stamps):
                    continue

                start_time = _get_ts_time(time_stamps[ts_idx], 'start_time')
                end_idx = min(ts_idx + token_count - 1, len(time_stamps) - 1)
                end_time = _get_ts_time(time_stamps[end_idx], 'end_time')

                body.append({
                    "from": round(start_time, 2),
                    "to": round(end_time, 2),
                    "sid": len(body) + 1,
                    "location": 2,
                    "content": seg_text,
                    "music": 0
                })
                ts_idx = end_idx + 1

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

def generate_subtitle_segments_from_timestamps(text: str, timestamps: list) -> list:
    """从统一格式的时间戳生成字幕段落

    Args:
        text: 转录文本（含标点）
        timestamps: 时间戳列表，格式为 [{"text": str, "start": float, "end": float}, ...]

    字级时间戳（GGUF/Qwen3-ASR，每段1字）需要按标点合并为短语；
    句级时间戳（FunASR，每段已是短语）直接使用。
    """
    import re

    if not timestamps:
        return generate_subtitle_segments(text)

    # 判断是否为字级时间戳：前 10 段平均文本长度 <= 2 视为字级
    avg_len = sum(len(ts.get("text", "")) for ts in timestamps[:10]) / min(len(timestamps), 10)
    if avg_len <= 2:
        return _merge_char_timestamps(text, timestamps)

    # 句级时间戳直接使用，但过滤过长段落
    body = []
    for i, ts in enumerate(timestamps):
        seg_text = ts.get("text", "").strip()
        if not seg_text:
            continue
        body.append({
            "from": round(ts.get("start", 0), 2),
            "to": round(ts.get("end", 0), 2),
            "sid": i + 1,
            "location": 2,
            "content": seg_text,
            "music": 0
        })

    return body if body else generate_subtitle_segments(text)


def _merge_char_timestamps(text: str, timestamps: list) -> list:
    """将字级时间戳按标点合并为字幕段落

    策略：先按句末标点分句，再对过长句子按逗号拆分。
    每段字幕取对应范围内首字的 start 和末字的 end。
    """
    import re

    max_len = config.max_segment_length

    # 拼出时间戳的纯文本（不含标点），用于和 text 对齐
    ts_chars = "".join(ts.get("text", "") for ts in timestamps)

    # 按标点分句
    raw_sentences = re.split(r'(?<=[。！？])', text)
    segments = []
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= max_len:
            segments.append(sent)
        else:
            for part in re.split(r'(?<=[，、；：])', sent):
                part = part.strip()
                if part:
                    segments.append(part)

    # 将每个文本段映射到时间戳范围
    # 从 text 中去掉标点，逐段推进 ts_chars 的偏移
    body = []
    ts_offset = 0

    for seg_text in segments:
        # 去掉标点后的纯文本长度，用于在 ts_chars 中定位
        clean_seg = re.sub(r'[^\w]', '', seg_text, flags=re.UNICODE)
        seg_len = len(clean_seg)

        if seg_len == 0 or ts_offset >= len(ts_chars):
            continue

        # 在 ts_chars 中找到匹配位置
        match_pos = ts_chars.find(clean_seg, ts_offset)
        if match_pos < 0:
            # 逐字推进兜底
            match_pos = ts_offset

        start_idx = match_pos
        end_idx = min(match_pos + seg_len - 1, len(timestamps) - 1)

        if start_idx < len(timestamps) and end_idx < len(timestamps):
            body.append({
                "from": round(timestamps[start_idx].get("start", 0), 2),
                "to": round(timestamps[end_idx].get("end", 0), 2),
                "sid": len(body) + 1,
                "location": 2,
                "content": seg_text,
                "music": 0
            })

        ts_offset = end_idx + 1

    return body if body else generate_subtitle_segments(text)


class TranscriptionService:
    """转录服务类，封装所有转录相关逻辑"""

    def __init__(self, model_manager):
        self.model_manager = model_manager

    async def process_transcription(self, audio_file_path: str, original_filename: str = None, audio_url: str = None, bvid: str = None, audio_id: str = None, no_cache: bool = False, file_path_for_cache: str = None):
        """处理音频转录的主函数"""
        timing = {
            "cache_check": 0.0,
            "model_load": 0.0,
            "duration_detect": 0.0,
            "transcription": 0.0,
            "subtitle_generate": 0.0,
            "cache_save": 0.0,
            "total": 0.0
        }
        total_start = time.time()

        # 检查转录缓存（除非禁用缓存）
        cache_check_start = time.time()
        if not no_cache:
            if file_path_for_cache:
                cached_result = cache_manager.get_cached_transcript(file_path=file_path_for_cache)
                if cached_result:
                    cached_result.pop('cached_at', None)
                    logger.info(f"使用缓存的转录结果，音频时长: {cached_result.get('audio_duration', 'unknown')}秒")
                    return cached_result
            elif audio_id and bvid:
                cached_result = cache_manager.get_cached_transcript(None, bvid, audio_id)
                if cached_result:
                    cached_result.pop('cached_at', None)
                    logger.info(f"使用缓存的转录结果，音频时长: {cached_result.get('audio_duration', 'unknown')}秒")
                    return cached_result
            elif audio_url or bvid:
                cached_result = cache_manager.get_cached_transcript(audio_url, bvid)
                if cached_result:
                    cached_result.pop('cached_at', None)
                    logger.info(f"使用缓存的转录结果，音频时长: {cached_result.get('audio_duration', 'unknown')}秒")
                    return cached_result
        timing["cache_check"] = time.time() - cache_check_start

        try:
            # 1. 触发懒加载
            model_load_start = time.time()
            backend = self.model_manager.load_model_if_needed()
            timing["model_load"] = time.time() - model_load_start
        except Exception as e:
            duration_start = time.time()
            audio_duration = get_audio_duration(audio_file_path)
            timing["duration_detect"] = time.time() - duration_start
            timing["total"] = time.time() - total_start
            return {
                "status": "error",
                "message": f"Model load failed: {str(e)}",
                "type": config.subtitle_config["type"],
                "version": config.subtitle_config["version"],
                "audio_duration": round(audio_duration, 2),
                "processing_time": 0.0,
                "rtf": 0.0,
                "timing": timing
            }

        try:
            # 2. 获取音频时长
            filename_to_log = original_filename or audio_file_path
            duration_start = time.time()
            audio_duration = get_audio_duration(audio_file_path)
            timing["duration_detect"] = time.time() - duration_start
            if audio_duration > 0:
                logger.info(f"音频时长: {audio_duration:.2f}秒")
            else:
                logger.warning("无法获取音频时长")

            logger.info(f"开始识别: {filename_to_log}")

            # 3. 调用后端转录
            transcription_start_time = time.time()
            result = backend.transcribe(audio_file_path)
            processing_time = time.time() - transcription_start_time
            timing["transcription"] = processing_time

            # 刷新活跃时间
            self.model_manager.last_active_time = time.time()

            transcript_text = result.text
            detected_lang = result.language
            timestamps = result.timestamps

            # 优先使用后端计算的 RTF，否则本地计算
            if result.performance and result.performance.get("rtf"):
                rtf_ratio = result.performance["rtf"]
            else:
                rtf_ratio = processing_time / audio_duration if audio_duration > 0 else 0.0

            logger.info(f"\n{'='*50}")
            logger.info(f"转录完成! ({backend.name})")
            logger.info(f"{'='*50}")
            logger.info(f"音频时长:     {format_duration(audio_duration)} ({audio_duration:.2f}秒)")
            logger.info(f"处理时长:     {format_duration(processing_time)} ({processing_time:.2f}秒)")
            logger.info(f"RTF比值:      {rtf_ratio:.3f}")
            if rtf_ratio < 1:
                logger.info(f"状态:         实时处理 (RTF < 1)")
            else:
                logger.info(f"状态:         非实时处理 (RTF >= 1)")
            logger.info(f"{'='*50}\n")

            logger.info(f"检测到语言: {detected_lang}")

            # 4. 生成字幕格式
            subtitle_start = time.time()
            if timestamps:
                subtitle_body = generate_subtitle_segments_from_timestamps(transcript_text, timestamps)
            else:
                subtitle_body = generate_subtitle_segments(transcript_text)
            timing["subtitle_generate"] = time.time() - subtitle_start

            subtitle_config = config.subtitle_config
            timing["total"] = time.time() - total_start

            response = {
                "font_size": subtitle_config["font_size"],
                "font_color": subtitle_config["font_color"],
                "background_alpha": subtitle_config["background_alpha"],
                "background_color": subtitle_config["background_color"],
                "Stroke": subtitle_config["stroke"],
                "type": subtitle_config["type"],
                "lang": detected_lang,
                "version": subtitle_config["version"],
                "body": subtitle_body,
                "device_used": backend.device,
                "audio_duration": round(audio_duration, 2),
                "processing_time": round(processing_time, 2),
                "rtf": round(rtf_ratio, 3),
                "timing": {k: round(v, 3) for k, v in timing.items()},
                "status": "success"
            }

            # 保存到缓存
            cache_save_start = time.time()
            if file_path_for_cache:
                cache_manager.save_transcript_to_cache(file_path=file_path_for_cache, transcript_data=response)
            elif audio_id and bvid:
                cache_manager.save_transcript_to_cache(None, response, bvid, audio_id)
            elif audio_url or bvid:
                cache_manager.save_transcript_to_cache(audio_url, response, bvid)
            timing["cache_save"] = time.time() - cache_save_start
            response["timing"]["cache_save"] = round(timing["cache_save"], 3)
            response["timing"]["total"] = round(time.time() - total_start, 3)

            return response

        except Exception as e:
            if "out of memory" in str(e).lower():
                self.model_manager.unload_model()

            audio_duration = 0.0
            if 'audio_duration' in locals():
                audio_duration = locals()['audio_duration']
            else:
                duration_start = time.time()
                audio_duration = get_audio_duration(audio_file_path)
                timing["duration_detect"] = time.time() - duration_start

            processing_time = 0.0
            rtf_ratio = 0.0
            if 'transcription_start_time' in locals():
                processing_time = time.time() - locals()['transcription_start_time']
                timing["transcription"] = processing_time
                if audio_duration > 0:
                    rtf_ratio = processing_time / audio_duration

            logger.error(f"\n{'='*50}")
            logger.error(f"转录失败!")
            logger.error(f"{'='*50}")
            logger.error(f"音频时长:     {format_duration(audio_duration)} ({audio_duration:.2f}秒)")
            logger.error(f"处理时长:     {format_duration(processing_time)} ({processing_time:.2f}秒)")
            logger.error(f"RTF比值:      {rtf_ratio:.3f}")
            logger.error(f"错误信息:     {str(e)}")
            logger.error(f"{'='*50}\n")

            timing["total"] = time.time() - total_start

            return {
                "status": "error",
                "message": str(e),
                "type": config.subtitle_config["type"],
                "version": config.subtitle_config["version"],
                "body": [],
                "audio_duration": round(audio_duration, 2),
                "processing_time": round(processing_time, 2),
                "rtf": round(rtf_ratio, 3),
                "timing": {k: round(v, 3) for k, v in timing.items()}
            }
