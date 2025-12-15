import time
import logging
from config import config

logger = logging.getLogger(__name__)

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
            return {
                "status": "error",
                "message": f"Model load failed: {str(e)}",
                "type": config.subtitle_config["type"],
                "version": config.subtitle_config["version"]
            }

        try:
            # 2. 转录
            filename_to_log = original_filename or audio_file_path
            logger.info(f"开始识别: {filename_to_log}")

            res = asr_model.generate(
                input=audio_file_path,
                batch_size_s=config.batch_size_s,
                disable_pbar=True
            )

            # 刷新活跃时间
            self.model_manager.last_active_time = time.time()

            # 获取转录文本
            transcript_text = res[0]["text"] if res else ""

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
                "status": "success"
            }

        except Exception as e:
            if "out of memory" in str(e).lower():
                self.model_manager.unload_model()  # 遇到错误赶紧释放资源
            return {
                "status": "error",
                "message": str(e),
                "type": config.subtitle_config["type"],
                "version": config.subtitle_config["version"],
                "body": []
            }