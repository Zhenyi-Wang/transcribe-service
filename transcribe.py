import time
import os
import asyncio
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

# ================= ASR context 钳制 =================
# 与 qwen_asr_gguf 默认定容公式同源（chunk_size=40s / memory_num=1，当前无配置承载，
# 两个 backend 均以库默认值构造；未来配置化时此处改为读取配置）。
# frames_per_chunk = int(chunk_size*13)；n_ubatch = ceil((audio_frames+500)/512)*512
# context 预算 = n_ubatch − audio_frames − 32（固定模板头尾），1 字符 ≈ 1 token 悲观换算
_ASR_CTX_CHUNK_SIZE = 40.0
_ASR_CTX_MEMORY_NUM = 1


def clamp_asr_context(context):
    """按引擎定容公式钳制 context 长度（理论上限；逐 chunk 精确拟合由引擎守卫负责）。"""
    if not context:
        return None
    frames_per_chunk = int(_ASR_CTX_CHUNK_SIZE * 13)
    audio_frames = frames_per_chunk * (_ASR_CTX_MEMORY_NUM + 1)
    n_ubatch = max(512, ((audio_frames + 500 + 511) // 512) * 512)
    budget = n_ubatch - audio_frames - 32
    return context[:budget]


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

def _assign_speakers_to_timestamps(timestamps: list, turns: list) -> list:
    """给每个时间戳项附加 speaker 键（spec 三级归属规则，不改输入 dict）

    1. 与某 turn 有重叠 → 归重叠面积最大者（字主体归属优先）
    2. 与所有 turn 无重叠（换人间隙/超范围）→ 归最近 turn（项中点到 turn 区间的距离）
    """
    def _nearest_speaker(t: float) -> int:
        best_speaker, best_dist = None, float("inf")
        for turn in turns:
            d = max(turn.start - t, 0.0, t - turn.end)
            if d < best_dist:
                best_dist, best_speaker = d, turn.speaker
        return best_speaker

    out = []
    for ts in timestamps:
        s, e = ts.get("start", 0.0), ts.get("end", 0.0)
        best_speaker, best_overlap = None, 0.0
        for turn in turns:
            overlap = min(e, turn.end) - max(s, turn.start)
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, turn.speaker
        if best_speaker is None:
            best_speaker = _nearest_speaker((s + e) / 2)
        out.append({**ts, "speaker": best_speaker})
    return out


def _aggregate_speakers(body: list) -> list:
    """从 body 段聚合说话人汇总；-1/缺失不计入，id 升序"""
    stat = {}
    for seg in body:
        spk = seg.get("speaker")
        if spk is None or spk < 0:
            continue
        d = stat.setdefault(spk, {"id": spk, "duration": 0.0, "segments": 0})
        d["duration"] += seg.get("to", 0.0) - seg.get("from", 0.0)
        d["segments"] += 1
    return [{"id": d["id"], "duration": round(d["duration"], 1), "segments": d["segments"]}
            for d in sorted(stat.values(), key=lambda x: x["id"])]


def _posthoc_align_speakers(body: list, turns: list) -> list:
    """句级段的后置说话人对齐：与 turn 最大重叠 ≥ 段时长 50% 才赋标签，否则 -1"""
    for seg in body:
        s, e = seg.get("from", 0.0), seg.get("to", 0.0)
        best, best_ov = None, 0.0
        for turn in turns:
            ov = min(e, turn.end) - max(s, turn.start)
            if ov > best_ov:
                best_ov, best = ov, turn.speaker
        seg["speaker"] = best if (best is not None and best_ov >= (e - s) * 0.5) else -1
    return body


def generate_subtitle_segments_from_timestamps(text: str, timestamps: list, lang: str = "zh",
                                                audio_duration: float = None, turns: list = None) -> list:
    """从统一格式的时间戳生成字幕段落

    Args:
        text: 转录文本（含标点）
        timestamps: 时间戳列表，格式为 [{"text": str, "start": float, "end": float}, ...]
        lang: 检测到的语言代码（zh/en/ja/ko 等），用于决定空格处理策略
        audio_duration: 音频总时长（秒），用于钳制兜底估算段不超出音频末尾
        turns: SpeakerTurn 列表（diarize 结果）；None/单说话人时不注入 speaker，
            行为与历史版本一致。

    字级时间戳（GGUF/Qwen3-ASR，每段1字）需要按标点合并为短语；
    句级时间戳（FunASR，每段已是短语）直接使用。
    """
    import re

    if not timestamps:
        return generate_subtitle_segments(text)

    # speaker 模式判定：turns 有效且聚类 ≥2 人（单人退化），否则保持 timestamps 原样
    speaker_mode = False
    if turns and len({t.speaker for t in turns}) >= 2:
        timestamps = _assign_speakers_to_timestamps(timestamps, turns)
        speaker_mode = True

    # CJK 语言不使用空格分词，其他语言（英法德西等）使用空格
    _CJK_LANGS = {"zh", "ja", "ko", "yue", ""}

    # 判断是否为字级时间戳：前 10 段平均文本长度 <= 2 视为字级
    avg_len = sum(len(ts.get("text", "")) for ts in timestamps[:10]) / min(len(timestamps), 10)
    if avg_len <= 2:
        if lang in _CJK_LANGS:
            body = _merge_char_timestamps(text, timestamps, audio_duration)
            return body if speaker_mode else _strip_speaker(body)
        # 非 CJK 语言的碎片级时间戳（Qwen3-ForcedAligner 对天城文按基字符+
        # 组合符号对齐，2026-08 印地语事故）：碎片保留空格，先按空格重组为
        # 词级，再走空格语言分段（其强制拆分兜底可防单条超长字幕）
        regrouped = _regroup_fragments_by_space(timestamps)
        if regrouped:
            timestamps = regrouped

    use_spaces = lang not in _CJK_LANGS

    if use_spaces:
        body = _segment_by_punctuation(timestamps, text)
    else:
        # 非空格语言（日/韩）/ 句级（funasr）：简单按 min_len 合并
        body = _segment_simple(timestamps, text)
        if speaker_mode:
            body = _posthoc_align_speakers(body, turns)
    if speaker_mode:
        return body
    return _strip_speaker(body)


def _strip_speaker(body: list) -> list:
    """非 speaker 模式兜底：剥掉任何可能混入的 speaker 键（回归红线保障）"""
    for seg in body:
        seg.pop("speaker", None)
    return body


def _majority_speaker(intervals, w_start: float, w_end: float):
    """[(start, end, speaker), ...] 中与 [w_start, w_end] 重叠总面积最大的 speaker（按 speaker 聚合）；全无 speaker 返回 None"""
    totals = {}
    for s, e, sp in intervals:
        if sp is None:
            continue
        ov = max(0.0, min(w_end, e) - max(w_start, s))
        totals[sp] = totals.get(sp, 0.0) + ov
    if not totals:
        return None
    return max(totals, key=totals.get)


def _regroup_fragments_by_space(timestamps: list) -> list:
    """把碎片级时间戳按空格重组为词级时间戳。

    Qwen3-ForcedAligner 对天城文（印地语）等文字按基字符/组合符号碎片对齐，
    碎片保留空格信息（如 "ैं " 或独立 " "）。以尾随空格为词边界拼接碎片，
    每词取首碎片 start、末碎片 end，供 _segment_by_punctuation 分段。

    Returns:
        词级时间戳列表 [{"text", "start", "end"}, ...]，text 保留单个尾随空格
        （_segment_by_punctuation 依赖尾随空格判断词间距）；无可重组内容时为空列表。
        碎片带 speaker 键时，词继承重叠最大的碎片 speaker（跨说话人词取主体）。
    """
    words = []
    current = []  # [(frag_text, start, end, speaker)]

    def _flush_word():
        if not current:
            return
        word_text = "".join(w[0] for w in current).rstrip()
        if word_text:
            w_start, w_end = current[0][1], current[-1][2]
            spk = _majority_speaker([(s, e, sp) for (_, s, e, sp) in current], w_start, w_end)
            entry = {"text": word_text + " ", "start": w_start, "end": w_end}
            if spk is not None:
                entry["speaker"] = spk
            words.append(entry)
        current.clear()

    for ts in timestamps:
        frag = ts.get("text", "")
        if not frag:
            continue
        current.append((frag, ts.get("start", 0), ts.get("end", 0), ts.get("speaker")))
        if frag.endswith(" "):
            _flush_word()
    _flush_word()  # 尾部无空格的残余词

    # 最后一个词的尾随空格无后续词，去掉以免拼接出尾空白
    if words:
        words[-1]["text"] = words[-1]["text"].rstrip()
    return words


def _merge_char_timestamps(text: str, timestamps: list, audio_duration: float = None) -> list:
    """将字级时间戳按标点合并为字幕段落

    策略：先按句末标点分句，再对过长句子按逗号拆分。
    每段字幕取对应范围内首字的 start 和末字的 end。
    """
    import re

    max_len = config.max_segment_length
    min_len = 5

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
                if not part:
                    continue
                if len(part) <= max_len * 3:
                    segments.append(part)
                else:
                    # 无次级标点的超长句强制按长度拆分：分句标点不匹配的语言
                    # 曾整段输出单条字幕（2026-08 印地语事故），此处为兜底
                    limit = max_len * 3
                    segments.extend(part[i:i + limit] for i in range(0, len(part), limit))

    # 合并过短的段落（少于 min_len 字）到前一段
    merged = []
    for seg in segments:
        if merged and len(re.sub(r'[^\w]', '', seg, flags=re.UNICODE)) < min_len:
            merged[-1] += seg
        else:
            merged.append(seg)
    segments = merged

    # 将每个文本段映射到时间戳范围
    # 从 text 中去掉标点，逐段推进 ts_chars 的偏移
    #
    # 文本段一个都不能丢：时间戳与文本失配（音乐复读导致 aligner 截断、
    # 英文词空格导致 find 错位）时降级为估算时间兜底，而不是静默 continue。
    # 曾因 continue 丢弃尾部 60%+ 文本（讲章批量转录事故，2026-07）。
    body = []
    ts_offset = 0
    last_end_time = 0.0  # 上一段（含兜底段）的结束时间，兜底段从此推进
    last_speaker = None  # 上一段（含兜底段）的说话人，估算兜底段继承

    for seg_text in segments:
        # 去掉标点后的纯文本长度，用于在 ts_chars 中定位
        clean_seg = re.sub(r'[^\w]', '', seg_text, flags=re.UNICODE)
        seg_len = len(clean_seg)

        if seg_len == 0:
            continue

        # 在 ts_chars 中找到匹配位置
        match_pos = -1
        if ts_offset < len(ts_chars):
            match_pos = ts_chars.find(clean_seg, ts_offset)
            if match_pos < 0:
                # 逐字推进兜底
                match_pos = ts_offset

        if 0 <= match_pos < len(timestamps):
            # 有真实时间戳：取首尾字时间
            start_idx = match_pos
            end_idx = min(match_pos + seg_len - 1, len(timestamps) - 1)
            ts_offset = end_idx + 1

            # speaker 模式：项带 speaker 键且段内存在变化 → 段内边界拆分。
            # 拆分不破坏全局对齐：各子段 clean 文本拼接 == 原 clean_seg，
            # 子段时间范围取各自首尾项，ts_offset 推进不变。
            # 过短子段（<4 字且 <1s，即 diarization 边界噪声落在句中的漂移）
            # 被吸收进相邻子段，避免把词从中间劈开产生 0.4s 碎字幕。
            range_speakers = [timestamps[i].get("speaker") for i in range(start_idx, end_idx + 1)]
            if any(sp is not None for sp in range_speakers) and len(set(range_speakers)) > 1:
                cuts = [start_idx]
                for k in range(1, len(range_speakers)):
                    if range_speakers[k] != range_speakers[k - 1]:
                        cuts.append(start_idx + k)
                cuts.append(end_idx + 1)

                # 吸收过短子段：迭代并入相邻较大子段，直至全部达标或仅剩一组
                MIN_SUB_CHARS, MIN_SUB_DUR = 4, 1.0
                groups = [[cuts[i], cuts[i + 1]] for i in range(len(cuts) - 1)]  # [start, end)

                # 每个子段的主导 speaker（按项时长与子段区间重叠）
                def _group_speaker(a, b):
                    return _majority_speaker(
                        [(timestamps[i].get("start", 0), timestamps[i].get("end", 0), range_speakers[i - start_idx])
                         for i in range(a, b)],
                        timestamps[a].get("start", 0), timestamps[b - 1].get("end", 0))

                changed = True
                while changed and len(groups) > 1:
                    changed = False
                    for gi, (a, b) in enumerate(groups):
                        dur = timestamps[b - 1].get("end", 0) - timestamps[a].get("start", 0)
                        if (b - a) < MIN_SUB_CHARS and dur < MIN_SUB_DUR:
                            if gi == 0:
                                groups[1][0] = a
                            elif gi == len(groups) - 1:
                                groups[gi - 1][1] = b
                            else:
                                left, right = groups[gi - 1], groups[gi + 1]
                                if left[1] - left[0] >= right[1] - right[0]:
                                    left[1] = b
                                else:
                                    right[0] = a
                            groups.pop(gi)
                            changed = True
                            break

                # 相邻子段主导 speaker 相同 → 合并（漂移吸收后两侧常同属一人）
                gi = 0
                while gi < len(groups) - 1:
                    if _group_speaker(*groups[gi]) == _group_speaker(*groups[gi + 1]):
                        groups[gi][1] = groups[gi + 1][1]
                        groups.pop(gi + 1)
                    else:
                        gi += 1

                if len(groups) == 1:
                    # 全部吸收：整段输出，标主导 speaker
                    spk = _group_speaker(groups[0][0], groups[0][1])
                    seg_from = timestamps[start_idx].get("start", 0)
                    seg_to = timestamps[end_idx].get("end", 0)
                    if seg_to <= seg_from:
                        seg_to = seg_from + 0.5
                    body.append({
                        "from": round(seg_from, 2),
                        "to": round(seg_to, 2),
                        "sid": len(body) + 1,
                        "location": 2,
                        "content": seg_text,
                        "music": 0,
                        "speaker": spk,
                    })
                    last_end_time = max(last_end_time, seg_to)
                    last_speaker = spk
                    continue

                # 把 seg_text 按 clean 字符切分位置切成含标点子串（标点跟随前一个 clean 字符）
                cuts = [groups[0][0]] + [g[1] for g in groups[:-1]] + [groups[-1][1]]
                cut_clean_pos = {c - start_idx for c in cuts[1:-1]}
                text_parts = []
                buf, ci = [], 0
                for ch in seg_text:
                    if re.match(r'[\w]', ch, flags=re.UNICODE):
                        if ci in cut_clean_pos and buf:
                            text_parts.append("".join(buf))
                            buf = []
                        ci += 1
                    buf.append(ch)
                text_parts.append("".join(buf))
                text_parts = [p for p in text_parts if p.strip()]

                for gi in range(len(cuts) - 1):
                    g_start, g_end = cuts[gi], cuts[gi + 1] - 1  # 项索引闭区间
                    sub_speaker = _group_speaker(g_start, g_end + 1)
                    seg_from = timestamps[g_start].get("start", 0)
                    seg_to = timestamps[g_end].get("end", 0)
                    if seg_to <= seg_from:
                        seg_to = seg_from + 0.3  # 子段时间塌缩兜底，杜绝零时长字幕
                    body.append({
                        "from": round(seg_from, 2),
                        "to": round(seg_to, 2),
                        "sid": len(body) + 1,
                        "location": 2,
                        "content": text_parts[gi] if gi < len(text_parts) else "",
                        "music": 0,
                        "speaker": sub_speaker,
                    })
                last_end_time = max(last_end_time, timestamps[end_idx].get("end", 0))
                last_speaker = range_speakers[-1]  # 拆分后更新，供后续估算兜底段继承
                continue

            seg_from = timestamps[start_idx].get("start", 0)
            seg_to = timestamps[end_idx].get("end", 0)
            if seg_to <= seg_from:
                seg_to = seg_from + 0.5
            # speaker 模式下，正常段附加段内（恒一的）speaker
            range_speakers = [timestamps[i].get("speaker") for i in range(start_idx, end_idx + 1)]
            known = [sp for sp in range_speakers if sp is not None]
            body_extra = {"speaker": known[0]} if known else {}
        else:
            # 时间戳耗尽/失配：用上一段结束时间 + 按语速估算的时长兜底。
            # 估算段必须钳制到音频总时长：音乐复读导致时间戳耗尽时，估算会
            # 一路推进超出音频实际长度（2026-09 祷告会事故，尾部 30 段幻影
            # 时间戳超出音频 45s）。超界段钉在音频末尾，文本一个不丢。
            seg_from = last_end_time
            seg_to = last_end_time + max(seg_len * 0.3, 1.0)
            if audio_duration and audio_duration > 0:
                if seg_from >= audio_duration:
                    seg_from = seg_to = audio_duration
                elif seg_to > audio_duration:
                    seg_to = audio_duration
            body_extra = {"speaker": last_speaker} if last_speaker is not None else {}

        body.append({
            "from": round(seg_from, 2),
            "to": round(seg_to, 2),
            "sid": len(body) + 1,
            "location": 2,
            "content": seg_text,
            "music": 0,
            **body_extra,
        })
        last_end_time = max(last_end_time, seg_to)
        if body_extra.get("speaker") is not None:
            last_speaker = body_extra["speaker"]

    return body if body else generate_subtitle_segments(text)


def _segment_simple(timestamps: list, text: str) -> list:
    """非空格语言（日/韩）的简单合并：按 min_len 合并短词，不处理空格"""
    min_len = 5
    body = []
    for ts in timestamps:
        seg_text = ts.get("text", "").strip()
        if not seg_text:
            continue
        if body and len(seg_text) < min_len:
            body[-1]["content"] += seg_text
            body[-1]["to"] = round(ts.get("end", 0), 2)
        else:
            body.append({
                "from": round(ts.get("start", 0), 2),
                "to": round(ts.get("end", 0), 2),
                "sid": 0,
                "location": 2,
                "content": seg_text,
                "music": 0
            })
    for i, seg in enumerate(body):
        seg["sid"] = i + 1
    return body if body else generate_subtitle_segments(text)


def _segment_by_punctuation(timestamps: list, text: str) -> list:
    """空格分隔语言的标点感知分段

    积累单词直到遇到句末标点（.!?）后 flush 一个段落；
    段落过长时在逗号/分号（,;:）处 flush；
    无标点的超长段落按 max_len*3 强制 flush 避免单段过长。
    单独标点项（",", "." 等）总是附加到当前段落，不独立成段。
    item 带 speaker 键时（说话人模式），相邻项 speaker 变化同样触发 flush，
    段落继承其成员词的（恒一的）speaker。
    """
    import re
    max_len = config.max_segment_length
    has_speaker = any("speaker" in ts for ts in timestamps)

    body = []
    # current_words: [(word_text, start_time, end_time, speaker)]
    # word_text 对于段首词不含前置空格，后续词含前置空格
    current_words = []
    pending_space = False

    def _flush():
        if not current_words:
            return
        content = "".join(w[0] for w in current_words)
        entry = {
            "from": round(current_words[0][1], 2),
            "to": round(current_words[-1][2], 2),
            "sid": 0,
            "location": 2,
            "content": content,
            "music": 0
        }
        if has_speaker and current_words[0][3] is not None:
            entry["speaker"] = current_words[0][3]
        body.append(entry)
        current_words.clear()

    def _is_punct_only(s: str) -> bool:
        return all(c in '.!?,;:\'"()-।॥' for c in s)

    for ts in timestamps:
        raw_text = ts.get("text", "")
        seg_text = raw_text.strip()
        if not seg_text:
            if raw_text:
                pending_space = True
            continue

        # 单独标点项：总是附加到当前段落，不出现在段首
        if _is_punct_only(seg_text):
            if current_words:
                current_words.append((seg_text, ts.get("start", 0), ts.get("end", 0), ts.get("speaker")))
                # 句末标点触发 flush，确保 "ID." 之类不被拆散（।॥ 为印地语句读）
                if seg_text in '.!?।॥':
                    _flush()
            # 如果 current_words 为空，这是段首标点（如 " ."），丢弃
            pending_space = raw_text.rstrip() != raw_text
            continue

        # 说话人模式：相邻词 speaker 变化 → flush 当前段再开新段
        if has_speaker and current_words and \
                current_words[-1][3] is not None and ts.get("speaker") != current_words[-1][3]:
            _flush()

        # 构建词文本：段首词无前置空格，后续词根据 pending_space 决定
        word_text = (" " if pending_space and current_words else "") + seg_text
        start = ts.get("start", 0)
        end = ts.get("end", 0)

        current_words.append((word_text, start, end, ts.get("speaker")))
        total_len = sum(len(w[0]) for w in current_words)

        stripped = seg_text
        ends_sentence = stripped.endswith(('.', '!', '?', '।', '॥'))
        ends_clause = stripped.endswith((',', ';', ':'))

        if ends_sentence:
            _flush()
        elif ends_clause and total_len >= max_len:
            _flush()
        elif total_len >= max_len * 3:
            # 无标点超长段强制 flush
            _flush()

        pending_space = raw_text.rstrip() != raw_text

    _flush()

    for i, seg in enumerate(body):
        seg["sid"] = i + 1

    return body if body else generate_subtitle_segments(text)


def _diarize_samples(audio_file_path: str):
    """分离线程函数体：解码 + 推理（重依赖延迟导入；任何异常向上抛由编排层降级）"""
    from qwen_asr_gguf.inference.audio import load_audio
    from diarization.manager import get_manager
    samples = load_audio(audio_file_path)
    return get_manager().diarize(samples)


def _diarize_timeout(audio_duration: float) -> float:
    """分离超时上限（秒）：下限 60s，随音频时长放宽"""
    return max(60.0, audio_duration * 0.5)


async def _diarize_turns_or_error(audio_file_path: str, video_id, download_time: float, total_start: float):
    """仅分离公共执行体（不跑 ASR）：返回 (turns, error, timing)，成功时 error 为 None。

    未启用/超时/失败时 turns 为 None 且 error 为 status=error 响应——
    仅分离模式的调用方需要显式失败信号来决定是否拼接，不做静默降级。
    """
    timing = {"download": round(download_time, 3), "diarization": 0.0}

    def _err(message: str) -> dict:
        return {"status": "error", "message": message, "video_id": video_id,
                "timing": {**timing, "total": round(time.time() - total_start, 3)}}

    if not config.diarization_enabled:
        logger.warning("收到仅分离请求，但 diarization.enabled=false")
        return None, _err("diarization disabled"), timing

    audio_duration = get_audio_duration(audio_file_path)
    timeout = _diarize_timeout(audio_duration)
    diarize_start = time.time()
    try:
        turns = await asyncio.wait_for(asyncio.to_thread(_diarize_samples, audio_file_path), timeout=timeout)
    except asyncio.TimeoutError:
        timing["diarization"] = time.time() - diarize_start
        logger.warning(f"说话人分离超时（>{timeout:.0f}s）")
        return None, _err("diarization timeout"), timing
    except Exception as e:
        timing["diarization"] = time.time() - diarize_start
        logger.warning("说话人分离失败", exc_info=True)
        return None, _err(f"diarization failed: {e}"), timing
    timing["diarization"] = time.time() - diarize_start
    return turns, None, timing


def _turns_speaker_summary(turns: list) -> list:
    """turns → 按说话人聚合的摘要（id 升序）"""
    stat = {}
    for t in turns:
        d = stat.setdefault(t.speaker, {"id": t.speaker, "duration": 0.0, "turns": 0})
        d["duration"] += t.end - t.start
        d["turns"] += 1
    return [{"id": d["id"], "duration": round(d["duration"], 1), "turns": d["turns"]}
            for d in sorted(stat.values(), key=lambda x: x["id"])]


async def diarize_only(audio_file_path: str, video_id: str = None, download_time: float = 0.0) -> dict:
    """仅说话人分离（不跑 ASR）：返回说话人时间轴。

    单人也原样返回 turns（是否采用由调用方判断）。
    不涉及 ASR，不写转录缓存（音频下载缓存由下载层负责）。
    """
    total_start = time.time()
    turns, error, timing = await _diarize_turns_or_error(audio_file_path, video_id, download_time, total_start)
    if error:
        return error

    speakers = _turns_speaker_summary(turns)
    logger.info(f"仅说话人分离完成: {video_id} {len(speakers)} 人 / {len(turns)} turns")
    return {
        "status": "success",
        "video_id": video_id,
        "turns": [{"speaker": t.speaker, "start": round(t.start, 3), "end": round(t.end, 3)} for t in turns],
        "speakers": speakers,
        "timing": {**timing, "total": round(time.time() - total_start, 3)},
    }


async def diarize_merge_subtitle(body: list, audio_file_path: str, video_id: str = None,
                                 download_time: float = 0.0) -> dict:
    """官方字幕说话人标注（不跑 ASR）：调用方（noteflow 自动字幕路径）把拉到的
    官方字幕 body 发来，分离后按时间重叠回填 speaker 返回。

    对齐复用 funasr 退化路径的 _posthoc_align_speakers（重叠 ≥ 段时长 50% 赋标签，
    否则 -1 = 跨界段，分组输出中延续当前组）。单人是"成功识别但无需标注"：
    status=success + annotated=false + body 原样返回；真正的失败（超时/异常/未启用）
    仍为 error。不写转录缓存。
    """
    total_start = time.time()
    turns, error, timing = await _diarize_turns_or_error(audio_file_path, video_id, download_time, total_start)
    if error:
        return error

    if len({t.speaker for t in turns}) <= 1:
        logger.info("仅分离+字幕标注：分离结果仅 1 人，无需标注")
        return {"status": "success", "video_id": video_id,
                "body": body, "speakers": _turns_speaker_summary(turns),
                "annotated": False, "reason": "single_speaker",
                "timing": {**timing, "total": round(time.time() - total_start, 3)}}

    annotated = _posthoc_align_speakers(body, turns)
    speakers = _aggregate_speakers(annotated)
    logger.info(f"仅分离+字幕标注完成: {video_id} {len(speakers)} 人标注 {len(annotated)} 段")
    return {
        "status": "success",
        "video_id": video_id,
        "body": annotated,
        "speakers": speakers,
        "annotated": True,
        "reason": None,
        "timing": {**timing, "total": round(time.time() - total_start, 3)},
    }


class TranscriptionService:
    """转录服务类，封装所有转录相关逻辑"""

    def __init__(self, model_manager):
        self.model_manager = model_manager

    async def process_transcription(self, audio_file_path: str, original_filename: str = None, audio_url: str = None, bvid: str = None, audio_id: str = None, no_cache: bool = False, file_path_for_cache: str = None, context=None, diarize: bool = False):
        """处理音频转录的主函数"""
        context = clamp_asr_context(context)  # 理论上限钳制（精确拟合由引擎守卫负责）
        diarize_enabled = bool(diarize and config.diarization_enabled)
        turns = None              # 分离结果（None = 未启用/降级/单人）
        diarization_failed = False  # 区分「分离失败不写缓存」与「单人成功正常写缓存」
        timing = {
            "cache_check": 0.0,
            "model_load": 0.0,
            "duration_detect": 0.0,
            "transcription": 0.0,
            "subtitle_generate": 0.0,
            "cache_save": 0.0,
            "total": 0.0
        }
        if diarize and not config.diarization_enabled:
            logger.warning("收到 diarize=true 请求，但 diarization.enabled=false，按未启用处理")
            timing["diarization"] = 0.0
        total_start = time.time()

        # 检查转录缓存（除非禁用缓存）
        cache_check_start = time.time()
        if not no_cache:
            if file_path_for_cache:
                cached_result = cache_manager.get_cached_transcript(file_path=file_path_for_cache, context=context, diarize=diarize)
                if cached_result:
                    cached_result.pop('cached_at', None)
                    logger.info(f"使用缓存的转录结果，音频时长: {cached_result.get('audio_duration', 'unknown')}秒")
                    return cached_result
            elif audio_id and bvid:
                cached_result = cache_manager.get_cached_transcript(None, bvid, audio_id, context=context, diarize=diarize)
                if cached_result:
                    cached_result.pop('cached_at', None)
                    logger.info(f"使用缓存的转录结果，音频时长: {cached_result.get('audio_duration', 'unknown')}秒")
                    return cached_result
            elif audio_url or bvid:
                cached_result = cache_manager.get_cached_transcript(audio_url, bvid, context=context, diarize=diarize)
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

        self.model_manager.acquire()
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

            # 3. 调用后端转录 + 可选说话人分离（并行，独立线程）
            transcription_start_time = time.time()
            if diarize_enabled:
                diarize_timeout = _diarize_timeout(audio_duration)
                diarize_start = time.time()

                async def _diarize_job():
                    t0 = time.time()
                    try:
                        turns = await asyncio.to_thread(_diarize_samples, audio_file_path)
                        return turns, time.time() - t0
                    except Exception:
                        logger.warning("说话人分离失败，本次降级为无 speaker 输出（不写缓存）", exc_info=True)
                        return None, time.time() - t0

                asr_task = asyncio.create_task(
                    asyncio.to_thread(backend.transcribe, audio_file_path, None, context))
                diar_task = asyncio.create_task(
                    asyncio.wait_for(_diarize_job(), timeout=diarize_timeout))
                diar_elapsed = 0.0
                try:
                    result = await asr_task  # ASR 异常照旧冒泡给外层 except
                except BaseException:
                    # ASR 失败时回收分离任务，避免后台线程继续占用 GPU/推理锁
                    diar_task.cancel()
                    try:
                        await diar_task
                    except BaseException:
                        pass
                    raise
                processing_time = time.time() - transcription_start_time  # ASR 耗时快照（语义不变）
                try:
                    turns, diar_elapsed = await diar_task
                except asyncio.TimeoutError:
                    diar_elapsed = time.time() - diarize_start
                    logger.warning(f"说话人分离超时（>{diarize_timeout:.0f}s），降级且不写缓存")
                    turns, diarization_failed = None, True
                except Exception:
                    diar_elapsed = time.time() - diarize_start
                    logger.warning("说话人分离任务异常，降级且不写缓存", exc_info=True)
                    turns, diarization_failed = None, True
                if turns is None:
                    diarization_failed = True  # job 内部异常降级同样标记
                timing["diarization"] = diar_elapsed
            else:
                result = await asyncio.to_thread(backend.transcribe, audio_file_path, None, context)
                processing_time = time.time() - transcription_start_time
            timing["transcription"] = processing_time

            # 刷新活跃时间
            self.model_manager.last_active_time = time.time()

            transcript_text = result.text
            detected_lang = result.language
            timestamps = result.timestamps

            # 单说话人退化：聚类仅 1 人视为未启用（分离成功，缓存照常写入）
            if turns is not None and len({t.speaker for t in turns}) <= 1:
                logger.info("说话人分离结果仅 1 人，按单人视频处理（不标注）")
                turns = None

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
                subtitle_body = generate_subtitle_segments_from_timestamps(
                    transcript_text, timestamps, detected_lang,
                    audio_duration=audio_duration, turns=turns)
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

            # 说话人汇总（仅 speaker 模式输出时存在）
            if subtitle_body and "speaker" in subtitle_body[0]:
                response["speakers"] = _aggregate_speakers(subtitle_body)

            # 保存到缓存（分离失败降级时跳过，下次请求重试分离）
            cache_save_start = time.time()
            if diarization_failed:
                logger.info("分离降级：本次结果不写入缓存（下次请求将重试分离）")
            elif file_path_for_cache:
                cache_manager.save_transcript_to_cache(file_path=file_path_for_cache, transcript_data=response, context=context, diarize=diarize)
            elif audio_id and bvid:
                cache_manager.save_transcript_to_cache(None, response, bvid, audio_id, context=context, diarize=diarize)
            elif audio_url or bvid:
                cache_manager.save_transcript_to_cache(audio_url, response, bvid, context=context, diarize=diarize)
            timing["cache_save"] = time.time() - cache_save_start
            response["timing"]["cache_save"] = round(timing["cache_save"], 3)
            response["timing"]["total"] = round(time.time() - total_start, 3)

            return response

        except Exception as e:
            if "out of memory" in str(e).lower():
                self.model_manager.unload_model()

            # ffmpeg 超时通常意味着音频文件损坏，删除缓存避免重复踩坑
            if "ffmpeg 超时" in str(e):
                try:
                    if os.path.exists(audio_file_path):
                        os.remove(audio_file_path)
                        logger.warning(f"已删除疑似损坏的缓存文件: {audio_file_path}")
                except Exception:
                    pass

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

        finally:
            self.model_manager.release()
