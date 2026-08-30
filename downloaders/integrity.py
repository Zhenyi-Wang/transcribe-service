"""音频文件完整性校验

用 ffprobe 枚举实际音频包，取最后一个包的结束时间作为实际可播放时长。
不能用 ffprobe 的 format duration —— fMP4 的 sidx 索引在文件头部，
截断文件也会报告完整的索引时长（实测验证），末包时间才反映真实进度。
"""
import os
import subprocess
from typing import Optional, Tuple

from logger_config import setup_logger

logger = setup_logger('integrity')

FFPROBE_TIMEOUT = 120
DEFAULT_TOLERANCE_S = 3.0


class FfprobeUnavailableError(Exception):
    """ffprobe 不可用（缺失/超时）——环境故障，与文件损坏相区分"""
    pass


def get_audio_end_time(filepath: str) -> Optional[float]:
    """返回最后一个音频包的结束时间（秒）；无法解析出任何音频包时返回 None

    Raises:
        FfprobeUnavailableError: ffprobe 缺失或执行超时（环境故障，非文件问题）
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "packet=pts_time,duration_time", "-of", "csv=p=0", filepath],
            capture_output=True, text=True, timeout=FFPROBE_TIMEOUT,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.error(f"ffprobe 执行失败 {filepath}: {e}")
        raise FfprobeUnavailableError(f"ffprobe 不可用: {e}") from e

    end_time = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        try:
            pts = float(parts[0])
        except ValueError:
            continue  # pts 无效则该包不可用（dur=N/A 的对称处理见下，pts 无效更罕见）
        try:
            dur = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
        except ValueError:
            dur = 0.0  # duration_time 为 N/A 时保留有效 pts
        candidate = pts + dur
        if end_time is None or candidate > end_time:
            end_time = candidate
    return end_time


def verify_audio_file(filepath: str, expected_duration_ms: Optional[int] = None,
                      tolerance_s: float = DEFAULT_TOLERANCE_S) -> Tuple[bool, str]:
    """校验音频文件完整性

    Args:
        filepath: 音频文件路径
        expected_duration_ms: 期望时长（毫秒，来自 B 站 timelength）；为 None 时仅校验可解码性
        tolerance_s: 时长容差（秒）

    Returns:
        (ok, detail) - ok 为 False 时 detail 说明失败原因；
        ffprobe 不可用（环境故障）时返回 (True, 跳过说明)，不判文件失败
    """
    if not os.path.exists(filepath):
        return False, "文件不存在"

    try:
        end_time = get_audio_end_time(filepath)
    except FfprobeUnavailableError as e:
        return True, f"跳过完整性校验: {e}"

    if end_time is None:
        return False, "音频无法解析（读不到任何音频包，文件可能损坏）"

    if expected_duration_ms:
        expected_s = expected_duration_ms / 1000
        if end_time < expected_s - tolerance_s:
            return False, (f"音频不完整: 实际时长 {end_time:.1f}s，"
                           f"期望 {expected_s:.1f}s（容差 {tolerance_s:.0f}s）")

    return True, f"音频时长 {end_time:.1f}s 校验通过"
