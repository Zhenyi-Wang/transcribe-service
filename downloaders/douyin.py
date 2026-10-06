"""抖音视频音频下载器

技术路线（2026-10 实测）：
- 元数据/播放地址：iesdouyin slidesinfo 免签名接口（移动 UA、零 cookie）
- 无独立音频流：ffmpeg 远程流式提音频（-vn -c:a copy），网络流量≈视频体积、磁盘只落音频
- 直链当日过期：每次下载尝试前重新解析，不持久化直链
- 反爬红线：绝不带非抖音域 Referer（403）
"""
import os
import subprocess
import threading
from pathlib import Path
from typing import Optional, Tuple, Union

import requests

from logger_config import setup_logger
from cache_manager import cache_manager
from .bilibili_base import MAX_DOWNLOAD_ATTEMPTS
from .integrity import verify_audio_file

logger = setup_logger('douyin')

SLIDESINFO_URL = "https://www.iesdouyin.com/web/api/v2/aweme/slidesinfo/"
DOUYIN_MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1"
)
# CDN 拉流耗时上界：60 分钟视频 ≈1GB，家宽 1.55MB/s ≈11 分钟，留波动余量
FFMPEG_TIMEOUT_SEC = 20 * 60


class DouyinDownloader:
    """抖音视频（aweme_id）音频下载器"""

    def fetch_aweme_detail(self, aweme_id: str) -> Optional[dict]:
        """查询 slidesinfo 返回 aweme 详情 dict；未找到返回 None。

        图集内容首次查询可能为空，带 request_source=200 重试一次。
        """
        headers = {"User-Agent": DOUYIN_MOBILE_UA}  # 绝不带 Referer
        for request_source in (None, 200):
            url = f"{SLIDESINFO_URL}?aweme_ids=%5B{aweme_id}%5D"  # 裸数字数组（["id"] 带引号会返回 null，2026-10-06 实测）
            if request_source is not None:
                url += f"&request_source={request_source}"
            try:
                resp = requests.get(url, headers=headers, timeout=(10, 30))
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(f"slidesinfo 请求失败(request_source={request_source}): {e}")
                continue
            details = data.get("aweme_details") or []
            if details:
                return details[0]
            logger.info(f"slidesinfo 空结果(request_source={request_source}), filter: "
                        f"{data.get('filter_list')}")
        return None

    def _select_play_url(self, detail: dict) -> Optional[dict]:
        """选最低码率档的 play_addr（提音频不需要高画质，流量减半）；无 bit_rate 时回退顶层 play_addr"""
        bit_rate = detail.get("video", {}).get("bit_rate") or []
        if bit_rate:
            lowest = min(bit_rate, key=lambda b: b.get("bit_rate", 0))
            play = lowest.get("play_addr") or {}
        else:
            play = detail.get("video", {}).get("play_addr") or {}
        url_list = play.get("url_list") or []
        return {"url": url_list[0], "uri": play.get("uri", "")} if url_list else None

    def download_douyin_audio(self, aweme_id: str, save_dir: str = "tmp") -> Tuple[bool, Union[str, dict]]:
        """下载抖音视频音频（ffmpeg 流式提取），返回约定与 B 站下载器一致：
        (True, {"file_path", "audio_url", "audio_id"}) 或 (False, 错误信息)
        """
        detail = self.fetch_aweme_detail(aweme_id)
        if not detail:
            return False, "抖音视频未找到（可能已删除或私密）"
        if detail.get("images"):
            return False, "图集暂不支持（无语音主体，仅配乐）"

        duration_ms = detail.get("duration") or 0
        # 缓存键与 B 站同构：bvid=aweme_id, audio_id='dy'
        cached_file = cache_manager.get_cached_file(None, aweme_id, ".m4a", "dy", 1)
        if cached_file:
            ok, detail_msg = verify_audio_file(cached_file, expected_duration_ms=duration_ms or None)
            if ok:
                logger.info(f"使用缓存文件: {cached_file}")
                return True, {"file_path": cached_file, "audio_url": f"cached://{aweme_id}",
                              "audio_id": "dy"}
            logger.warning(f"缓存文件校验失败，删除重下: {cached_file} - {detail_msg}")
            try:
                os.remove(cached_file)
            except OSError:
                pass

        Path(save_dir).mkdir(exist_ok=True)
        filepath = os.path.join(save_dir, f"{aweme_id}_audio_dy.m4a")
        last_error = None
        for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
            play = self._select_play_url(detail) if attempt == 1 else None
            # 直链当日过期：非首次尝试重新解析
            if play is None:
                detail = self.fetch_aweme_detail(aweme_id)
                if not detail:
                    return False, "抖音视频未找到（可能已删除或私密）"
                if detail.get("images"):
                    return False, "图集暂不支持（无语音主体，仅配乐）"
                play = self._select_play_url(detail)
            if not play:
                last_error = "无法获取播放地址"
                logger.warning(f"第 {attempt}/{MAX_DOWNLOAD_ATTEMPTS} 次{last_error}")
                continue

            # 临时名必须以 .m4a 结尾：ffmpeg 按输出扩展名推断容器格式
            part_path = os.path.join(save_dir, f"{aweme_id}_audio_dy.p{os.getpid()}.t{threading.get_ident()}.a{attempt}.m4a")
            cmd = [
                "ffmpeg", "-user_agent", DOUYIN_MOBILE_UA,
                "-i", play["url"], "-vn", "-c:a", "copy", "-y", part_path,
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                last_error = f"ffmpeg 超时（>{FFMPEG_TIMEOUT_SEC}s）"
                logger.warning(f"第 {attempt}/{MAX_DOWNLOAD_ATTEMPTS} 次{last_error}")
                try:
                    os.remove(part_path)
                except OSError:
                    pass
                continue
            if proc.returncode != 0 or not os.path.exists(part_path):
                stderr_tail = (proc.stderr or b"")[-500:].decode("utf-8", "ignore")
                last_error = f"ffmpeg 提取音频失败: {stderr_tail}"
                logger.warning(f"第 {attempt}/{MAX_DOWNLOAD_ATTEMPTS} 次{last_error}")
                try:
                    os.remove(part_path)
                except OSError:
                    pass
                continue

            ok, detail_msg = verify_audio_file(part_path, expected_duration_ms=duration_ms or None)
            if ok:
                os.replace(part_path, filepath)
                cached_path = cache_manager.save_to_cache(play["url"], filepath, aweme_id, "dy", 1)
                logger.info(f"抖音音频下载完成: {cached_path}")
                return True, {"file_path": cached_path, "audio_url": play["url"], "audio_id": "dy"}
            last_error = detail_msg
            logger.warning(f"第 {attempt}/{MAX_DOWNLOAD_ATTEMPTS} 次下载校验失败: {detail_msg}")
            try:
                os.remove(part_path)
            except OSError:
                pass

        return False, last_error or "下载失败"
