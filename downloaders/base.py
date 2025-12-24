from typing import Tuple, Union
from logger_config import setup_logger
from .bilibili_video import BilibiliVideoDownloader
from .bilibili_episode import BilibiliEpisodeDownloader

logger = setup_logger('downloader')


class Downloader:
    """下载器调度器，根据ID格式分发到对应的下载模块"""

    def __init__(self):
        self.video_downloader = BilibiliVideoDownloader()
        self.episode_downloader = BilibiliEpisodeDownloader()

    def _parse_id_type(self, id: str) -> Tuple[str, str]:
        """解析ID类型并提取纯ID

        Args:
            id: 视频/番剧ID

        Returns:
            (type, pure_id) - 类型('bvid', 'ep', 'unknown') 和 提取后的纯ID
        """
        id_lower = id.lower()

        # bvid 格式: BV1xx411c7mD
        if id_lower.startswith('bv'):
            return 'bvid', id

        # ep 格式: ep=2289525 或 ep2289525
        if id_lower.startswith('ep'):
            if '=' in id_lower:
                # ep=2289525 -> 2289525
                return 'ep', id.split('=')[1]
            else:
                # ep2289525 -> 2289525
                return 'ep', id[2:]

        return 'unknown', id

    def download(self, id: str, cookie: str, save_dir: str = "tmp") -> Tuple[bool, Union[str, dict]]:
        """根据ID类型下载音频

        Args:
            id: 视频/番剧ID (支持 BV... 或 ep2289525 格式)
            cookie: B站Cookie
            save_dir: 保存目录

        Returns:
            (success, result) - 成功时返回文件信息字典，失败时返回错误信息字符串

        示例:
            downloader.download("BV1xx411c7mD", cookie)  # bvid格式
            downloader.download("ep2289525", cookie)     # ep格式
        """
        id_type, pure_id = self._parse_id_type(id)

        logger.info(f"检测到ID类型: {id_type}, 原始ID: {id}")

        if id_type == 'bvid':
            return self.video_downloader.download(pure_id, cookie, save_dir)
        elif id_type == 'ep':
            return self.episode_downloader.download(pure_id, cookie, save_dir)
        else:
            logger.error(f"无法识别的ID格式: {id}")
            return False, f"无法识别的ID格式: {id}，支持的格式: bvid (BV...) 或 ep (ep2289525)"


# 向后兼容：保留原有的类名和接口
class BilibiliDownloader(Downloader):
    """B站音频下载器（向后兼容类）

    Deprecated: 建议直接使用 Downloader 类
    """

    def download_bilibili_audio(self, bvid: str, cookie: str, save_dir: str = "tmp") -> Tuple[bool, Union[str, dict]]:
        """下载B站音频的完整流程（向后兼容方法）

        Deprecated: 建议使用 download() 方法
        """
        return self.download(bvid, cookie, save_dir)
