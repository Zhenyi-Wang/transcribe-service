from typing import Tuple, Union
from logger_config import setup_logger

logger = setup_logger('bilibili_episode')


class BilibiliEpisodeDownloader:
    """B站番剧（ep格式）音频下载器"""

    def download(self, id: str, cookie: str, save_dir: str = "tmp") -> Tuple[bool, Union[str, dict]]:
        """下载B站番剧音频的完整流程

        Args:
            id: B站番剧EP ID (纯数字，例如: 2289525)
            cookie: B站Cookie
            save_dir: 保存目录

        Returns:
            (success, result) - 成功时返回文件信息字典，失败时返回错误信息字符串

        TODO: 实现番剧EP的音频下载逻辑
        """
        # TODO: 实现时使用 cookie 和 save_dir 参数
        _ = cookie, save_dir  # 避免未使用参数警告

        logger.warning(f"番剧EP下载功能暂未实现: ep{id}")
        return False, f"番剧EP下载功能暂未实现: ep{id}"
