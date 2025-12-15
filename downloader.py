import os
import time
import requests
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class BilibiliDownloader:
    """B站音频下载器"""

    def __init__(self):
        self.headers_template = {
            "Referer": "https://www.bilibili.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def get_audio_url(self, bvid: str, cid: str, cookie: str) -> Optional[str]:
        """获取B站音频URL"""
        try:
            url = f"https://api.bilibili.com/x/player/wbi/playurl?bvid={bvid}&cid={cid}&qn=80&fnval=16"

            headers = self.headers_template.copy()
            headers["Cookie"] = f"Cookie: {cookie}"

            response = requests.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            if 'data' in data and 'dash' in data['data'] and 'audio' in data['data']['dash']:
                return data['data']['dash']['audio'][0]['baseUrl']
            else:
                logger.error(f"无法获取音频URL: {data}")
                return None

        except Exception as e:
            logger.error(f"获取音频URL失败: {e}")
            return None

    def download_audio(self, audio_url: str, cookie: str, filename: str) -> Tuple[bool, str]:
        """下载音频文件"""
        try:
            headers = self.headers_template.copy()
            headers["Cookie"] = f"Cookie: {cookie}"

            logger.info(f"开始下载: {filename}")

            response = requests.get(audio_url, headers=headers, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            block_size = 8192
            start_time = time.time()

            logger.info(f"文件大小: {total_size / 1024 / 1024:.2f} MB")

            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

            total_time = time.time() - start_time
            avg_speed = downloaded_size / total_time / 1024 / 1024

            logger.info(f"下载完成！总耗时: {total_time:.2f} 秒，平均速度: {avg_speed:.2f} MB/s")
            return True, os.path.abspath(filename)

        except Exception as e:
            logger.error(f"下载失败: {e}")
            return False, str(e)

    def download_bilibili_audio(self, bvid: str, cid: str, cookie: str, save_dir: str = "tmp") -> Tuple[bool, str]:
        """下载B站音频的完整流程"""
        try:
            # 1. 获取音频URL
            logger.info(f"获取音频URL: bvid={bvid}, cid={cid}")
            audio_url = self.get_audio_url(bvid, cid, cookie)

            if not audio_url:
                return False, "无法获取音频URL"

            logger.info(f"音频URL: {audio_url}")

            # 2. 准备保存路径
            Path(save_dir).mkdir(exist_ok=True)
            filename = f"{bvid}_{cid}.m4s"
            filepath = os.path.join(save_dir, filename)

            # 3. 下载文件
            success, result = self.download_audio(audio_url, cookie, filepath)

            if success:
                return True, result  # 返回文件路径
            else:
                return False, result  # 返回错误信息

        except Exception as e:
            logger.error(f"下载流程失败: {e}")
            return False, str(e)