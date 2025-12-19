import os
import time
import requests
import json
import re
from pathlib import Path
from typing import Optional, Tuple
from logger_config import setup_logger

logger = setup_logger('downloader')

class BilibiliDownloader:
    """B站音频下载器"""

    def __init__(self):
        self.headers_template = {
            "Referer": "https://www.bilibili.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def get_audio_url(self, bvid: str, cookie: str) -> Optional[Tuple[str, dict]]:
        """从页面源码获取B站音频URL（支持新旧两种格式）"""
        try:
            video_url = f"https://www.bilibili.com/video/{bvid}"

            headers = self.headers_template.copy()
            headers["Cookie"] = cookie

            logger.info(f"获取视频页面: {video_url}")
            response = requests.get(video_url, headers=headers)
            response.raise_for_status()
            html_content = response.text

            # 从页面源码中提取 __playinfo__ 数据
            playinfo_pattern = r'<script>window\.__playinfo__=({.+?})</script>'
            playinfo_match = re.search(playinfo_pattern, html_content)

            if not playinfo_match:
                logger.error("无法在页面中找到 __playinfo__ 数据")
                return None

            playinfo_data = json.loads(playinfo_match.group(1))

            # 新版格式：dash分离音视频
            if 'data' in playinfo_data and 'dash' in playinfo_data['data'] and 'audio' in playinfo_data['data']['dash']:
                logger.info("使用新版格式（dash）获取音频")
                audio_list = playinfo_data['data']['dash']['audio']
                # 按比特率排序，选择最低音质（文件最小）
                audio_list_sorted = sorted(audio_list, key=lambda x: x['bandwidth'])
                audio = audio_list_sorted[0]

                audio_info = {
                    'url': audio['baseUrl'],
                    'id': audio['id'],
                    'bandwidth': audio['bandwidth'],
                    'codecs': audio['codecs'],
                    'format': 'dash'  # 标记为dash格式
                }

                logger.info(f"找到音频信息 - ID: {audio_info['id']}, "
                          f"比特率: {audio_info['bandwidth']} bps ({audio_info['bandwidth']/1000:.1f} kbps), "
                          f"编码: {audio_info['codecs']}")

                return audio_info['url'], audio_info

            # 旧版格式：durl混合音视频
            elif 'data' in playinfo_data and 'durl' in playinfo_data['data']:
                logger.info("使用旧版格式（durl）获取音视频")
                durl = playinfo_data['data']['durl']
                if isinstance(durl, list) and len(durl) > 0 and 'url' in durl[0]:
                    audio_info = {
                        'url': durl[0]['url'],
                        'id': 'video_audio',
                        'bandwidth': 0,  # 旧版格式没有比特率信息
                        'codecs': 'h264+aac',  # 假设编码格式
                        'format': 'durl'  # 标记为durl格式
                    }

                    logger.info(f"找到音视频流 - 注意：这是视频+音频的混合流")

                    return audio_info['url'], audio_info
                else:
                    logger.error("durl 数据格式不正确")
                    return None
            else:
                logger.error("无法从 playinfo 数据中提取音频信息，既没有 dash 也没有 durl")
                return None

        except Exception as e:
            logger.error(f"获取音频URL失败: {e}")
            return None

    def download_audio(self, audio_url: str, cookie: str, filename: str) -> Tuple[bool, str]:
        """下载音频文件"""
        try:
            headers = self.headers_template.copy()
            headers["Cookie"] = cookie

            logger.info(f"开始下载: {filename}")
            logger.info(f"实际下载URL: {audio_url}")

            response = requests.get(audio_url, headers=headers, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            block_size = 8192
            start_time = time.time()

            if total_size > 0:
                logger.info(f"文件大小: {total_size / 1024 / 1024:.2f} MB")

            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

            total_time = time.time() - start_time
            avg_speed = downloaded_size / total_time / 1024 / 1024 if total_time > 0 else 0

            actual_size = downloaded_size / 1024 / 1024
            logger.info(f"下载完成！文件大小: {actual_size:.2f} MB，"
                      f"总耗时: {total_time:.2f} 秒，"
                      f"平均速度: {avg_speed:.2f} MB/s")
            return True, os.path.abspath(filename)

        except Exception as e:
            logger.error(f"下载失败: {e}")
            return False, str(e)

    def download_bilibili_audio(self, bvid: str, cookie: str, save_dir: str = "tmp") -> Tuple[bool, str]:
        """下载B站音频的完整流程"""
        try:
            # 1. 获取音频URL（从页面源码，支持新旧格式）
            logger.info(f"获取音频URL: bvid={bvid}")
            result = self.get_audio_url(bvid, cookie)

            if not result:
                return False, "无法获取音频URL"

            audio_url, audio_info = result

            # 根据格式类型选择不同的文件扩展名
            if audio_info.get('format') == 'durl':
                logger.info(f"音频格式: 旧版durl（音视频混合流）")
                ext = '.mp4'
            else:
                logger.info(f"音频ID: {audio_info['id']}, 比特率: {audio_info['bandwidth']/1000:.1f} kbps")
                ext = '.m4s'

            # 2. 准备保存路径
            Path(save_dir).mkdir(exist_ok=True)
            # 使用BVID和音频ID作为文件名
            filename = f"{bvid}_audio_{audio_info['id']}{ext}"
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