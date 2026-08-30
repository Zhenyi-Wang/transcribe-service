import os
import time
import threading
import requests
import json
import re
from pathlib import Path
from typing import Optional, Tuple, Union
from logger_config import setup_logger
from cache_manager import cache_manager
from .integrity import verify_audio_file

logger = setup_logger('bilibili_video')

# 下载+校验最大尝试次数（首次 + 2 次重试）
MAX_DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT = (10, 300)  # (连接超时, 读取超时)



class BilibiliVideoDownloader:
    """B站视频（bvid格式）音频下载器"""

    def __init__(self):
        self.headers_template = {
            "Referer": "https://www.bilibili.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def get_audio_url(self, bvid: str, cookie: str, extract_audio_info_only: bool = False, page: int = 1) -> Optional[Tuple[str, dict]]:
        """从页面源码获取B站音频URL（支持新旧两种格式）

        Args:
            bvid: B站视频ID
            cookie: B站Cookie
            extract_audio_info_only: 是否只提取音频信息（不获取URL）
            page: 分P页码（默认 1）
        """
        try:
            if page > 1:
                video_url = f"https://www.bilibili.com/video/{bvid}/?p={page}"
            else:
                video_url = f"https://www.bilibili.com/video/{bvid}/"

            headers = self.headers_template.copy()
            headers["Cookie"] = cookie

            # 如果不是只提取音频信息，才打印获取页面的日志
            if not extract_audio_info_only:
                logger.info(f"获取视频页面: {video_url}")
            response = requests.get(video_url, headers=headers, timeout=(10, 30))
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
                    'format': 'dash',  # 标记为dash格式
                    'timelength': playinfo_data['data'].get('timelength')  # 视频时长(ms)，用于完整性校验
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
                        'format': 'durl',  # 标记为durl格式
                        'timelength': playinfo_data['data'].get('timelength')  # 视频时长(ms)
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

            response = requests.get(audio_url, headers=headers, stream=True, timeout=DOWNLOAD_TIMEOUT)
            response.raise_for_status()

            # 重复头会被 requests 合并为 "n, n"，取首段
            try:
                total_size = int(str(response.headers.get('content-length', 0)).split(",")[0].strip())
            except ValueError:
                total_size = 0
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

            # 完整性校验：实际字节数必须与 content-length 一致（连接中断时流会正常结束）
            if total_size > 0 and downloaded_size != total_size:
                try:
                    os.remove(filename)
                except OSError:
                    pass
                error_msg = (f"下载不完整: 已下载 {downloaded_size} 字节，"
                             f"期望 {total_size} 字节")
                logger.error(error_msg)
                return False, error_msg

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

    def get_audio_info(self, bvid: str, cookie: str, page: int = 1) -> Optional[dict]:
        """仅获取音频信息，不下载"""
        result = self.get_audio_url(bvid, cookie, extract_audio_info_only=True, page=page)
        if result:
            _, audio_info = result
            return audio_info
        return None

    def download(self, id: str, cookie: str, save_dir: str = "tmp", page: int = 1) -> Tuple[bool, Union[str, dict]]:
        """下载B站视频音频的完整流程

        Args:
            id: B站视频BVID
            cookie: B站Cookie
            save_dir: 保存目录
            page: 分P页码（默认 1）

        Returns:
            (success, result) - 成功时返回文件信息字典，失败时返回错误信息字符串
        """
        try:
            # 1. 首先获取音频信息（不包含URL）用于缓存检查
            audio_info = self.get_audio_info(id, cookie, page=page)
            if not audio_info:
                return False, "无法获取音频信息"

            # 根据格式类型选择不同的文件扩展名
            if audio_info.get('format') == 'durl':
                ext = '.mp4'
            else:
                ext = '.m4s'

            # 期望时长（ms），完整性校验用；缺失时降级为仅可解码性校验
            expected_ms = audio_info.get('timelength')

            # 2. 先检查缓存（使用BVID+page+音频ID作为缓存键），命中时校验完整性
            cached_file = cache_manager.get_cached_file(None, id, ext, str(audio_info['id']), page)
            if cached_file:
                ok, detail = verify_audio_file(cached_file, expected_duration_ms=expected_ms)
                if ok:
                    logger.info(f"使用缓存文件: {cached_file}")
                    return True, {
                        "file_path": cached_file,
                        "audio_url": f"cached://{id}_p{page}_{audio_info['id']}",
                        "audio_id": str(audio_info['id'])
                    }
                # 缓存文件不完整（历史截断残留），删除后走重新下载
                logger.warning(f"缓存文件校验失败，删除后重新下载: {cached_file} - {detail}")
                try:
                    os.remove(cached_file)
                except OSError as e:
                    logger.warning(f"删除不完整缓存文件失败: {e}")

            # 3. 准备保存路径（使用BVID+page+音频ID作为文件名）
            Path(save_dir).mkdir(exist_ok=True)
            filename = f"{id}_p{page}_audio_{audio_info['id']}{ext}"
            filepath = os.path.join(save_dir, filename)

            # 4. 下载 + 校验，失败重试（每次重新获取URL，避免签名过期）
            # 下载写到唯一临时名（并发同ID请求不互相踩踏），校验通过后原子落位
            last_error = None
            for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
                result = self.get_audio_url(id, cookie, page=page)
                if not result:
                    last_error = "无法获取音频URL"
                    logger.warning(f"第 {attempt}/{MAX_DOWNLOAD_ATTEMPTS} 次{last_error}")
                    continue

                audio_url, _ = result
                # 临时名含 pid+线程id，跨进程/线程并发同 ID 请求不互相踩踏
                part_path = f"{filepath}.{os.getpid()}.{threading.get_ident()}.part{attempt}"

                if audio_info.get('format') == 'durl':
                    logger.info(f"音频格式: 旧版durl（音视频混合流）")
                else:
                    logger.info(f"音频ID: {audio_info['id']}, 比特率: {audio_info['bandwidth']/1000:.1f} kbps")

                success, dl_result = self.download_audio(audio_url, cookie, part_path)
                if not success:
                    last_error = f"下载失败: {dl_result}"
                    logger.warning(f"第 {attempt}/{MAX_DOWNLOAD_ATTEMPTS} 次{last_error}")
                    continue

                # 5. 下载完成，校验音频时长完整性
                ok, detail = verify_audio_file(part_path, expected_duration_ms=expected_ms)
                if ok:
                    logger.info(detail)
                    # 原子落位到正式文件名
                    os.replace(part_path, filepath)
                    # 6. 保存到缓存（使用BVID+page+音频ID作为缓存键）
                    cached_path = cache_manager.save_to_cache(audio_url, filepath, id, str(audio_info['id']), page)
                    return True, {
                        "file_path": cached_path,
                        "audio_url": audio_url,
                        "audio_id": str(audio_info['id'])
                    }
                last_error = detail
                logger.warning(f"第 {attempt}/{MAX_DOWNLOAD_ATTEMPTS} 次下载校验失败: {detail}")
                try:
                    os.remove(part_path)
                except OSError as e:
                    logger.warning(f"删除不完整文件失败: {e}")

            return False, f"{last_error}（已尝试 {MAX_DOWNLOAD_ATTEMPTS} 次）"

        except Exception as e:
            logger.error(f"下载流程失败: {e}")
            return False, str(e)
