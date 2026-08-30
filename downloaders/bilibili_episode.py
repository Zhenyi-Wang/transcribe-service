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

logger = setup_logger('bilibili_episode')

# 下载+校验最大尝试次数（首次 + 2 次重试）
MAX_DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT = (10, 300)  # (连接超时, 读取超时)


class BilibiliEpisodeDownloader:
    """B站番剧（ep格式）音频下载器"""

    def __init__(self):
        self.headers_template = {
            "Referer": "https://www.bilibili.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def get_audio_url(self, ep_id: str, cookie: str, extract_audio_info_only: bool = False) -> Optional[Tuple[str, dict]]:
        """从番剧页面获取音频URL

        Args:
            ep_id: B站番剧EP ID (纯数字)
            cookie: B站Cookie
            extract_audio_info_only: 是否只提取音频信息（不获取URL）
        """
        try:
            episode_url = f"https://www.bilibili.com/bangumi/play/ep{ep_id}"

            headers = self.headers_template.copy()
            headers["Cookie"] = cookie

            if not extract_audio_info_only:
                logger.info(f"获取番剧页面: {episode_url}")
            response = requests.get(episode_url, headers=headers, timeout=(10, 30))
            response.raise_for_status()
            html_content = response.text

            # 从页面源码中提取 playurlSSRData 数据
            # 找到 playurlSSRData 的起始位置
            start_marker = 'const playurlSSRData = '
            start_idx = html_content.find(start_marker)
            if start_idx == -1:
                logger.error("页面中不包含 playurlSSRData")
                return None

            logger.debug(f"找到 playurlSSRData 起始位置: {start_idx}")

            # 从起始位置开始找到第一个 '{'
            json_start = html_content.find('{', start_idx)
            if json_start == -1:
                logger.error("无法找到JSON起始花括号")
                return None

            # 从JSON起始位置开始找到匹配的右花括号
            brace_count = 0
            json_end = -1
            for i in range(json_start, len(html_content)):
                if html_content[i] == '{':
                    brace_count += 1
                elif html_content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1  # 包含右花括号
                        break

            if json_end == -1:
                logger.error("无法找到匹配的JSON结束花括号")
                return None

            json_str = html_content[json_start:json_end]
            logger.debug(f"提取的JSON长度: {len(json_str)} 字符")

            try:
                playurl_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {e}")
                logger.debug(f"JSON前200字符: {json_str[:200]}")
                return None

            # 检查响应状态
            if playurl_data.get('status') != 200:
                logger.error(f"API返回错误状态: {playurl_data.get('status')}")
                return None

            # 从 data.result.video_info.dash.audio 获取音频流
            result = playurl_data.get('data', {}).get('result', {})
            video_info = result.get('video_info', {})

            if 'dash' not in video_info or 'audio' not in video_info['dash']:
                logger.error("无法从 playurlSSRData 中找到音频流")
                return None

            audio_list = video_info['dash']['audio']
            if not audio_list:
                logger.error("音频流列表为空")
                return None

            # 按比特率排序，选择最低音质（文件最小）
            audio_list_sorted = sorted(audio_list, key=lambda x: x['bandwidth'])
            audio = audio_list_sorted[0]

            audio_info = {
                'url': audio['base_url'],
                'id': audio['id'],
                'bandwidth': audio['bandwidth'],
                'codecs': audio['codecs'],
                'format': 'dash',
                'timelength': result.get('timelength')  # 视频时长(ms)，用于完整性校验
            }

            logger.info(f"找到音频信息 - ID: {audio_info['id']}, "
                      f"比特率: {audio_info['bandwidth']} bps ({audio_info['bandwidth']/1000:.1f} kbps), "
                      f"编码: {audio_info['codecs']}")

            return audio_info['url'], audio_info

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

    def get_audio_info(self, ep_id: str, cookie: str) -> Optional[dict]:
        """仅获取音频信息，不下载"""
        result = self.get_audio_url(ep_id, cookie, extract_audio_info_only=True)
        if result:
            _, audio_info = result
            return audio_info
        return None

    def download(self, id: str, cookie: str, save_dir: str = "tmp") -> Tuple[bool, Union[str, dict]]:
        """下载B站番剧音频的完整流程

        Args:
            id: B站番剧EP ID (纯数字，例如: 2289525)
            cookie: B站Cookie
            save_dir: 保存目录

        Returns:
            (success, result) - 成功时返回文件信息字典，失败时返回错误信息字符串
        """
        try:
            # 1. 首先获取音频信息（不包含URL）用于缓存检查
            audio_info = self.get_audio_info(id, cookie)
            if not audio_info:
                return False, "无法获取音频信息"

            ext = '.m4s'  # dash格式使用.m4s扩展名

            # 期望时长（ms），完整性校验用；缺失时降级为仅可解码性校验
            expected_ms = audio_info.get('timelength')

            # 2. 先检查缓存（使用EP ID+音频ID作为缓存键），命中时校验完整性
            ep_id_str = f"ep{id}"
            cached_file = cache_manager.get_cached_file(None, ep_id_str, ext, str(audio_info['id']))
            if cached_file:
                ok, detail = verify_audio_file(cached_file, expected_duration_ms=expected_ms)
                if ok:
                    logger.info(f"使用缓存文件: {cached_file}")
                    return True, {
                        "file_path": cached_file,
                        "audio_url": f"cached://{ep_id_str}_{audio_info['id']}",
                        "audio_id": str(audio_info['id'])
                    }
                # 缓存文件不完整（历史截断残留），删除后走重新下载
                logger.warning(f"缓存文件校验失败，删除后重新下载: {cached_file} - {detail}")
                try:
                    os.remove(cached_file)
                except OSError as e:
                    logger.warning(f"删除不完整缓存文件失败: {e}")

            # 3. 准备保存路径
            Path(save_dir).mkdir(exist_ok=True)
            filename = f"ep{id}_audio_{audio_info['id']}{ext}"
            filepath = os.path.join(save_dir, filename)

            # 4. 下载 + 校验，失败重试（每次重新获取URL，避免签名过期）
            # 下载写到唯一临时名（并发同ID请求不互相踩踏），校验通过后原子落位
            last_error = None
            for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
                result = self.get_audio_url(id, cookie)
                if not result:
                    last_error = "无法获取音频URL"
                    logger.warning(f"第 {attempt}/{MAX_DOWNLOAD_ATTEMPTS} 次{last_error}")
                    continue

                audio_url, _ = result
                # 临时名含 pid+线程id，跨进程/线程并发同 ID 请求不互相踩踏
                part_path = f"{filepath}.{os.getpid()}.{threading.get_ident()}.part{attempt}"
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
                    # 6. 保存到缓存（使用EP ID+音频ID作为缓存键）
                    cached_path = cache_manager.save_to_cache(audio_url, filepath, ep_id_str, str(audio_info['id']))
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
