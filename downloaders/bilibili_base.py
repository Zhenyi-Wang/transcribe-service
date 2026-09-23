"""B站下载器公共基类

模板方法模式：download() 骨架（缓存校验 → 临时名下载 → 时长校验 → 原子落位 → 入缓存 → 重试）
在此实现，子类只提供真实差异的钩子（页面解析、缓存键、文件名、耗尽提示）。
"""
import os
import time
import threading
import requests
from pathlib import Path
from typing import Optional, Tuple, Union

from logger_config import setup_logger
from cache_manager import cache_manager
from .integrity import verify_audio_file

logger = setup_logger('bilibili_base')

# 下载+校验最大尝试次数（首次 + 2 次重试）
MAX_DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT = (10, 300)  # (连接超时, 读取超时)


class TrialSegmentError(Exception):
    """B站未登录仅返回试看片段（durl 总时长远小于视频时长），cookie 失效所致"""
    pass


class BilibiliDownloaderBase:
    """B站音频下载器基类，子类实现 get_audio_url 与各钩子"""

    def __init__(self):
        self.headers_template = {
            "Referer": "https://www.bilibili.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    # ---------- 子类钩子 ----------

    def get_audio_url(self, id: str, cookie: str, extract_audio_info_only: bool = False,
                      page: int = 1) -> Optional[Tuple[str, dict]]:
        """获取音频URL与信息（子类实现：页面抓取与解析）"""
        raise NotImplementedError

    def _cache_bvid(self, id: str) -> str:
        """缓存键中的视频 ID 成分（子类实现）"""
        raise NotImplementedError

    def _audio_filename(self, id: str, page: int, audio_info: dict, ext: str) -> str:
        """下载文件名（子类实现）"""
        raise NotImplementedError

    def _cached_url(self, cache_bvid: str, page: int, audio_id: str) -> str:
        """缓存命中时返回的 audio_url 展示值（格式变更会影响上游缓存键，子类按需覆盖）"""
        return f"cached://{cache_bvid}_p{page}_{audio_id}"

    def _exhausted_error(self, last_error: Optional[str]) -> str:
        """重试耗尽时的最终错误信息（子类可覆盖以附加提示）"""
        return f"{last_error}（已尝试 {MAX_DOWNLOAD_ATTEMPTS} 次）"

    # ---------- 共享实现 ----------

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

    def get_audio_info(self, id: str, cookie: str, page: int = 1) -> Optional[dict]:
        """仅获取音频信息，不下载"""
        result = self.get_audio_url(id, cookie, extract_audio_info_only=True, page=page)
        if result:
            _, audio_info = result
            return audio_info
        return None

    def download(self, id: str, cookie: str, save_dir: str = "tmp", page: int = 1) -> Tuple[bool, Union[str, dict]]:
        """下载B站音频的完整流程

        Args:
            id: B站视频ID（BVID 或 EP ID）
            cookie: B站Cookie
            save_dir: 保存目录
            page: 分P页码（仅普通视频有效，番剧忽略）

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

            # 2. 先检查缓存，命中时校验完整性
            cache_bvid = self._cache_bvid(id)
            audio_id = str(audio_info['id'])
            cached_file = cache_manager.get_cached_file(None, cache_bvid, ext, audio_id, page)
            if cached_file:
                ok, detail = verify_audio_file(cached_file, expected_duration_ms=expected_ms)
                if ok:
                    logger.info(f"使用缓存文件: {cached_file}")
                    return True, {
                        "file_path": cached_file,
                        "audio_url": self._cached_url(cache_bvid, page, audio_id),
                        "audio_id": audio_id
                    }
                # 缓存文件不完整（历史截断残留），删除后走重新下载
                logger.warning(f"缓存文件校验失败，删除后重新下载: {cached_file} - {detail}")
                try:
                    os.remove(cached_file)
                except OSError as e:
                    logger.warning(f"删除不完整缓存文件失败: {e}")

            # 3. 准备保存路径
            Path(save_dir).mkdir(exist_ok=True)
            filepath = os.path.join(save_dir, self._audio_filename(id, page, audio_info, ext))

            # 4. 下载 + 校验，失败重试（每次重新获取URL，避免签名过期）
            # 下载写到唯一临时名（并发同ID请求不互相踩踏），校验通过后原子落位
            last_error = None
            for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
                try:
                    result = self.get_audio_url(id, cookie, page=page)
                except TrialSegmentError as e:
                    # cookie 失效重试无意义，立即失败
                    return False, str(e)
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
                    # 6. 保存到缓存
                    cached_path = cache_manager.save_to_cache(audio_url, filepath, cache_bvid, audio_id, page)
                    return True, {
                        "file_path": cached_path,
                        "audio_url": audio_url,
                        "audio_id": audio_id
                    }
                last_error = detail
                logger.warning(f"第 {attempt}/{MAX_DOWNLOAD_ATTEMPTS} 次下载校验失败: {detail}")
                try:
                    os.remove(part_path)
                except OSError as e:
                    logger.warning(f"删除不完整文件失败: {e}")

            return False, self._exhausted_error(last_error)

        except TrialSegmentError as e:
            return False, str(e)
        except Exception as e:
            logger.error(f"下载流程失败: {e}")
            return False, str(e)
