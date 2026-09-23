import json
import re
from typing import Optional, Tuple, Union

import requests

from logger_config import setup_logger
from .bilibili_base import BilibiliDownloaderBase, TrialSegmentError  # noqa: F401 (re-export)

logger = setup_logger('bilibili_video')


class BilibiliVideoDownloader(BilibiliDownloaderBase):
    """B站视频（bvid格式）音频下载器"""

    def get_audio_url(self, bvid: str, cookie: str, extract_audio_info_only: bool = False, page: int = 1) -> Optional[Tuple[str, dict]]:
        """从页面源码获取B站音频URL（支持新旧两种格式）

        Args:
            bvid: B站视频ID
            cookie: B站Cookie
            extract_audio_info_only: 是否只提取音频信息（不获取URL）
            page: 分P页码（默认 1）

        Raises:
            TrialSegmentError: 未登录仅返回试看片段（durl 总时长 << 视频时长），
                               或多段 durl 暂不支持
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
                    # 多段 durl 需要分段下载合并，明确报不支持而非静默只取第一段
                    if len(durl) > 1:
                        raise TrialSegmentError(
                            f"多段 durl（{len(durl)} 段）暂不支持，请反馈")
                    # 未登录时 B站仅返回试看片段：durl 时长远小于视频时长
                    timelength = playinfo_data['data'].get('timelength')
                    durl_total_ms = sum(x.get('length', 0) for x in durl if isinstance(x, dict))
                    if timelength and durl_total_ms > 0 and durl_total_ms < timelength - 3000:
                        raise TrialSegmentError(
                            f"B站仅返回试看片段（durl 总时长 {durl_total_ms/1000:.0f}s，"
                            f"视频时长 {timelength/1000:.0f}s），疑似 B站 cookie 失效，请更新 cookie")
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

        except TrialSegmentError:
            raise
        except Exception as e:
            logger.error(f"获取音频URL失败: {e}")
            return None

    def _cache_bvid(self, id: str) -> str:
        return id

    def _audio_filename(self, id: str, page: int, audio_info: dict, ext: str) -> str:
        return f"{id}_p{page}_audio_{audio_info['id']}{ext}"

    def download(self, id: str, cookie: str, save_dir: str = "tmp", page: int = 1) -> Tuple[bool, Union[str, dict]]:
        """下载B站视频音频（透传 page，其余流程见基类）"""
        return super().download(id, cookie, save_dir, page=page)
