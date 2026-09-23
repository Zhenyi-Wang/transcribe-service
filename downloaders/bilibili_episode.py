import json
from typing import Optional, Tuple

import requests

from logger_config import setup_logger
from .bilibili_base import BilibiliDownloaderBase, MAX_DOWNLOAD_ATTEMPTS

logger = setup_logger('bilibili_episode')


class BilibiliEpisodeDownloader(BilibiliDownloaderBase):
    """B站番剧（ep格式）音频下载器"""

    def get_audio_url(self, ep_id: str, cookie: str, extract_audio_info_only: bool = False,
                      page: int = 1) -> Optional[Tuple[str, dict]]:
        """从番剧页面获取音频URL

        Args:
            ep_id: B站番剧EP ID (纯数字)
            cookie: B站Cookie
            extract_audio_info_only: 是否只提取音频信息（不获取URL）
            page: 未使用（番剧无分P），保持与基类签名一致
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

    def _cache_bvid(self, id: str) -> str:
        return f"ep{id}"

    def _audio_filename(self, id: str, page: int, audio_info: dict, ext: str) -> str:
        return f"ep{id}_audio_{audio_info['id']}{ext}"

    def _cached_url(self, cache_bvid: str, page: int, audio_id: str) -> str:
        # 保持历史格式（无 _p 成分），避免影响上游转录缓存键
        return f"cached://{cache_bvid}_{audio_id}"

    def _exhausted_error(self, last_error: Optional[str]) -> str:
        final_msg = f"{last_error}（已尝试 {MAX_DOWNLOAD_ATTEMPTS} 次）"
        if last_error and "音频不完整" in last_error:
            # 仅 ffprobe 时长不足场景提示（网络中断的"下载不完整"不归因 cookie）
            # 番剧未登录预览走 dash（无 durl 可直接比对），重试耗尽后补充提示
            final_msg += "；番剧未登录仅提供预览片段，疑似 B站 cookie 失效，请更新 cookie"
        return final_msg
