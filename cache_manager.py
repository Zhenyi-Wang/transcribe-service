import os
import hashlib
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
from config import config
from logger_config import setup_logger

logger = setup_logger('cache_manager')

class CacheManager:
    """缓存管理器"""

    def __init__(self):
        self.cache_dir = Path(config.cache_dir)
        self.cache_enabled = config.cache_enabled
        self.cache_days = config.cache_days

        if self.cache_enabled:
            self.cache_dir.mkdir(exist_ok=True)
            # 创建转录结果缓存子目录
            self.transcript_dir = self.cache_dir / "transcripts"
            self.transcript_dir.mkdir(exist_ok=True)
            logger.info(f"缓存已启用，目录: {self.cache_dir}, 保存天数: {self.cache_days}")
        else:
            logger.info("缓存已禁用")

    def _get_cache_key(self, url: str = None, bvid: str = None, audio_id: str = None) -> str:
        """生成缓存键"""
        # 优先使用BVID+音频ID作为缓存键，更稳定
        if bvid and audio_id:
            content = f"{bvid}_{audio_id}"
        elif url and bvid:
            # 兼容旧版本，使用URL+BVID
            content = url + bvid
        elif url:
            # 仅使用URL（音频文件缓存）
            content = url
        else:
            # 如果都没有，使用空字符串
            content = ""
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str, ext: str = '.mp3') -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}{ext}"

    def get_cached_file(self, url: str, bvid: str = None, ext: str = '.mp3', audio_id: str = None) -> Optional[str]:
        """获取缓存文件"""
        if not self.cache_enabled:
            return None

        # 优先使用BVID+音频ID作为缓存键
        if bvid and audio_id:
            cache_key = self._get_cache_key(bvid=bvid, audio_id=audio_id)
        else:
            cache_key = self._get_cache_key(url=url, bvid=bvid)
        cache_path = self._get_cache_path(cache_key, ext)

        if cache_path.exists():
            # 检查文件是否过期
            file_age = time.time() - cache_path.stat().st_mtime
            if file_age <= self.cache_days * 24 * 3600:
                logger.info(f"使用缓存文件: {cache_path.name}")
                return str(cache_path)
            else:
                # 删除过期文件
                cache_path.unlink()
                logger.info(f"删除过期缓存文件: {cache_path.name}")

        return None

    def save_to_cache(self, url: str, file_path: str, bvid: str = None, audio_id: str = None) -> str:
        """保存文件到缓存"""
        if not self.cache_enabled:
            return file_path

        # 获取文件扩展名
        ext = Path(file_path).suffix
        if not ext:
            ext = '.mp3'  # 默认扩展名

        # 优先使用BVID+音频ID作为缓存键
        if bvid and audio_id:
            cache_key = self._get_cache_key(bvid=bvid, audio_id=audio_id)
        else:
            cache_key = self._get_cache_key(url=url, bvid=bvid)
        cache_path = self._get_cache_path(cache_key, ext)

        try:
            # 复制文件到缓存目录
            import shutil
            shutil.copy2(file_path, cache_path)
            logger.info(f"文件已缓存: {cache_path.name}")
            return str(cache_path)
        except Exception as e:
            logger.error(f"缓存文件失败: {e}")
            return file_path

    def get_cached_transcript(self, url: str = None, bvid: str = None, audio_id: str = None) -> Optional[Dict[str, Any]]:
        """获取缓存的转录结果"""
        if not self.cache_enabled:
            return None

        # 优先使用BVID+音频ID作为缓存键
        if bvid and audio_id:
            cache_key = self._get_cache_key(bvid=bvid, audio_id=audio_id)
        else:
            cache_key = self._get_cache_key(url=url, bvid=bvid)
        cache_path = self.transcript_dir / f"{cache_key}.json"

        if cache_path.exists():
            # 检查文件是否过期
            file_age = time.time() - cache_path.stat().st_mtime
            if file_age <= self.cache_days * 24 * 3600:
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        transcript_data = json.load(f)
                    logger.info(f"使用缓存的转录结果: {cache_path.name}")
                    return transcript_data
                except Exception as e:
                    logger.error(f"读取转录缓存失败: {e}")
                    cache_path.unlink()
            else:
                # 删除过期文件
                cache_path.unlink()
                logger.info(f"删除过期的转录缓存: {cache_path.name}")

        return None

    def save_transcript_to_cache(self, url: str = None, transcript_data: Dict[str, Any] = None, bvid: str = None, audio_id: str = None) -> None:
        """保存转录结果到缓存"""
        if not self.cache_enabled or not transcript_data:
            return

        # 优先使用BVID+音频ID作为缓存键
        if bvid and audio_id:
            cache_key = self._get_cache_key(bvid=bvid, audio_id=audio_id)
        else:
            cache_key = self._get_cache_key(url=url, bvid=bvid)
        cache_path = self.transcript_dir / f"{cache_key}.json"

        try:
            # 添加缓存时间戳
            transcript_data['cached_at'] = time.time()
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)
            logger.info(f"转录结果已缓存: {cache_path.name}")
        except Exception as e:
            logger.error(f"缓存转录结果失败: {e}")

    def cleanup_expired_cache(self):
        """清理过期缓存"""
        if not self.cache_enabled or not self.cache_dir.exists():
            return

        logger.info("开始清理过期缓存...")
        current_time = time.time()
        expired_count = 0

        # 清理音频文件缓存
        for file_path in self.cache_dir.iterdir():
            if file_path.is_file() and file_path.suffix not in ['.json']:
                file_age = (current_time - file_path.stat().st_mtime) / 24 / 3600
                if file_age > self.cache_days:
                    try:
                        file_path.unlink()
                        expired_count += 1
                        logger.debug(f"删除过期缓存: {file_path.name}")
                    except Exception as e:
                        logger.error(f"删除缓存文件失败 {file_path.name}: {e}")

        # 清理转录结果缓存
        if self.transcript_dir.exists():
            for file_path in self.transcript_dir.iterdir():
                if file_path.is_file() and file_path.suffix == '.json':
                    file_age = (current_time - file_path.stat().st_mtime) / 24 / 3600
                    if file_age > self.cache_days:
                        try:
                            file_path.unlink()
                            expired_count += 1
                            logger.debug(f"删除过期转录缓存: {file_path.name}")
                        except Exception as e:
                            logger.error(f"删除转录缓存文件失败 {file_path.name}: {e}")

        logger.info(f"清理完成，删除了 {expired_count} 个过期缓存文件")

# 全局缓存管理器实例
cache_manager = CacheManager()