import os
import hashlib
import math
import time
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from config import config
from logger_config import setup_logger

logger = setup_logger('cache_manager')

class CacheManager:
    """缓存管理器"""

    # 转录缓存条目内的私有过期标记（epoch 秒）；读取返回前会剥掉，不外泄给调用方
    _EXPIRES_KEY = '_cache_expires_at'

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

    def _get_cache_key(self, url: str = None, bvid: str = None, audio_id: str = None, file_path: str = None, page: int = 1, context: str = None, diarize: bool = False) -> str:
        """生成缓存键（context/diarize 改变转录结果时纳入派生，仅转录结果缓存链路传入）"""
        # 优先使用文件路径作为缓存键
        if file_path:
            content = file_path
        # 优先使用BVID+page+音频ID作为缓存键，更稳定
        elif bvid and audio_id:
            content = f"{bvid}_p{page}_{audio_id}"
        elif url and bvid:
            # 兼容旧版本，使用URL+BVID
            content = url + bvid
        elif url:
            # 仅使用URL（音频文件缓存）
            content = url
        else:
            # 如果都没有，使用空字符串
            content = ""
        if context:
            # context 改变转录结果，必须参与 key；sha1[:10] 足够区分且不膨胀文件名
            content = f"{content}#ctx:{hashlib.sha1(context.encode('utf-8')).hexdigest()[:10]}"
        if diarize:
            # 说话人分离改变转录结果（body 带 speaker）；拼接顺序固定在 #ctx 之后
            content = f"{content}#diar:1"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str, ext: str = '.mp3') -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}{ext}"

    def get_cached_file(self, url: str, bvid: str = None, ext: str = '.mp3', audio_id: str = None, page: int = 1) -> Optional[str]:
        """获取缓存文件"""
        if not self.cache_enabled:
            return None

        # 优先使用BVID+page+音频ID作为缓存键
        if bvid and audio_id:
            cache_key = self._get_cache_key(bvid=bvid, audio_id=audio_id, page=page)
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

    def save_to_cache(self, url: str, file_path: str, bvid: str = None, audio_id: str = None, page: int = 1) -> str:
        """保存文件到缓存"""
        if not self.cache_enabled:
            return file_path

        # 获取文件扩展名
        ext = Path(file_path).suffix
        if not ext:
            ext = '.mp3'  # 默认扩展名

        # 优先使用BVID+page+音频ID作为缓存键
        if bvid and audio_id:
            cache_key = self._get_cache_key(bvid=bvid, audio_id=audio_id, page=page)
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

    @staticmethod
    def _discard_transcript_cache(cache_path):
        try:
            cache_path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"清理转录缓存失败（仍视为未命中）: {e}")

    def get_cached_transcript(self, url: str = None, bvid: str = None, audio_id: str = None, file_path: str = None, context: str = None, diarize: bool = False, include_asr: bool = False) -> Optional[Dict[str, Any]]:
        """获取缓存的转录结果

        include_asr: 为 True 时保留条目内的私有 ASR 基线字段 _asr（词级时间戳），
        供分离失败冷却后仅重试分离使用；默认 False 时剥除，正常路径 / API 不泄露内部字段。
        """
        if not self.cache_enabled:
            return None

        # 优先使用文件路径作为缓存键
        if file_path:
            cache_key = self._get_cache_key(file_path=file_path, context=context, diarize=diarize)
        # 优先使用BVID+音频ID作为缓存键
        elif bvid and audio_id:
            cache_key = self._get_cache_key(bvid=bvid, audio_id=audio_id, context=context, diarize=diarize)
        else:
            cache_key = self._get_cache_key(url=url, bvid=bvid, context=context, diarize=diarize)
        cache_path = self.transcript_dir / f"{cache_key}.json"

        if cache_path.exists():
            # 文件可能在 exists 与 stat/open 之间被外部清理。
            try:
                file_age = time.time() - cache_path.stat().st_mtime
            except FileNotFoundError:
                return None
            if file_age <= self.cache_days * 24 * 3600:
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        transcript_data = json.load(f)
                    # 显式过期时间（短期冷却缓存）优先于 cache_days；异常标记按无标记处理，回落 cache_days
                    expires_at = transcript_data.pop(self._EXPIRES_KEY, None)
                    if isinstance(expires_at, (int, float)) and not isinstance(expires_at, bool) and time.time() >= expires_at:
                        self._discard_transcript_cache(cache_path)
                        logger.info(f"删除已到期的短期转录缓存: {cache_path.name}")
                        return None
                    # 缓存元数据不外泄：返回给调用方的 dict 不含 cached_at / 私有过期标记
                    transcript_data.pop('cached_at', None)
                    # 私有 ASR 基线默认剥除，仅 include_asr=True 的内部链路可取回
                    if not include_asr:
                        transcript_data.pop('_asr', None)
                    logger.info(f"使用缓存的转录结果: {cache_path.name}")
                    return transcript_data
                except Exception as e:
                    logger.error(f"读取转录缓存失败: {e}")
                    self._discard_transcript_cache(cache_path)
            else:
                self._discard_transcript_cache(cache_path)
                logger.info(f"删除过期的转录缓存: {cache_path.name}")

        return None

    @staticmethod
    def _validate_ttl(ttl_seconds) -> None:
        """校验显式 TTL：非法值直接抛 ValueError，绝不静默按无 TTL（永久缓存）处理"""
        if ttl_seconds is None:
            return
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
            raise ValueError(f"ttl_seconds 必须是数值秒数，收到: {ttl_seconds!r}")
        if not math.isfinite(ttl_seconds):
            raise ValueError(f"ttl_seconds 不允许为 NaN/Inf，收到: {ttl_seconds!r}")
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds 必须为正数，收到: {ttl_seconds!r}")

    def save_transcript_to_cache(self, url: str = None, transcript_data: Dict[str, Any] = None, bvid: str = None, audio_id: str = None, file_path: str = None, context: str = None, diarize: bool = False, ttl_seconds: float = None) -> None:
        """保存转录结果到缓存

        ttl_seconds: 可选显式过期时间（秒）。提供时条目在 cache_days 之外额外受该过期约束
        （读取端到期即 miss），用于分离失败降级结果等需要短期冷却缓存的场景。
        """
        # 显式 TTL 先校验（即使缓存被禁用也拒绝非法值，暴露调用方 bug）
        self._validate_ttl(ttl_seconds)
        if not self.cache_enabled or not transcript_data:
            return

        # 优先使用文件路径作为缓存键
        if file_path:
            cache_key = self._get_cache_key(file_path=file_path, context=context, diarize=diarize)
        # 优先使用BVID+音频ID作为缓存键
        elif bvid and audio_id:
            cache_key = self._get_cache_key(bvid=bvid, audio_id=audio_id, context=context, diarize=diarize)
        else:
            cache_key = self._get_cache_key(url=url, bvid=bvid, context=context, diarize=diarize)
        cache_path = self.transcript_dir / f"{cache_key}.json"

        tmp_path = None
        try:
            # 复制后再加缓存元数据：不污染调用方的 transcript_data
            now = time.time()
            payload = dict(transcript_data)
            payload['cached_at'] = now
            if ttl_seconds is not None:
                payload[self._EXPIRES_KEY] = now + ttl_seconds
            # 同目录临时文件 + 原子替换：并发写/读不会读到半截 JSON
            fd, tmp_name = tempfile.mkstemp(dir=str(self.transcript_dir), prefix=f".{cache_path.stem}.", suffix=".tmp")
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.chmod(tmp_path, 0o644)  # mkstemp 默认 0600，保持与旧写入方式一致的权限
            os.replace(tmp_path, cache_path)
            tmp_path = None  # 已原子落盘，finally 不再清理
            logger.info(f"转录结果已缓存: {cache_path.name}")
        except Exception as e:
            logger.error(f"缓存转录结果失败: {e}")
        finally:
            # 故障清理只针对自己创建的临时文件，不动其他缓存/其他写入者的临时文件
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

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