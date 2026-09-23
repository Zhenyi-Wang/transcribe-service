"""说话人分离模块。顶层零重依赖：pyannote/torch 仅在 manager._load() 内延迟导入。"""
from .manager import DiarizationManager, SpeakerTurn, get_manager

__all__ = ["DiarizationManager", "SpeakerTurn", "get_manager"]
