"""ASR 后端抽象接口"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class TranscribeResult:
    """统一的转录结果"""
    text: str
    language: str = "zh"
    timestamps: Optional[List[Dict[str, Any]]] = None  # [{"text", "start", "end"}, ...]
    performance: Optional[Dict[str, Any]] = field(default_factory=dict)  # {"rtf", "time", ...}


class ASRBackend(ABC):
    """ASR 后端抽象接口

    所有后端实现必须继承此类，实现以下抽象方法。
    通过 BackendFactory.create(config) 统一创建实例。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """后端名称，如 'gguf', 'qwen3-asr', 'funasr'"""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """运行设备，如 'cuda', 'cpu'"""
        pass

    @abstractmethod
    def load(self) -> None:
        """加载模型

        首次调用时加载模型到内存/GPU。
        如果模型已加载，应快速返回或重新加载。
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """释放模型资源"""
        pass

    @abstractmethod
    def transcribe(self, audio_file: str, language: str = None) -> TranscribeResult:
        """执行转录

        Args:
            audio_file: 音频文件路径
            language: 目标语言（可选，后端可能自动检测）

        Returns:
            TranscribeResult: 统一格式的转录结果
        """
        pass
