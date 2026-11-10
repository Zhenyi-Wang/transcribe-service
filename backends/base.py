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
    def transcribe(self, audio_file: str, language: str = None, context: str = None,
                   progress_cb=None, request_id: str = None) -> TranscribeResult:
        """执行转录

        Args:
            audio_file: 音频文件路径
            language: 目标语言（可选，后端可能自动检测）
            context: ASR 偏置文本（system 段；仅 gguf/asr-engine 后端消费，其余忽略）
            progress_cb: 可选进度回调 progress_cb(done, total)，在转录线程内、chunk
                完成边界调用（spec §5 transcribing 行；仅 gguf 实际承载——共享库
                chunk 循环回调；asr-engine 进度经远端采样，其余后端接受但忽略）
            request_id: 本次计算的唯一 id（spec §4.3/§6 引擎侧第 3 条，归属化取消与
                远端进度匹配的 key；None = 遗留调用，后端按不可归属取消处理）

        Returns:
            TranscribeResult: 统一格式的转录结果
        """
        pass

    def cancel_request(self, request_id: str) -> None:
        """归属化取消（spec §6 引擎侧第 3 条）：请求后端取消 request_id 对应的计算。

        基线 no-op——不支持归属化取消的后端安全调用；gguf（进程内 pending 集合）
        与 asr-engine（POST /admin/cancel）覆写。取消不改变返回值结构：计算自然
        结束后照常返回（result.performance["cancelled"] 标记），调用方按需 join。
        """
        return None

    def probe_capability(self) -> str:
        """进度/取消能力自述：'progress+scoped' 或 'legacy'（spec §6 引擎侧第 3 条）。

        - progress+scoped：有真实 percent 且支持 request_id 归属化取消
        - legacy：无 percent，取消降级「标志 + join」（消费方 JobManager 职责）

        默认 legacy（qwen3/funasr 恒 legacy）；gguf 恒 progress+scoped；
        asr-engine 经 GET /v1/progress 探测决定。
        """
        return "legacy"
