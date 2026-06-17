"""后端工厂"""
from typing import Dict, Type
from .base import ASRBackend


class BackendFactory:
    """后端工厂，根据配置创建对应后端"""

    _registry: Dict[str, Type[ASRBackend]] = {}

    @classmethod
    def register(cls, name: str, backend_class: Type[ASRBackend]):
        """注册后端实现"""
        cls._registry[name] = backend_class

    @classmethod
    def create(cls, config) -> ASRBackend:
        """根据配置创建后端实例

        Args:
            config: 配置对象（Config 实例）

        Returns:
            ASRBackend: 后端实例

        Raises:
            ValueError: 未知后端类型
        """
        backend_name = config.backend_name

        if backend_name not in cls._registry:
            raise ValueError(f"未知后端: {backend_name}。可用后端: {list(cls._registry.keys())}")

        backend_class = cls._registry[backend_name]
        return backend_class(config)

    @classmethod
    def available_backends(cls) -> list:
        """返回所有可用后端名称"""
        return list(cls._registry.keys())


# 延迟导入并注册后端
def _register_backends():
    """注册所有后端实现"""
    try:
        from .gguf_backend import GGUFBackend
        BackendFactory.register("gguf", GGUFBackend)
    except ImportError as e:
        import logging
        logging.getLogger(__name__).warning(f"GGUF 后端不可用: {e}")

    try:
        from .qwen3_backend import Qwen3Backend
        BackendFactory.register("qwen3-asr", Qwen3Backend)
    except ImportError as e:
        import logging
        logging.getLogger(__name__).warning(f"Qwen3-ASR 后端不可用: {e}")

    try:
        from .funasr_backend import FunASRBackend
        BackendFactory.register("funasr", FunASRBackend)
    except ImportError as e:
        import logging
        logging.getLogger(__name__).warning(f"FunASR 后端不可用: {e}")

    try:
        from .asr_engine_backend import ASREngineClientBackend
        BackendFactory.register("asr-engine", ASREngineClientBackend)
    except ImportError as e:
        import logging
        logging.getLogger(__name__).warning(f"ASR-Engine 后端不可用: {e}")


# 模块导入时自动注册
_register_backends()
