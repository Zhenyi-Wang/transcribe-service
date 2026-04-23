import os
import yaml
from typing import Dict, Any

class Config:
    """配置管理类"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"配置文件 {self.config_path} 不存在！"
                f"请复制 config.yaml.example 为 config.yaml 并根据需要修改配置"
            )

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误: {e}")
        except Exception as e:
            raise RuntimeError(f"读取配置文件失败: {e}")

    def get(self, key: str, default=None):
        """获取配置项"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    @property
    def idle_timeout(self) -> int:
        """闲置超时时间（秒）"""
        return self.get('server.idle_timeout', 300)

    @property
    def check_interval(self) -> int:
        """检查间隔（秒）"""
        return self.get('server.check_interval', 10)

    @property
    def asr_model(self) -> str:
        """ASR 模型名称"""
        return self.get('model.asr_model', 'Qwen/Qwen3-ASR-1.7B')

    @property
    def forced_aligner(self) -> str:
        """时间戳对齐模型"""
        return self.get('model.forced_aligner', 'Qwen/Qwen3-ForcedAligner-0.6B')

    @property
    def dtype(self) -> str:
        """模型精度"""
        return self.get('model.dtype', 'bfloat16')

    @property
    def max_new_tokens(self) -> int:
        """最大生成 token 数"""
        return self.get('model.max_new_tokens', 4096)

    @property
    def max_segment_length(self) -> int:
        """最大段落长度"""
        return self.get('processing.max_segment_length', 20)

    @property
    def duration_per_segment(self) -> float:
        """每段字幕持续时间（秒）"""
        return self.get('processing.duration_per_segment', 3.0)


    # 字幕样式配置
    @property
    def subtitle_config(self) -> Dict[str, Any]:
        """字幕样式配置"""
        return self.get('subtitle', {
            "font_size": 0.4,
            "font_color": "#FFFFFF",
            "background_alpha": 0.5,
            "background_color": "#9C27B0",
            "stroke": "none",
            "type": "manual_transcribe",
            "version": "v1"
        })

    # API配置
    @property
    def api_config(self) -> Dict[str, Any]:
        """API配置"""
        return self.get('api', {
            "host": "0.0.0.0",
            "port": 8000,
            "token": ""
        })

    @property
    def api_token(self) -> str:
        """API访问令牌"""
        return self.get('api.token', "")

    # 缓存配置
    @property
    def cache_enabled(self) -> bool:
        """是否启用缓存"""
        return self.get('cache.enabled', True)

    @property
    def cache_dir(self) -> str:
        """缓存目录"""
        return self.get('cache.dir', 'cache')

    @property
    def cache_days(self) -> int:
        """缓存保存天数"""
        return self.get('cache.days', 7)

# 全局配置实例
config = Config()