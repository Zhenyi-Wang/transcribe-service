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
    def model_name(self) -> str:
        """模型名称"""
        return self.get('model.name', 'paraformer-zh')

    @property
    def vad_model(self) -> str:
        """VAD模型名称"""
        return self.get('model.vad_model', 'fsmn-vad')

    @property
    def punc_model(self) -> str:
        """标点模型名称"""
        return self.get('model.punc_model', 'ct-punc')

    @property
    def disable_update(self) -> bool:
        """禁用模型更新检查"""
        return self.get('model.disable_update', True)

    @property
    def batch_size_s(self) -> int:
        """批处理大小（秒）"""
        return self.get('processing.batch_size_s', 300)

    @property
    def max_segment_length(self) -> int:
        """最大段落长度"""
        return self.get('processing.max_segment_length', 20)

    @property
    def duration_per_segment(self) -> float:
        """每段字幕持续时间（秒）"""
        return self.get('processing.duration_per_segment', 3.0)

    @property
    def chinese_ratio_threshold(self) -> float:
        """中文比例阈值"""
        return self.get('processing.chinese_ratio_threshold', 0.3)

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

# 全局配置实例
config = Config()