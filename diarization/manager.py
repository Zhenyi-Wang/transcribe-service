"""说话人分离管理器：pyannote-hybrid 管线的懒加载与推理。

延迟导入约定：pyannote/torch 的 import 全部在 _load()/diarize() 函数体内，
本模块顶层（含 __init__.py）不允许出现重依赖 import。
"""
import logging
import os
import threading
from dataclasses import dataclass
from typing import List

import numpy as np

from config import config

logger = logging.getLogger("diarization")


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: int


class DiarizationManager:
    """懒加载 + 常驻的分离管线。加载失败抛异常（调用方降级），不静默回退。"""

    def __init__(self, backend: str, cluster_threshold: float, min_cluster_size: int,
                 num_speakers: int, embedding_model: str, hf_token: str):
        self.backend = backend
        self.cluster_threshold = cluster_threshold
        self.min_cluster_size = min_cluster_size
        self.num_speakers = num_speakers
        self.embedding_model = embedding_model
        self.hf_token = hf_token
        self._pipeline = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()  # pyannote 管线非并发安全，推理串行化

    def _load(self):
        with self._load_lock:
            if self._pipeline is not None:
                return
            if self.backend != "pyannote-hybrid":
                raise ValueError(f"未实现的 diarization.backend: {self.backend!r}")
            if self.hf_token:
                # huggingface_hub 凭据链读取；setdefault 不覆盖用户既有环境
                os.environ.setdefault("HF_TOKEN", self.hf_token)
            # 以下均为延迟导入（见模块 docstring）
            import torch
            from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

            pipeline = SpeakerDiarization(
                segmentation="pyannote/segmentation-3.0",
                embedding=os.path.expanduser(self.embedding_model),  # 本地 ONNX（~ 展开）→ ONNXWeSpeakerPretrainedSpeakerEmbedding
                clustering="AgglomerativeClustering",
                segmentation_batch_size=32,
                embedding_batch_size=32,
            )
            pipeline.instantiate({
                "segmentation": {"min_duration_off": 0.0},
                "clustering": {"method": "centroid",
                               "threshold": self.cluster_threshold,
                               "min_cluster_size": self.min_cluster_size},
            })
            pipeline.to(torch.device("cuda"))
            self._pipeline = pipeline
            logger.info("说话人分离管线加载完成（pyannote-hybrid, cuda）")

    @staticmethod
    def _compact_turns(raw) -> list:
        """[(label, start, end), ...] → 紧凑编号的 SpeakerTurn 列表（按 label 首次出现顺序）"""
        label_to_id, turns = {}, []
        for label, start, end in raw:
            if label not in label_to_id:
                label_to_id[label] = len(label_to_id)
            turns.append(SpeakerTurn(start=start, end=end, speaker=label_to_id[label]))
        return turns

    def diarize(self, samples: np.ndarray, sample_rate: int = 16000) -> List[SpeakerTurn]:
        self._load()
        with self._infer_lock:
            import torch
            waveform = torch.from_numpy(samples).unsqueeze(0)  # (1, T)
            audio = {"waveform": waveform, "sample_rate": sample_rate}
            kwargs = {"num_speakers": self.num_speakers} if self.num_speakers > 0 else {}
            diarization = self._pipeline(audio, **kwargs)
        raw = [(label, turn.start, turn.end) for turn, _, label in diarization.itertracks(yield_label=True)]
        turns = self._compact_turns(raw)
        logger.info(f"说话人分离完成: {len(set(t.speaker for t in turns))} 人 / {len(turns)} turns")
        return turns


_manager = None
_manager_lock = threading.Lock()


def get_manager() -> DiarizationManager:
    """进程级单例；首次调用时按 config 构造（懒加载，模型在首次 diarize 时才加载）"""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = DiarizationManager(
                    backend=config.diarization_backend,
                    cluster_threshold=config.diarization_cluster_threshold,
                    min_cluster_size=config.diarization_min_cluster_size,
                    num_speakers=config.diarization_num_speakers,
                    embedding_model=config.diarization_embedding_model,
                    hf_token=config.diarization_hf_token,
                )
    return _manager
