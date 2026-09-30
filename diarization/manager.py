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

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def unload(self, should_abort=None) -> bool:
        """卸载管线释放显存(服务暂停联动)。先等在跑的推理结束;未加载返回 False。

        should_abort: 中止回调(Callable[[], bool]),在获取 _infer_lock 之后、检查/清空
        _pipeline 之前调用——等锁期间世代可能已变(如等待长推理时用户 /resume),
        返回 True 即提前中止,防"恢复后又被旧释放链卸载"。

        锁序 _infer_lock → _load_lock:与 diarize(_infer_lock 内 _load)一致,无死锁。
        可在后台线程长时间等待(长视频推理分钟级)——调用方不得在请求线程同步调用。
        """
        with self._infer_lock:
            if should_abort is not None and should_abort():
                return False  # 等锁期间世代已变,提前中止(管线保持原状)
            with self._load_lock:
                if self._pipeline is None:
                    return False
                self._pipeline = None
                import gc
                import torch  # 延迟导入(模块 docstring 约定)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("说话人分离管线已卸载")
                return True

    def diarize(self, samples: np.ndarray, sample_rate: int = 16000) -> List[SpeakerTurn]:
        # _load 在 _infer_lock 内:加载→推理构成同一锁保护的完整生命周期,
        # 与 unload(_infer_lock → _load_lock)无交错窗口
        with self._infer_lock:
            self._load()
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


def unload_global(should_abort=None) -> bool:
    """卸载单例管线并丢弃单例(下次 get_manager 按 config 重建);should_abort 透传给 unload。

    中止(世代已变)时管线仍在 → 保留单例供后续推理直接复用,/status 的
    diarization_loaded 保持如实;其余情形(正常卸载/本就未加载)照旧丢弃单例。
    """
    global _manager
    m = _manager
    if m is None:
        return False
    released = m.unload(should_abort=should_abort)
    if not released and m.is_loaded:
        return False  # 等锁期间被中止,管线未动,单例保留
    with _manager_lock:
        _manager = None
    return released


def is_loaded() -> bool:
    return _manager is not None and _manager.is_loaded
