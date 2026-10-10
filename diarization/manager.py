"""说话人分离管理器：pyannote-hybrid 管线的懒加载与推理。

延迟导入约定：pyannote/torch 的 import 全部在 _load()/diarize() 函数体内，
本模块顶层（含 __init__.py）不允许出现重依赖 import。

内存边界（可靠性护栏，触发即该请求降级，不影响服务其它部分）：
- max_num_embeddings：聚类子采样上限。pyannote 3.3.2 的 AgglomerativeClustering
  默认 np.inf，filter_embeddings 从不子采样，scipy linkage 对全量 embedding
  建 O(N²) 距离矩阵（长音频 GB 级）；设有限值后随机下采样到该上限。
- max_speakers：自动模式聚类簇数上限，随管线调用传入（约束 set_num_clusters
  的 max_clusters 并截断 speaker count）；显式 num_speakers 超过它直接拒绝。
- max_reconstruction_mb：reconstruct 重建数组双缓冲字节预算（见
  _bounded_speaker_diarization_class），分配前核算，超限抛可降级异常。
"""
import logging
import math
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from config import config

logger = logging.getLogger("diarization")

# 边界安全默认值（与 config.py 属性默认一致）
DEFAULT_MAX_SPEAKERS = 16
DEFAULT_MAX_NUM_EMBEDDINGS = 1000
DEFAULT_MAX_RECONSTRUCTION_MB = 512

# cudnn_conv_algo_search 合法取值（ORT CUDA EP 选项；EXHAUSTIVE=ORT 自身默认）
CUDNN_CONV_ALGO_CHOICES = ("DEFAULT", "HEURISTIC", "EXHAUSTIVE")

_FLOAT64_ITEMSIZE = 8  # reconstruct 重建数组 dtype（np.nan * np.zeros 默认 float64）


class DiarizationResourceLimitError(RuntimeError):
    """分离管线触发资源边界（重建预算超限/形状异常/簇数越界）。

    编排层以 except Exception 兜底降级，本异常仅为可读性与日志定案服务。
    """


def _require_positive_int(name: str, value) -> int:
    """硬验证正整数值：拒绝 0/负数/NaN/inf/非数/非整数（bool 视为非数）"""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"diarization.{name} 必须为正整数，得到 {value!r}")
    v = float(value)
    if not math.isfinite(v) or v <= 0 or v != int(v):
        raise ValueError(
            f"diarization.{name} 必须为正整数（得到 {value!r}；拒绝 0/负数/NaN/inf/非整数）")
    return int(v)


def _require_positive_number(name: str, value) -> float:
    """硬验证正的有限数值（允许小数）：拒绝 0/负数/NaN/inf/非数"""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"diarization.{name} 必须为正数，得到 {value!r}")
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(
            f"diarization.{name} 必须为正的有限数值（得到 {value!r}；拒绝 0/负数/NaN/inf）")
    return v


def _validate_num_speakers(num_speakers, max_speakers: int) -> int:
    """num_speakers 验证：整数有限（0/负数=自动，沿用既有语义）；
    显式人数超过 max_speakers 明确拒绝——不静默裁剪。"""
    if isinstance(num_speakers, bool) or not isinstance(num_speakers, (int, float, np.integer, np.floating)) \
            or not math.isfinite(float(num_speakers)) or float(num_speakers) != int(num_speakers):
        raise ValueError(f"diarization.num_speakers 必须为整数（-1/0 自动），得到 {num_speakers!r}")
    ns = int(num_speakers)
    if ns > max_speakers:
        raise ValueError(
            f"diarization.num_speakers={ns} 超过 max_speakers={max_speakers}："
            f"显式人数上限不允许静默裁剪，请调大 diarization.max_speakers 或降低 num_speakers")
    return ns


def _validate_cudnn_conv_algo_search(value) -> str:
    if not isinstance(value, str) or value.strip().upper() not in CUDNN_CONV_ALGO_CHOICES:
        raise ValueError(
            f"diarization.cudnn_conv_algo_search 必须是 {'/'.join(CUDNN_CONV_ALGO_CHOICES)} 之一，"
            f"得到 {value!r}")
    return value.strip().upper()


@contextmanager
def _strip_cudnn_conv_algo_search_default(target: str = "EXHAUSTIVE"):
    """建 ONNX 会话期间改写 pyannote 硬编码的 cudnn_conv_algo_search="DEFAULT"。

    pyannote 3.3.2 为 wespeaker embedding 硬编码该选项（speaker_verification.py，
    仅 CUDA EP）。本机实测（docs/2026-10-09_diarization-onnx-conv-fallback.md）：
    DEFAULT 下 ORT 逐 Conv 刷 "Fallback mode" 告警且 embedding 慢 3~10 倍；
    改写为 EXHAUSTIVE（等价于原来的"剥掉选项"，ORT 自身默认即 EXHAUSTIVE）后
    零告警且最快。DEFAULT 为何触发慢路径属 ORT C++ 内部机理，未定案——这里只
    固化实测对应关系"DEFAULT 慢、EXHAUSTIVE 快"，不做"算法找不到/CPU 回退"
    之类的机制断言。

    行为：仅匹配 CUDAExecutionProvider 且该选项值为 "DEFAULT" 的条目，把值改写
    为 target；其余（裸 CUDA provider / CPU EP / 值非 DEFAULT）原样透传——GGUF
    encoder 不受影响，pyannote 未来移除硬编码后本 patch 自动变 no-op。target
    可选 DEFAULT/HEURISTIC/EXHAUSTIVE（"DEFAULT" 即 A/B 基线，原样透传）。
    with 结束（含异常路径）恢复 ort.InferenceSession。
    """
    import onnxruntime as ort  # 延迟导入（模块 docstring 约定）
    original = ort.InferenceSession

    def patched(path_or_bytes, sess_options=None, providers=None, **kwargs):
        if isinstance(providers, list):
            rewritten = []
            for p in providers:
                if (isinstance(p, tuple) and len(p) == 2 and p[0] == "CUDAExecutionProvider"
                        and isinstance(p[1], dict) and p[1].get("cudnn_conv_algo_search") == "DEFAULT"):
                    opts = dict(p[1])
                    opts["cudnn_conv_algo_search"] = target
                    rewritten.append(("CUDAExecutionProvider", opts))
                else:
                    rewritten.append(p)
            providers = rewritten
        return original(path_or_bytes, sess_options=sess_options, providers=providers, **kwargs)

    ort.InferenceSession = patched
    try:
        yield
    finally:
        ort.InferenceSession = original


def check_reconstruction_budget(seg_shape, hard_clusters, max_bytes=None,
                                max_speakers=None) -> dict:
    """reconstruct 大分配前的预算检查（纯函数，供重建守卫与测试直接调用）。

    pyannote 的 reconstruct 无条件分配 float64 (num_chunks, num_frames,
    num_clusters)：np.nan * np.zeros(...) 期间 zeros 与乘积两个数组并存（双缓冲），
    num_clusters 仅由 hard_clusters 决定、无上限。这里在分配前：

    - 校验 hard_clusters 形状与 segmentations 前两轴一致
    - 按实际 hard_clusters 核算簇数（> max_speakers 视为上游约束失效，拒绝）
    - 按双缓冲核算预计字节，超过 max_bytes 抛 DiarizationResourceLimitError

    返回 {"num_chunks", "num_frames", "num_clusters", "bytes", "bytes_doubled"} 供日志。
    """
    num_chunks, num_frames, local_num_speakers = (int(x) for x in seg_shape)
    hard_clusters = np.asarray(hard_clusters)
    if hard_clusters.shape != (num_chunks, local_num_speakers):
        raise DiarizationResourceLimitError(
            f"reconstruct 形状不一致: segmentations={tuple(seg_shape)} vs "
            f"hard_clusters={hard_clusters.shape}，拒绝重建")
    num_clusters = max(int(np.max(hard_clusters)) + 1, 0) if hard_clusters.size else 0
    if max_speakers is not None and num_clusters > max_speakers:
        raise DiarizationResourceLimitError(
            f"reconstruct 实际簇数 {num_clusters} 超过 max_speakers={max_speakers}"
            f"（上游聚类约束失效），拒绝重建")
    # 双缓冲：np.nan * np.zeros(...) 的 zeros 与乘积数组并存；后续 aggregate 还会
    # 再分配同量级缓冲，故 2x 是预算下限（按"至少双缓冲"口径核算）
    bytes_needed = num_chunks * num_frames * num_clusters * _FLOAT64_ITEMSIZE
    if max_bytes is not None and 2 * bytes_needed > int(max_bytes):
        raise DiarizationResourceLimitError(
            f"reconstruct 预计分配 {2 * bytes_needed / 1048576:.1f}MB（双缓冲）超过上限 "
            f"{int(max_bytes) / 1048576:.0f}MB：num_chunks={num_chunks} "
            f"num_frames={num_frames} num_clusters={num_clusters}。"
            f"可调大 diarization.max_reconstruction_mb 或降低音频时长")
    return {"num_chunks": num_chunks, "num_frames": num_frames,
            "num_clusters": num_clusters, "bytes": bytes_needed,
            "bytes_doubled": 2 * bytes_needed}


# 带重建守卫的管线子类缓存（_bounded_speaker_diarization_class 懒构造一次）
_BOUNDED_PIPELINE_CLASS = None


def _bounded_speaker_diarization_class():
    """延迟构造带重建内存边界的 SpeakerDiarization 子类（模块级缓存一次）。

    用子类覆盖 reconstruct 而非改 site-packages / 全局 np patch；基类引用在
    首次调用时才 import（模块 docstring 延迟导入约定）。
    """
    global _BOUNDED_PIPELINE_CLASS
    if _BOUNDED_PIPELINE_CLASS is not None:
        return _BOUNDED_PIPELINE_CLASS

    from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization  # 延迟导入

    class BoundedSpeakerDiarization(SpeakerDiarization):
        """SpeakerDiarization + reconstruct 内存边界。

        覆盖版在父实现分配前核对形状与实际簇数、按双缓冲字节核算预算，
        超限抛 DiarizationResourceLimitError（编排层降级），不发生大分配。
        """

        _max_reconstruction_bytes = None
        _max_speakers = None

        def reconstruct(self, segmentations, hard_clusters, count):
            info = check_reconstruction_budget(
                segmentations.data.shape, hard_clusters,
                max_bytes=self._max_reconstruction_bytes,
                max_speakers=self._max_speakers)
            logger.info(
                "[diarization][reconstruct] chunks=%d frames=%d clusters=%d "
                "预计分配=%.1fMB(双缓冲) rss=%s",
                info["num_chunks"], info["num_frames"], info["num_clusters"],
                info["bytes_doubled"] / 1048576.0, _proc_rss_mb())
            return super().reconstruct(segmentations, hard_clusters, count)

    _BOUNDED_PIPELINE_CLASS = BoundedSpeakerDiarization
    return _BOUNDED_PIPELINE_CLASS


def _proc_rss_mb() -> str:
    """当前进程 RSS（Linux /proc/self/status VmRSS）；不可用时返回 n/a（日志辅助）"""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return f"{int(line.split()[1]) / 1024:.0f}MB"
    except (OSError, ValueError, IndexError):
        pass
    return "n/a"


def _writable_samples(samples: np.ndarray) -> np.ndarray:
    """只读解码缓冲不能直接交给 PyTorch；可写数组保留零拷贝。"""
    return samples if samples.flags.writeable else samples.copy()


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: int


class DiarizationManager:
    """懒加载 + 常驻的分离管线。加载失败抛异常（调用方降级），不静默回退。"""

    def __init__(self, backend: str, cluster_threshold: float, min_cluster_size: int,
                 num_speakers: int, embedding_model: str, hf_token: str,
                 max_speakers: int = DEFAULT_MAX_SPEAKERS,
                 max_num_embeddings: int = DEFAULT_MAX_NUM_EMBEDDINGS,
                 max_reconstruction_mb: float = DEFAULT_MAX_RECONSTRUCTION_MB,
                 cudnn_conv_algo_search: Optional[str] = None):
        # 边界参数先于 num_speakers 验证（后者依赖 max_speakers）；
        # 全部硬验证，非法配置在构造期即失败（get_manager/工作进程皆走此路径）
        self.max_speakers = _require_positive_int("max_speakers", max_speakers)
        self.max_num_embeddings = _require_positive_int("max_num_embeddings", max_num_embeddings)
        self.max_reconstruction_mb = _require_positive_number("max_reconstruction_mb",
                                                              max_reconstruction_mb)
        self.num_speakers = _validate_num_speakers(num_speakers, self.max_speakers)
        if cudnn_conv_algo_search is None:
            cudnn_conv_algo_search = config.diarization_cudnn_conv_algo_search
        self.cudnn_conv_algo_search = _validate_cudnn_conv_algo_search(cudnn_conv_algo_search)
        self.backend = backend
        self.cluster_threshold = cluster_threshold
        self.min_cluster_size = min_cluster_size
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
            pipeline_cls = _bounded_speaker_diarization_class()

            # 建会话期间改写 pyannote 硬编码的 cudnn_conv_algo_search=DEFAULT（见 helper docstring）
            with _strip_cudnn_conv_algo_search_default(self.cudnn_conv_algo_search):
                pipeline = pipeline_cls(
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
                # 内存边界接线（在 instantiate 之后写入，避免任何参数重置路径覆盖）：
                # - clustering.max_num_embeddings：pyannote 构造时只传 metric，默认 np.inf
                #   （linkage 全量 O(N²) 距离矩阵）；有限值让 filter_embeddings 随机下采样
                pipeline.clustering.max_num_embeddings = self.max_num_embeddings
                pipeline._max_reconstruction_bytes = int(self.max_reconstruction_mb * 1048576)
                pipeline._max_speakers = self.max_speakers
                pipeline.to(torch.device("cuda"))
            self._pipeline = pipeline
            logger.info(
                "说话人分离管线加载完成（pyannote-hybrid, cuda, "
                f"max_speakers={self.max_speakers}, "
                f"max_num_embeddings={self.max_num_embeddings}, "
                f"max_reconstruction_mb={self.max_reconstruction_mb:g}, "
                f"cudnn_conv_algo_search={self.cudnn_conv_algo_search}）")

    def _log_pipeline_stage(self, step_name, artefact, **kwargs):
        """管线阶段采样日志 hook（事后定案用：形状/embedding 槽位数/RSS）。

        pyannote 在阶段末尾调 hook(step_name, artefact)（无 total/completed），
        批内进度调用带 total/completed——只记录阶段末尾，避免逐 batch 刷屏。
        日志自身异常不外泄（不影响推理）。
        """
        if "completed" in kwargs:
            return
        try:
            detail = ""
            data = getattr(artefact, "data", artefact)
            shape = getattr(data, "shape", None)
            if shape:
                detail = f"shape={tuple(shape)}"
                if step_name == "embeddings" and len(shape) == 3:
                    detail += f" embedding_slots={shape[0] * shape[1]}"  # 含静音/NaN 槽位，不等同有效训练数
            logger.info("[diarization][stage] %s %s rss=%s", step_name, detail, _proc_rss_mb())
        except Exception:
            logger.debug("管线阶段日志失败", exc_info=True)

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
            samples = _writable_samples(samples)
            waveform = torch.from_numpy(samples).unsqueeze(0)  # (1, T)
            audio = {"waveform": waveform, "sample_rate": sample_rate}
            # max_speakers 恒传：自动模式约束 set_num_clusters 上界并截断 speaker
            # count；显式 num_speakers 时被 pyannote 覆盖（__init__ 已拒绝超限）
            kwargs = {"max_speakers": self.max_speakers,
                      "hook": self._log_pipeline_stage}
            if self.num_speakers > 0:
                kwargs["num_speakers"] = self.num_speakers
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
                    max_speakers=config.diarization_max_speakers,
                    max_num_embeddings=config.diarization_max_num_embeddings,
                    max_reconstruction_mb=config.diarization_max_reconstruction_mb,
                    cudnn_conv_algo_search=config.diarization_cudnn_conv_algo_search,
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
