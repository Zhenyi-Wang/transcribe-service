"""pyannote 分离管线内存边界测试。

覆盖：三边界配置（max_speakers / max_num_embeddings / max_reconstruction_mb）
+ cudnn_conv_algo_search 可配置化。全部离线：不加载模型，torch/pyannote/onnxruntime
以 fake 模块注入，不触 GPU。
"""
import logging
import sys
import types

import numpy as np
import pytest

from diarization import manager as dm
from diarization.manager import (
    DiarizationManager,
    DiarizationResourceLimitError,
    check_reconstruction_budget,
)

MB = 1024 * 1024


def _make_manager(**overrides):
    kwargs = dict(backend="pyannote-hybrid", cluster_threshold=0.7046,
                  min_cluster_size=12, num_speakers=-1,
                  embedding_model="~/models/diarization/fake.onnx", hf_token="")
    kwargs.update(overrides)
    return DiarizationManager(**kwargs)


# ========== Config 属性与默认值 ==========

def _config_from(tmp_path, yaml_text):
    import config as config_mod
    p = tmp_path / "config.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    return config_mod.Config(str(p))


def test_config_memory_limit_defaults(tmp_path):
    """新字段缺省 = 安全默认：max_speakers=16 / max_num_embeddings=1000 /
    max_reconstruction_mb=512 / cudnn_conv_algo_search=EXHAUSTIVE"""
    c = _config_from(tmp_path, "backend:\n  name: qwen3-asr\n")
    assert c.diarization_max_speakers == 16
    assert c.diarization_max_num_embeddings == 1000
    assert c.diarization_max_reconstruction_mb == 512
    assert c.diarization_cudnn_conv_algo_search == "EXHAUSTIVE"


def test_config_memory_limit_overrides(tmp_path):
    c = _config_from(tmp_path, (
        "diarization:\n"
        "  max_speakers: 8\n"
        "  max_num_embeddings: 300\n"
        "  max_reconstruction_mb: 256.5\n"
        '  cudnn_conv_algo_search: "HEURISTIC"\n'
    ))
    assert c.diarization_max_speakers == 8
    assert c.diarization_max_num_embeddings == 300
    assert c.diarization_max_reconstruction_mb == 256.5
    assert c.diarization_cudnn_conv_algo_search == "HEURISTIC"


# ========== __init__ 硬验证（拒绝 0/负/NaN/inf/非数/非整数） ==========

@pytest.mark.parametrize("bad", [0, -1, -16, float("nan"), float("inf"), -float("inf"),
                                 np.inf, float("2.5"), "16", None, True])
def test_init_rejects_invalid_max_speakers(bad):
    with pytest.raises(ValueError):
        _make_manager(max_speakers=bad)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), "1000", 1.5, None])
def test_init_rejects_invalid_max_num_embeddings(bad):
    with pytest.raises(ValueError):
        _make_manager(max_num_embeddings=bad)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), "512", None])
def test_init_rejects_invalid_max_reconstruction_mb(bad):
    with pytest.raises(ValueError):
        _make_manager(max_reconstruction_mb=bad)


@pytest.mark.parametrize("kwargs", [
    dict(max_speakers=1), dict(max_speakers=16),
    dict(max_num_embeddings=1), dict(max_num_embeddings=np.int64(500)),
    dict(max_reconstruction_mb=0.5), dict(max_reconstruction_mb=512.5),
])
def test_init_accepts_valid_boundaries(kwargs):
    m = _make_manager(**kwargs)
    if "max_speakers" in kwargs:
        assert m.max_speakers == kwargs["max_speakers"]
    if "max_num_embeddings" in kwargs:
        assert m.max_num_embeddings == kwargs["max_num_embeddings"]
    if "max_reconstruction_mb" in kwargs:
        assert m.max_reconstruction_mb == kwargs["max_reconstruction_mb"]


def test_init_explicit_num_speakers_over_max_rejected():
    """显式 num_speakers > max_speakers → 明确 ValueError，不静默裁剪"""
    with pytest.raises(ValueError, match="num_speakers.*max_speakers|max_speakers.*num_speakers"):
        _make_manager(num_speakers=20, max_speakers=16)


@pytest.mark.parametrize("ns", [-1, 0, 1, 16])
def test_init_num_speakers_within_max_ok(ns):
    assert _make_manager(num_speakers=ns, max_speakers=16).num_speakers == ns


def test_init_num_speakers_non_integer_rejected():
    with pytest.raises(ValueError):
        _make_manager(num_speakers=2.5)


# ========== cudnn_conv_algo_search 可配置 ==========

@pytest.mark.parametrize("value,expected", [
    ("default", "DEFAULT"), ("Exhaustive", "EXHAUSTIVE"), ("HEURISTIC", "HEURISTIC"),
])
def test_cudnn_enum_normalized(value, expected):
    assert _make_manager(cudnn_conv_algo_search=value).cudnn_conv_algo_search == expected


@pytest.mark.parametrize("bad", ["OBSERVE", "", 123, ["EXHAUSTIVE"]])
def test_cudnn_invalid_rejected(bad):
    with pytest.raises(ValueError):
        _make_manager(cudnn_conv_algo_search=bad)


def test_cudnn_none_reads_config(monkeypatch):
    """构造参数 None = 读 config（工作进程显式传值之外的默认路径）"""
    monkeypatch.setattr(type(dm.config), "diarization_cudnn_conv_algo_search",
                        property(lambda self: "heuristic"))
    assert _make_manager(cudnn_conv_algo_search=None).cudnn_conv_algo_search == "HEURISTIC"


# ========== cudnn patch helper（改写 DEFAULT + finally 恢复） ==========

def _fake_ort(monkeypatch):
    """注入假 onnxruntime 模块；返回记录 providers 的 dict"""
    recorded = {}

    def fake_session(path_or_bytes, sess_options=None, providers=None, **kwargs):
        recorded["providers"] = providers

    monkeypatch.setitem(sys.modules, "onnxruntime",
                        types.SimpleNamespace(InferenceSession=fake_session))
    return recorded


def _call_patched(recorded, providers):
    sys.modules["onnxruntime"].InferenceSession("model.onnx", providers=providers)
    return recorded["providers"]


def test_cudnn_helper_replaces_default_and_restores(monkeypatch):
    """pyannote 目标条目 DEFAULT → EXHAUSTIVE（其余选项保留），退出恢复原函数"""
    recorded = _fake_ort(monkeypatch)
    original = sys.modules["onnxruntime"].InferenceSession
    providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT",
                                            "device_id": 0}),
                 "CPUExecutionProvider"]
    with dm._strip_cudnn_conv_algo_search_default("EXHAUSTIVE"):
        assert sys.modules["onnxruntime"].InferenceSession is not original
        got = _call_patched(recorded, providers)
    assert got == [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "EXHAUSTIVE",
                                              "device_id": 0}),
                   "CPUExecutionProvider"]
    assert sys.modules["onnxruntime"].InferenceSession is original


def test_cudnn_helper_target_default_is_passthrough(monkeypatch):
    """target=DEFAULT（A/B 基线）：值原样保留，不改写不剥除"""
    recorded = _fake_ort(monkeypatch)
    providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})]
    with dm._strip_cudnn_conv_algo_search_default("DEFAULT"):
        assert _call_patched(recorded, providers) == providers


def test_cudnn_helper_passthrough_non_target(monkeypatch):
    """裸 CUDA provider（GGUF encoder）/ CPU EP / 值非 DEFAULT → 原样透传"""
    recorded = _fake_ort(monkeypatch)
    variants = [
        ["CUDAExecutionProvider"],
        [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})],
        ["CPUExecutionProvider"],
    ]
    with dm._strip_cudnn_conv_algo_search_default("EXHAUSTIVE"):
        for p in variants:
            assert _call_patched(recorded, p) == p


def test_cudnn_helper_restores_on_exception(monkeypatch):
    recorded = _fake_ort(monkeypatch)
    original = sys.modules["onnxruntime"].InferenceSession
    with pytest.raises(RuntimeError):
        with dm._strip_cudnn_conv_algo_search_default("EXHAUSTIVE"):
            raise RuntimeError("boom")
    assert sys.modules["onnxruntime"].InferenceSession is original


# ========== reconstruct 重建预算检查 ==========

def _clusters(chunks, speakers, num_clusters):
    c = np.arange(chunks * speakers, dtype=np.int8) % max(num_clusters, 1)
    return c.reshape(chunks, speakers)


def test_budget_under_limit_returns_info():
    hard = _clusters(100, 3, 3)
    info = check_reconstruction_budget((100, 589, 3), hard, max_bytes=512 * MB)
    assert info["num_chunks"] == 100 and info["num_frames"] == 589
    assert info["num_clusters"] == 3
    assert info["bytes"] == 100 * 589 * 3 * 8  # float64
    assert info["bytes_doubled"] == 2 * info["bytes"]


def test_budget_over_limit_raises():
    """(8000 chunks × 589 帧 × 8 簇) 双缓冲 576MB > 512MB → 分配前抛可降级异常"""
    hard = _clusters(8000, 3, 8)
    with pytest.raises(DiarizationResourceLimitError, match="max_reconstruction_mb"):
        check_reconstruction_budget((8000, 589, 3), hard, max_bytes=512 * MB)


def test_budget_exact_limit_passes_below_fails():
    c, f, k = 200, 100, 16
    hard = _clusters(c, 3, k)
    seg_shape = (c, f, 3)
    exact = 2 * c * f * k * 8
    assert check_reconstruction_budget(seg_shape, hard, max_bytes=exact)["bytes"] > 0
    with pytest.raises(DiarizationResourceLimitError):
        check_reconstruction_budget(seg_shape, hard, max_bytes=exact - 1)


def test_budget_shape_mismatch_raises():
    with pytest.raises(DiarizationResourceLimitError, match="形状"):
        check_reconstruction_budget((10, 50, 3), np.zeros((10, 4), dtype=np.int8),
                                    max_bytes=512 * MB)


def test_budget_clusters_over_max_speakers_raises():
    """实际簇数超 max_speakers（上游约束失效）→ 拒绝重建，即使字节未超限"""
    hard = _clusters(10, 3, 20)
    with pytest.raises(DiarizationResourceLimitError, match="max_speakers"):
        check_reconstruction_budget((10, 50, 3), hard,
                                    max_bytes=512 * MB, max_speakers=16)


def test_budget_all_unassigned_clusters_clamped_to_zero():
    hard = -2 * np.ones((5, 3), dtype=np.int8)
    info = check_reconstruction_budget((5, 50, 3), hard, max_bytes=512 * MB)
    assert info["num_clusters"] == 0 and info["bytes"] == 0


def test_resource_limit_error_is_degradable():
    """编排层 except Exception 兜底 → 必须是 Exception 子类"""
    assert issubclass(DiarizationResourceLimitError, Exception)


# ========== _load 接线（fake pyannote/torch，不触真依赖） ==========

def test_load_sets_pipeline_boundaries(monkeypatch):
    captured = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            captured["init_count"] = captured.get("init_count", 0) + 1
            captured["init_kwargs"] = kwargs
            self.clustering = types.SimpleNamespace(max_num_embeddings=np.inf)
            self._max_reconstruction_bytes = None
            self._max_speakers = None

        def instantiate(self, params):
            captured["instantiate"] = params
            # instantiate 时仍是 pyannote 默认 np.inf，边界在之后写入
            captured["max_num_embeddings_at_instantiate"] = self.clustering.max_num_embeddings

        def to(self, device):
            captured["device"] = device

    fake_pyannote = types.SimpleNamespace(SpeakerDiarization=FakePipeline)
    fake_torch = types.SimpleNamespace(device=lambda spec: ("device", spec))
    monkeypatch.setitem(sys.modules, "pyannote.audio.pipelines.speaker_diarization", fake_pyannote)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(dm, "_BOUNDED_PIPELINE_CLASS", None)

    used = {}
    real_strip = dm._strip_cudnn_conv_algo_search_default

    def recording_strip(target="EXHAUSTIVE"):
        used["target"] = target
        return real_strip(target)

    monkeypatch.setattr(dm, "_strip_cudnn_conv_algo_search_default", recording_strip)

    m = _make_manager(max_speakers=9, max_num_embeddings=500,
                      max_reconstruction_mb=256, cudnn_conv_algo_search="HEURISTIC")
    m._load()
    p = m._pipeline

    assert isinstance(p, FakePipeline)
    assert captured["max_num_embeddings_at_instantiate"] == np.inf
    assert p.clustering.max_num_embeddings == 500   # 接线让 filter_embeddings 子采样生效
    assert p._max_reconstruction_bytes == 256 * MB
    assert p._max_speakers == 9
    assert used["target"] == "HEURISTIC"            # cudnn 策略传入 patch
    assert captured["device"] == ("device", "cuda")
    assert captured["init_count"] == 1
    m._load()  # 幂等：已加载直接返回
    assert captured["init_count"] == 1


# ========== diarize 传参（max_speakers 限制自动聚类 + 阶段日志 hook） ==========

class _FakeWave:
    def unsqueeze(self, n):
        return self


class _FakeDiarization:
    def itertracks(self, yield_label):
        seg = types.SimpleNamespace(start=0.0, end=1.0)
        return iter([(seg, "t0", "SPEAKER_00"), (seg, "t1", "SPEAKER_01")])


class _FakeCallablePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return _FakeDiarization()


def _loaded_manager(**overrides):
    m = _make_manager(**overrides)
    m._pipeline = _FakeCallablePipeline()
    return m


def test_diarize_auto_passes_max_speakers_and_hook(monkeypatch):
    """自动模式：传 max_speakers（限制 set_num_clusters 上界 + count 截断）+ hook"""
    monkeypatch.setitem(sys.modules, "torch",
                        types.SimpleNamespace(from_numpy=lambda a: _FakeWave()))
    fp = _FakeCallablePipeline()
    m = _loaded_manager(num_speakers=-1, max_speakers=16)
    m._pipeline = fp
    turns = m.diarize(np.zeros(16000, dtype=np.float32))
    _, kwargs = fp.calls[0]
    assert kwargs["max_speakers"] == 16
    assert "num_speakers" not in kwargs
    assert callable(kwargs["hook"])
    assert [(t.speaker, t.start, t.end) for t in turns] == [(0, 0.0, 1.0), (1, 0.0, 1.0)]


def test_diarize_explicit_num_speakers_kept(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch",
                        types.SimpleNamespace(from_numpy=lambda a: _FakeWave()))
    fp = _FakeCallablePipeline()
    m = _loaded_manager(num_speakers=2, max_speakers=16)
    m._pipeline = fp
    m.diarize(np.zeros(16000, dtype=np.float32))
    _, kwargs = fp.calls[0]
    assert kwargs["num_speakers"] == 2
    assert kwargs["max_speakers"] == 16  # 恒传，num_speakers 存在时被 pyannote 覆盖


def test_stage_hook_logs_final_stages_only(caplog):
    """带 total/completed 的批内进度调用跳过；阶段末尾记录形状与 embedding 槽位数"""
    m = _make_manager()
    with caplog.at_level(logging.INFO, logger="diarization"):
        m._log_pipeline_stage("embeddings", types.SimpleNamespace(shape=(32, 256)),
                              total=100, completed=1)  # 批内进度 → 跳过
        m._log_pipeline_stage("embeddings", types.SimpleNamespace(shape=(100, 3, 256)))
        m._log_pipeline_stage("segmentation", types.SimpleNamespace(
            data=types.SimpleNamespace(shape=(100, 589, 3))))
    texts = [r.getMessage() for r in caplog.records]
    assert len(texts) == 2
    assert "300" in texts[0]        # 槽位数 = 100×3（静音/NaN 槽位仍计入形状）
    assert "rss=" in texts[0]
    assert "(100, 589, 3)" in texts[1]


def test_stage_hook_swallows_logging_errors():
    """日志 hook 自身异常不外泄（不影响推理）"""
    m = _make_manager()
    m._log_pipeline_stage("embeddings", None)  # artefact=None 不应抛


# ========== get_manager 按 config 构造 ==========

def test_get_manager_reads_config_boundaries(monkeypatch):
    monkeypatch.setattr(type(dm.config), "diarization_max_speakers", property(lambda self: 7))
    monkeypatch.setattr(type(dm.config), "diarization_max_num_embeddings",
                        property(lambda self: 400))
    monkeypatch.setattr(type(dm.config), "diarization_max_reconstruction_mb",
                        property(lambda self: 128))
    monkeypatch.setattr(type(dm.config), "diarization_cudnn_conv_algo_search",
                        property(lambda self: "HEURISTIC"))
    monkeypatch.setattr(dm, "_manager", None)
    m = dm.get_manager()
    assert m.max_speakers == 7
    assert m.max_num_embeddings == 400
    assert m.max_reconstruction_mb == 128
    assert m.cudnn_conv_algo_search == "HEURISTIC"
