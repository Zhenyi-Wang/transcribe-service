import numpy as np
import pytest

from diarization.manager import DiarizationManager, SpeakerTurn


def test_speaker_turn_fields():
    t = SpeakerTurn(start=1.0, end=2.5, speaker=0)
    assert (t.start, t.end, t.speaker) == (1.0, 2.5, 0)


def test_backend_validation_rejects_unknown():
    """backend 非法值 → diarize 抛 ValueError（走降级路径，不静默回退）"""
    m = DiarizationManager(
        backend="sherpa-onnx", cluster_threshold=0.7046, min_cluster_size=12,
        num_speakers=-1, embedding_model="~/nonexistent.onnx", hf_token="",
    )
    with pytest.raises(ValueError):
        m.diarize(np.zeros(16000, dtype=np.float32))


def test_turns_compact_relabel():
    """pyannote 标签重映射为紧凑 0..N-1（按首次出现顺序）——通过注入假管线测重映射逻辑"""
    m = DiarizationManager.__new__(DiarizationManager)  # 跳过 __init__
    raw = [("SPEAKER_05", 0.0, 1.0), ("SPEAKER_02", 1.0, 2.0), ("SPEAKER_05", 2.0, 3.0)]
    turns = m._compact_turns(raw)
    assert [(t.speaker, t.start, t.end) for t in turns] == [
        (0, 0.0, 1.0), (1, 1.0, 2.0), (0, 2.0, 3.0)
    ]


@pytest.mark.parametrize("readonly", [True, False])
def test_diarize_prepares_writable_input_without_copying_writable_arrays(monkeypatch, readonly):
    import sys
    import types

    original = np.array([0.25, -0.5, 0.75], dtype=np.float32)
    samples = np.frombuffer(original.tobytes(), dtype=np.float32) if readonly else original
    captured = {}

    class Tensor:
        def __init__(self, array):
            self.array = array

        def unsqueeze(self, axis):
            assert axis == 0
            return self

    class Pipeline:
        def __call__(self, audio, **kwargs):
            captured["prepared"] = audio["waveform"].array
            assert audio["sample_rate"] == 16000
            return types.SimpleNamespace(itertracks=lambda **kw: [])

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(from_numpy=Tensor))
    manager = DiarizationManager(
        backend="pyannote-hybrid", cluster_threshold=0.7046, min_cluster_size=12,
        num_speakers=-1, embedding_model="~/nonexistent.onnx", hf_token="",
    )
    manager._pipeline = Pipeline()  # 只替换昂贵的模型，真实执行准备输入与锁内推理路径
    assert manager.diarize(samples) == []
    prepared = captured["prepared"]
    assert prepared.flags.writeable
    assert prepared.dtype == np.float32
    np.testing.assert_array_equal(prepared, [0.25, -0.5, 0.75])
    if readonly:
        assert not np.shares_memory(prepared, samples)
        assert not samples.flags.writeable
    else:
        assert prepared is samples


def test_writable_preparation_releases_original_readonly_storage():
    import weakref
    from diarization import manager as M
    assert hasattr(M, "_writable_samples"), "工作器需要在加载模型前准备可写输入"
    source = np.frombuffer(np.array([0.25, -0.5], dtype=np.float32).tobytes(), dtype=np.float32)
    reference = weakref.ref(source)
    prepared = M._writable_samples(source)
    del source
    assert reference() is None
    assert prepared.flags.writeable
    np.testing.assert_array_equal(prepared, [0.25, -0.5])
