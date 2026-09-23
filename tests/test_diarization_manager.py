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
