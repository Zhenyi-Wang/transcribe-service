import pytest
from unittest.mock import patch, MagicMock

import transcribe as T
from backends.base import TranscribeResult
from diarization.manager import SpeakerTurn


class FakeBackend:
    name = "fake"
    device = "cpu"

    def transcribe(self, path, lang=None, context=None):
        chars = "甲乙甲乙二人对话内容持续输出一直到十六字整"  # 21 字（断言不依赖字数）
        return TranscribeResult(
            text=chars,
            language="zh",
            timestamps=[{"text": c, "start": i * 0.5, "end": i * 0.5 + 0.4}
                        for i, c in enumerate(chars)],
            performance={"rtf": 0.1},
        )


def _make_service():
    svc = T.TranscriptionService(MagicMock())
    svc.model_manager.load_model_if_needed.return_value = FakeBackend()
    return svc


def _turns():
    return [SpeakerTurn(0.0, 4.0, 0), SpeakerTurn(4.0, 16.0, 1)]


def _enable_diarization(monkeypatch):
    """config.diarization_enabled 是类 property，monkeypatch 用 property 覆盖"""
    monkeypatch.setattr(type(T.config), "diarization_enabled",
                        property(lambda self: True))


@pytest.mark.asyncio
async def test_diarize_true_attaches_speakers(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 8.0)
    _enable_diarization(monkeypatch)
    svc = _make_service()
    with patch.object(T, "_diarize_samples", lambda path: _turns()):
        resp = await svc.process_transcription(str(wav), "a.wav", no_cache=True, diarize=True)
    speakers = {seg["speaker"] for seg in resp["body"]}
    assert speakers == {0, 1}
    assert "speakers" in resp and resp["speakers"][0]["id"] == 0
    assert "diarization" in resp["timing"]


@pytest.mark.asyncio
async def test_diarize_failure_degrades_and_skips_cache(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 8.0)
    _enable_diarization(monkeypatch)
    svc = _make_service()

    def boom(path):
        raise RuntimeError("model gone")
    with patch.object(T, "_diarize_samples", boom), \
         patch.object(T.cache_manager, "save_transcript_to_cache") as save_mock:
        resp = await svc.process_transcription(str(wav), "a.wav", no_cache=False,
                                               file_path_for_cache=str(wav), diarize=True)
    assert all("speaker" not in seg for seg in resp["body"])
    assert "speakers" not in resp
    save_mock.assert_not_called()  # 有 file_path_for_cache 且被跳过 → 证明降级不写缓存


@pytest.mark.asyncio
async def test_diarize_single_speaker_degrades_but_caches(tmp_path, monkeypatch):
    """单人（分离成功）→ 无 speaker 标注但正常写缓存（区别于失败降级）"""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 8.0)
    _enable_diarization(monkeypatch)
    svc = _make_service()
    with patch.object(T, "_diarize_samples", lambda path: [SpeakerTurn(0.0, 16.0, 0)]), \
         patch.object(T.cache_manager, "save_transcript_to_cache") as save_mock:
        resp = await svc.process_transcription(str(wav), "a.wav", no_cache=False,
                                               file_path_for_cache=str(wav), diarize=True)
    assert all("speaker" not in seg for seg in resp["body"])
    save_mock.assert_called_once()


@pytest.mark.asyncio
async def test_diarize_false_baseline(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 8.0)
    svc = _make_service()
    resp = await svc.process_transcription(str(wav), "a.wav", no_cache=True, diarize=False)
    assert all("speaker" not in seg for seg in resp["body"])
    assert "speakers" not in resp
    assert "diarization" not in resp["timing"]


@pytest.mark.asyncio
async def test_diarize_timeout_degrades(tmp_path, monkeypatch):
    """分离超时 → 降级且不写缓存（patch 超时函数为极小值触发真超时）"""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 8.0)
    monkeypatch.setattr(T, "_diarize_timeout", lambda d: 0.05)
    _enable_diarization(monkeypatch)
    svc = _make_service()

    def slow(path):
        import time as _t
        _t.sleep(1.0)  # 超过 0.05s 超时
        return _turns()
    with patch.object(T, "_diarize_samples", slow), \
         patch.object(T.cache_manager, "save_transcript_to_cache") as save_mock:
        resp = await svc.process_transcription(str(wav), "a.wav", no_cache=False,
                                               file_path_for_cache=str(wav), diarize=True)
    assert all("speaker" not in seg for seg in resp["body"])
    assert resp["status"] == "success"
    save_mock.assert_not_called()


@pytest.mark.asyncio
async def test_diarize_true_but_disabled(tmp_path, monkeypatch):
    """diarize=true 但 enabled=false：按未启用处理，timing.diarization==0.0

    显式 patch 为 False 而非依赖默认值——生产 config.yaml 已启用分离。
    """
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 8.0)
    monkeypatch.setattr(type(T.config), "diarization_enabled",
                        property(lambda self: False))
    svc = _make_service()
    with patch.object(T, "_diarize_samples") as should_not_call:
        resp = await svc.process_transcription(str(wav), "a.wav", no_cache=True, diarize=True)
    assert all("speaker" not in seg for seg in resp["body"])
    assert resp["timing"]["diarization"] == 0.0
    should_not_call.assert_not_called()
