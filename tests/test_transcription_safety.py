"""重复请求与失败冷却的回归测试，模型/分离用小型替身，缓存真实落盘。"""
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import transcribe as T
from backends.base import TranscribeResult
from cache_manager import CacheManager
from diarization.manager import SpeakerTurn


class CountingBackend:
    name = "fake"
    device = "cpu"

    def __init__(self):
        self.calls = 0

    def transcribe(self, path, lang=None, context=None):
        self.calls += 1
        return TranscribeResult(
            text="甲说乙说", language="zh",
            timestamps=[{"text": c, "start": i * 2.0, "end": i * 2.0 + 1.0}
                        for i, c in enumerate("甲说乙说")],
        )


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(type(T.config), "diarization_enabled", property(lambda self: True))
    monkeypatch.setattr(type(T.config), "cache_dir", property(lambda self: str(tmp_path / "cache")))
    monkeypatch.setattr(type(T.config), "cache_enabled", property(lambda self: True))
    monkeypatch.setattr(T, "cache_manager", CacheManager())
    monkeypatch.setattr(T, "get_audio_duration", lambda path: 8.0)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake-audio")
    backend = CountingBackend()
    manager = MagicMock()
    manager.load_model_if_needed.return_value = backend
    service = T.TranscriptionService(manager)
    return service, backend, str(audio)


@pytest.mark.parametrize("duration", [0.0, -1.0, float("nan"), float("inf")])
def test_unknown_duration_does_not_become_short_job_timeout(duration):
    assert 600 <= T._diarize_timeout(duration) <= 7200


@pytest.mark.asyncio
async def test_failure_cools_down_and_keeps_completed_asr(env, monkeypatch):
    service, backend, path = env
    calls = []

    def fail(audio_path, timeout=None):
        calls.append(audio_path)
        raise RuntimeError("temporary diarization error")

    monkeypatch.setattr(T, "_diarize_samples", fail)
    first = await service.process_transcription(path, file_path_for_cache=path, diarize=True)
    second = await service.process_transcription(path, file_path_for_cache=path, diarize=True)
    assert first["status"] == second["status"] == "success"
    assert first["diarization"]["status"] == second["diarization"]["status"] == "degraded"
    assert first["body"] == second["body"]
    assert backend.calls == 1
    assert len(calls) == 1
    assert T.cache_manager.get_cached_transcript(file_path=path, diarize=False)["body"] == first["body"]


@pytest.mark.asyncio
async def test_after_cooldown_retries_only_diarization(env, monkeypatch):
    service, backend, path = env

    async def fail(audio_path, timeout):
        raise RuntimeError("try later")

    monkeypatch.setattr(T, "_diarize_samples", fail)
    first = await service.process_transcription(path, file_path_for_cache=path, diarize=True)
    assert first["status"] == "success"
    # 明确只清掉冷却结果，模拟该条短 TTL 过期；ASR 的正常 TTL 仍有效。
    cache_key = T.cache_manager._get_cache_key(file_path=path, diarize=True)
    fallback_path = T.cache_manager.transcript_dir / f"{cache_key}.json"
    assert fallback_path.exists()
    fallback_path.unlink()
    calls = []

    async def recovered(audio_path, timeout):
        calls.append(audio_path)
        return [SpeakerTurn(0.0, 3.1, 0), SpeakerTurn(4.0, 8.0, 1)]

    monkeypatch.setattr(T, "_diarize_samples", recovered)
    second = await service.process_transcription(path, file_path_for_cache=path, diarize=True)
    assert second["status"] == "success"
    assert backend.calls == 1, "重试分离不应重复完整 ASR"
    assert len(calls) == 1
    assert second["diarization"]["status"] == "success"
    assert "speakers" in second


@pytest.mark.asyncio
async def test_duplicate_inflight_requests_share_work_not_response_objects(env, monkeypatch):
    service, backend, path = env
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def infer(audio_path, timeout):
        calls.append(audio_path)
        started.set()
        await release.wait()
        return [SpeakerTurn(0.0, 3.1, 0), SpeakerTurn(4.0, 8.0, 1)]

    monkeypatch.setattr(T, "_diarize_samples", infer)
    first = asyncio.create_task(service.process_transcription(path, no_cache=True, diarize=True))
    await asyncio.wait_for(started.wait(), 2)
    second = asyncio.create_task(service.process_transcription(path, no_cache=True, diarize=True))
    await asyncio.sleep(0)
    release.set()
    a, b = await asyncio.gather(first, second)
    assert a["status"] == b["status"] == "success"
    assert backend.calls == len(calls) == 1
    a["body"][0]["content"] = "modified by endpoint"
    assert b["body"][0]["content"] != "modified by endpoint"


@pytest.mark.asyncio
async def test_cancelled_duplicate_does_not_cancel_owner(env, monkeypatch):
    service, backend, path = env
    started, release = asyncio.Event(), asyncio.Event()

    async def infer(audio_path, timeout):
        started.set()
        await release.wait()
        return [SpeakerTurn(0.0, 8.0, 0)]

    monkeypatch.setattr(T, "_diarize_samples", infer)
    owner = asyncio.create_task(service.process_transcription(path, no_cache=True, diarize=True))
    await asyncio.wait_for(started.wait(), 2)
    duplicate = asyncio.create_task(service.process_transcription(path, no_cache=True, diarize=True))
    await asyncio.sleep(0)
    duplicate.cancel()
    with pytest.raises(asyncio.CancelledError):
        await duplicate
    release.set()
    assert (await owner)["status"] == "success"
    assert backend.calls == 1


@pytest.mark.asyncio
async def test_asr_failure_cancels_and_awaits_diarization_cleanup(env, monkeypatch):
    service, backend, path = env
    started, cleaned = asyncio.Event(), asyncio.Event()

    async def infer(audio_path, timeout):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    def fail(*args):
        raise RuntimeError("ASR failed")

    backend.transcribe = fail
    monkeypatch.setattr(T, "_diarize_samples", infer)
    response = await service.process_transcription(path, no_cache=True, diarize=True)
    assert response["status"] == "error"
    assert cleaned.is_set(), "不能只取消等待却遗留分离工作"


@pytest.mark.asyncio
async def test_cancelled_owner_does_not_cancel_remaining_follower(env, monkeypatch):
    service, backend, path = env
    started, release = asyncio.Event(), asyncio.Event()

    async def infer(audio_path, timeout):
        started.set()
        await release.wait()
        return [SpeakerTurn(0.0, 8.0, 0)]

    monkeypatch.setattr(T, "_diarize_samples", infer)
    owner = asyncio.create_task(service.process_transcription(path, no_cache=True, diarize=True))
    await asyncio.wait_for(started.wait(), 2)
    follower = asyncio.create_task(service.process_transcription(path, no_cache=True, diarize=True))
    await asyncio.sleep(0)
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    release.set()
    assert (await follower)["status"] == "success"
    assert backend.calls == 1


@pytest.mark.asyncio
async def test_shared_job_keeps_upload_alive_after_owner_cleanup(env, tmp_path, monkeypatch):
    service, backend, _ = env
    monkeypatch.chdir(tmp_path)
    upload_dir = tmp_path / "tmp"
    upload_dir.mkdir()
    upload = upload_dir / "upload.wav"
    upload.write_bytes(b"fake-audio")
    started, release = asyncio.Event(), asyncio.Event()

    async def infer(audio_path, timeout):
        started.set()
        await release.wait()
        assert Path(audio_path).read_bytes() == b"fake-audio"
        return [SpeakerTurn(0.0, 8.0, 0)]

    monkeypatch.setattr(T, "_diarize_samples", infer)
    owner = asyncio.create_task(service.process_transcription(str(upload), no_cache=True, diarize=True))
    await asyncio.wait_for(started.wait(), 2)
    follower = asyncio.create_task(service.process_transcription(str(upload), no_cache=True, diarize=True))
    await asyncio.sleep(0)
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    upload.unlink()  # 路由 finally 删除 owner 的上传文件
    release.set()
    assert (await follower)["status"] == "success"
    assert backend.calls == 1
    assert list(upload_dir.iterdir()) == []  # 共享任务自身持有的音频也已清理


@pytest.mark.asyncio
async def test_last_waiter_cancellation_stops_shared_job(env, monkeypatch):
    service, backend, path = env
    started, cleaned = asyncio.Event(), asyncio.Event()

    async def infer(audio_path, timeout):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    monkeypatch.setattr(T, "_diarize_samples", infer)
    task = asyncio.create_task(service.process_transcription(path, no_cache=True, diarize=True))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()
    assert not service._inflight


@pytest.mark.asyncio
async def test_busy_failure_uses_short_cooldown_not_one_hour(env, monkeypatch):
    import time
    from diarization.worker import DiarizationBusyError
    service, backend, path = env

    async def busy(audio_path, timeout):
        raise DiarizationBusyError("busy")

    monkeypatch.setattr(T, "_diarize_samples", busy)
    response = await service.process_transcription(path, file_path_for_cache=path, diarize=True)
    assert response["status"] == "success"
    assert response["diarization"]["status"] == "degraded"
    assert 0 < response["diarization"]["retry_after"] - time.time() < 90
    assert T.cache_manager.get_cached_transcript(file_path=path, diarize=False)


@pytest.mark.asyncio
async def test_new_caller_does_not_join_a_cancelled_job_during_cleanup(env, monkeypatch):
    service, backend, path = env
    started, cleaning, finish_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def infer(audio_path, timeout):
        calls.append(audio_path)
        if len(calls) == 1:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await finish_cleanup.wait()
        return [SpeakerTurn(0.0, 8.0, 0)]

    monkeypatch.setattr(T, "_diarize_samples", infer)
    old = asyncio.create_task(service.process_transcription(path, no_cache=True, diarize=True))
    await asyncio.wait_for(started.wait(), 2)
    old.cancel()
    await asyncio.wait_for(cleaning.wait(), 2)
    new = asyncio.create_task(service.process_transcription(path, no_cache=True, diarize=True))
    await asyncio.sleep(0)
    finish_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await old
    assert (await new)["status"] == "success"
    assert len(calls) == 2



