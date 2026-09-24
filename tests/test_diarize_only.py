"""diarize_only：仅说话人分离路径（不跑 ASR）的单元测试。"""
import pytest
from unittest.mock import patch, MagicMock

import asyncio

import transcribe as T
from diarization.manager import SpeakerTurn


def _enable_diarization(monkeypatch):
    monkeypatch.setattr(type(T.config), "diarization_enabled",
                        property(lambda self: True))


@pytest.mark.asyncio
async def test_success_returns_turns_and_speakers(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 100.0)
    _enable_diarization(monkeypatch)
    turns = [SpeakerTurn(0.0, 40.0, 0), SpeakerTurn(40.0, 100.0, 1)]
    with patch.object(T, "_diarize_samples", lambda path: turns):
        resp = await T.diarize_only(str(wav), "BV1xx", download_time=1.5)
    assert resp["status"] == "success"
    assert resp["video_id"] == "BV1xx"
    assert resp["turns"] == [
        {"speaker": 0, "start": 0.0, "end": 40.0},
        {"speaker": 1, "start": 40.0, "end": 100.0},
    ]
    assert resp["speakers"] == [
        {"id": 0, "duration": 40.0, "turns": 1},
        {"id": 1, "duration": 60.0, "turns": 1},
    ]
    assert resp["timing"]["download"] == 1.5
    assert resp["timing"]["diarization"] >= 0.0
    assert resp["timing"]["total"] >= resp["timing"]["diarization"]


@pytest.mark.asyncio
async def test_single_speaker_returned_as_is(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 10.0)
    _enable_diarization(monkeypatch)
    with patch.object(T, "_diarize_samples", lambda path: [SpeakerTurn(0.0, 10.0, 0)]):
        resp = await T.diarize_only(str(wav))
    assert resp["status"] == "success"
    assert resp["speakers"] == [{"id": 0, "duration": 10.0, "turns": 1}]


@pytest.mark.asyncio
async def test_disabled_returns_error(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(type(T.config), "diarization_enabled",
                        property(lambda self: False))
    with patch.object(T, "_diarize_samples") as should_not_call:
        resp = await T.diarize_only(str(wav))
    assert resp["status"] == "error"
    assert "disabled" in resp["message"]
    should_not_call.assert_not_called()


@pytest.mark.asyncio
async def test_diarization_failure_returns_error(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 10.0)
    _enable_diarization(monkeypatch)

    def boom(path):
        raise RuntimeError("model gone")
    with patch.object(T, "_diarize_samples", boom):
        resp = await T.diarize_only(str(wav))
    assert resp["status"] == "error"
    assert "model gone" in resp["message"]
    assert resp["timing"]["total"] >= 0.0


@pytest.mark.asyncio
async def test_timeout_returns_error(tmp_path, monkeypatch):
    """分离超时 → status=error（区别于转录路径的静默降级：此处调用方需要显式失败信号）"""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 10.0)
    monkeypatch.setattr(T, "_diarize_timeout", lambda d: 0.05)
    _enable_diarization(monkeypatch)

    def slow(path):
        import time as _t
        _t.sleep(1.0)
        return [SpeakerTurn(0.0, 10.0, 0)]
    with patch.object(T, "_diarize_samples", slow):
        resp = await T.diarize_only(str(wav))
    assert resp["status"] == "error"
    assert "timeout" in resp["message"]


@pytest.mark.asyncio
async def test_timeout_cancels_underlying_thread_job(tmp_path, monkeypatch):
    """wait_for 超时后 asyncio.to_thread 无法中断线程，但等待本身必须解除（响应不被卡死）"""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 10.0)
    monkeypatch.setattr(T, "_diarize_timeout", lambda d: 0.05)
    _enable_diarization(monkeypatch)

    def hang(path):
        import time as _t
        _t.sleep(5.0)
        return []
    with patch.object(T, "_diarize_samples", hang):
        resp = await asyncio.wait_for(T.diarize_only(str(wav)), timeout=3.0)
    assert resp["status"] == "error"


# ==================== 分离+字幕标注拼接模式（diarize_merge_subtitle） ====================

SUBTITLE_BODY = [
    {"from": 0.0, "to": 3.0, "sid": 1, "location": 2, "content": "甲说", "music": 0},
    {"from": 3.5, "to": 6.0, "sid": 2, "location": 2, "content": "还是甲说", "music": 0},
    {"from": 8.0, "to": 12.0, "sid": 3, "location": 2, "content": "乙说", "music": 0},
]


@pytest.mark.asyncio
async def test_merge_success_annotates_body(tmp_path, monkeypatch):
    """拼接模式：body 按时间重叠标注 speaker，返回标注后 body + speakers 摘要"""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 20.0)
    _enable_diarization(monkeypatch)
    turns = [SpeakerTurn(0.0, 7.0, 0), SpeakerTurn(7.0, 20.0, 1)]
    with patch.object(T, "_diarize_samples", lambda path: turns):
        resp = await T.diarize_merge_subtitle(SUBTITLE_BODY, str(wav), "BV1xx", download_time=1.0)
    assert resp["status"] == "success"
    assert [seg["speaker"] for seg in resp["body"]] == [0, 0, 1]
    # 原有字段保留
    assert resp["body"][0]["content"] == "甲说"
    assert resp["speakers"] == [
        {"id": 0, "duration": 5.5, "segments": 2},
        {"id": 1, "duration": 4.0, "segments": 1},
    ]
    assert resp["annotated"] is True and resp["reason"] is None


@pytest.mark.asyncio
async def test_merge_cross_boundary_segment_gets_minus_one(tmp_path, monkeypatch):
    """跨界段（重叠 < 段时长 50%）标 -1——与 funasr 退化路径同一语义"""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 20.0)
    _enable_diarization(monkeypatch)
    turns = [SpeakerTurn(0.0, 3.0, 0), SpeakerTurn(7.0, 20.0, 1)]
    body = [{"from": 2.0, "to": 10.0, "content": "横跨两人的长段"}]  # 段长 8s，重叠各 1s/3s，最大 3s < 4s
    with patch.object(T, "_diarize_samples", lambda path: turns):
        resp = await T.diarize_merge_subtitle(body, str(wav))
    assert resp["status"] == "success"
    assert resp["body"][0]["speaker"] == -1


@pytest.mark.asyncio
async def test_merge_single_speaker_success_unannotated(tmp_path, monkeypatch):
    """单人 = 成功识别但无需标注：success + annotated=false + body 原样返回"""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 20.0)
    _enable_diarization(monkeypatch)
    with patch.object(T, "_diarize_samples", lambda path: [SpeakerTurn(0.0, 20.0, 0)]):
        resp = await T.diarize_merge_subtitle(SUBTITLE_BODY, str(wav))
    assert resp["status"] == "success"
    assert resp["annotated"] is False
    assert resp["reason"] == "single_speaker"
    assert resp["body"] == SUBTITLE_BODY  # 原样，无 speaker 键
    assert resp["speakers"] == [{"id": 0, "duration": 20.0, "turns": 1}]


@pytest.mark.asyncio
async def test_merge_disabled_returns_error(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(type(T.config), "diarization_enabled",
                        property(lambda self: False))
    with patch.object(T, "_diarize_samples") as should_not_call:
        resp = await T.diarize_merge_subtitle(SUBTITLE_BODY, str(wav))
    assert resp["status"] == "error"
    assert "disabled" in resp["message"]
    should_not_call.assert_not_called()


def test_endpoint_merge_mode_routes_to_merge(tmp_path, monkeypatch):
    """diarize_only=true 且带 body → 走拼接分支并透传 body/page"""
    from fastapi.testclient import TestClient
    import server

    client = TestClient(server.app)
    captured = {}

    async def fake_merge(body, path, video_id, download_time):
        captured.update({"body": body, "video_id": video_id, "download_time": download_time})
        return {"status": "success", "video_id": video_id, "body": body, "speakers": []}

    async def fake_turns(path, video_id, download_time):
        raise AssertionError("turns mode should not be called in merge mode")

    monkeypatch.setattr(server, "diarize_merge_subtitle", fake_merge)
    monkeypatch.setattr(server, "diarize_only", fake_turns)
    monkeypatch.setattr(
        server.downloader, "download_bilibili_audio",
        lambda *a, **k: (True, {"file_path": "/tmp/fake.m4s", "audio_url": "http://x", "audio_id": "id1"}))
    headers = {}
    if server.config.api_token:
        headers["Authorization"] = f"Bearer {server.config.api_token}"
    r = client.post("/transcribe_url",
                    json={"bvid": "BV1xx", "cookie": "c", "diarize_only": True,
                          "body": [{"from": 0, "to": 1, "content": "hi"}]},
                    headers=headers)
    assert r.status_code == 200
    assert r.json()["body"][0]["content"] == "hi"
    assert captured["video_id"] == "BV1xx"
    assert captured["body"] == [{"from": 0, "to": 1, "content": "hi"}]


def test_request_model_body_default_none():
    import server
    req = server.BilibiliTranscribeRequest(bvid="BV1", cookie="c")
    assert req.body is None
    req2 = server.BilibiliTranscribeRequest(bvid="BV1", cookie="c", body=[{"from": 0}])
    assert req2.body == [{"from": 0}]


# ==================== 端点分流 ====================

def test_endpoint_diarize_only_short_circuits(tmp_path, monkeypatch):
    """/transcribe_url diarize_only=true → 调 diarize_only 且不进转录流程"""
    from fastapi.testclient import TestClient
    import server

    client = TestClient(server.app)
    captured = {}

    async def fake_diarize_only(path, video_id, download_time):
        captured.update({"path": path, "video_id": video_id, "download_time": download_time})
        return {"status": "success", "video_id": video_id, "turns": [], "speakers": []}

    monkeypatch.setattr(server, "diarize_only", fake_diarize_only)
    monkeypatch.setattr(
        server.downloader, "download_bilibili_audio",
        lambda *a, **k: (True, {"file_path": "/tmp/fake.m4s", "audio_url": "http://x", "audio_id": "id1"}))
    with patch.object(server.transcription_service, "process_transcription") as should_not_transcribe:
        headers = {}
        if server.config.api_token:
            headers["Authorization"] = f"Bearer {server.config.api_token}"
        r = client.post("/transcribe_url",
                        json={"bvid": "BV1xx", "cookie": "c", "diarize_only": True, "page": 2},
                        headers=headers)
    assert r.status_code == 200
    assert r.json()["status"] == "success"
    assert captured["video_id"] == "BV1xx"
    assert captured["download_time"] >= 0.0
    should_not_transcribe.assert_not_called()


def test_endpoint_diarize_only_default_false(tmp_path, monkeypatch):
    """不传 diarize_only → 走正常转录流程"""
    from fastapi.testclient import TestClient
    import server

    client = TestClient(server.app)
    monkeypatch.setattr(
        server.downloader, "download_bilibili_audio",
        lambda *a, **k: (True, {"file_path": "/tmp/fake.m4s", "audio_url": "http://x", "audio_id": "id1"}))

    async def fake_process(*args, **kwargs):
        return {"status": "success", "body": []}
    monkeypatch.setattr(server.transcription_service, "process_transcription", fake_process)
    headers = {}
    if server.config.api_token:
        headers["Authorization"] = f"Bearer {server.config.api_token}"
    r = client.post("/transcribe_url", json={"bvid": "BV1xx", "cookie": "c"}, headers=headers)
    assert r.status_code == 200
    assert r.json()["status"] == "success"


def test_request_model_defaults():
    import server
    req = server.BilibiliTranscribeRequest(bvid="BV1", cookie="c")
    assert req.diarize_only is False
