import pytest
from fastapi.testclient import TestClient

import server


def test_models_accept_diarize():
    req = server.BilibiliTranscribeRequest(bvid="BV1", cookie="c", diarize=True)
    assert req.diarize is True
    req2 = server.WebdavTranscribeRequest(path="x", diarize=True)
    assert req2.diarize is True
    # 默认 False
    assert server.BilibiliTranscribeRequest(bvid="BV1", cookie="c").diarize is False
    assert server.WebdavTranscribeRequest(path="x").diarize is False


def test_transcribe_endpoint_accepts_form_field(tmp_path, monkeypatch):
    client = TestClient(server.app)
    wav = tmp_path / "t.wav"
    wav.write_bytes(b"RIFF0000")

    captured = {}

    async def fake_process(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "success", "body": []}

    monkeypatch.setattr(server.transcription_service, "process_transcription", fake_process)
    headers = {}
    if server.config.api_token:
        headers["Authorization"] = f"Bearer {server.config.api_token}"
    r = client.post("/transcribe", files={"file": ("t.wav", wav.read_bytes(), "audio/wav")},
                    data={"diarize": "true"}, headers=headers)
    assert r.status_code == 200
    assert captured.get("diarize") is True


def test_transcribe_endpoint_diarize_default_false(tmp_path, monkeypatch):
    client = TestClient(server.app)
    wav = tmp_path / "t.wav"
    wav.write_bytes(b"RIFF0000")

    captured = {}

    async def fake_process(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "success", "body": []}

    monkeypatch.setattr(server.transcription_service, "process_transcription", fake_process)
    headers = {}
    if server.config.api_token:
        headers["Authorization"] = f"Bearer {server.config.api_token}"
    r = client.post("/transcribe", files={"file": ("t.wav", wav.read_bytes(), "audio/wav")},
                    headers=headers)
    assert r.status_code == 200
    assert captured.get("diarize") is False


def test_transcribe_url_passthrough(tmp_path, monkeypatch):
    """/transcribe_url 透传 diarize 到 process_transcription"""
    client = TestClient(server.app)
    captured = {}

    async def fake_process(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "success", "body": []}

    monkeypatch.setattr(
        server.downloader, "download_bilibili_audio",
        lambda *a, **k: (True, {"file_path": "/tmp/fake.m4s", "audio_url": "http://x", "audio_id": "id1"}))
    monkeypatch.setattr(server.transcription_service, "process_transcription", fake_process)
    headers = {}
    if server.config.api_token:
        headers["Authorization"] = f"Bearer {server.config.api_token}"
    r = client.post("/transcribe_url", json={"bvid": "BV1xx", "cookie": "c", "diarize": True},
                    headers=headers)
    assert r.status_code == 200
    assert captured.get("diarize") is True


def test_transcribe_file_passthrough(tmp_path, monkeypatch):
    """/transcribe_file 透传 diarize 到 process_transcription"""
    client = TestClient(server.app)
    f = tmp_path / "x.mp3"
    f.write_bytes(b"x")

    captured = {}

    async def fake_process(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "success", "body": []}

    monkeypatch.setattr(server.transcription_service, "process_transcription", fake_process)
    # webdav.base_path 指向 tmp_path，其余配置读取委托原方法
    original_get = server.config.get
    monkeypatch.setattr(
        server.config, "get",
        lambda key, default=None: str(tmp_path) if key == "webdav.base_path" else original_get(key, default))
    headers = {}
    if server.config.api_token:
        headers["Authorization"] = f"Bearer {server.config.api_token}"
    r = client.post("/transcribe_file", json={"path": "x.mp3", "diarize": True}, headers=headers)
    assert r.status_code == 200
    assert captured.get("diarize") is True
