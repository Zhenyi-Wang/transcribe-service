"""asr 暂停 503 → UpstreamPausedError → 端点 503 paused 的传播链测试。"""
import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException

import server


def _headers():
    return {"Authorization": f"Bearer {server.config.api_token}"} if server.config.api_token else {}


def test_backend_raises_upstream_paused():
    from backends.asr_engine_backend import UpstreamPausedError
    import httpx
    from unittest.mock import patch

    def fake_post(*a, **kw):
        # Response 必须带 request,否则 raise_for_status 在构造 HTTPStatusError 前失败
        req = httpx.Request("POST", "http://asr-engine.test/v1/audio/transcriptions")
        return httpx.Response(503, request=req,
                              json={"paused": True, "resume_at": "2099-01-01T00:00:00+08:00",
                                    "detail": "ASR 引擎暂停中"})

    with patch.object(httpx, "post", side_effect=fake_post):
        from backends.asr_engine_backend import ASREngineClientBackend
        b = ASREngineClientBackend(server.config)
        with pytest.raises(UpstreamPausedError):
            b.transcribe("/etc/hostname")


def test_endpoint_returns_503_paused(monkeypatch, tmp_path):
    client = TestClient(server.app)
    from backends.asr_engine_backend import UpstreamPausedError

    async def fake_process(*a, **kw):
        raise UpstreamPausedError("ASR 引擎暂停中", resume_at="2099-01-01T00:00:00+08:00")

    # 拦截下载,避免测试触网
    monkeypatch.setattr(server.downloader, "download_bilibili_audio",
                        lambda *a, **kw: (True, {"file_path": str(tmp_path / "a.mp3"), "audio_url": "u"}))
    monkeypatch.setattr(server.transcription_service, "process_transcription", fake_process)
    r = client.post("/transcribe_url", json={"bvid": "BV1", "cookie": "c"}, headers=_headers())
    assert r.status_code == 503
    assert r.json()["paused"] is True and r.json()["resume_at"]
