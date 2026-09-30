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
        with pytest.raises(UpstreamPausedError) as excinfo:
            b.transcribe("/etc/hostname")
    # 异常属性透传:resume_at 取自 fake body,paused_at 缺省为 None
    assert excinfo.value.resume_at == "2099-01-01T00:00:00+08:00"
    assert excinfo.value.paused_at is None


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


def test_transcribe_file_returns_503_paused(tmp_path, monkeypatch):
    """/transcribe_file 内层重抛 → 外层包装转 503 paused(而非 except Exception 转 error dict)"""
    client = TestClient(server.app)
    from backends.asr_engine_backend import UpstreamPausedError

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "x.mp3").write_bytes(b"x")

    async def fake_process(*a, **kw):
        raise UpstreamPausedError("ASR 引擎暂停中", resume_at="2099-01-01T00:00:00+08:00",
                                  paused_at="2099-01-01T00:00:00+08:00")

    monkeypatch.setattr(server.transcription_service, "process_transcription", fake_process)
    # webdav.base_path 指向 tmp_path(文件真实存在以通过存在性检查),其余配置读取委托原方法
    original_get = server.config.get
    monkeypatch.setattr(
        server.config, "get",
        lambda key, default=None: str(tmp_path) if key == "webdav.base_path" else original_get(key, default))
    r = client.post("/transcribe_file", json={"path": "inbox/x.mp3"}, headers=_headers())
    assert r.status_code == 503
    body = r.json()
    assert body["paused"] is True and body["resume_at"] and body["paused_at"]
