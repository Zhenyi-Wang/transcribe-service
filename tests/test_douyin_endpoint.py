"""/transcribe_douyin 端点单测：成功/下载失败fatal/暂停503/aweme_id校验
构造模式对齐 tests/test_upstream_paused.py：TestClient(server.app)（不用 with 块，
与现有测试一致）、monkeypatch 拦截下载与转录、Bearer 头按 config.api_token。
"""
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

import server


def _client():
    return TestClient(server.app)


def _headers():
    return {"Authorization": f"Bearer {server.config.api_token}"} if server.config.api_token else {}


def test_invalid_aweme_id_rejected():
    client = _client()  # 不用 with 块：避免触发 lifespan/startup 预加载模型（对齐 test_upstream_paused.py）
    r = client.post("/transcribe_douyin", json={"aweme_id": "abc"}, headers=_headers())
    assert r.status_code == 422  # pydantic pattern 校验


def test_success(monkeypatch, tmp_path):
    client = _client()
    f = tmp_path / "x.m4a"
    f.write_bytes(b"fake")
    monkeypatch.setattr(server.downloader_douyin, "download_douyin_audio",
                        lambda *a, **kw: (True, {"file_path": str(f), "audio_url": "https://cdn/x",
                                                  "audio_id": "dy"}))
    mock_ts = AsyncMock(return_value={
        "status": "success", "type": "qwen3", "version": "1",
        "body": [{"from": 0.0, "to": 1.0, "content": "你好"}],
        "rtf": 0.1, "timing": {"total": 1.0}})
    monkeypatch.setattr(server.transcription_service, "process_transcription", mock_ts)
    r = client.post("/transcribe_douyin", json={"aweme_id": "7376234567890123456"}, headers=_headers())
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert "download" in data["timing"]
    # process_transcription 第 4 个位置参数（bvid 位）= aweme_id 作转录缓存键
    assert mock_ts.call_args.args[3] == "7376234567890123456"


def test_download_failure_returns_error_status(monkeypatch):
    client = _client()
    monkeypatch.setattr(server.downloader_douyin, "download_douyin_audio",
                        lambda *a, **kw: (False, "图集暂不支持（无语音主体，仅配乐）"))
    r = client.post("/transcribe_douyin", json={"aweme_id": "7376234567890123456"}, headers=_headers())
    assert r.status_code == 200
    assert r.json()["status"] == "error"
    assert "图集" in r.json()["message"]


def test_paused_503(monkeypatch):
    """中间件层拦截（PAUSED_REJECT_PATHS），不执行下载"""
    client = _client()
    monkeypatch.setattr(server.pause_manager, "is_paused", lambda: True)
    dl = AsyncMock()
    monkeypatch.setattr(server.downloader_douyin, "download_douyin_audio", dl)
    r = client.post("/transcribe_douyin", json={"aweme_id": "7376234567890123456"}, headers=_headers())
    assert r.status_code == 503
    dl.assert_not_called()
