"""暂停端点:503 拦截(先于处理)/惰性到期/resume/校验/鉴权优先/status。"""
import time

import pytest
from fastapi.testclient import TestClient

import server
from pause_manager import PauseManager


@pytest.fixture
def paused_env(tmp_path, monkeypatch):
    # 隔离:通知不发、释放链不跑、asr 管理转发打桩——测试绝不能碰常驻服务/生产 asr
    pm = PauseManager(state_file=tmp_path / "pause.json", notify_fn=lambda: None)
    monkeypatch.setattr(server, "pause_manager", pm)
    monkeypatch.setattr(server, "_pause_asr_engine", lambda hours: "mocked")
    monkeypatch.setattr(server, "_resume_asr_engine", lambda: "mocked")
    monkeypatch.setattr(server, "_release_gpu_for_pause", lambda gen: {"mocked": True})
    monkeypatch.setattr(server, "_query_asr_paused", lambda: False)

    async def fake_process(*a, **kw):
        return {"status": "success", "body": []}

    monkeypatch.setattr(server.transcription_service, "process_transcription", fake_process)
    return pm


def _h():
    return {"Authorization": f"Bearer {server.config.api_token}"} if server.config.api_token else {}


def test_pause_blocks_all_endpoints(paused_env):
    c = TestClient(server.app)
    assert c.post("/pause", json={"hours": 1}, headers=_h()).status_code == 200
    for path, kw in [("/transcribe", {"files": {"file": ("t.wav", b"RIFF", "audio/wav")}}),
                     ("/transcribe_url", {"json": {"bvid": "BV1", "cookie": "c"}}),
                     ("/transcribe_file", {"json": {"path": "inbox/x.mp3"}})]:
        r = c.post(path, headers=_h(), **kw)
        assert r.status_code == 503
        assert r.json()["paused"] is True and r.json()["resume_at"]
        assert int(r.headers["Retry-After"]) >= 1


def test_block_precedes_processing(paused_env, monkeypatch):
    c = TestClient(server.app)
    c.post("/pause", json={"hours": 1}, headers=_h())
    called = []

    async def spy(*a, **kw):
        called.append(1)
        return {"status": "success"}

    monkeypatch.setattr(server.transcription_service, "process_transcription", spy)
    assert c.post("/transcribe_file", json={"path": "x"}, headers=_h()).status_code == 503
    assert called == []


def test_lazy_expiry_and_resume(paused_env):
    c = TestClient(server.app)
    c.post("/pause", json={"hours": 1}, headers=_h())
    paused_env._paused_until = time.time() - 1
    assert c.post("/transcribe_file", json={"path": "x"}, headers=_h()).status_code == 200
    c.post("/pause", json={"hours": 1}, headers=_h())
    r = c.post("/resume", headers=_h()).json()
    assert r["paused"] is False and r["was_paused"] is True  # 逐键断言(响应含 asr_engine 字段)
    s = c.get("/status", headers=_h()).json()
    assert s["paused"] is False and "asr_engine_paused" in s


def test_validation_and_auth_order(paused_env):
    c = TestClient(server.app)
    for bad in (0, -1, 49, "abc"):
        assert c.post("/pause", json={"hours": bad}, headers=_h()).status_code == 422
    c.post("/pause", json={"hours": 1}, headers=_h())
    assert c.post("/transcribe_file", json={"path": "x"}).status_code == 401  # 鉴权优先
