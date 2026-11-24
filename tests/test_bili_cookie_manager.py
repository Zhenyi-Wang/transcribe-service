"""bili_cookie_manager 单元测试（全部离线：subprocess / requests 均打桩）。"""
import json
import subprocess

import pytest
import requests

import bili_cookie_manager as m


@pytest.fixture
def cookie_env(tmp_path, monkeypatch):
    """隔离的 Cookie 文件 / 防抖标记 / 刷新脚本路径。"""
    cookie_path = tmp_path / "bili_cookie.local"
    marker_path = tmp_path / "bili_refresh.last"
    script_path = tmp_path / "refresh.sh"
    script_path.write_text("#!/usr/bin/env bash\nexit 0\n")
    log_path = tmp_path / "bili_refresh.log"
    monkeypatch.setattr(m, "COOKIE_PATH", cookie_path)
    monkeypatch.setattr(m, "REFRESH_MARKER_PATH", marker_path)
    monkeypatch.setattr(m, "REFRESH_SCRIPT", script_path)
    monkeypatch.setattr(m, "REFRESH_LOG", log_path)
    return {"cookie": cookie_path, "marker": marker_path, "script": script_path}


# ---------- load / save ----------

def test_load_missing_returns_empty(cookie_env):
    assert m.load_cookie() == ""


def test_save_then_load_roundtrip_and_perm(cookie_env):
    m.save_cookie("SESSDATA=abc; buvid3=x")
    assert m.load_cookie() == "SESSDATA=abc; buvid3=x"
    import os
    assert oct(os.stat(cookie_env["cookie"]).st_mode & 0o777) == "0o600"


# ---------- sync_from_production ----------

def _fake_run(stdout="", returncode=0, raise_exc=None):
    def run(cmd, capture_output=True, text=True, timeout=None):
        if raise_exc is not None:
            raise raise_exc
        class P:
            pass
        p = P()
        p.stdout = stdout
        p.returncode = returncode
        return p
    return run


def test_sync_success_writes_file(cookie_env, monkeypatch):
    payload = json.dumps({"data": {"bilibili": {"cookie": "SESSDATA=zzz"}}})
    monkeypatch.setattr(m.subprocess, "run", _fake_run(stdout=payload))
    assert m.sync_from_production() is True
    assert m.load_cookie() == "SESSDATA=zzz"


def test_sync_ssh_failure(cookie_env, monkeypatch):
    monkeypatch.setattr(m.subprocess, "run", _fake_run(returncode=255))
    assert m.sync_from_production() is False


def test_sync_timeout(cookie_env, monkeypatch):
    monkeypatch.setattr(m.subprocess, "run",
                        _fake_run(raise_exc=subprocess.TimeoutExpired("ssh", 25)))
    assert m.sync_from_production() is False


def test_sync_bad_json(cookie_env, monkeypatch):
    monkeypatch.setattr(m.subprocess, "run", _fake_run(stdout="not json"))
    assert m.sync_from_production() is False


def test_sync_empty_cookie(cookie_env, monkeypatch):
    payload = json.dumps({"data": {"bilibili": {"cookie": ""}}})
    monkeypatch.setattr(m.subprocess, "run", _fake_run(stdout=payload))
    assert m.sync_from_production() is False


# ---------- probe_login ----------

def _fake_get(json_body=None, raise_exc=None):
    def get(url, timeout=None, headers=None):
        if raise_exc is not None:
            raise raise_exc
        class R:
            pass
        r = R()
        r.json = lambda: json_body
        return r
    return get


def test_probe_empty_cookie_no_http(cookie_env, monkeypatch):
    called = []
    monkeypatch.setattr(m.requests, "get",
                        lambda *a, **k: called.append(1) or (_ for _ in ()).throw(AssertionError()))
    assert m.probe_login("") is False
    assert called == []


def test_probe_logged_in(cookie_env, monkeypatch):
    monkeypatch.setattr(m.requests, "get",
                        _fake_get(json_body={"code": 0, "data": {"isLogin": True}}))
    assert m.probe_login("SESSDATA=ok") is True


def test_probe_not_logged_in(cookie_env, monkeypatch):
    monkeypatch.setattr(m.requests, "get",
                        _fake_get(json_body={"code": -101, "message": "账号未登录"}))
    assert m.probe_login("SESSDATA=dead") is False


def test_probe_network_error(cookie_env, monkeypatch):
    monkeypatch.setattr(m.requests, "get",
                        _fake_get(raise_exc=requests.ConnectionError("boom")))
    assert m.probe_login("SESSDATA=ok") is False


# ---------- trigger_unattended_refresh ----------

def test_trigger_respects_debounce(cookie_env, monkeypatch):
    import time
    cookie_env["marker"].write_text(str(time.time()))   # 刚触发过
    popen_called = []
    monkeypatch.setattr(m.subprocess, "Popen",
                        lambda *a, **k: popen_called.append(1))
    assert m.trigger_unattended_refresh() is False
    assert popen_called == []


def test_trigger_spawns_detached(cookie_env, monkeypatch):
    popen_kwargs = {}
    def fake_popen(*a, **kw):
        popen_kwargs.update(kw)
        class P:
            pass
        return P()
    monkeypatch.setattr(m.subprocess, "Popen", fake_popen)
    assert m.trigger_unattended_refresh() is True
    assert popen_kwargs.get("start_new_session") is True
    assert cookie_env["marker"].exists()


# ---------- ensure_fresh_cookie ----------

def test_ensure_keeps_current_when_valid(cookie_env, monkeypatch):
    m.save_cookie("SESSDATA=good")
    called = []
    monkeypatch.setattr(m, "probe_login", lambda ck: called.append(ck) or True)
    monkeypatch.setattr(m, "sync_from_production",
                        lambda: called.append("sync") or (_ for _ in ()).throw(AssertionError()))
    cookie, note = m.ensure_fresh_cookie("BV1")
    assert cookie == "SESSDATA=good" and note == "cookie 有效"
    assert len(called) == 1   # 只探测了一次，未触发同步


def test_ensure_syncs_when_expired(cookie_env, monkeypatch):
    m.save_cookie("SESSDATA=dead")
    probe_results = {"SESSDATA=dead": False, "SESSDATA=fresh": True}
    monkeypatch.setattr(m, "probe_login",
                        lambda ck: probe_results.get(ck, False))
    monkeypatch.setattr(m, "sync_from_production",
                        lambda: m.save_cookie("SESSDATA=fresh") or True)
    cookie, note = m.ensure_fresh_cookie("BV1")
    assert cookie == "SESSDATA=fresh" and "同步" in note


def test_ensure_triggers_refresh_when_both_dead(cookie_env, monkeypatch):
    m.save_cookie("SESSDATA=dead")
    triggered = []
    monkeypatch.setattr(m, "probe_login", lambda ck: False)
    monkeypatch.setattr(m, "sync_from_production", lambda: True)
    monkeypatch.setattr(m, "trigger_unattended_refresh",
                        lambda: triggered.append(1) or True)
    cookie, note = m.ensure_fresh_cookie("BV1")
    assert triggered == [1]
    assert "均失效" in note
