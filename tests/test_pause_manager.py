"""PauseManager 单元:状态/落盘/覆盖/世代/惰性恢复。"""
import json
import time

from pause_manager import PauseManager


def test_pause_resume_roundtrip(tmp_path):
    f = tmp_path / "p.json"
    pm = PauseManager(state_file=f)
    assert not pm.is_paused()
    pm.pause(0.02)
    assert pm.is_paused() and pm.remaining_seconds() > 0
    assert f.exists()
    # 新实例从文件恢复
    pm2 = PauseManager(state_file=f)
    assert pm2.is_paused()
    assert pm2.resume() is True
    assert not pm2.is_paused() and not f.exists()


def test_expired_and_corrupted_state(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"paused_until": "2000-01-01T00:00:00", "paused_at": "2000-01-01T00:00:00"}), encoding="utf-8")
    assert not PauseManager(state_file=f).is_paused() and not f.exists()
    f.write_text("{broken", encoding="utf-8")
    assert not PauseManager(state_file=f).is_paused() and not f.exists()


def test_pause_overrides_and_lazy_expiry(tmp_path):
    pm = PauseManager(state_file=tmp_path / "p.json")
    pm.pause(1)
    later = pm.pause(3)
    assert pm.resume_at().replace(microsecond=0) == later.replace(microsecond=0)
    pm._paused_until = time.time() - 1
    assert not pm.is_paused()


def test_generation_increases(tmp_path):
    pm = PauseManager(state_file=tmp_path / "p.json")
    g0 = pm.generation
    pm.pause(0.01); g1 = pm.generation
    pm.resume(); g2 = pm.generation
    assert g1 == g0 + 1 and g2 == g1 + 1


def test_notify_scheduled_and_fired(tmp_path):
    fired = []
    pm = PauseManager(state_file=tmp_path / "p.json", notify_fn=lambda: fired.append(1))
    pm.pause(0.0001)         # 0.36s 后到期
    time.sleep(0.8)
    assert fired == [1]      # 到点通知已触发(到期即发,不做 is_paused 检查)
    pm.pause(0.0001)
    pm.resume()              # resume 立即通知并清定时
    time.sleep(0.3)
    assert len(fired) == 2


def test_unload_flag(tmp_path):
    pm = PauseManager(state_file=tmp_path / "p.json")
    assert not pm.should_unload()
    pm.pause(0.01)
    assert pm.should_unload()
    pm.mark_unload_attempted()
    assert not pm.should_unload()
