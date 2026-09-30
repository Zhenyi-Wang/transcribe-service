"""暂停状态管理:落盘持久化 + 惰性恢复 + 恢复通知调度 + 世代令牌。

时间权威唯一在本模块(pause_state.json);恢复通知到点推送(notify_fn),
resume 立即推送。世代令牌供后台释放链在每个耗时步骤前检查防交错。
"""
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from logger_config import setup_logger

logger = setup_logger('pause')

MAX_PAUSE_HOURS = 48


class PauseManager:
    def __init__(self, state_file="pause_state.json", notify_url="", notify_token="", notify_fn=None):
        self._state_file = Path(state_file)
        self._paused_until = None  # epoch 秒
        self._paused_at = None
        self._unload_attempted = False
        self._generation = 0
        self._notify_url = notify_url
        self._notify_token = notify_token
        self._notify_fn = notify_fn or (self._default_notify if notify_url else None)
        self._notify_timer = None
        self._lock = threading.Lock()
        self._load()

    # ---------- 状态文件 ----------
    def _load(self):
        if not self._state_file.exists():
            return
        try:
            data = json.loads(self._state_file.read_text(encoding="utf-8"))
            until = datetime.fromisoformat(data["paused_until"]).timestamp()
            paused_at = datetime.fromisoformat(data.get("paused_at")).timestamp()
        except Exception as e:
            logger.warning(f"暂停状态文件损坏,视为未暂停: {self._state_file} ({e})")
            self._try_unlink(); return
        if until <= time.time():
            logger.info("已过期的暂停状态文件,删除并视为未暂停")
            self._try_unlink(); return
        self._paused_until = until
        self._paused_at = paused_at
        logger.info(f"恢复暂停状态: 至 {self.resume_at():%Y-%m-%d %H:%M:%S}")
        self._schedule_notify()  # 重启后按剩余时长重设通知

    def _write_state(self):  # 调用方持锁
        tmp = self._state_file.with_suffix(".json.tmp")
        payload = {"paused_until": datetime.fromtimestamp(self._paused_until).isoformat(),
                   "paused_at": datetime.fromtimestamp(self._paused_at or time.time()).isoformat()}
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self._state_file)
        except Exception as e:
            logger.warning(f"暂停状态写盘失败(仅影响重启后恢复): {e}")

    def _try_unlink(self):
        try:
            self._state_file.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"删除暂停状态文件失败: {e}")

    # ---------- 通知 ----------
    def _default_notify(self):
        """推送恢复信号到 noteflow 唤醒端点;重试 3 次。由 server.py 注入自定义时本函数不生效。"""
        import httpx
        headers = {"Content-Type": "application/json"}
        if self._notify_token:
            headers["Authorization"] = f"Bearer {self._notify_token}"
        for attempt in range(3):
            try:
                resp = httpx.post(self._notify_url, json={"type": "transcribe_resumed"},
                                  headers=headers, timeout=5.0)
                if resp.status_code == 200:
                    logger.info("恢复通知已送达")
                    return
                logger.warning(f"恢复通知响应异常: {resp.status_code}")
            except Exception as e:
                logger.warning(f"恢复通知失败({attempt + 1}/3): {e}")
            time.sleep(5 * (attempt + 1))
        logger.warning("恢复通知三次失败(noteflow 兜底定时/手动 reset 兜底)")

    def _schedule_notify(self):
        """按剩余时长设一次性通知定时(覆盖旧定时)"""
        with self._lock:
            if self._notify_timer is not None:
                self._notify_timer.cancel()
                self._notify_timer = None
            if not self.is_paused() or self._notify_fn is None:
                return
            delay = max(0.1, self.remaining_seconds())
            self._notify_timer = threading.Timer(delay, self._fire_notify)
            self._notify_timer.daemon = True
            self._notify_timer.start()

    def _fire_notify(self):
        # 到点即发,不检查 is_paused():定时到点时暂停必然已到期(is_paused=False),
        # 检查反而会吞掉"到期自动恢复"这条主通知路径;被提前 resume 的场景由 clearTimeout 保证
        try:
            self._notify_fn()
        except Exception as e:
            logger.warning(f"通知回调异常: {e}")

    # ---------- 对外 ----------
    def pause(self, hours: float) -> datetime:
        now = time.time()
        with self._lock:
            self._paused_until = now + hours * 3600
            self._paused_at = now
            self._unload_attempted = False
            self._generation += 1
            self._write_state()
        self._schedule_notify()
        return datetime.fromtimestamp(self._paused_until)

    def resume(self) -> bool:
        with self._lock:
            was_paused = self.is_paused()
            self._paused_until = None
            self._paused_at = None
            self._unload_attempted = False
            self._generation += 1
            self._try_unlink()
            if self._notify_timer is not None:
                self._notify_timer.cancel()
                self._notify_timer = None
        if was_paused and self._notify_fn is not None:
            threading.Thread(target=self._notify_fn, daemon=True).start()  # 立即通知,不阻塞响应
        if was_paused:
            logger.info("服务已恢复(模型懒加载)")
        return was_paused

    def is_paused(self) -> bool:
        until = self._paused_until
        return until is not None and time.time() < until

    def resume_at(self):
        until = self._paused_until
        if until is None or time.time() >= until:
            return None
        return datetime.fromtimestamp(until)

    def paused_at(self):
        until = self._paused_until
        paused_at = self._paused_at
        if until is None or paused_at is None or time.time() >= until:
            return None
        return datetime.fromtimestamp(paused_at)

    def remaining_seconds(self) -> float:
        until = self._paused_until
        if until is None:
            return 0.0
        remaining = until - time.time()
        return remaining if remaining > 0 else 0.0

    def status(self) -> dict:
        resume_at = self.resume_at()
        return {"paused": self.is_paused(),
                "resume_at": resume_at.astimezone().isoformat() if resume_at else None,
                "remaining_seconds": round(max(0.0, self.remaining_seconds()), 1)}

    def should_unload(self) -> bool:
        return self.is_paused() and not self._unload_attempted

    def mark_unload_attempted(self):
        self._unload_attempted = True

    @property
    def generation(self) -> int:
        return self._generation
