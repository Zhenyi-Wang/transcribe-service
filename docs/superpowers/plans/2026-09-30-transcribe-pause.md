# transcribe 暂停功能实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** transcribe-service 暂停 N 小时释放 GPU(桌面 bat 触发),noteflow 任务挂起不烧重试、恢复后自动/被唤醒重跑。

**Architecture:** 时间权威唯一在 transcribe-service(pause_state.json 落盘 + 到点/手动推送通知);asr-engine 承载暂停语义(拒新+卸载);noteflow 以 `deferred` 状态挂起(零 DDL),被动接收唤醒信号 + 挂起时设兜底 setTimeout。错误类型经 5 处包装点透传。

**Tech Stack:** FastAPI/uvicorn(transcribe、asr-engine,conda funasr 环境,pytest);Nuxt3/Prisma/MySQL/vitest(mryk24,`pnpm vitest run`);Windows bat + curl.exe。

**Spec:** `docs/superpowers/specs/2026-09-30-transcribe-pause-design.md`(本计划与其配套,执行者两个都读)

## Global Constraints

- Python 用 `~/miniconda3/envs/funasr/bin/python -m pytest`(工作目录:对应项目根)
- mryk24 测试:`pnpm vitest run tests/unit/noteflow/`;类型:`pnpm typecheck`;构建:`pnpm build`(schema 改动后必须先 `pnpm prisma:generate`——本轮零 DDL 但 client 需与 schema 一致)
- 提交规范:Conventional Commits、按功能点分次、消息中**不得出现 Claude/Co-Authored-By**;提交与 push 已获用户授权(出门模式),完成即执行
- 生产 DB **零变更**(deferred 只是 String 列新值)
- 中文注释/文案;resume_at 一律带时区 ISO(`astimezone().isoformat()`)
- asr-engine 新管理端点无 token(回环惯例)
- noteflow 唤醒端点 token:runtimeConfig `noteflowInternalToken`(env `NUXT_NOTEFLOW_INTERNAL_TOKEN`),空值放行(本地开发友好),transcribe 侧 `pause.notify_token` 配同值
- 运行环境:home 机 tmux(`transcribe`:31080,`asr`:31090);mryk24 本地容器(挂载 `.output`,重启生效);生产 mryk VPS(sync.sh)

---

### Task 1: asr-engine 暂停态

**Files:**
- Modify: `/home/zhenyi/ownprojects/asr-engine/asr_engine/model_manager.py`
- Modify: `/home/zhenyi/ownprojects/asr-engine/asr_engine/server.py`
- Test: `/home/zhenyi/ownprojects/asr-engine/tests/test_admin_pause.py`(新)

**Interfaces(Produces,后续 task 依赖):**
- `POST /admin/pause` body `{"hours": float}` → `{"ok": true, "paused": true, "resume_at": "<iso>"}`;in_use 时仍置暂停态返回 ok:true(卸载由其 monitor 补做)
- `POST /admin/resume` → `{"ok": true, "was_paused": bool}`
- `/health` 增加 `"paused": bool`
- `/v1/audio/transcriptions` 暂停期返回**扁平** `503 {"paused": true, "resume_at": "<iso>", "paused_at": "<iso>", "detail": "ASR 引擎暂停中"}`(JSONResponse,不嵌 detail——与 transcribe 的 503 格式一致,backend 与 noteflow 都按顶层解析;`paused_at` 是暂停起始时刻,noteflow 用它识别迟到的旧事件)
- ModelManager 新方法:`pause(hours) -> float`(返回 epoch 秒,内部记 `_paused_at` 起始 epoch)、`resume() -> bool`、`is_paused() -> bool`、`resume_at() -> datetime`、`paused_at() -> datetime`(均为 naive 本地时间,端点用 `.astimezone().isoformat()` 输出)

- [ ] **Step 1: 写失败测试** `tests/test_admin_pause.py`

```python
"""admin pause/resume 端点与暂停期拒新测试。"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from asr_engine.server import app
    return TestClient(app)


def test_pause_then_transcribe_rejected(client):
    with patch("asr_engine.server.manager") as mock_manager:
        import datetime
        mock_manager.is_paused.return_value = True
        mock_manager.resume_at.return_value = datetime.datetime(2099, 1, 1, tzinfo=datetime.timezone.utc)
        mock_manager.paused_at.return_value = datetime.datetime(2099, 1, 1, tzinfo=datetime.timezone.utc)
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("t.wav", b"RIFF", "audio/wav")},
            data={"model": "qwen3-asr-q4k"},
        )
    assert resp.status_code == 503
    body = resp.json()
    assert body["paused"] is True and body["resume_at"]


def test_admin_pause_calls_manager(client):
    with patch("asr_engine.server.manager") as mock_manager:
        mock_manager.pause.return_value = 4070880000.0  # epoch 秒
        mock_manager.in_use = False
        r = client.post("/admin/pause", json={"hours": 2})
    assert r.status_code == 200 and r.json()["ok"] is True
    mock_manager.pause.assert_called_once_with(2.0)


def test_admin_resume_and_health(client):
    with patch("asr_engine.server.manager") as mock_manager:
        mock_manager.resume.return_value = True
        mock_manager.is_paused.return_value = False
        mock_manager.engine.is_loaded = False
        mock_manager.in_use = False
        assert client.post("/admin/resume").json() == {"ok": True, "was_paused": True}
        assert client.get("/health").json()["paused"] is False


def test_model_manager_pause_semantics():
    from asr_engine.model_manager import ModelManager
    from asr_engine.config import get_config
    m = ModelManager(get_config())
    assert not m.is_paused()
    m.pause(0.01)
    assert m.is_paused()
    assert m.resume() is True and not m.is_paused()
```

- [ ] **Step 2: 跑测试确认失败**(模块/方法不存在)

Run: `cd /home/zhenyi/ownprojects/asr-engine && ~/miniconda3/envs/funasr/bin/python -m pytest tests/test_admin_pause.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 ModelManager.pause/resume**

`model_manager.py` 的 `__init__` 加 `self._paused_until = None` 与 `self._paused_at = None`;`_monitor_loop` 的 idle 卸载分支前加:`if self.is_paused() and not self.in_use: self.unload(); continue`(暂停期到点即卸载,补做 in_use 场景)。新增:

```python
def pause(self, hours: float) -> float:
    """置暂停截止(epoch 秒,覆盖语义);不阻塞——卸载由 monitor/idle 兜底"""
    import time as _t
    now = _t.time()
    self._paused_at = now              # 起始时刻(旧事件识别用,__init__ 初始化为 None)
    self._paused_until = now + hours * 3600
    logger.info("engine paused for %sh until epoch %s", hours, self._paused_until)
    if not self.in_use:
        self.unload()
    return self._paused_until

def resume(self) -> bool:
    was = self.is_paused()
    self._paused_until = None
    self._paused_at = None
    if was:
        logger.info("engine resumed")
    return was

def is_paused(self) -> bool:
    import time as _t
    return self._paused_until is not None and _t.time() < self._paused_until

def resume_at(self):
    if not self.is_paused():
        return None
    import datetime
    return datetime.datetime.fromtimestamp(self._paused_until)

def paused_at(self):
    if not self.is_paused():
        return None
    import datetime
    return datetime.datetime.fromtimestamp(self._paused_at)
```

- [ ] **Step 4: server.py 端点**

`asr_engine/server.py` 加(在 `/admin/cancel` 后);`transcribe` 函数体开头(`_check_token` 之后)加拦截;`/health` 加字段:

```python
from pydantic import BaseModel


class PauseRequest(BaseModel):
    hours: float


@app.post("/admin/pause")
def admin_pause(req: PauseRequest):
    if not (0 < req.hours <= 48):
        return {"ok": False, "reason": "hours out of range (0, 48]"}
    until = manager.pause(req.hours)
    from datetime import datetime
    return {"ok": True, "paused": True,
            "resume_at": datetime.fromtimestamp(until).astimezone().isoformat(),
            "engine_loaded": manager.engine.is_loaded}


@app.post("/admin/resume")
def admin_resume():
    return {"ok": True, "was_paused": manager.resume()}


# /health 返回加: "paused": manager.is_paused()

# transcribe() 内 _check_token(authorization) 之后(直接 return 扁平 JSONResponse,与 transcribe-service 格式一致):
from fastapi.responses import JSONResponse
if manager.is_paused():
    return JSONResponse(
        status_code=503,
        content={"paused": True,
                 "resume_at": manager.resume_at().astimezone().isoformat(),
                 "paused_at": manager.paused_at().astimezone().isoformat(),
                 "detail": "ASR 引擎暂停中"})
```

- [ ] **Step 5: 跑测试通过**

Run: 同 Step 2。Expected: 4 passed

- [ ] **Step 6: 提交**

```bash
cd /home/zhenyi/ownprojects/asr-engine && git add -A && git commit -m "feat: asr-engine 暂停态(/admin/pause /admin/resume,暂停期拒新+卸载)"
```

---

### Task 2: transcribe-service PauseManager

**Files:**
- Create: `/home/zhenyi/ownprojects/transcribe-service/pause_manager.py`
- Modify: `/home/zhenyi/ownprojects/transcribe-service/.gitignore`(加 `pause_state.json`)
- Test: `/home/zhenyi/ownprojects/transcribe-service/tests/test_pause_manager.py`(新)

**Interfaces(Produces):**

```python
class PauseManager:
    def __init__(self, state_file="pause_state.json", notify_url="", notify_token="", notify_fn=None)
    def pause(self, hours: float) -> datetime            # 覆盖语义;落盘;重设通知定时
    def resume(self) -> bool                             # 清态+清标志+clearTimeout+立即通知
    def is_paused(self) -> bool                          # 惰性时间戳
    def resume_at(self) -> datetime | None
    def paused_at(self) -> datetime | None               # 本次暂停起始(旧事件识别)
    def remaining_seconds(self) -> float
    def status(self) -> dict                             # {paused, resume_at(带时区), remaining_seconds}
    def should_unload(self) -> bool                      # is_paused and not _unload_attempted
    def mark_unload_attempted(self) -> None
    @property generation -> int                          # 世代令牌,pause/resume 各自 +1
```

通知:到点(=pause 的剩余时长)`notify_fn()`;构造时若 `is_paused()` 按剩余时长重设;`notify_fn` 由 server.py 注入(默认 `None` 不通知)。

- [ ] **Step 1: 写失败测试** `tests/test_pause_manager.py`

```python
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
```

- [ ] **Step 2: 跑测试确认失败**(ImportError)

Run: `cd /home/zhenyi/ownprojects/transcribe-service && ~/miniconda3/envs/funasr/bin/python -m pytest tests/test_pause_manager.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 pause_manager.py**

```python
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
        except Exception as e:
            logger.warning(f"暂停状态文件损坏,视为未暂停: {self._state_file} ({e})")
            self._try_unlink(); return
        if until <= time.time():
            logger.info("已过期的暂停状态文件,删除并视为未暂停")
            self._try_unlink(); return
        self._paused_until = until
        try:
            self._paused_at = datetime.fromisoformat(data["paused_at"]).timestamp()
        except Exception:
            self._paused_at = None
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
        return self._paused_until is not None and time.time() < self._paused_until

    def resume_at(self):
        if not self.is_paused():
            return None
        return datetime.fromtimestamp(self._paused_until)

    def paused_at(self):
        if not self.is_paused() or self._paused_at is None:
            return None
        return datetime.fromtimestamp(self._paused_at)

    def remaining_seconds(self) -> float:
        return self._paused_until - time.time() if self.is_paused() else 0.0

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
```

`.gitignore` 追加:

```
# 暂停状态文件(运行时产物)
pause_state.json
```

- [ ] **Step 4: 跑测试通过**

Run: 同 Step 2。Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
cd /home/zhenyi/ownprojects/transcribe-service && git add pause_manager.py tests/test_pause_manager.py .gitignore && git commit -m "feat: PauseManager 暂停状态管理(落盘/惰性恢复/到点通知/世代令牌)"
```

---

### Task 3: diarization 锁修正 + unload

**Files:**
- Modify: `/home/zhenyi/ownprojects/transcribe-service/diarization/manager.py`
- Test: `/home/zhenyi/ownprojects/transcribe-service/tests/test_diarization_unload.py`(新)

**Interfaces(Produces):** `DiarizationManager.unload() -> bool`(锁序 `_infer_lock → _load_lock`,内置 gc+empty_cache)、`is_loaded` property、模块级 `unload_global() -> bool` / `is_loaded() -> bool`。**锁修正**:`diarize()` 的 `self._load()` 移入 `_infer_lock` 内(加载→推理同一把锁保护)。

- [ ] **Step 1: 写失败测试** `tests/test_diarization_unload.py`

```python
"""diarization unload 语义与锁结构测试。"""
import threading


def test_unload_clears_pipeline_and_flags():
    from diarization.manager import DiarizationManager
    m = DiarizationManager.__new__(DiarizationManager)
    m._pipeline = object()
    m._load_lock = threading.Lock()
    m._infer_lock = threading.Lock()

    import gc as gc_mod
    emptied = {}

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            emptied["cache"] = True

    import sys
    fake = type(sys)("fake_torch")
    fake.cuda = FakeCuda
    import pytest
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(gc_mod, "collect", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", fake)
    try:
        assert m.unload() is True
        assert m._pipeline is None and m.is_loaded is False
        assert emptied.get("cache") is True
        assert m.unload() is False  # 未加载再卸返回 False
    finally:
        monkeypatch.undo()


def test_diarize_loads_inside_infer_lock():
    """锁修正验证:diarize 源码中 _load 调用必须出现在 with self._infer_lock 之后"""
    import inspect
    from diarization import manager as dm
    src = inspect.getsource(dm.DiarizationManager.diarize)
    infer_pos = src.index("with self._infer_lock")
    load_pos = src.index("self._load()")
    assert load_pos > infer_pos, "_load() 必须在 _infer_lock 内(防卸载竞态)"
```

- [ ] **Step 2: 确认失败**。Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/test_diarization_unload.py -v`(cwd=transcribe-service 根)Expected: FAIL

- [ ] **Step 3: 实现**

`diarize()` 改为(锁修正):

```python
    def diarize(self, samples: np.ndarray, sample_rate: int = 16000) -> List[SpeakerTurn]:
        # _load 在 _infer_lock 内:加载→推理构成同一锁保护的完整生命周期,
        # 与 unload(_infer_lock → _load_lock)无交错窗口
        with self._infer_lock:
            self._load()
            import torch
            waveform = torch.from_numpy(samples).unsqueeze(0)
            audio = {"waveform": waveform, "sample_rate": sample_rate}
            kwargs = {"num_speakers": self.num_speakers} if self.num_speakers > 0 else {}
            diarization = self._pipeline(audio, **kwargs)
        raw = [(label, turn.start, turn.end) for turn, _, label in diarization.itertracks(yield_label=True)]
        turns = self._compact_turns(raw)
        logger.info(f"说话人分离完成: {len(set(t.speaker for t in turns))} 人 / {len(turns)} turns")
        return turns
```

新增(`diarize` 前):

```python
    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def unload(self) -> bool:
        """卸载管线释放显存(服务暂停联动)。先等在跑的推理结束;未加载返回 False。

        锁序 _infer_lock → _load_lock:与 diarize(_infer_lock 内 _load)一致,无死锁。
        可在后台线程长时间等待(长视频推理分钟级)——调用方不得在请求线程同步调用。
        """
        with self._infer_lock:
            with self._load_lock:
                if self._pipeline is None:
                    return False
                self._pipeline = None
                import gc
                import torch  # 延迟导入(模块 docstring 约定)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("说话人分离管线已卸载")
                return True
```

模块级(`get_manager` 后):

```python
def unload_global() -> bool:
    """卸载单例管线并丢弃单例(下次 get_manager 按 config 重建)"""
    global _manager
    m = _manager
    if m is None:
        return False
    released = m.unload()
    with _manager_lock:
        _manager = None
    return released


def is_loaded() -> bool:
    return _manager is not None and _manager.is_loaded
```

- [ ] **Step 4: 跑测试通过**。Run: 同 Step 2。Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add diarization/manager.py tests/test_diarization_unload.py && git commit -m "fix: diarize 加载移入推理锁内消除卸载竞态;新增 unload/is_loaded(暂停联动)"
```

---

### Task 4: UpstreamPausedError 上行传播

**Files:**
- Modify: `/home/zhenyi/ownprojects/transcribe-service/backends/asr_engine_backend.py`
- Modify: `/home/zhenyi/ownprojects/transcribe-service/transcribe.py`(一处:process_transcription 大 catch 前重抛;另一处在 server.py 的 `/transcribe_file` 外层 catch,见 Step 3)
- Modify: `/home/zhenyi/ownprojects/transcribe-service/server.py`(三端点捕获→503;/transcribe_file 外层 catch 同样先重抛)
- Test: `/home/zhenyi/ownprojects/transcribe-service/tests/test_upstream_paused.py`(新)

**Interfaces(Produces):** `backends/asr_engine_backend.py` 定义 `class UpstreamPausedError(RuntimeError)`(属性 `resume_at: str | None`、`paused_at: str | None`);backend 收到 asr 503+`paused` body 时抛它;`transcribe.py` 与 server 端点让它原样穿透为 `503 {"detail","paused","resume_at","paused_at"}`。

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 确认失败**。Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/test_upstream_paused.py -v`。Expected: FAIL

- [ ] **Step 3: 实现**

`backends/asr_engine_backend.py`(模块级,类定义前):

```python
class UpstreamPausedError(RuntimeError):
    """asr-engine 暂停(503 + paused body)。需原样穿透到 server 端点转 503,不得转 error dict。"""

    def __init__(self, message: str, resume_at: str = None, paused_at: str = None):
        super().__init__(message)
        self.resume_at = resume_at
        self.paused_at = paused_at
```

`transcribe()` 的 `except httpx.HTTPStatusError` 分支内(asr 暂停 body 为**扁平**结构,读顶层):

```python
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503:
                    try:
                        body = e.response.json()
                    except Exception:
                        body = {}
                    if isinstance(body, dict) and body.get("paused") is True:
                        raise UpstreamPausedError(
                            body.get("detail") or "ASR 引擎暂停中",
                            resume_at=body.get("resume_at"), paused_at=body.get("paused_at"))
                raise RuntimeError(...)  # 原有逻辑不变
```

`transcribe.py`:`process_transcription` 的大 `except Exception as e:`(约 1116 行)**之前**插入独立分支:

```python
        except UpstreamPausedError:
            raise  # 暂停信号原样上抛,由端点转 503;不得转 error dict
```

并在文件头部 import 区加 `from backends.asr_engine_backend import UpstreamPausedError`(若循环依赖则函数内延迟导入)。

`server.py`:三个端点的调用外各包一层(以 `/transcribe_url` 为例,其余两个同款;**直接 return 扁平 JSONResponse**,与中间件 503 格式完全一致——noteflow 读顶层 `body.paused`;本 task 阶段不依赖 pause_manager 单例,Task 5 再增强):

```python
        try:
            result = await transcription_service.process_transcription(...)
        except UpstreamPausedError as e:
            from datetime import datetime as _dt
            resume_iso = e.resume_at or _dt.now().astimezone().isoformat()
            paused_iso = e.paused_at or _dt.now().astimezone().isoformat()
            return JSONResponse(status_code=503, content={
                "detail": "服务暂停中,稍后自动恢复", "paused": True,
                "resume_at": resume_iso, "paused_at": paused_iso})
```

注意 `/transcribe_file` 现有 `except Exception as e:` 外层 catch 要在其**前面**加 `except UpstreamPausedError: raise`,让上述包装统一处理。

- [ ] **Step 4: 跑测试通过**。Run: 同 Step 2。Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add backends/asr_engine_backend.py transcribe.py server.py tests/test_upstream_paused.py && git commit -m "feat: asr 暂停信号上行传播(UpstreamPausedError 原样穿透为 503 paused)"
```

---

### Task 5: transcribe-service 端点/中间件/释放链/通知配置

**Files:**
- Modify: `/home/zhenyi/ownprojects/transcribe-service/server.py`(主改动)
- Modify: `/home/zhenyi/ownprojects/transcribe-service/config.py`(pause 配置 property)
- Modify: `/home/zhenyi/ownprojects/transcribe-service/config.yaml.example`(示例)
- Modify: `/home/zhenyi/ownprojects/transcribe-service/config.yaml`(实际配置,加 pause 组)
- Test: `/home/zhenyi/ownprojects/transcribe-service/tests/test_pause_endpoints.py`(新)

**Interfaces(Consumes):** Task 2 PauseManager、Task 3 diarization unload_global、Task 4 UpstreamPausedError。
**Interfaces(Produces):** `POST /pause {hours}`(响应含扁平释放状态)、`POST /resume`、`GET /status`(含 asr_engine_paused);中间件 503 拦截(含 diarize_only——同路径);`_release_gpu_for_pause(gen)`;monitor_loop 补做。

- [ ] **Step 1: 写失败测试** `tests/test_pause_endpoints.py`

```python
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
```

- [ ] **Step 2: 确认失败**。Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/test_pause_endpoints.py -v`。Expected: FAIL

- [ ] **Step 3: 实现**

`config.py` 加 property:

```python
    @property
    def pause_config(self) -> dict:
        return self.get('pause', {})

    @property
    def pause_notify_url(self) -> str:
        return self.pause_config.get('notify_url', '')

    @property
    def pause_notify_token(self) -> str:
        return self.pause_config.get('notify_token', '')
```

`config.yaml`(与 example 同步)加:

```yaml
pause:
  notify_url: ""            # noteflow 唤醒端点,生产填 https://<mryk域名>/api/noteflow/internal/transcribe-wake
  notify_token: ""          # 与 mryk24 NUXT_NOTEFLOW_INTERNAL_TOKEN 同值
```

`server.py`:

```python
# import 区追加
import asyncio
import math
from pause_manager import PauseManager, MAX_PAUSE_HOURS
from diarization import manager as diarization_mgr
from backends.asr_engine_backend import UpstreamPausedError
from pydantic import BaseModel, Field

# 单例区(manager/downloader 之后)
pause_manager = PauseManager(notify_url=config.pause_notify_url,
                              notify_token=config.pause_notify_token)
PAUSED_REJECT_PATHS = {"/transcribe", "/transcribe_url", "/transcribe_file"}


def _fmt_resume_at(resume_at):
    if resume_at is None:
        return "未知"
    from datetime import datetime
    return f"{resume_at:%H:%M}" if resume_at.date() == datetime.now().date() else f"{resume_at:%m-%d %H:%M}"


def _paused_503_response():
    resume_at = pause_manager.resume_at()
    if resume_at is None:  # 窄 TOCTOU:检查与构造之间被 /resume
        resume_at = datetime.now()
    retry_after = max(1, math.ceil(pause_manager.remaining_seconds()))
    paused_at = pause_manager.paused_at()
    return JSONResponse(
        status_code=503,
        content={"detail": f"服务暂停中,预计 {_fmt_resume_at(resume_at)} 恢复",
                 "paused": True,
                 "resume_at": resume_at.astimezone().isoformat(),
                 "paused_at": paused_at.astimezone().isoformat() if paused_at else None},
        headers={"Retry-After": str(retry_after)})


def _pause_asr_engine(hours: float) -> str:
    """转发暂停到 asr-engine;容错(其 idle 卸载兜底)"""
    import httpx
    try:
        resp = httpx.post(f"{config.asr_engine_url.rstrip('/')}/admin/pause",
                          json={"hours": hours}, timeout=5.0)
        return "paused" if resp.status_code == 200 else f"http_{resp.status_code}"
    except Exception as e:
        logger.warning(f"转发暂停到 asr-engine 失败(容错): {e}")
        return "failed"


def _resume_asr_engine() -> str:
    """转发恢复到 asr-engine(手动提前恢复必须同步,否则 asr 挂到原期限);容错"""
    import httpx
    try:
        resp = httpx.post(f"{config.asr_engine_url.rstrip('/')}/admin/resume", timeout=5.0)
        return "resumed" if resp.status_code == 200 else f"http_{resp.status_code}"
    except Exception as e:
        logger.warning(f"转发恢复到 asr-engine 失败(容错,其到点自恢复兜底): {e}")
        return "failed"


def _release_gpu_for_pause(gen: int) -> dict:
    """后台释放链(仅本地卸载):世代令牌防交错;绝不在请求线程同步调用(会等推理锁)。

    asr-engine 的暂停转发不在此链中——/pause 端点同步调用 _pause_asr_engine(快速 HTTP),
    不受本地 in_use/推理锁阻塞,也不会因等锁延迟或延长上游暂停窗口。
    """
    result = {"backend": "not_loaded", "diarization": "not_loaded"}
    if not pause_manager.is_paused():
        return {"skipped": "resumed"}
    pause_manager.mark_unload_attempted()
    if manager._backend is not None and not manager.in_use:
        try:
            manager.unload_model()
            result["backend"] = "released"
        except Exception as e:
            logger.warning(f"暂停卸载 backend 失败: {e}")
            result["backend"] = "failed"
    if pause_manager.generation != gen:
        return result  # 等锁期间已 resume,中止
    try:
        result["diarization"] = "released" if diarization_mgr.unload_global() else "not_loaded"
    except Exception as e:
        logger.warning(f"暂停卸载 diarization 失败: {e}")
        result["diarization"] = "failed"
    logger.info(f"暂停本地释放链完成: {result}")
    return result
```

中间件(token 校验通过后、`call_next` 前):

```python
    if request.url.path in PAUSED_REJECT_PATHS and request.method == "POST" and pause_manager.is_paused():
        return _paused_503_response()
```

monitor_loop(整函数替换):

```python
def monitor_loop():
    while True:
        time.sleep(config.check_interval)
        try:
            if pause_manager.should_unload():
                # 补做本地卸载(等 diarization 推理锁是排空语义);asr 转发已在 /pause 同步完成
                _release_gpu_for_pause(pause_manager.generation)
            elif (not pause_manager.is_paused() and manager._backend is not None
                  and not manager.in_use):
                if time.time() - manager.last_active_time > config.idle_timeout:
                    manager.unload_model()
        except Exception as e:
            logger.warning(f"monitor_loop 异常(忽略继续): {e}")
```

(注:后台释放链不转发 asr——`/pause` 已同步转发过;monitor 补做只处理本地卸载。in_use 时 backend 分支跳过,但 diarization 的等锁排空仍会执行——函数返回即视为已尝试,`_unload_attempted` 已置位,monitor 不再重试,这是预期行为:关键显存(diarization)已被排空处理。)

端点(追加在 `/transcribe_file` 后):

```python
class PauseRequest(BaseModel):
    hours: float = Field(gt=0, le=MAX_PAUSE_HOURS, description="暂停时长(小时,支持小数)")


@app.post("/pause")
async def pause_service(request: PauseRequest):
    """暂停 N 小时:立即拒新请求;同步转发 asr 暂停(快速 HTTP);本地释放链后台执行"""
    resume_at = pause_manager.pause(request.hours)  # generation 在此 +1
    gen = pause_manager.generation                   # 捕获新世代供释放链校验
    logger.info(f"服务暂停 {request.hours}h,至 {_fmt_resume_at(resume_at)}")
    asr_status = await asyncio.to_thread(_pause_asr_engine, request.hours)
    asyncio.get_running_loop().run_in_executor(None, _release_gpu_for_pause, gen)
    return {"paused": True,
            "resume_at": resume_at.astimezone().isoformat(),
            "resume_at_display": _fmt_resume_at(resume_at),
            "hours": request.hours,
            "asr_engine": asr_status,
            "local_release": "background(在途任务跑完后完成,通常 ≤90s,长视频说话人分离可能更久;最终状态可查 /status)"}


@app.post("/resume")
async def resume_service():
    """提前恢复:清暂停态+世代自增使旧释放链中止+转发 asr-engine 恢复+立即通知 noteflow 唤醒"""
    was_paused = pause_manager.resume()
    asr_status = await asyncio.to_thread(_resume_asr_engine) if was_paused else "not_paused"
    logger.info(f"收到恢复请求(此前{'处于' if was_paused else '不在'}暂停状态,asr: {asr_status})")
    return {"paused": False, "was_paused": was_paused, "asr_engine": asr_status}


def _query_asr_paused() -> bool:
    """查询 asr-engine 暂停态;不可达时报告 False(其进程不在=未暂停)"""
    try:
        import httpx
        return httpx.get(f"{config.asr_engine_url.rstrip('/')}/health", timeout=3.0).json().get("paused", False)
    except Exception:
        return False


@app.get("/status")
async def service_status():
    return {**pause_manager.status(),
            "backend": manager.backend,
            "model_loaded": manager._backend is not None,
            "in_use": manager.in_use,
            "diarization_loaded": diarization_mgr.is_loaded(),
            "asr_engine_paused": _query_asr_paused()}
```

`startup_event` 与 `__main__` 预加载块加:

```python
    if pause_manager.is_paused():
        logger.info(f"启动时处于暂停状态(至 {_fmt_resume_at(pause_manager.resume_at())}),跳过预加载")
        return  # __main__ 分支为跳过预加载(不 return 函数)
```

- [ ] **Step 4: 跑全套测试**(新 + 既有回归)

Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/ -q`
Expected: 全部 passed(含既有 106+)

- [ ] **Step 5: 提交**

```bash
git add server.py config.py config.yaml.example tests/test_pause_endpoints.py && git commit -m "feat: /pause /resume /status 端点与后台释放链(世代令牌+通知配置)"
```

(实际 `config.yaml` 在 .gitignore 且未跟踪,显式 add 会报错——**只本地修改、不入库**;提交仅含 example)

---

### Task 6: noteflow errors + markTaskDeferred + worker 分支 + 兜底定时

**Files:**
- Create: `/home/zhenyi/ownprojects/mryk24/server/utils/noteflow/errors.ts`
- Create: `/home/zhenyi/ownprojects/mryk24/server/utils/noteflow/pause-state.ts`(中立微模块,防 scheduler↔pause-wake 循环 import)
- Create: `/home/zhenyi/ownprojects/mryk24/server/utils/noteflow/pause-wake.ts`
- Modify: `/home/zhenyi/ownprojects/mryk24/server/utils/noteflow/scheduler.ts`
- Modify: `/home/zhenyi/ownprojects/mryk24/server/api/noteflow/internal/worker.ts`
- Test: `/home/zhenyi/ownprojects/mryk24/tests/unit/noteflow/service-paused.test.ts`(新)

**Interfaces(Produces):**
- `errors.ts`:`ServicePausedError(message, resumeAt?)`、`isServicePaused(err)`(name 判定)、`parsePausedResponse(response) -> Promise<ServicePausedError | null>`
- `pause-state.ts`:`getLastResumeSignalAt() -> number`(epoch ms,0=从未)、`markResumeSignal() -> void`(唤醒时调用)
- `scheduler.ts`:`markTaskDeferred(taskId, error) -> Promise<'deferred' | 'fallback-failed'>`(`resumeAt <= now` **或** `resumeAt <= lastResumeSignalAt`(迟到旧事件,即使未过期)→ `markTaskFailed` 常规路径)
- `pause-wake.ts`:`scheduleFallbackWake(resumeAt: Date)`、`cancelFallbackWake()`、`wakeDeferredTasks() -> Promise<number>`(包装版:markResumeSignal + scheduler.resumeDeferredTasks;**刻意改名防 Nuxt auto-import 与 scheduler 的 DB 版重名拿错**)、`startupSelfHeal()`(**不经包装层、不记恢复信号**——启动≠服务恢复,记了会把跨重启的旧暂停 503 判 stale 烧重试)
- worker 分支:`isServicePaused(error)` → `markTaskDeferred` + `scheduleFallbackWake`

- [ ] **Step 1: 写失败测试** `tests/unit/noteflow/service-paused.test.ts`

```typescript
/**
 * 暂停联动:错误识别 / markTaskDeferred(resume_at 过期回退) / 兜底定时 / worker 分支
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';

// Mock prisma(自包含工厂,相对路径与被测模块一致)
vi.mock('../../../server/utils/prisma', () => {
  const noteFlowTask = {
    findFirst: vi.fn(), updateMany: vi.fn(), update: vi.fn(), findUnique: vi.fn(), count: vi.fn(),
  };
  return {
    prisma: {
      noteFlowTask,
      $transaction: vi.fn(async (cb: any) => cb({ noteFlowTask })),
    },
  };
});

vi.mock('../../../server/utils/noteflow/config', () => ({
  getAllConfig: vi.fn(async () => ({ retry: { workerMaxAttempts: 3, maxAttempts: 3, backoffMs: 5000, multiplier: 2 } })),
  TASK_CONFIG: { HEARTBEAT: { VIDEO: 15000, DEFAULT: 30000 } },
}));

import { prisma } from '../../../server/utils/prisma';
import { ServicePausedError, isServicePaused, parsePausedResponse } from '../../../server/utils/noteflow/errors';
import { markTaskDeferred } from '../../../server/utils/noteflow/scheduler';
import { resumeDeferredTasks } from '../../../server/utils/noteflow/pause-wake';

const mockNoteFlowTask = prisma.noteFlowTask as any;

function pausedResponse(body: unknown, status = 503): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

describe('ServicePausedError', () => {
  it('name 判定;普通错误/非错误不误判', () => {
    expect(isServicePaused(new ServicePausedError('暂停', '2099-01-01T00:00:00+08:00'))).toBe(true);
    expect(isServicePaused(new Error('FunASR API 错误: 503'))).toBe(false);
    expect(isServicePaused(null)).toBe(false);
  });

  it('parsePausedResponse:paused body 解析 resume_at;无 paused/非 JSON → null', async () => {
    const err = await parsePausedResponse(pausedResponse({ detail: '服务暂停中', paused: true, resume_at: '2099-06-01T04:00:00+08:00' }));
    expect(err).toBeInstanceOf(ServicePausedError);
    expect(err!.resumeAt!.toISOString()).toBe('2099-05-31T20:00:00.000Z');
    expect(await parsePausedResponse(pausedResponse({ detail: 'x' }))).toBeNull();
    expect(await parsePausedResponse(new Response('oops', { status: 503 }))).toBeNull();
  });
});

describe('markTaskDeferred', () => {
  beforeEach(() => vi.clearAllMocks());

  it('未过期:置 deferred,不碰 retryCount', async () => {
    mockNoteFlowTask.update.mockResolvedValue({});
    const r = await markTaskDeferred('t1', new ServicePausedError('暂停', '2099-06-01T04:00:00+08:00'));
    expect(r).toBe('deferred');
    const data = mockNoteFlowTask.update.mock.calls[0][0].data;
    expect(data.status).toBe('deferred');
    expect(data.error).toContain('转录服务暂停中');
    expect(data.retryCount).toBeUndefined();
  });

  it('resume_at 已过期:回退 markTaskFailed 常规路径(烧一次 retryCount)', async () => {
    mockNoteFlowTask.update.mockResolvedValue({});
    mockNoteFlowTask.findUnique.mockResolvedValue({ retryCount: 0 });
    const r = await markTaskDeferred('t1', new ServicePausedError('暂停', '2000-01-01T00:00:00+08:00'));
    expect(r).toBe('fallback-failed');
    const data = mockNoteFlowTask.update.mock.calls.at(-1)[0].data;
    expect(data.status).toBe('pending');
    expect(data.retryCount).toEqual({ increment: 1 });
  });

  it('resume_at 缺失:同回退路径(无法判定的旧事件不凭空挂起)', async () => {
    mockNoteFlowTask.update.mockResolvedValue({});
    mockNoteFlowTask.findUnique.mockResolvedValue({ retryCount: 0 });
    const r = await markTaskDeferred('t1', new ServicePausedError('暂停'));
    expect(r).toBe('fallback-failed');
  });

  it('竞态:恢复信号已到、迟到的 503 起始于信号之前(截止未过期)→ 仍回退不挂起', async () => {
    const { resumeDeferredTasks } = await import('../../../server/utils/noteflow/pause-wake');
    mockNoteFlowTask.update.mockResolvedValue({});
    mockNoteFlowTask.updateMany.mockResolvedValue({ count: 1 }); // pause-wake 包装层的 DB 调用
    await resumeDeferredTasks(); // 触发 markResumeSignal(记下当前时刻 T)
    mockNoteFlowTask.findUnique.mockResolvedValue({ retryCount: 0 });
    const oldPauseStart = new Date(Date.now() - 60_000).toISOString();   // T-60s 开始的旧暂停
    const futureDeadline = new Date(Date.now() + 3600_000).toISOString(); // 截止未过期
    const r = await markTaskDeferred(
      't1', new ServicePausedError('暂停', futureDeadline, oldPauseStart));
    expect(r).toBe('fallback-failed'); // pausedAt <= 恢复信号 → 旧事件
  });

  it('竞态(反向):恢复信号之后开始的新暂停 → 正常挂起', async () => {
    const { resumeDeferredTasks } = await import('../../../server/utils/noteflow/pause-wake');
    mockNoteFlowTask.update.mockResolvedValue({});
    mockNoteFlowTask.updateMany.mockResolvedValue({ count: 1 });
    await resumeDeferredTasks();
    mockNoteFlowTask.findUnique.mockResolvedValue({ retryCount: 0 });
    const newPauseStart = new Date(Date.now() + 1000).toISOString();     // 信号之后开始
    const futureDeadline = new Date(Date.now() + 3600_000).toISOString();
    const r = await markTaskDeferred(
      't1', new ServicePausedError('暂停', futureDeadline, newPauseStart));
    expect(r).toBe('deferred');
  });
});

describe('resumeDeferredTasks', () => {
  beforeEach(() => vi.clearAllMocks());

  it('批量 deferred→pending,清挂起文案,返回数量', async () => {
    mockNoteFlowTask.updateMany.mockResolvedValue({ count: 2 });
    const n = await resumeDeferredTasks();
    expect(n).toBe(2);
    const arg = mockNoteFlowTask.updateMany.mock.calls[0][0];
    expect(arg.where.status).toBe('deferred');
    expect(arg.data.status).toBe('pending');
    expect(arg.data.error).toBeNull();
  });
});
```

- [ ] **Step 2: 确认失败**。Run: `cd /home/zhenyi/ownprojects/mryk24 && pnpm vitest run tests/unit/noteflow/service-paused.test.ts`。Expected: FAIL(模块不存在)

- [ ] **Step 3: 实现**

`errors.ts`:

```typescript
/**
 * 转录服务暂停错误:任务应挂起(deferred)等待恢复,不烧 retryCount。
 * 由 extractors 识别 503 paused body 抛出,worker 捕获后 markTaskDeferred。
 */
export class ServicePausedError extends Error {
  readonly resumeAt: Date | null;
  /** 本次暂停的起始时刻——旧事件判定用它(截止时刻拦不住提前恢复竞态,见 markTaskDeferred) */
  readonly pausedAt: Date | null;

  constructor(message: string, resumeAt?: string | null, pausedAt?: string | null) {
    super(message);
    this.name = 'ServicePausedError';
    this.resumeAt = resumeAt ? new Date(resumeAt) : null;
    this.pausedAt = pausedAt ? new Date(pausedAt) : null;
  }
}

/** name 判定(抗 Nuxt 打包边界的跨模块实例) */
export function isServicePaused(err: unknown): err is ServicePausedError {
  return err instanceof Error && err.name === 'ServicePausedError';
}

/** 解析 503 响应体;无 paused:true 返回 null(走普通错误路径) */
export async function parsePausedResponse(response: Response): Promise<ServicePausedError | null> {
  try {
    const body = await response.json();
    if (body?.paused === true) {
      return new ServicePausedError(body.detail || '转录服务暂停中', body.resume_at, body.paused_at);
    }
  } catch {
    // 非 JSON body
  }
  return null;
}
```

`scheduler.ts`(import 区加 `import { isServicePaused, type ServicePausedError } from './errors';` 与 `import { getLastResumeSignalAt } from './pause-state';`,`markTaskCancelled` 后加):

```typescript
/**
 * 暂停挂起:任务置 deferred(不碰 retryCount),挂起文案含预计恢复时间(Asia/Shanghai 格式)。
 * "迟到的旧暂停事件"判定(→ 回退 markTaskFailed):
 * ① resumeAt 已过期;或 ② pausedAt <= 最近恢复信号时刻——本次暂停开始于最近一次唤醒之前。
 *    注意必须比较【起始】而非截止:提前恢复场景(截止 18:00、17:00 恢复、17:01 迟到 503 带
 *    resume_at=18:00)中截止晚于信号,比较截止拦不住;比较起始(暂停开始 ≤ 17:00)才正确。
 *    pausedAt 缺失(旧版服务/字段丢失)时退化为仅 ①(保守挂起,兜底定时救)。
 */
export async function markTaskDeferred(
  taskId: string,
  error: ServicePausedError
): Promise<'deferred' | 'fallback-failed'> {
  if (!isServicePaused(error)) {
    throw new Error('markTaskDeferred 仅接受 ServicePausedError');
  }
  const stale =
    !error.resumeAt ||
    error.resumeAt.getTime() <= Date.now() ||
    (error.pausedAt !== null && error.pausedAt.getTime() <= getLastResumeSignalAt());
  if (stale) {
    // 旧暂停事件(服务已恢复):按普通失败重试,自收敛
    await markTaskFailed(taskId, error.message);
    return 'fallback-failed';
  }
  const resumeText = error.resumeAt.toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai' });
  await prisma.noteFlowTask.update({
    where: { id: taskId },
    data: {
      status: 'deferred',
      error: `System: 转录服务暂停中(预计 ${resumeText} 恢复),恢复后自动重跑`,
    },
  });
  return 'deferred';
}

/** 唤醒的 DB 实现:全部 deferred → pending(清挂起文案);返回数量。幂等。
 * 恢复信号时刻的更新在 pause-wake 的包装层(避免 scheduler 依赖 pause-wake 造成循环)。 */
export async function resumeDeferredTasks(): Promise<number> {
  const result = await prisma.noteFlowTask.updateMany({
    where: { status: 'deferred' },
    data: { status: 'pending', error: null },
  });
  return result.count;
}
```

`pause-state.ts`(中立微模块——scheduler 与 pause-wake 都要读写的"恢复信号时刻",独立出来防循环 import):

```typescript
/**
 * 暂停恢复信号时刻(进程内):每次唤醒(推送信号/兜底定时/自愈)时更新。
 * markTaskDeferred 用它识别"迟到的旧暂停事件"——503 的 resume_at 即使未过期,
 * 若早于最近一次恢复信号,说明它属于已恢复的旧暂停,不再挂起。
 */
let lastResumeSignalAt = 0;

export function getLastResumeSignalAt(): number {
  return lastResumeSignalAt;
}

export function markResumeSignal(): void {
  lastResumeSignalAt = Date.now();
}
```

`pause-wake.ts`(新;注意 prisma 相对路径是 `../prisma`——本文件在 `server/utils/noteflow/` 下):

```typescript
/**
 * 暂停唤醒:noteflow 侧兜底定时(固定时间 setTimeout,无轮询无探测)。
 * 主路径是 transcribe-service 的推送(唤醒端点);本定时只在其失效时兜底。
 */
import { prisma } from '../prisma';
import { markResumeSignal } from './pause-state';

const WORKER_URL = 'http://localhost:8000/api/noteflow/internal/worker';

let fallbackTimer: NodeJS.Timeout | null = null;

function triggerWorker(): void {
  fetch(WORKER_URL, { method: 'POST' }).catch((err: unknown) => {
    console.error('[NoteFlow] 暂停唤醒触发 worker 失败:', (err as Error).message);
  });
}

/** 唤醒(信号/定时/自愈共用):更新恢复信号时刻 + 批量置回 pending + 触发 worker */
export async function resumeDeferredTasks(): Promise<number> {
  const { resumeDeferredTasks: doResume } = await import('./scheduler');
  markResumeSignal();
  return doResume();
}

/** 挂起时调用:对齐最近一次挂起的 resume_at 设一次性兜底唤醒(覆盖旧定时) */
export function scheduleFallbackWake(resumeAt: Date): void {
  if (fallbackTimer) {
    clearTimeout(fallbackTimer);
    fallbackTimer = null;
  }
  const delay = resumeAt.getTime() - Date.now();
  if (delay <= 0) {
    return; // 已过期,由 markTaskDeferred 的回退路径处理
  }
  fallbackTimer = setTimeout(async () => {
    fallbackTimer = null;
    const n = await resumeDeferredTasks();
    if (n > 0) {
      console.warn(`[NoteFlow] 暂停兜底定时到期,唤醒 ${n} 个挂起任务`);
      triggerWorker();
    }
  }, delay);
  if (typeof fallbackTimer.unref === 'function') {
    fallbackTimer.unref(); // 不阻止进程退出
  }
}

/** 被信号唤醒后清除兜底定时 */
export function cancelFallbackWake(): void {
  if (fallbackTimer) {
    clearTimeout(fallbackTimer);
    fallbackTimer = null;
  }
}

/** 重启自愈:存在 deferred(内存定时已丢)→ 直接唤醒触发;若服务仍暂停会重新挂起重建信号链 */
export async function startupSelfHeal(): Promise<void> {
  const n = await prisma.noteFlowTask.count({ where: { status: 'deferred' } });
  if (n > 0) {
    console.warn(`[NoteFlow] 启动发现 ${n} 个挂起任务,自愈唤醒`);
    await resumeDeferredTasks();
    triggerWorker();
  }
}
```

(注:`resumeDeferredTasks` 的 DB 实现在 scheduler.ts,本模块包装一层注入 `markResumeSignal`;re-export 保持单一入口。scheduler.ts **不得** import pause-wake,只 import pause-state——无循环。)

`worker.ts`(import 区加两条;runWorker catch 的 `PromptConfigError` 分支后加):

```typescript
import { isServicePaused, type ServicePausedError } from '~/server/utils/noteflow/errors';
import { scheduleFallbackWake } from '~/server/utils/noteflow/pause-wake';

    } else if (isServicePaused(error)) {
      // 转录服务暂停:挂起(deferred,不烧 retryCount)+ 对齐兜底定时
      const paused = error as ServicePausedError;
      const outcome = await markTaskDeferred(task.id, paused);
      if (outcome === 'deferred' && paused.resumeAt) {
        scheduleFallbackWake(paused.resumeAt);
      }
      console.warn(`[NoteFlow] 转录服务暂停,任务 ${task.id} ${outcome}(resume=${paused.resumeAt?.toISOString() ?? '未知'})`);
    } else {
```

(markTaskDeferred 加入 worker.ts 顶部 scheduler import 列表)

- [ ] **Step 4: 跑测试通过**。Run: 同 Step 2。Expected: 全 passed

- [ ] **Step 5: 提交**

```bash
cd /home/zhenyi/ownprojects/mryk24 && git add server/utils/noteflow/errors.ts server/utils/noteflow/pause-state.ts server/utils/noteflow/pause-wake.ts server/utils/noteflow/scheduler.ts server/api/noteflow/internal/worker.ts tests/unit/noteflow/service-paused.test.ts && git commit -m "feat: noteflow 暂停挂起(deferred 状态+markTaskDeferred+兜底定时+worker 分支)"
```

---

### Task 7: 唤醒端点 + 白名单放行

**Files:**
- Create: `/home/zhenyi/ownprojects/mryk24/server/api/noteflow/internal/transcribe-wake.post.ts`
- Modify: `/home/zhenyi/ownprojects/mryk24/nuxt.config.ts`(runtimeConfig)
- Modify: `/home/zhenyi/ownprojects/mryk24/server/api/noteflow/[id]/reset.post.ts`(白名单+触发)
- Modify: `/home/zhenyi/ownprojects/mryk24/server/api/noteflow/[id]/cancel.post.ts`(白名单)
- Modify: `/home/zhenyi/ownprojects/mryk24/server/plugins/noteflow-init.ts`(启动自愈)
- Modify: `/home/zhenyi/ownprojects/mryk24/.env.prod`(生产 token;本地 `.env` 同步)

**Interfaces:** `POST /api/noteflow/internal/transcribe-wake`(header `Authorization: Bearer <NUXT_NOTEFLOW_INTERNAL_TOKEN>`,空配置放行)→ `{"woken": <n>}`;reset 接受 deferred 并触发 worker;cancel 接受 deferred。

- [ ] **Step 1: 实现**

(端点自身**不写 vitest 单测**:Nitro 自动导入全局(defineEventHandler/useRuntimeConfig 等)在纯 vitest node 环境不可用,仓库 36 个既有测试无一 import 过端点 handler,此路未验证。端点行为由 Task 11 端到端用例覆盖——spec §7 用例 4/5/6 全部经过它。)

- [ ] **Step 2: 实现**

`transcribe-wake.post.ts`:

```typescript
/**
 * transcribe-service 恢复通知端点(内部):deferred → pending + 触发 worker。
 * token 来自 runtimeConfig(env NUXT_NOTEFLOW_INTERNAL_TOKEN);未配置时放行(本地开发)。
 * wakeDeferredTasks 是 pause-wake 的包装版(内含 markResumeSignal,
 * 迟到的旧暂停 503 靠它识别不再挂起)——名字与 scheduler 的 DB 版本
 * resumeDeferredTasks 刻意不同,防 Nuxt auto-import 拿错。
 */
import { wakeDeferredTasks, cancelFallbackWake } from '~/server/utils/noteflow/pause-wake';

export default defineEventHandler(async (event) => {
  const config = useRuntimeConfig();
  const expected = config.noteflowInternalToken;
  if (expected) {
    const auth = getHeader(event, 'authorization') || '';
    if (auth !== `Bearer ${expected}`) {
      throw createError({ statusCode: 401, statusMessage: 'invalid wake token' });
    }
  }
  cancelFallbackWake();
  const woken = await wakeDeferredTasks();
  if (woken > 0) {
    console.warn(`[NoteFlow] 收到转录服务恢复通知,唤醒 ${woken} 个挂起任务`);
    fetch('http://localhost:8000/api/noteflow/internal/worker', { method: 'POST' }).catch(() => {});
  }
  return { isSuccess: true, data: { woken } };
});
```

`nuxt.config.ts` 的 `runtimeConfig`(若无则新增键):

```typescript
  runtimeConfig: {
    noteflowInternalToken: '', // env: NUXT_NOTEFLOW_INTERNAL_TOKEN
  },
```

`reset.post.ts`:状态白名单数组(现为 `['failed','cancelled','completed','skipped']`,在 ~26 行)加 `'deferred'`;重置数据对象已含 `error: null`(确认即可,无需改);文件尾(reset 成功后)加触发:

```typescript
  fetch('http://localhost:8000/api/noteflow/internal/worker', { method: 'POST' }).catch(() => {});
```

`cancel.post.ts`:白名单(现 `['pending','processing']`,~30 行)加 `'deferred'`。

`noteflow-init.ts`(启动链,DB 恢复之后)加:

```typescript
  const { startupSelfHeal } = await import('~/server/utils/noteflow/pause-wake');
  await startupSelfHeal().catch((err: unknown) => {
    console.error('[NoteFlow] 暂停自愈失败(不阻塞启动):', (err as Error).message);
  });
```

`.env.prod` 与本地 `.env` 各加一行(值临时生成,两处与 transcribe config.yaml 一致;**生成后写入,勿入库日志**——`.env.prod` 本就在 gitignore):

```
NUXT_NOTEFLOW_INTERNAL_TOKEN=<随机串>
```

- [ ] **Step 3: 跑测试 + typecheck**

Run: `pnpm vitest run tests/unit/noteflow/service-paused.test.ts && pnpm typecheck`
Expected: passed / 无错误

- [ ] **Step 4: 提交**

```bash
git add server/api/noteflow/internal/transcribe-wake.post.ts server/api/noteflow/[id]/reset.post.ts server/api/noteflow/[id]/cancel.post.ts nuxt.config.ts server/plugins/noteflow-init.ts tests/unit/noteflow/service-paused.test.ts && git commit -m "feat: noteflow 唤醒端点(transcribe-wake)+reset/cancel 放行 deferred+启动自愈"
```

---

### Task 8: extractors 识别与透传(4 处 TS)

**Files:**
- Modify: `/home/zhenyi/ownprojects/mryk24/server/utils/noteflow/extractors/bilibili.ts`(3 处 + diarizeSubtitle)
- Modify: `/home/zhenyi/ownprojects/mryk24/server/utils/noteflow/extractors/webdav.ts`(2 处)

**改动模式(每处同款,先判型重抛;注意个别 catch 的参数名是 `err` 而非 `error`,按现场变量名适配)**:

```typescript
if (isServicePaused(error)) { throw error; }
```

1. **bilibili.ts `transcribeAudio`**:`!response.ok` 分支(transcribeAudio 内 ~834 行)加识别(在抛 `FunASR API 错误` 前):

```typescript
      if (!response.ok) {
        if (response.status === 503) {
          const paused = await parsePausedResponse(response);
          if (paused) {
            fatal = true; // 短路退避重试循环
            throw paused;
          }
        }
        throw new Error(`FunASR API 错误: ${response.status}`);
      }
```

catch 块(~884 行)`lastError = getErrorMessage(error)` 前:

```typescript
      if (isServicePaused(error)) { throw error; } // 保住类型与 resumeAt,不走退避
```

2. **bilibili.ts `extractBilibiliSubtitle` 总 catch**(~1260 行)`const errMsg` 前:同款透传(抛出前可 `addLog` warn 记录)。
3. **bilibili.ts `diarizeSubtitle`**:`!response.ok` 分支(~967 行)同款识别抛出;其**上层降级 catch**(~1157-1168 行,记警告后返回无标注字幕处)首行同款透传。
4. **webdav.ts `extractWebdavSubtitle`**:`!response.ok`(~156 行)同款识别;外层 catch(~199 行)首行同款透传。

两个文件 import 区各加:

```typescript
import { isServicePaused, parsePausedResponse } from '../errors';
```

- [ ] **Step 1: 实现**(本 task 是机械透传,测试用既有套件回归 + typecheck 保证;识别逻辑已在 Task 6 覆盖 `parsePausedResponse`)

- [ ] **Step 2: 跑全量单测 + typecheck**

Run: `pnpm vitest run tests/unit/noteflow/ && pnpm typecheck`
Expected: 全 passed / 无错误

- [ ] **Step 3: 提交**

```bash
git add server/utils/noteflow/extractors/bilibili.ts server/utils/noteflow/extractors/webdav.ts && git commit -m "feat: extractors 识别暂停 503 并全程透传(含官方字幕说话人标注挂起)"
```

---

### Task 9: 前端 deferred 展示与操作

**Files:**
- Modify: `/home/zhenyi/ownprojects/mryk24/pages/noteflow/tasks.vue`(状态定义区 ~42-150 + 卡片 error 渲染条件 + 操作按钮条件,实施时 grep 定位)
- Modify: `/home/zhenyi/ownprojects/mryk24/composables/useNoteFlow.ts`(TaskCounts interface ~144-152 行、NoteFlowTask.status 联合类型)
- Modify: `/home/zhenyi/ownprojects/mryk24/server/api/noteflow/list.ts`(初始 counts 对象 ~22-30 行加 `deferred: 0`)

- [ ] **Step 1: 实现**

```typescript
// tasks.vue:
// statusLabels(~56 行)加:
  deferred: "挂起",
// statusColors(~139 行)加:
  deferred: "blue",
// statusTabs(页面实际构建筛选的数组,~72 行附近,仿既有项含 icon)加:
  { value: "deferred", label: `${statusLabels.deferred} (${statusCounts.value.deferred || 0})`,
    icon: 'i-heroicons-pause-circle' },
// statusCounts(~42 行)加:
  deferred: 0,
// 卡片 error 显示条件:移动端与桌面端各一处(现 status === 'failed' 才显示 error 红块),
// 改为 ['failed', 'deferred'].includes(status) ——否则挂起文案(预计恢复时间)不可见
// 操作按钮:取消按钮显示条件(现仅 processing)改为 ['processing', 'deferred'].includes(status);
// 重置按钮的可见状态集合加 'deferred'(与后端白名单对齐,否则页面无法重试挂起任务)

// useNoteFlow.ts:
// TaskCounts 闭合 interface(TS 多余属性检查会挂 typecheck)加: deferred: number
// NoteFlowTask.status 联合类型(若有显式 union)加 'deferred'
```

(结构提示:桌面端操作按钮是按状态逐个拆分的独立 `v-if` 块(重置按钮有 failed/cancelled/completed 三个),加 deferred 需新增一个同款按钮块或合并条件;移动端是集合条件。`list.ts` 初始 counts 对象(~22-30 行)顺手加 `deferred: 0`,避免无挂起任务时响应缺键;groupBy 动态填充会覆盖它。)

- [ ] **Step 2: 构建验证**

Run: `pnpm build && pnpm typecheck`
Expected: 构建成功 / 无类型错误

- [ ] **Step 3: 提交**

```bash
git add pages/noteflow/tasks.vue composables/useNoteFlow.ts server/api/noteflow/list.ts && git commit -m "feat: 任务页挂起(deferred)状态展示与取消/重置操作"
```

---

### Task 10: bat 脚本 + start.sh

**Files:**
- Create: `/home/zhenyi/ownprojects/transcribe-service/scripts/暂停转录.bat`
- Create: `/home/zhenyi/ownprojects/transcribe-service/scripts/恢复转录.bat`
- Modify: `/home/zhenyi/ownprojects/transcribe-service/start.sh`(第 18 行 `./run.sh` → `./run.sh --no-reload`)

**scripts/暂停转录.bat**(UTF-8 无 BOM;`<TOKEN>` 为 config.yaml api.token 值):

```bat
@echo off
chcp 65001 >nul
title 暂停转录服务
echo ==========================================
echo   转录服务暂停（到时自动恢复）
echo ==========================================
echo.
set /p HOURS=请输入暂停小时数，支持小数 [直接回车=2]：

if "%HOURS%"=="" set HOURS=2

echo %HOURS%|findstr /r "^[0-9][0-9.]*$" >nul
if errorlevel 1 (
    echo 输入无效：%HOURS%
    pause
    exit /b 1
)

echo.
echo 正在暂停服务...
curl.exe -s -X POST http://localhost:31080/pause -H "Authorization: Bearer <TOKEN>" -H "Content-Type: application/json" -d "{\"hours\": %HOURS%}"
echo.
echo.
echo （paused=true 即成功；本地释放按 local_release 字段说明在后台完成，最终状态可查 /status）
pause
```

**scripts/恢复转录.bat**:

```bat
@echo off
chcp 65001 >nul
title 恢复转录服务
echo 正在恢复转录服务...
echo.
curl.exe -s -X POST http://localhost:31080/resume -H "Authorization: Bearer <TOKEN>"
echo.
echo 当前状态：
curl.exe -s http://localhost:31080/status -H "Authorization: Bearer <TOKEN>"
echo.
echo.
echo （paused=false 即已恢复；模型将在下一个转录请求时自动加载）
pause
```

- [ ] **Step 1: 写两个 bat + 改 start.sh**(bat 中 token 一律写 `<TOKEN>` 占位——**真实 token 绝不入库**,Task 12 部署时替换到桌面副本;**顺手删除仓库根目录上一轮回退残留的 `暂停转录.bat`/`恢复转录.bat`(内含真实 token)**)

```bash
rm -f /home/zhenyi/ownprojects/transcribe-service/暂停转录.bat /home/zhenyi/ownprojects/transcribe-service/恢复转录.bat
```
- [ ] **Step 2: 验证 bat 编码**:`file scripts/*.bat` 显示 UTF-8(无 BOM);`grep -c '<TOKEN>' scripts/*.bat` 每文件 ≥1(占位在)
- [ ] **Step 3: 提交**

```bash
git add scripts/ start.sh && git commit -m "feat: 桌面暂停/恢复 bat;start.sh 切换 no-reload 根除假死隐患"
```

---

### Task 11: 本地全链路验证(spec §7 用例)

**Files:** 无代码变更;操作验证。**前置**:三项目代码全部就位。

- [x] **Step 1: 环境准备** ✅(mryk24 用 `docker compose up -d` 重建而非 restart——compose 补丁的 env 需重建才进容器)

```bash
# transcribe-service / asr-engine 以新代码重启
tmux kill-session -t transcribe; bash /home/zhenyi/ownprojects/transcribe-service/start.sh
tmux kill-session -t asr; cd /home/zhenyi/ownprojects/asr-engine && bash start.sh
# mryk24 本地:generate→build→重启容器
cd /home/zhenyi/ownprojects/mryk24 && pnpm prisma:generate && pnpm build && docker restart mryk24
```

- [x] **Step 2: 本地 noteflow 指向本地 transcribe + token** ✅(容器无 wget,改用容器内 node fetch 实测;⚠️ `funasr.apiToken` 须填 transcribe `server.token` 而非 `pause.notify_token`——前者是转录 API 鉴权,后者只用于唤醒端点;.env 的 token 行须独立成行,追加到注释行尾不生效)

```bash
# 容器→宿主可达地址实测(容器内 localhost 是容器自身!):
docker exec mryk24 sh -c 'wget -qO- http://<网关IP>:31080/status 2>&1 | head -c 100'   # 网关IP: docker network inspect mryk24_mryk24 | grep Gateway
# 本地 DB 配置(测完还原为 funasr.test):
docker exec mryk24-db mariadb -uroot -pexample mryk -e "UPDATE NoteFlowConfig SET value='http://<网关IP>:31080/transcribe_url' WHERE \`key\`='funasr.apiUrl'; UPDATE NoteFlowConfig SET value='<token>' WHERE \`key\`='funasr.apiToken';"
# 本地 .env 加 NUXT_NOTEFLOW_INTERNAL_TOKEN=<同 transcribe config.yaml pause.notify_token>;transcribe config.yaml pause.notify_url 填 http://<宿主可达mryk24地址>:10002/api/noteflow/internal/transcribe-wake
docker restart mryk24
```

- [x] **Step 3: 逐条跑 spec §7 用例 1-11** ✅ **11/11 通过**(9b/9c 按 brief 跳过实操;9d cancel-deferred 顺手实操通过)。BV 选取:BV1bban6dEJo(73s,有 AI 字幕→用例 3/4/9)、BV1x9ab6rE2C(152s,无字幕→用例 2/5);BV1HMaR6MEB5 实测 2845s 过长弃用。逐条证据见 `.superpowers/sdd/2026-09-30-transcribe-pause/task-11-report.md`。用例 4 注意:推送与兜底定时在 resume_at 同秒竞速,mryk24 日志可能是任一字样,均为正确路径

重点判定:
- 用例 2/3:deferred 后 `retryCount` 保持 0、error 含恢复时间、tasks.vue 显示"挂起"
- 用例 4:0.02h 窗口到期后,transcribe 日志出现"恢复通知已送达"、mryk24 日志出现"收到转录服务恢复通知"、任务自动 completed
- 用例 5:/resume 后 ≤5s 任务回到 processing
- 用例 6:临时把 notify_url 改错并重启 transcribe → 仍由 noteflow 兜底定时唤醒
- 用例 7:`docker restart mryk24`(挂起中)→ 自愈
- 用例 8:挂起中 `tmux kill-session -t transcribe && bash start.sh` → /status 仍 paused、通知定时重设(到期仍送达)
- 用例 9a:暂停后立即发转录请求(绕过中间件需直连 asr:先 /pause 再立刻 curl asr /v1/audio/transcriptions 收 503;UpstreamPausedError 路径用单测已覆盖,此处验证端到端一次)
- 用例 9b-9d:按 spec 描述模拟(9b:resume 后手动以旧 resume_at 构造 ServicePausedError 场景已由单测覆盖;此处跳过实操)

- [x] **Step 4: 还原本地配置** ✅(funasr.apiUrl/apiToken 已还原;另还原测试期临时改动:git.repoPath→obsidian、ai.model→deepseek-v4.1-flash:cloud、autoSubtitleDiarize 行删除、test-% 任务全删、/home/zhenyi/git/test 清理;.env token 保留)

- [x] **Step 5: 结果记录** ✅ 报告:`.superpowers/sdd/2026-09-30-transcribe-pause/task-11-report.md`(compose 补丁提交 mryk24@6e31405;concerns:token 两值区分/同秒竞速日志/生产 ai.model 402)

---

### Task 12: 部署与收尾(已授权:push / sync / tellme)

- [ ] **Step 1: 生产配置就位**

```bash
# mryk24 生产 env(.env.prod 已含 NUXT_NOTEFLOW_INTERNAL_TOKEN;sync 会同步)
# transcribe-service config.yaml:
#   pause.notify_url: https://<mryk24公网域名>/api/noteflow/internal/transcribe-wake
#   pause.notify_token: <同上 token>
# 验证公网可达:curl -s -X POST <notify_url> -H "Authorization: Bearer <token>" → {"isSuccess":true,...}
```

- [ ] **Step 2: 全量测试最后一跑**(三项目),全绿才继续
- [ ] **Step 3: push 三仓库**

```bash
for d in /home/zhenyi/ownprojects/transcribe-service /home/zhenyi/ownprojects/asr-engine /home/zhenyi/ownprojects/mryk24; do
  git -C $d push || echo "PUSH FAILED: $d"
done
```

(无 remote 的仓库记录跳过)

- [ ] **Step 4: mryk24 生产部署**

```bash
cd /home/zhenyi/ownprojects/mryk24 && bash sync.sh
# 部署后验证:ssh mryk "docker logs --tail 5 mryk24"(NoteFlow 启动正常)
```

- [ ] **Step 5: 重启本机两服务(生产切 no-reload)**

```bash
tmux kill-session -t transcribe; bash /home/zhenyi/ownprojects/transcribe-service/start.sh
tmux kill-session -t asr; cd /home/zhenyi/ownprojects/asr-engine && bash start.sh
# 实测进程参数:ps aux | grep -E 'server.py|asr_engine.server' 无 --reload
```

- [ ] **Step 6: bat 上桌面(替换真实 token,桌面副本不入库)**

```bash
TOKEN=$(grep -E '^\s*token:' /home/zhenyi/ownprojects/transcribe-service/config.yaml | head -1 | sed 's/.*token:\s*//')
for f in /home/zhenyi/ownprojects/transcribe-service/scripts/*.bat; do
  sed "s/<TOKEN>/$TOKEN/" "$f" > "/mnt/d/OneDrive/Desktop_home/$(basename "$f")"
done
grep -c "$TOKEN" /mnt/d/OneDrive/Desktop_home/*.bat   # 确认已替换
```

- [ ] **Step 7: 生产冒烟(spec §7 生产验证)**:暂停 0.03h → 公网 /transcribe_url 503 → 生产选一个已完成短视频任务重置 → 观察 deferred → 恢复 → completed → tellme:

```bash
tellme "暂停功能已部署完成:三项目已 push、mryk24 已 sync、bat 已上桌面,生产冒烟通过"
```

- [ ] **Step 8: 文档沉淀**:按项目规范在 transcribe-service `docs/2026-09-30_pause-endpoint.md` 写最终文档(上轮已删,重写,内容据实)+ CLAUDE.md 索引;mryk24 侧不改文档(联动说明并入前述文档)

---

## Self-Review 记录

- Spec 覆盖:§3(端点/状态/释放链/通知)→ Task 2/4/5;§3.3.1 → Task 4;§4 → Task 1;§5.1/5.2/5.3 → Task 6/7/8;§5.4 → Task 9;§6 → Task 10;§7 → Task 11/12;§8 → Task 12;§9 边界 → 用例 9a-9d + 各单测。无缺口
- 关键修订(review 驱动):asr 转发与本地释放链解耦(/pause 同步转发 asr,释放链仅本地);迟到 503 用 pause-state 恢复信号时刻判定(不止时间过期);asr 503 响应扁平化与 transcribe 一致;bat token 占位不入库;测试全程隔离不触真实服务;TaskCounts/NoteFlowTask.status 类型补齐
- 类型一致:PauseManager.generation(int)/should_unload/mark_unload_attempted;UpstreamPausedError.resume_at(str|None);ServicePausedError.resumeAt(Date|null);markTaskDeferred → 'deferred'|'fallback-failed';resumeDeferredTasks → number(scheduler 纯 DB 版 / pause-wake 包装版含 markResumeSignal,端点与测试用包装版)
- 占位符:bat 的 `<TOKEN>`/`<网关IP>`/`<BV>` 是实施时填充的运行时值,非代码占位 ✓
