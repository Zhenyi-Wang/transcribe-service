"""JobManager 执行 worker、取消生命周期、采样器测试（Task 7）。

spec：docs/superpowers/specs/2026-11-10-transcribe-job-queue-design.md
§4.3（取消——六形态矩阵与即时状态矩阵表）、§5（进度）、§6（执行模型全部条目）。

覆盖（brief Step 1 清单）：
1.  终态映射：响应 dict status=="error" → job error；否则 done（含 timing.download 注入）；
2.  queued 即时 cancelled + 取件竞态（陈旧就绪引用跳过）；
3.  独占 × 自串行（chunk 级取消）→ cancelling → join thread_future → cancelled；
    含分离变体（R4A fold）：ASR join 与分离进程回收顺序；
4.  共享 × 自串行 × 无分离 → 即时 cancelled（不 join、不动 entry）；
5.  共享 × 非自串行（capability legacy）→ cancelling + join 后才取下一件；
6.  分离冷却重试路径（_annotate_cached_asr，thread_future 恒空）→ 协程级取消（R1A）；
7.  diarize_only running 取消（无 entry 路径）→ 协程取消 → 分离进程回收（R1A）；
8.  遗留先创建 + job 加入 + 遗留离开 → job 独占取消命中 entry（R1A）；
9.  独占 × 非自串行放弃式 → join 自然结束 → 探针丢弃结果、不入缓存（R2A）；
10. 独占取消 + performance.cancelled → cancelled 且缓存写入未被调；
11. pause gate：暂停不取件，恢复后继续；
12. UpstreamPausedError → error + paused 三键；_upstream_pause_probe True 期间不取新件；
13. 采样器：本地双槽拼装；asr-engine 远端样本归属校验 + phase 应用时校验；
14. worker 自愈：worker 被外部 cancel → 当前 job 落 error 带原因 → 重拉；
15. capability 消费：负缓存内 legacy（percent 关）；注入时钟过 TTL 后重探恢复（R1B）；
16. 下载中取消 → executor 清理部分文件 → cancelled（R1A）；
17. upload 终态后内部键文件删除（R2A fold）；
18. 取消标志已置位但 UpstreamPausedError 先到 → 暂停语义胜出（error+paused
    三键，非 cancelled）；JobCancelledError 同交错仍 cancelled（Task 10，R3）；
19. timing.download 注入直接断言：注入/自带不覆盖/既有键保留/webdav 与
    未记录耗时不注入（Task 10，Task 7 评审 Minor-6 交接）。
20. fixwave F1：legacy capability 独占取消不发 scoped HTTP（标志+join 自然结束、
    不入缓存）；
21. fixwave F2：worker 自愈先 join thread_future（ASR 线程自然结束）再收尾——
    模型计数释放晚于线程结束，终态 error 不入缓存、不发 backend 信号；
22. fixwave F4：执行体经 phase_cb 写 formatting 槽（percent=100，spec §5）；
23. fixwave F5：取消路径探测/信号一律 to_thread，事件循环不被阻塞、锁内无网络调用。
"""
import asyncio
import os
import threading
import time
from unittest.mock import MagicMock

import pytest

import transcribe as T
from backends.asr_engine_backend import UpstreamPausedError
from backends.base import TranscribeResult
from cache_manager import CacheManager
from diarization.manager import SpeakerTurn
from downloaders.base import DownloadCancelled
from jobs import JobManager


# ---------- 配置桩 ----------

class StubConfig:
    def __init__(self, **values):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


def bili_params(diarize=False, diarize_only=False):
    return {"bvid": "BV1xx", "page": 1, "context": None, "diarize": diarize,
            "diarize_only": diarize_only, "body": None, "no_cache": False}


def wd(path, **over):
    p = {"path": path, "context": None, "diarize": False, "no_cache": False}
    p.update(over)
    return p


# ---------- 桩后端 ----------

class SelfSerialBackend:
    """gguf/asr-engine 语义桩：progress+scoped + 归属化 chunk 级取消。

    transcribe 线程阻塞在 per-request 门上；cancel_request 置取消标记并放行门，
    线程随即返回 performance.cancelled=True（模拟 chunk 边界停止）。
    """

    name = "stub-gguf"
    device = "cpu"

    def __init__(self):
        self.cancel_requests = []
        self.events = []                      # (事件名, time.monotonic())
        self._gates = {}
        self._cancelled = {}
        self._lock = threading.Lock()

    def probe_capability(self):
        return "progress+scoped"

    def _gate(self, request_id):
        with self._lock:
            return self._gates.setdefault(request_id, threading.Event())

    def transcribe(self, path, lang=None, context=None, progress_cb=None, request_id=None):
        self.events.append(("asr_start", time.monotonic()))
        self._gate(request_id).wait(timeout=5)
        perf = {"cancelled": True} if self._cancelled.get(request_id) else {}
        self.events.append(("asr_end", time.monotonic()))
        return TranscribeResult(
            text="甲说乙说", language="zh",
            timestamps=[{"text": c, "start": i * 2.0, "end": i * 2.0 + 1.0}
                        for i, c in enumerate("甲说乙说")],
            performance=perf)

    def cancel_request(self, request_id):
        self.cancel_requests.append(request_id)
        with self._lock:
            self._cancelled[request_id] = True
        self._gate(request_id).set()

    def release(self, request_id=None):
        """自然放行（不置取消标记）。"""
        with self._lock:
            ids = list(self._gates) if request_id is None else [request_id]
        for i in ids:
            self._gate(i).set()

    def event_mono(self, name):
        for n, mono in self.events:
            if n == name:
                return mono
        return None


class LegacyBackend:
    """qwen3/funasr 语义桩：legacy capability、无 chunk 钩子，取消靠自然结束。"""

    name = "stub-qwen3"
    device = "cpu"

    def __init__(self):
        self.cancel_requests = []
        self.events = []
        self.gate = threading.Event()

    def probe_capability(self):
        return "legacy"

    def transcribe(self, path, lang=None, context=None, progress_cb=None, request_id=None):
        self.events.append(("asr_start", time.monotonic()))
        self.gate.wait(timeout=5)     # 不响应 cancel_request，跑到自然结束
        self.events.append(("asr_end", time.monotonic()))
        return TranscribeResult(
            text="甲说乙说", language="zh",
            timestamps=[{"text": c, "start": i * 2.0, "end": i * 2.0 + 1.0}
                        for i, c in enumerate("甲说乙说")],
            performance={})

    def cancel_request(self, request_id):
        self.cancel_requests.append(request_id)   # 记录但无 chunk 级效果

    def release(self):
        self.gate.set()

    def event_mono(self, name):
        for n, mono in self.events:
            if n == name:
                return mono
        return None


class InstantBackend:
    """即时完成桩：缓存播种/annotate 等不需要阻塞控制时使用。"""

    name = "stub-gguf"
    device = "cpu"

    def __init__(self):
        self.cancel_requests = []

    def probe_capability(self):
        return "progress+scoped"

    def transcribe(self, path, lang=None, context=None, progress_cb=None, request_id=None):
        return TranscribeResult(text="甲说乙说", language="zh",
                                timestamps=[{"text": "甲", "start": 0.0, "end": 1.0}],
                                performance={})

    def cancel_request(self, request_id):
        self.cancel_requests.append(request_id)


class CapProbeBackend:
    """asr-engine 能力探测语义桩：404 负缓存 10min（注入时钟）→ TTL 后重探恢复。"""

    name = "asr-engine"
    device = "remote"

    def __init__(self, clock):
        self.clock = clock          # callable -> float（time.monotonic() 语义）
        self.http_calls = 0
        self._pos = False
        self._neg_until = None
        self._upgrade_pending = False   # True：下次真实探测返回 progress+scoped（模拟引擎升级）

    def probe_capability(self):
        if self._pos:
            return "progress+scoped"
        now = self.clock()
        if self._neg_until is not None and now < self._neg_until:
            return "legacy"
        self.http_calls += 1        # 真实探测（负缓存命中不发生）
        if self._upgrade_pending:
            self._pos = True
            self._neg_until = None
            return "progress+scoped"
        self._neg_until = now + 600  # 404 → 负缓存 10min
        return "legacy"


# ---------- 公共装置 ----------

@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(type(T.config), "diarization_enabled", property(lambda self: True))
    monkeypatch.setattr(type(T.config), "cache_dir", property(lambda self: str(tmp_path / "cache")))
    monkeypatch.setattr(type(T.config), "cache_enabled", property(lambda self: True))
    monkeypatch.setattr(T, "cache_manager", CacheManager())
    monkeypatch.setattr(T, "get_audio_duration", lambda path: 8.0)
    return tmp_path


def make_service(backend):
    manager = MagicMock()
    manager.load_model_if_needed.return_value = backend
    return T.TranscriptionService(manager)


def build_mgr(backend, service, factory, **over):
    mgr = JobManager()
    mgr._executor_factory = factory
    mgr._cancel_computation = service.cancel_computation if service else None
    mgr._entry_lookup = service._inflight.get if service else None
    mgr._backend_lookup = (lambda: backend) if backend is not None else None
    mgr.sample_interval = 0.05
    for k, v in over.items():
        setattr(mgr, k, v)
    return mgr


def service_factory(service, path, *, diarize=False, file_path_for_cache=None,
                    audio_url=None, bvid=None, audio_id=None, backend_name=None):
    """按真实形态构建执行体：先回填 inflight_key（与 process_transcription 同参），
    再经 sinks 以队列参数调用 process_transcription。"""
    async def factory(job, sinks):
        if backend_name:
            sinks["set_backend_name"](backend_name)
        key = service.build_inflight_key(path, audio_url=audio_url, bvid=bvid,
                                         audio_id=audio_id,
                                         file_path_for_cache=file_path_for_cache,
                                         context=None, diarize=diarize, no_cache=False)
        sinks["set_inflight_key"](key)
        sinks["phase"]("transcribing")
        return await service.process_transcription(
            path, audio_url=audio_url, bvid=bvid, audio_id=audio_id,
            file_path_for_cache=file_path_for_cache, context=None, diarize=diarize,
            progress_cb=sinks["progress"], cancelled=sinks["cancelled"],
            diarize_wait_deadline=sinks["diarize_wait_deadline"], stage_cb=sinks["stage"])
    return factory


def simple_factory(kinds=None):
    """按 request['kind'] 分派的简单桩执行体：ok=成功，err=error dict，
    block=阻塞（asyncio 门），up=抛 UpstreamPausedError。"""
    async def factory(job, sinks):
        kind = (job.request or {}).get("kind", "ok")
        if kind == "ok":
            return {"status": "success", "body": [{"text": "hi"}]}
        if kind == "err":
            return {"status": "error", "message": "boom"}
        if kind == "block":
            await (job.request["gate"]).wait()
            return {"status": "success", "body": []}
        if kind == "up":
            raise UpstreamPausedError("engine paused",
                                      resume_at="2026-11-10T10:00:00",
                                      paused_at="2026-11-10T09:00:00")
        raise AssertionError(f"unknown kind {kind}")
    return factory


async def wait_pred(pred, timeout=3.0):
    for _ in range(int(timeout * 100)):
        if pred():
            return True
        await asyncio.sleep(0.01)
    raise AssertionError("条件未在期限内满足")


async def wait_entry(service, timeout=3.0):
    for _ in range(int(timeout * 100)):
        for entry in service._inflight.values():
            return entry
        await asyncio.sleep(0.01)
    raise AssertionError("entry 未注册")


async def wait_entry_asr(service, timeout=3.0):
    for _ in range(int(timeout * 100)):
        for entry in service._inflight.values():
            if entry["asr_started"] and entry["thread_future"] is not None:
                return entry
        await asyncio.sleep(0.01)
    raise AssertionError("entry 未进入 ASR 执行")


def install_diar_stub(monkeypatch, gate, reaped):
    """_diarize_samples 桩：发 stage、阻塞在 asyncio 门上；取消路径记录回收。"""
    turns = [SpeakerTurn(0.0, 8.0, 0), SpeakerTurn(2.0, 8.0, 1)]

    async def stub(audio_path, timeout, stage_cb=None):
        if stage_cb:
            stage_cb("decoding")
        try:
            await gate.wait()
        except asyncio.CancelledError:
            reaped.append(True)
            raise
        return turns

    monkeypatch.setattr(T, "_diarize_samples", stub)


# ---------- 1. 终态映射 ----------

@pytest.mark.asyncio
async def test_status_error_mapping():
    mgr = build_mgr(None, None, simple_factory())
    e = await mgr.submit("webdav", wd("a", kind="err"), job_id="e")
    s = await mgr.submit("webdav", wd("b", kind="ok"), job_id="s")
    await mgr.start()
    await wait_pred(lambda: mgr._jobs["e"].status == "error")
    await wait_pred(lambda: mgr._jobs["s"].status == "done")
    je, js = mgr._jobs["e"], mgr._jobs["s"]
    assert je.result == {"status": "error", "message": "boom"}
    assert je.finished_at is not None and je.started_at is not None
    assert js.result == {"status": "success", "body": [{"text": "hi"}]}
    assert e["job_id"] == "e" and s["job_id"] == "s"
    assert mgr.get("e")["status"] == "error" and mgr.get("s")["status"] == "done"
    await mgr.stop()


# ---------- 2. queued 取消 + 取件竞态 ----------

@pytest.mark.asyncio
async def test_cancel_queued_and_take_race():
    gate = asyncio.Event()
    mgr = build_mgr(None, None, simple_factory())
    j1 = await mgr.submit("webdav", wd("1", kind="block", gate=gate), job_id="j1")
    await mgr.start()
    await wait_pred(lambda: mgr._jobs["j1"].status == "running")
    j2 = await mgr.submit("webdav", wd("2", kind="ok"), job_id="j2")
    j3 = await mgr.submit("webdav", wd("3", kind="ok"), job_id="j3")
    assert mgr.get(j3["job_id"])["queue_position"] == 2

    res = await mgr.cancel(j2["job_id"])
    assert res == {"job_id": j2["job_id"], "status": "cancelled"}
    assert mgr.get(j3["job_id"])["queue_position"] == 1
    assert j2["job_id"] not in mgr._ready

    # 取件竞态：陈旧就绪引用（cancel 后残留）被 worker 取件时跳过，绝不 mark_running
    mgr._ready.append(j2["job_id"])
    gate.set()
    await wait_pred(lambda: mgr._jobs["j3"].status == "done")
    assert mgr._jobs[j2["job_id"]].status == "cancelled", "陈旧引用不得复活"
    assert j2["job_id"] not in mgr._ready, "陈旧引用已被取件循环清出"
    assert mgr._jobs["j1"].status == "done"
    await mgr.stop()


# ---------- 3. 独占 × 自串行（含分离变体，R4A fold） ----------

@pytest.mark.asyncio
async def test_cancel_running_sole_self_serial(env, monkeypatch):
    backend = SelfSerialBackend()
    service = make_service(backend)
    path = str(env / "a.wav")
    open(path, "wb").write(b"audio")
    saves = []
    monkeypatch.setattr(T.cache_manager, "save_transcript_to_cache",
                        lambda *a, **k: saves.append(k))
    mgr = build_mgr(backend, service, service_factory(service, path,
                                                      file_path_for_cache=path))
    r = await mgr.submit("webdav", wd(path), job_id="j")
    await mgr.start()
    entry = await wait_entry_asr(service)

    res = await mgr.cancel(r["job_id"])
    assert res == {"job_id": r["job_id"], "status": "cancelling"}, "即时返回 cancelling"
    assert mgr._jobs[r["job_id"]].status == "cancelling"
    await wait_pred(lambda: backend.cancel_requests == [entry["request_id"]])  # join 协程先跑

    await wait_pred(lambda: mgr._jobs[r["job_id"]].status == "cancelled")
    assert saves == [], "取消不得写缓存"
    assert service._inflight == {}, "entry 已回收"
    # chunk 级取消：ASR 线程早于终态结束（join 语义）
    assert backend.event_mono("asr_end") < mgr._jobs[r["job_id"]].finished_mono
    await mgr.stop()


@pytest.mark.asyncio
async def test_cancel_running_sole_self_serial_with_diarize(env, monkeypatch):
    """running·含分离 → join 行：ASR join 与分离进程回收均先于终态。"""
    backend = SelfSerialBackend()
    service = make_service(backend)
    path = str(env / "a.wav")
    open(path, "wb").write(b"audio")
    saves = []
    monkeypatch.setattr(T.cache_manager, "save_transcript_to_cache",
                        lambda *a, **k: saves.append(k))
    diar_gate = asyncio.Event()
    reaped = []
    install_diar_stub(monkeypatch, diar_gate, reaped)
    mgr = build_mgr(backend, service, service_factory(service, path,
                                                      file_path_for_cache=path, diarize=True))
    r = await mgr.submit("bilibili", bili_params(diarize=True), job_id="j")
    await mgr.start()
    entry = await wait_entry_asr(service)
    await mgr.cancel(r["job_id"])
    await wait_pred(lambda: mgr._jobs[r["job_id"]].status == "cancelled")
    job = mgr._jobs[r["job_id"]]
    assert backend.cancel_requests == [entry["request_id"]]
    assert reaped == [True], "分离子任务经既有 except 链回收"
    assert saves == [], "部分结果不入缓存"
    assert backend.event_mono("asr_end") < job.finished_mono, "先 join ASR 线程"
    assert job.result is None, "cancelled 终态无 result"
    await mgr.stop()


# ---------- 4. 共享 × 自串行 × 无分离 → 即时 cancelled（不 join） ----------

@pytest.mark.asyncio
async def test_cancel_running_shared_self_serial(env):
    backend = SelfSerialBackend()
    service = make_service(backend)
    path = str(env / "a.wav")
    open(path, "wb").write(b"audio")
    # 遗留等待者先创建计算
    legacy = asyncio.create_task(
        service.process_transcription(path, file_path_for_cache=path, diarize=False))
    entry = await wait_entry_asr(service)
    assert entry["waiters"] == 1

    mgr = build_mgr(backend, service, service_factory(service, path,
                                                      file_path_for_cache=path))
    r = await mgr.submit("webdav", wd(path), job_id="j")
    await mgr.start()
    await wait_pred(lambda: entry["waiters"] == 2)

    res = await mgr.cancel(r["job_id"])
    assert res == {"job_id": r["job_id"], "status": "cancelled"}, "共享×自串行即时终态"
    assert mgr._jobs[r["job_id"]].status == "cancelled"
    assert not entry["cancel_flag"].is_set(), "共享路径不置 entry 标志"
    assert backend.cancel_requests == [], "共享路径不调 cancel_computation"
    assert all(n != "asr_end" for n, _ in backend.events), "即时终态不 join：计算仍在运行"
    # 计算完整跑完给遗留等待者
    backend.release()
    legacy_result = await asyncio.wait_for(legacy, 5)
    assert legacy_result["status"] == "success"
    assert mgr._jobs[r["job_id"]].status == "cancelled"
    await mgr.stop()


# ---------- 5. 共享 × 非自串行 → cancelling + join 后才取下一件 ----------

@pytest.mark.asyncio
async def test_cancel_running_shared_nonserial(env):
    backend = LegacyBackend()
    service = make_service(backend)
    path = str(env / "a.wav")
    open(path, "wb").write(b"audio")
    legacy = asyncio.create_task(
        service.process_transcription(path, file_path_for_cache=path, diarize=False))
    entry = await wait_entry_asr(service)

    markers = []

    async def factory(job, sinks):
        if (job.request or {}).get("kind") == "join_me":
            return await service_factory(service, path, file_path_for_cache=path)(job, sinks)
        markers.append(("next_start", time.monotonic()))
        return {"status": "success"}

    mgr = build_mgr(backend, service, factory)
    r = await mgr.submit("webdav", wd(path, kind="join_me"), job_id="j")
    j2 = await mgr.submit("webdav", wd("next", kind="ok"), job_id="j2")
    await mgr.start()
    await wait_pred(lambda: entry["waiters"] == 2)

    res = await mgr.cancel(r["job_id"])
    assert res == {"job_id": r["job_id"], "status": "cancelling"}, "共享×legacy → cancelling"
    await asyncio.sleep(0.15)
    assert mgr._jobs[r["job_id"]].status == "cancelling", "计算退出前保持 cancelling"
    assert mgr._jobs[j2["job_id"]].status == "queued", "join 完成前 worker 不取下一件"
    assert not entry["cancel_flag"].is_set(), "共享路径不置标志"

    backend.release()
    await wait_pred(lambda: mgr._jobs[r["job_id"]].status == "cancelled")
    await wait_pred(lambda: mgr._jobs[j2["job_id"]].status == "done")
    legacy_result = await asyncio.wait_for(legacy, 5)
    assert legacy_result["status"] == "success"
    asr_end = backend.event_mono("asr_end")
    assert asr_end < mgr._jobs[r["job_id"]].finished_mono
    assert markers and markers[0][1] > asr_end, "下一件在计算退出（asr_end）之后才开始"
    await mgr.stop()


# ---------- 6. 分离冷却重试路径（annotate，thread_future 恒空）→ 协程级取消（R1A） ----------

@pytest.mark.asyncio
async def test_cancel_annotate_running(env, monkeypatch):
    backend = InstantBackend()
    service = make_service(backend)
    path = str(env / "a.wav")
    open(path, "wb").write(b"audio")
    # 播种 ASR 基线缓存（diarize=False 正常转录一次）
    await service.process_transcription(path, file_path_for_cache=path, diarize=False)
    assert T.cache_manager.get_cached_transcript(file_path=path, diarize=False, include_asr=True)

    saves = []
    monkeypatch.setattr(T.cache_manager, "save_transcript_to_cache",
                        lambda *a, **k: saves.append(k))
    diar_gate = asyncio.Event()
    reaped = []
    install_diar_stub(monkeypatch, diar_gate, reaped)

    mgr = build_mgr(backend, service, service_factory(service, path,
                                                      file_path_for_cache=path, diarize=True))
    r = await mgr.submit("bilibili", bili_params(diarize=True), job_id="j")
    await mgr.start()
    entry = await wait_entry(service)
    await wait_pred(lambda: mgr._jobs[r["job_id"]].status == "running")

    res = await mgr.cancel(r["job_id"])
    assert res["status"] == "cancelling"
    assert backend.cancel_requests == [], "未开始 ASR 的取消不得向 pending-cancel 写永不消费的条目"
    await wait_pred(lambda: mgr._jobs[r["job_id"]].status == "cancelled")
    assert reaped == [True], "协程级取消 → 分离进程回收"
    assert saves == [], "取消不得写分离降级缓存"
    assert service._inflight == {}, "entry 已随拥有者任务收尾移除"
    await mgr.stop()


# ---------- 7. diarize_only running 取消（无 entry 路径，R1A） ----------

@pytest.mark.asyncio
async def test_cancel_diarize_only_running(env, monkeypatch):
    backend = InstantBackend()
    service = make_service(backend)
    path = str(env / "a.wav")
    open(path, "wb").write(b"audio")
    diar_gate = asyncio.Event()
    reaped = []
    install_diar_stub(monkeypatch, diar_gate, reaped)

    cancel_calls = []

    def traced_cancel(key):
        cancel_calls.append(key)
        return service.cancel_computation(key)

    async def factory(job, sinks):
        sinks["phase"]("formatting")   # 下载段已过，处于分离段
        return await T.diarize_only(path, video_id="BV1xx",
                                    wait_deadline=sinks["diarize_wait_deadline"],
                                    stage_cb=sinks["stage"])

    mgr = build_mgr(backend, service, factory)
    mgr._cancel_computation = traced_cancel
    r = await mgr.submit("bilibili", bili_params(diarize_only=True), job_id="j")
    await mgr.start()
    await wait_pred(lambda: mgr._jobs[r["job_id"]].status == "running")

    assert (await mgr.cancel(r["job_id"]))["status"] == "cancelling"
    await wait_pred(lambda: mgr._jobs[r["job_id"]].status == "cancelled")
    assert reaped == [True], "分离进程经 IPC await 取消立即回收"
    assert cancel_calls == [], "无 entry 路径不得调 cancel_computation"
    await mgr.stop()


# ---------- 8. 遗留先创建 + job 加入 + 遗留离开 → job 独占取消命中 entry（R1A） ----------

@pytest.mark.asyncio
async def test_cancel_legacy_created_shared(env):
    backend = SelfSerialBackend()
    service = make_service(backend)
    path = str(env / "a.wav")
    open(path, "wb").write(b"audio")
    legacy = asyncio.create_task(
        service.process_transcription(path, file_path_for_cache=path, diarize=False))
    entry = await wait_entry_asr(service)

    mgr = build_mgr(backend, service, service_factory(service, path,
                                                      file_path_for_cache=path))
    r = await mgr.submit("webdav", wd(path), job_id="j")
    await mgr.start()
    await wait_pred(lambda: entry["waiters"] == 2)

    legacy.cancel()                     # 遗留离开：job 成为唯一等待者
    with pytest.raises(asyncio.CancelledError):
        await legacy
    await wait_pred(lambda: entry["waiters"] == 1)

    assert (await mgr.cancel(r["job_id"]))["status"] == "cancelling"
    await wait_pred(lambda: backend.cancel_requests == [entry["request_id"]])  # join 协程先跑
    await wait_pred(lambda: mgr._jobs[r["job_id"]].status == "cancelled")
    assert service._inflight == {}
    await mgr.stop()


# ---------- 9. 独占 × 非自串行放弃式（R2A） ----------

@pytest.mark.asyncio
async def test_cancel_running_sole_nonserial(env, monkeypatch):
    backend = LegacyBackend()
    service = make_service(backend)
    path = str(env / "a.wav")
    open(path, "wb").write(b"audio")
    saves = []
    monkeypatch.setattr(T.cache_manager, "save_transcript_to_cache",
                        lambda *a, **k: saves.append(k))
    mgr = build_mgr(backend, service, service_factory(service, path,
                                                      file_path_for_cache=path))
    r = await mgr.submit("webdav", wd(path), job_id="j")
    await mgr.start()
    entry = await wait_entry_asr(service)

    assert (await mgr.cancel(r["job_id"]))["status"] == "cancelling"
    await wait_pred(lambda: entry["cancel_flag"].is_set())
    await asyncio.sleep(0.15)
    assert mgr._jobs[r["job_id"]].status == "cancelling", "放弃式：join 自然结束前保持 cancelling"
    assert all(n != "asr_end" for n, _ in backend.events), "线程未受 chunk 级中断"

    backend.release()                   # 自然跑完
    await wait_pred(lambda: mgr._jobs[r["job_id"]].status == "cancelled")
    assert backend.event_mono("asr_end") < mgr._jobs[r["job_id"]].finished_mono
    assert saves == [], "探针按 entry.cancel_flag 丢弃结果，不入缓存"
    await mgr.stop()


# ---------- 10. 独占取消 + performance.cancelled → 缓存写入未被调 ----------

@pytest.mark.asyncio
async def test_probe_blocks_cache(env, monkeypatch):
    backend = SelfSerialBackend()
    service = make_service(backend)
    path = str(env / "a.wav")
    open(path, "wb").write(b"audio")
    saves = []
    monkeypatch.setattr(T.cache_manager, "save_transcript_to_cache",
                        lambda *a, **k: saves.append(k))
    mgr = build_mgr(backend, service, service_factory(service, path,
                                                      file_path_for_cache=path))
    r = await mgr.submit("webdav", wd(path), job_id="j")
    await mgr.start()
    await wait_entry_asr(service)
    await mgr.cancel(r["job_id"])
    await wait_pred(lambda: mgr._jobs[r["job_id"]].status == "cancelled")
    assert saves == [], "缓存写入函数未被调"
    assert T.cache_manager.get_cached_transcript(file_path=path, diarize=False) is None, \
        "缓存无部分结果残留"
    await mgr.stop()


# ---------- 11. pause gate ----------

@pytest.mark.asyncio
async def test_pause_gate():
    paused = [True]
    mgr = build_mgr(None, None, simple_factory())
    mgr._pause_gate = lambda: paused[0]
    r1 = await mgr.submit("webdav", wd("1", kind="ok"), job_id="j1")
    await mgr.start()
    await asyncio.sleep(0.15)
    assert mgr._jobs["j1"].status == "queued", "暂停期间不取件"

    paused[0] = False
    await wait_pred(lambda: mgr._jobs["j1"].status == "done", timeout=5)

    paused[0] = True
    r2 = await mgr.submit("webdav", wd("2", kind="ok"), job_id="j2")
    await asyncio.sleep(0.15)
    assert mgr._jobs["j2"].status == "queued", "取件门逐次生效"
    paused[0] = False
    await wait_pred(lambda: mgr._jobs["j2"].status == "done", timeout=5)
    await mgr.stop()


# ---------- 12. 上游暂停：error 三键 + 自冷 ----------

@pytest.mark.asyncio
async def test_upstream_paused_selfcool():
    upstream_paused = [False]
    mgr = build_mgr(None, None, simple_factory())
    mgr._upstream_pause_probe = lambda: upstream_paused[0]
    up = await mgr.submit("webdav", wd("u", kind="up"), job_id="up")
    await mgr.start()
    await wait_pred(lambda: mgr._jobs["up"].status == "error")
    job = mgr._jobs["up"]
    assert job.result["status"] == "error"
    assert job.result["paused"] is True
    assert job.result["resume_at"] == "2026-11-10T10:00:00"
    assert job.result["paused_at"] == "2026-11-10T09:00:00"

    upstream_paused[0] = True
    nxt = await mgr.submit("webdav", wd("n", kind="ok"), job_id="n")
    await asyncio.sleep(0.15)
    assert mgr._jobs["n"].status == "queued", "上游暂停期间自冷不取新件"

    upstream_paused[0] = False
    await wait_pred(lambda: mgr._jobs["n"].status == "done", timeout=5)
    await mgr.stop()


# ---------- 12b. 取消标志已置位但 UpstreamPausedError 先到 → 暂停语义胜出（Task 10，R3） ----------

@pytest.mark.asyncio
async def test_pause_signal_wins_over_cancel_flag():
    """取消期间 503 穿透回归（spec §9/评审 R3）：取消标志已在执行体侧置位
    （用户点了取消，取消状态机尚未接管终态），执行体先收到 UpstreamPausedError
    → 暂停语义胜出：error + paused 三键（非 cancelled、非泛化 error dict）；
    JobCancelledError 同交错不受影响（仍 cancelled）。"""
    async def factory(job, sinks):
        kind = (job.request or {}).get("kind", "pause_win")
        job.cancel_event.set()          # 取消的一瞬：标志已置位
        await asyncio.sleep(0)
        if kind == "pause_win":
            raise UpstreamPausedError("asr engine paused",
                                      resume_at="2026-11-10T10:00:00",
                                      paused_at="2026-11-10T09:00:00")
        raise T.JobCancelledError("cancelled before cache write")

    mgr = build_mgr(None, None, factory)
    await mgr.submit("webdav", wd("u", kind="pause_win"), job_id="up")
    await mgr.submit("webdav", wd("c", kind="cancel_win"), job_id="cn")
    await mgr.start()
    await wait_pred(lambda: mgr._jobs["up"].status == "error")
    await wait_pred(lambda: mgr._jobs["cn"].status == "cancelled")

    job = mgr._jobs["up"]
    assert job.result["status"] == "error"
    assert job.result["paused"] is True
    assert job.result["resume_at"] == "2026-11-10T10:00:00"
    assert job.result["paused_at"] == "2026-11-10T09:00:00"
    assert job.result["message"] == "asr engine paused", "暂停信号原样穿透"
    assert mgr._jobs["cn"].result is None, "JobCancelledError 路径终态不变"
    await mgr.stop()


# ---------- 12c. timing.download 注入直接断言（Task 7 评审 Minor-6 交接） ----------

@pytest.mark.asyncio
async def test_timing_download_injection():
    """终态映射的 timing.download 注入直接测试：sinks note_download_time →
    bilibili/douyin 成功响应注入；响应自带 download 不被覆盖（setdefault）、
    既有 timing 键保留；webdav 不注入；未记录耗时不注入。"""
    async def factory(job, sinks):
        kind = (job.request or {}).get("kind")
        if kind == "note":
            sinks["note_download_time"](2.5)
            return {"status": "success", "body": [], "timing": {"total": 3.2}}
        if kind == "preset":
            sinks["note_download_time"](2.5)
            return {"status": "success", "body": [], "timing": {"download": 9.9}}
        if kind == "note_plain":
            sinks["note_download_time"](2.5)
            return {"status": "success", "body": []}
        if kind == "no_note":
            return {"status": "success", "body": []}
        raise AssertionError(f"unknown kind {kind}")

    mgr = build_mgr(None, None, factory)
    await mgr.submit("bilibili", dict(bili_params(), kind="note"), job_id="b")
    await mgr.submit("bilibili", dict(bili_params(), kind="preset"), job_id="p")
    await mgr.submit("douyin", {"aweme_id": "1234567890123456789", "kind": "note",
                                "context": None, "diarize": False, "no_cache": False},
                     job_id="d")
    await mgr.submit("webdav", wd("w", kind="note_plain"), job_id="w")
    await mgr.submit("bilibili", dict(bili_params(), kind="no_note"), job_id="n")
    await mgr.start()
    await wait_pred(lambda: all(mgr._jobs[j].status == "done"
                                for j in ("b", "p", "d", "w", "n")))

    rb = mgr._jobs["b"].result
    assert rb["timing"]["download"] == 2.5, "note_download_time → 终态 timing.download"
    assert rb["timing"]["total"] == 3.2, "响应既有 timing 键保留"
    assert mgr._jobs["p"].result["timing"]["download"] == 9.9, "响应自带 download 不被覆盖"
    assert mgr._jobs["d"].result["timing"]["download"] == 2.5, "douyin 同样注入"
    assert "timing" not in mgr._jobs["w"].result, "webdav 不注入 download"
    assert "timing" not in mgr._jobs["n"].result, "未记录下载耗时不注入"
    await mgr.stop()


# ---------- 13. 采样器：本地双槽 + 远端归属/phase 校验 ----------

@pytest.mark.asyncio
async def test_sampler_slots(env):
    # (a) 本地：执行体写双槽 → GET 拼装（gguf 语义：progress+scoped → percent 开）
    hold = asyncio.Event()

    async def local_factory(job, sinks):
        sinks["phase"]("transcribing")
        sinks["progress"](30, 100)
        sinks["stage"]("embedding")
        await hold.wait()
        return {"status": "success"}

    mgr = build_mgr(InstantBackend(), None, local_factory, sample_interval=0.05)
    r = await mgr.submit("webdav", wd("a"), job_id="loc")
    await mgr.start()
    await wait_pred(lambda: mgr.get("loc")["percent"] == 30.0)
    v = mgr.get("loc")
    assert v["phase"] == "transcribing" and v["message"] == "30/100"
    assert v["diarize_stage"] == "embedding", "分离槽独立拼装"
    hold.set()
    await wait_pred(lambda: mgr._jobs["loc"].status == "done")
    await mgr.stop()

    # (b) 远端：asr-engine 采样，request_id 归属 + phase 应用时校验
    # （SelfSerialBackend 阻塞 ASR 线程，job 在采样断言期间保持 running）
    backend = SelfSerialBackend()
    service = make_service(backend)
    path = str(env / "a.wav")
    open(path, "wb").write(b"audio")
    box = {"v": None}
    polls = []

    async def poll(job, request_id):
        polls.append(request_id)
        return box["v"]

    factory = service_factory(service, path, file_path_for_cache=path,
                              backend_name="asr-engine")
    mgr2 = build_mgr(backend, service, factory, sample_interval=0.05)
    mgr2._progress_poll = poll
    r2 = await mgr2.submit("webdav", wd(path), job_id="rem")
    await mgr2.start()
    entry = await wait_entry_asr(service)
    rid = entry["request_id"]

    # request_id 不匹配 → 不采信
    box["v"] = {"active": True, "request_id": "OTHER", "chunk": 5, "total": 10}
    await asyncio.sleep(0.2)
    assert polls, "采样器在轮询"
    assert mgr2.get("rem")["percent"] is None

    # 匹配但本地 phase 非 transcribing（应用时校验）→ 不应用
    job = mgr2._jobs["rem"]
    job.transcript_slot = {"phase": "downloading"}
    box["v"] = {"active": True, "request_id": rid, "chunk": 5, "total": 10}
    await asyncio.sleep(0.2)
    assert mgr2.get("rem")["percent"] is None
    assert mgr2.get("rem")["phase"] == "downloading", "迟到样本不得拉低 phase"

    # 匹配且 phase=transcribing → 应用（仅 percent/message，phase 保持本地值）
    job.transcript_slot = {"phase": "transcribing"}
    await asyncio.sleep(0.2)
    assert mgr2.get("rem")["percent"] == 50.0
    assert mgr2.get("rem")["message"] == "5/10"
    assert mgr2.get("rem")["phase"] == "transcribing"

    # inactive → 本轮忽略，已应用值保持
    box["v"] = {"active": False, "request_id": rid, "chunk": 9, "total": 10}
    await asyncio.sleep(0.2)
    assert mgr2.get("rem")["percent"] == 50.0

    backend.release(entry["request_id"])   # 放行 → 正常完成
    await wait_pred(lambda: mgr2._jobs["rem"].status == "done", timeout=6)
    await mgr2.stop()


# ---------- 14. worker 自愈 ----------

@pytest.mark.asyncio
async def test_worker_selfheal():
    gate = asyncio.Event()
    mgr = build_mgr(None, None, simple_factory())
    r1 = await mgr.submit("webdav", wd("1", kind="block", gate=gate), job_id="j1")
    await mgr.start()
    await wait_pred(lambda: mgr._jobs["j1"].status == "running")
    dead_worker = mgr._worker_task

    dead_worker.cancel()                # 外部取消 worker
    await wait_pred(lambda: mgr._jobs["j1"].status == "error")
    job = mgr._jobs["j1"]
    assert "worker" in (job.result or {}).get("message", ""), "落 error 带原因"
    assert job.finished_at is not None
    assert mgr._worker_task is not None and mgr._worker_task is not dead_worker, "已重拉"
    assert not mgr._worker_task.done()
    assert mgr._current_job is None, "无遗留引用"

    r2 = await mgr.submit("webdav", wd("2", kind="ok"), job_id="j2")
    await wait_pred(lambda: mgr._jobs["j2"].status == "done", timeout=5)
    assert mgr._jobs["j2"].status == "done", "新 worker 正常取件"
    await mgr.stop()


# ---------- 14b. stop() 对活跃 job 走取消语义收尾（评审修复轮 1，Important-1） ----------

@pytest.mark.asyncio
async def test_stop_finalizes_running_job(env):
    backend = SelfSerialBackend()
    service = make_service(backend)
    path = str(env / "a.wav")
    open(path, "wb").write(b"audio")
    mgr = build_mgr(backend, service, service_factory(service, path,
                                                      file_path_for_cache=path))
    r = await mgr.submit("webdav", wd(path), job_id="j")
    j2 = await mgr.submit("webdav", wd("next", kind="ok"), job_id="j2")
    await mgr.start()
    entry = await wait_entry_asr(service)

    await mgr.stop()    # running 期间直接停
    job = mgr._jobs[r["job_id"]]
    assert job.status == "cancelled", "stop 收尾 running job → cancelled"
    assert "service stopped" in (job.message or ""), "message 注明 service stopped"
    assert job.finished_at is not None
    assert backend.cancel_requests == [entry["request_id"]], "复用 cancel_computation 归属化取消"
    assert service._inflight == {}, "计算 entry 已回收（无孤儿执行体持模型）"
    assert mgr._worker_task is None and mgr._current_job is None
    assert j2["job_id"] in mgr._ready, "queued 任务保持排队"

    # 收尾后可安全重启：新 worker 取下一件
    await mgr.start()
    await wait_pred(lambda: mgr._jobs["j2"].status == "done", timeout=5)
    await mgr.stop()


@pytest.mark.asyncio
async def test_stop_finalizes_cancelling_and_idle():
    # cancelling 中途 stop：等既有 join 收尾，不留活 job
    backend = LegacyBackend()
    service = make_service(backend)
    mgr = build_mgr(backend, service, simple_factory())
    mgr._executor_factory = None        # 手工构造 running/cancelling 态：不拉 worker
    r = await mgr.submit("webdav", wd("1"), job_id="j")
    job = mgr._jobs["j"]
    job.status = "running"
    job.exec_task = asyncio.create_task(asyncio.sleep(5))

    async def fake_join():
        await asyncio.sleep(0.05)
        if job.status == "cancelling":
            job.status = "cancelled"    # 模拟 join 完成落态
    job.status = "cancelling"
    job.cancel_join_task = asyncio.create_task(fake_join())
    mgr._current_job = job              # stop 收尾以 _current_job 为锚

    await mgr.stop()
    assert job.status == "cancelled"
    assert job.cancel_join_task.done()

    # idle stop（无活跃 job）幂等 + 重启语义
    mgr2 = build_mgr(None, None, simple_factory())
    r2 = await mgr2.submit("webdav", wd("x"), job_id="q")
    await mgr2.start()
    await mgr2.stop()
    assert mgr2._jobs["q"].status == "queued", "queued 任务 stop 后保持排队"
    await mgr2.start()
    await wait_pred(lambda: mgr2._jobs["q"].status == "done", timeout=5)
    await mgr2.stop()


# ---------- 15. capability 消费：负缓存 + 注入时钟重探恢复（R1B） ----------

@pytest.mark.asyncio
async def test_probe_capability_cache():
    clock = [1000.0]
    backend = CapProbeBackend(lambda: clock[0])
    mgr = JobManager()
    mgr._backend_lookup = lambda: backend

    assert mgr._capability() == "legacy"
    assert mgr._capability() == "legacy"
    assert backend.http_calls == 1, "负缓存内不重复探测（消费 Task 5 后端缓存）"

    # legacy 态：percent 关（progress sink 不写 percent）
    from jobs import Job
    job = Job(job_id="t", source="webdav", request={})
    sinks = mgr._make_sinks(job)
    sinks["progress"](5, 10)
    assert job.transcript_slot.get("percent") is None, "legacy 态 percent 关闭"
    assert job.transcript_slot.get("message") == "5/10"

    # 负缓存 10min 过期（注入时钟）→ 重探 → 引擎已升级 → 恢复 progress+scoped
    backend._upgrade_pending = True      # 模拟引擎侧升级（TTL 到期后探测到 /v1/progress）
    clock[0] += 610
    assert mgr._capability() == "progress+scoped"
    assert backend.http_calls == 2
    sinks["progress"](5, 10)
    assert job.transcript_slot.get("percent") == 50.0, "恢复后 percent 开启"

    # 无 backend / 探测异常 → legacy 兜底
    mgr2 = JobManager()
    mgr2._backend_lookup = None
    assert mgr2._capability() == "legacy"


# ---------- 16. 下载中取消 → executor 清理部分文件（R1A） ----------

@pytest.mark.asyncio
async def test_download_cancel_cleanup(env):
    part_file = env / "dl.mp4.part1"

    async def factory(job, sinks):
        def download():
            with open(part_file, "wb") as f:
                f.write(b"partial-bytes")
            sinks["cancel_event"].wait(timeout=5)
            if sinks["cancel_event"].is_set():
                raise DownloadCancelled("下载已取消")
            return True, {"file_path": str(part_file)}

        sinks["phase"]("downloading")
        try:
            success, _ = await asyncio.to_thread(download)
        except DownloadCancelled:
            part_file.unlink(missing_ok=True)   # executor 职责：清理部分文件
            raise
        if not success:
            return {"status": "error", "message": "download failed"}
        return {"status": "success"}

    mgr = build_mgr(None, None, factory)
    r = await mgr.submit("bilibili", bili_params(), job_id="j")
    await mgr.start()
    await wait_pred(lambda: mgr._jobs["j"].status == "running"
                    and mgr.get("j")["phase"] == "downloading")
    assert part_file.exists()

    assert (await mgr.cancel(r["job_id"]))["status"] == "cancelling"
    await wait_pred(lambda: mgr._jobs["j"].status == "cancelled")
    assert not part_file.exists(), ".part 部分文件已清理"
    assert mgr._jobs["j"].result is None
    await mgr.stop()


# ---------- 17. upload 终态后内部键文件删除（R2A fold） ----------

@pytest.mark.asyncio
async def test_upload_tmp_cleanup(tmp_path):
    mgr = build_mgr(None, None, simple_factory())
    f1 = tmp_path / "jobfile_done.mp3"
    f1.write_bytes(b"x")
    r1 = await mgr.submit("upload", {"diarize": False}, upload_tmp_path=str(f1), job_id="u1")
    await mgr.start()
    await wait_pred(lambda: mgr._jobs["u1"].status == "done")
    assert not f1.exists(), "done 终态删除内部键文件"
    assert str(f1) not in mgr._active_uploads
    await mgr.stop()

    # queued 取消同样删除
    mgr2 = build_mgr(None, None, simple_factory())
    f2 = tmp_path / "jobfile_cancelled.mp3"
    f2.write_bytes(b"y")
    await mgr2.submit("upload", {"diarize": False}, upload_tmp_path=str(f2), job_id="u2")
    await mgr2.cancel("u2")
    assert mgr2._jobs["u2"].status == "cancelled"
    assert not f2.exists(), "cancelled 终态删除内部键文件"


# ---------- 18. F1：legacy capability 独占取消不发 scoped HTTP ----------

@pytest.mark.asyncio
async def test_cancel_sole_legacy_no_scoped_signal(env, monkeypatch):
    """fixwave F1：legacy capability（旧引擎取消端点为全局语义）下独占取消——
    cancel_request 不被调（不发 scoped HTTP，避免误杀他人共享同一引擎的转录）；
    降级「标志 + join 自然结束」，探针丢弃结果、不入缓存。"""
    backend = LegacyBackend()
    service = make_service(backend)
    path = str(env / "a.wav")
    open(path, "wb").write(b"audio")
    saves = []
    monkeypatch.setattr(T.cache_manager, "save_transcript_to_cache",
                        lambda *a, **k: saves.append(k))
    mgr = build_mgr(backend, service, service_factory(service, path,
                                                      file_path_for_cache=path))
    r = await mgr.submit("webdav", wd(path), job_id="j")
    await mgr.start()
    entry = await wait_entry_asr(service)

    assert (await mgr.cancel(r["job_id"]))["status"] == "cancelling"
    await wait_pred(lambda: entry["cancel_flag"].is_set())
    await asyncio.sleep(0.15)
    assert backend.cancel_requests == [], "legacy 独占取消不得发 cancel_request（scoped HTTP）"
    assert mgr._jobs[r["job_id"]].status == "cancelling", "join 自然结束前保持 cancelling"
    assert all(n != "asr_end" for n, _ in backend.events), "线程未受中断"

    backend.release()                   # 自然跑完
    await wait_pred(lambda: mgr._jobs[r["job_id"]].status == "cancelled")
    assert backend.event_mono("asr_end") < mgr._jobs[r["job_id"]].finished_mono
    assert saves == [], "探针按 entry.cancel_flag 丢弃结果，不入缓存"
    await mgr.stop()


# ---------- 19. F2：自愈 join thread_future 后才释放资源 ----------

class HealBackend:
    """自愈场景桩：ASR 线程阻塞在 threading 门上（无 chunk 级取消响应），
    cancel_request 只记录（自愈路径应保持为空）。"""

    name = "stub-heal"
    device = "cpu"

    def __init__(self):
        self.cancel_requests = []
        self.events = []
        self.gate = threading.Event()

    def probe_capability(self):
        return "progress+scoped"

    def transcribe(self, path, lang=None, context=None, progress_cb=None, request_id=None):
        self.events.append(("asr_start", time.monotonic()))
        self.gate.wait(timeout=10)
        self.events.append(("asr_end", time.monotonic()))
        return TranscribeResult(
            text="甲说乙说", language="zh",
            timestamps=[{"text": c, "start": i * 2.0, "end": i * 2.0 + 1.0}
                        for i, c in enumerate("甲说乙说")],
            performance={})

    def cancel_request(self, request_id):
        self.cancel_requests.append(request_id)

    def event_mono(self, name):
        for n, mono in self.events:
            if n == name:
                return mono
        return None


@pytest.mark.asyncio
async def test_self_heal_joins_thread_future_before_release(env, monkeypatch):
    """fixwave F2：worker 异常退出时复用取消传播顺序——置 entry 标志 →
    先 join thread_future（ASR 线程自然结束）→ 才收尾 exec_task。
    模型计数释放必须晚于 ASR 线程结束；自愈不发 backend 信号；
    终态 error（message 注明 worker 异常退出），结果不入缓存。"""
    backend = HealBackend()
    service = make_service(backend)
    releases = []
    service.model_manager.release = lambda: releases.append(time.monotonic())
    path = str(env / "a.wav")
    open(path, "wb").write(b"audio")
    saves = []
    monkeypatch.setattr(T.cache_manager, "save_transcript_to_cache",
                        lambda *a, **k: saves.append(k))
    mgr = build_mgr(backend, service, service_factory(service, path,
                                                      file_path_for_cache=path))
    r = await mgr.submit("webdav", wd(path), job_id="j")
    await mgr.start()
    entry = await wait_entry_asr(service)

    dead_worker = mgr._worker_task
    dead_worker.cancel()                # worker 死于 ASR 线程运行中
    await asyncio.sleep(0.2)            # 自愈已进入 thread_future join（线程仍阻塞）
    assert releases == [], "thread_future 完成前不得释放模型计数（F2 资源提前释放）"
    assert mgr._jobs[r["job_id"]].status == "running", "join 期间未落终态"

    backend.gate.set()                  # ASR 线程自然结束（无人消费探针）
    await wait_pred(lambda: mgr._jobs[r["job_id"]].status == "error", timeout=15)
    job = mgr._jobs[r["job_id"]]
    assert "worker" in (job.result or {}).get("message", ""), "落 error 注明 worker 异常"
    assert releases and releases[0] >= backend.event_mono("asr_end"), \
        "模型计数释放在 ASR 线程结束之后"
    assert backend.cancel_requests == [], "自愈置标志 + join，不发 backend 信号"
    assert saves == [], "取消标志命中，结果不入缓存"
    assert mgr._worker_task is not dead_worker and not mgr._worker_task.done(), "已重拉"
    assert mgr._current_job is None, "无遗留引用"
    await mgr.stop()


# ---------- 20. F4：formatting phase 由执行体经 phase_cb 写槽 ----------

@pytest.mark.asyncio
async def test_formatting_phase_written_by_executor(env):
    """fixwave F4：执行体传 phase_cb（formatting → percent 100）→ 完成后 GET 的
    phase 即 formatting（spec §5：formatting 直接 100）；转录开始不重复置
    transcribing。"""
    backend = InstantBackend()
    service = make_service(backend)
    path = str(env / "a.wav")
    open(path, "wb").write(b"audio")

    async def factory(job, sinks):
        key = service.build_inflight_key(path, file_path_for_cache=path,
                                         context=None, diarize=False, no_cache=False)
        sinks["set_inflight_key"](key)
        sinks["phase"]("transcribing")   # 转录开始由执行体置位（不重复）
        return await service.process_transcription(
            path, file_path_for_cache=path, context=None, diarize=False,
            progress_cb=sinks["progress"], cancelled=sinks["cancelled"],
            phase_cb=lambda name: sinks["phase"](name, 100 if name == "formatting" else None))

    mgr = build_mgr(backend, service, factory)
    r = await mgr.submit("webdav", wd(path), job_id="j")
    await mgr.start()
    await wait_pred(lambda: mgr._jobs["j"].status == "done")
    v = mgr.get("j")
    assert v["phase"] == "formatting", "终态前最后一槽为 formatting"
    assert v["percent"] == 100.0, "formatting percent 直接 100"
    await mgr.stop()


# ---------- 21. F5：取消路径探测/信号离线程，事件循环不被阻塞 ----------

@pytest.mark.asyncio
async def test_cancel_path_offloads_http(env):
    """fixwave F5：probe_capability（≤5s）与 cancel_request（≤10s）可能发 HTTP
    → 取消流程一律 to_thread；探测/HTTP 期间事件循环保持可调度（心跳推进），
    且 JobManager 锁临界区内不发网络调用。"""
    class SlowHttpBackend(SelfSerialBackend):
        def probe_capability(self):
            time.sleep(0.3)
            return "progress+scoped"

        def cancel_request(self, request_id):
            time.sleep(0.3)
            super().cancel_request(request_id)

    backend = SlowHttpBackend()
    service = make_service(backend)
    path = str(env / "a.wav")
    open(path, "wb").write(b"audio")
    mgr = build_mgr(backend, service, service_factory(service, path,
                                                      file_path_for_cache=path))
    r = await mgr.submit("webdav", wd(path), job_id="j")
    await mgr.start()
    entry = await wait_entry_asr(service)

    ticks = [0]

    async def heartbeat():
        while True:
            ticks[0] += 1
            await asyncio.sleep(0.01)

    hb = asyncio.create_task(heartbeat())

    cancel_task = asyncio.create_task(mgr.cancel(r["job_id"]))
    await asyncio.sleep(0.15)           # cancel 进行中：锁外探测在 to_thread（0.3s）
    assert ticks[0] >= 5, \
        f"事件循环被取消路径 HTTP 阻塞（0.15s 内心跳仅 {ticks[0]} 次）"
    assert (await cancel_task)["status"] == "cancelling"

    await wait_pred(lambda: mgr._jobs[r["job_id"]].status == "cancelled", timeout=10)
    hb.cancel()
    assert ticks[0] >= 30, "整个取消流程心跳持续推进（探测/信号均未阻塞事件循环）"
    assert backend.cancel_requests == [entry["request_id"]], "scoped 信号照发（progress+scoped）"
    await mgr.stop()
