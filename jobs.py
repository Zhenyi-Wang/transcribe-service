"""JobManager：/jobs 异步任务队列的存储、提交语义、查询、取消状态机与清扫。

spec：docs/superpowers/specs/2026-11-10-transcribe-job-queue-design.md
§4.1（提交）/§4.2（查询）/§4.3（取消——六形态矩阵）/§5（进度）/§6（执行模型）/§7（配置）。

Task 3 范围（沿用）：存储（dict + 就绪 deque）、提交语义、GET/query/列表形状、
取消状态机骨架、60s 异步清扫。
Task 7 范围（本模块当前状态）：
- worker：单 asyncio.Task 消费循环（pause 门 → 上游暂停探询 → 锁内出队+状态复查），
  `_executor_factory(job, sinks)` 执行体由 server.py 注入（未注入时不拉起 worker，
  任务保持 queued）；worker 自愈（done_callback：当前 job 落 error 带原因 → 重拉）。
- 终态映射：执行体响应 dict.status=="error" → job error；否则 done
  （bilibili/douyin 注入 timing.download）；JobCancelledError/DownloadCancelled/
  协程取消 → cancelled；UpstreamPausedError → error + paused/resume_at/paused_at。
- 取消生命周期（spec §4.3 即时状态矩阵，`_cancel_exec` 钩子绑 `_cancel_running`）：
  queued 即时 cancelled（Task 3）；running 按 entry 独占/共享 × 自串行（capability
  推导）× 含分离 六形态分派 join/协程取消；cancelling 计入活跃。
- 采样器：独立 asyncio 任务（~1s）；asr-engine 路径轮询注入的 `_progress_poll`，
  request_id 归属匹配 + 应用时本地 phase=="transcribing" 才更新转录槽 percent/message。
- upload 内部键文件终态删除（`_mark_finished`）。

对 server 层（Task 8）的装配约定：
- 注入点：`_executor_factory(job, sinks) -> 协程`（返回与端点同构的响应 dict）；
  `_cancel_computation(inflight_key) -> entry|None`（= transcription_service.cancel_computation）；
  `_entry_lookup(inflight_key) -> entry|None`（= transcription_service._inflight.get）；
  `_backend_lookup() -> ASRBackend|None`（= model_manager.backend）；
  `_progress_poll(job, request_id) -> dict|None`（asr-engine GET /v1/progress 异步轮询）；
  `_pause_gate()`（sync () -> bool）/ `_upstream_pause_probe()`（sync () -> bool，
  **必须为非阻塞快照读取（缓存值），不得在事件循环内做同步 HTTP**——现状
  `server._query_asr_paused` 是 3s 同步 get，直接绑会卡死轮询；Task 8 装配时绑
  缓存值或 to_thread 刷新的快照。直接 import server.py 单例会成环——评审 R5A，
  均由 Task 8 装配传入）。
- sinks 契约（dict，Task 8 执行体消费）：
  progress(done, total)：转录槽 percent/message（ASR 线程内调用、不得抛；legacy
    capability 时 percent 关闭）；
  stage(name)：分离槽 diarize_stage；
  phase(name, percent=None, message=None)：本地 phase 驱动（转录槽整体替换）；
  cancelled()：取消探针；cancel_event：threading.Event（下载段 cancel_flag）；
  note_download_time(seconds)：下载耗时（终态映射注入 timing.download）；
  set_inflight_key(key)/set_backend_name(name)：回填 job 字段（取消矩阵/采样器依赖）；
  diarize_wait_deadline：分离忙退避绝对期限（monotonic，600s，Task 3 传参侧钳制上限）。
- 执行体取消职责：捕获 DownloadCancelled → 清理 .part/部分文件 → 重新抛出
  （终态 cancelled 由 JobManager 映射）；下载完成后应检查 sinks["cancelled"]()。

对 server 层（Task 8）的响应通道约定：
- 新任务 → 202 形状 dict `{"job_id", "status": "queued"}`
- 重放/幂等命中 → 200 形状 dict（GET 完整结构 + `"dedup": true`）
- job_id 不合法 → raise ValueError（server 层转 422）
- 队列满 → raise QueueFullError（`detail="queue full"` + `retry_after` 秒数，
  server 层转 429 + Retry-After 响应头/body 双写）
- get/cancel 未知 id → 返回 None（server 层转 404）

活跃口径 = queued / running / cancelling（identity 去重与 max_queue 计数一致，spec §4.3）。
"""
import asyncio
import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from logger_config import setup_logger
from transcribe import JobCancelledError, clamp_asr_context
from backends.asr_engine_backend import UpstreamPausedError
from downloaders.base import DownloadCancelled

logger = setup_logger(__name__)

# job_id 服务端不透明：格式校验即全部语义（noteflow 格式 {cuid}-{attempt}，不解析 attempt）
JOB_ID_RE = re.compile(r"^[A-Za-z0-9._:@-]{1,128}$")

ACTIVE_STATUSES = ("queued", "running", "cancelling")
TERMINAL_STATUSES = ("done", "error", "cancelled")

QUEUE_FULL_DETAIL = "queue full"

DEFAULT_MAX_QUEUE = 1000                # 含 running 与 cancelling
DEFAULT_RECORD_TTL_MINUTES = 60         # 终态记录目标保留期
DEFAULT_MAX_RESULT_BYTES = 268435456    # 终态 result 总字节预算（256MB）
DEFAULT_RETRY_AFTER_SECONDS = 30        # 429 Retry-After；config jobs.retry_after_seconds 可调

SWEEP_INTERVAL_SECONDS = 60             # 清扫周期（spec §6：~60s asyncio 任务）
LIST_TERMINAL_CAP = 100                 # GET /jobs 列表终态 cap（cap 只作用于终态）

WORKER_IDLE_WAIT_SECONDS = 0.5          # worker 空闲等待（新提交经 event 即刻唤醒）
SAMPLE_INTERVAL_SECONDS = 1.0           # 采样器周期（spec §5/§6：~1s）
DIARIZE_WAIT_SECONDS = 600.0            # 分离忙退避期限（10min，spec §6 有界退避）
ASR_ENGINE_BACKEND_NAME = "asr-engine"  # 走远端采样的 backend name（ASREngineClientBackend.name）


class QueueFullError(Exception):
    """队列满（max_queue 含 running 与 cancelling）——server 层转 429 + Retry-After。"""

    def __init__(self, retry_after: int = DEFAULT_RETRY_AFTER_SECONDS):
        super().__init__(QUEUE_FULL_DETAIL)
        self.detail = QUEUE_FULL_DETAIL
        self.retry_after = retry_after


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _result_size(result) -> int:
    """终态 result 的序列化字节量（预算核算口径）。"""
    if result is None:
        return 0
    try:
        return len(json.dumps(result, ensure_ascii=False, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return 0


def _body_digest(body) -> Optional[str]:
    """body 摘要 = JSON 序列化后 md5（spec §4.1 评审 R6 定死）。"""
    if body is None:
        return None
    payload = json.dumps(body, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _identity_key(source: str, params: dict):
    """提交幂等 identity 代理键（spec §4.1）：
    (source, bvid+page | aweme_id | path, clamp(context), diarize, diarize_only,
     md5(json.dumps(body)), no_cache)。
    cookie 不入键（凭据差异不影响幂等）；upload 无稳定 identity → None。
    """
    if source == "bilibili":
        locator = (params.get("bvid"), params.get("page") or 1)
    elif source == "douyin":
        locator = params.get("aweme_id")
    elif source == "webdav":
        locator = params.get("path")
    else:
        return None
    context = clamp_asr_context(params.get("context"))
    return (source, locator, context,
            bool(params.get("diarize")), bool(params.get("diarize_only")),
            _body_digest(params.get("body")), bool(params.get("no_cache")))


def _unlink_quiet(path) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


@dataclass
class Job:
    """单条任务记录（纯内存，重启即清——spec 非目标：持久化/重启恢复）。"""
    job_id: str
    source: str
    request: dict                        # 提交参数摘要（cookie 已剔除，永不回显）
    status: str = "queued"               # queued|running|cancelling|done|error|cancelled
    phase: Optional[str] = None          # downloading|transcribing|formatting|None
    percent: Optional[float] = None
    message: Optional[str] = None
    diarize_stage: Optional[str] = None  # decoding|segmenting|embedding|clustering|reconstructing|None
    submitted_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[dict] = None        # done=端点同构响应；error=错误 dict（Task 7 映射）
    upload_path: Optional[str] = None    # upload 内部键文件路径（终态删除归 Task 7）
    inflight_key: Optional[tuple] = None  # 执行体回填（build_inflight_key，Task 7）
    transcript_slot: dict = field(default_factory=dict)  # 转录槽 {phase, percent, message}
    diarize_slot: dict = field(default_factory=dict)     # 分离槽 {diarize_stage}
    backend_name: Optional[str] = None
    identity: Optional[tuple] = None     # 提交幂等代理键
    submitted_mono: float = 0.0          # 单调时钟（排序/容量核算用）
    finished_mono: Optional[float] = None
    # Task 7 执行期字段：
    cancel_event: threading.Event = field(default_factory=threading.Event)  # 取消标志（下载段/探针）
    download_seconds: Optional[float] = None    # 下载耗时（终态映射注入 timing.download）
    exec_task: Optional[object] = None   # 本 job 的 `_execute_job` asyncio.Task（协程级取消句柄）
    cancel_join_task: Optional[object] = None  # 取消 join 协程（cancelling → 终态的推进者）
    # fixwave F3 提交点：执行体已开始不可回退的副作用（webdav 归档）置位——
    # 取消在其后到达时 done 胜出取消（_execute_job/_cancel_join 消费）
    commit_done: bool = False


class JobManager:
    """单例（Task 8 于 server.py 装配）。全部状态经 `self._lock`（asyncio）串行化。"""

    def __init__(self, config=None):
        get = config.get if config is not None else (lambda key, default=None: default)
        self.max_queue = int(get("jobs.max_queue", DEFAULT_MAX_QUEUE))
        self.record_ttl_minutes = float(get("jobs.record_ttl_minutes", DEFAULT_RECORD_TTL_MINUTES))
        self.max_result_bytes = int(get("jobs.max_result_bytes", DEFAULT_MAX_RESULT_BYTES))
        self.retry_after_seconds = int(get("jobs.retry_after_seconds", DEFAULT_RETRY_AFTER_SECONDS))

        self._jobs: dict = {}              # job_id -> Job（活跃 + 终态）
        self._ready: deque = deque()       # 就绪队列（仅 queued 的 job_id，FIFO）
        self._active_uploads: set = set()  # 活跃 upload 路径登记（tmp 清理豁免，Task 8 消费）
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()       # 新提交唤醒 worker（WORKER_IDLE_WAIT 兜底）

        # Task 7 注入点（Task 8 于 server.py 装配；直接 import server.py 单例会成环——评审 R5A）：
        self._executor_factory = None      # (job, sinks) -> 协程，返回端点同构响应 dict
        self._cancel_computation = None    # (inflight_key) -> entry|None（service.cancel_computation）
        self._entry_lookup = None          # (inflight_key) -> entry|None（service._inflight.get）
        self._backend_lookup = None        # () -> ASRBackend|None（model_manager.backend）
        self._progress_poll = None         # async (job, request_id) -> dict|None（GET /v1/progress）
        self._pause_gate = None            # () -> bool（pause_manager.is_paused）
        # 契约：必须为非阻塞快照读取（缓存值），不得在事件循环内做同步 HTTP
        # （server._query_asr_paused 现状是 3s 同步 get，直接绑会卡死 worker 取件
        # 与自冷轮询）——Task 8 装配时绑缓存值或经 to_thread 刷新的快照。
        self._upstream_pause_probe = None  # () -> bool（worker 自冷探询，见上方契约）

        # running 取消联动钩子（Task 3 预留）：绑真实实现，测试可覆写
        self._cancel_exec = self._cancel_running
        self._current_job = None           # worker 正在执行的 job（自愈清理用）
        self._healing = False              # 自愈窗口旗标（_execute_job 让位 error 落态）
        # capability 最近一次观测值（fixwave F5）：供锁内必须同步返回的即时判定点
        # （_cancel_running 共享×无分离单元）免 HTTP 读取；由 _capability/_capability_async
        # 每次探测后刷新，None = 尚未探测过
        self._cap_snapshot: Optional[str] = None
        self.sample_interval = SAMPLE_INTERVAL_SECONDS

        # 自愈 join 兜底上限（fixwave F2）：worker 已死、无人消费探针时 ASR 线程
        # 最多跑到自然结束——join 与取消路径同语义，只设灾备上限不设短超时
        self.heal_join_timeout = 3600.0

        self._sweep_task: Optional[asyncio.Task] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._sampler_task: Optional[asyncio.Task] = None
        self._heal_task: Optional[asyncio.Task] = None
        self._running = False

    # ================= 提交（spec §4.1） =================

    async def submit(self, source: str, params: dict, job_id: Optional[str] = None,
                     upload_tmp_path: Optional[str] = None) -> dict:
        """提交任务。返回 202 形状或 200 重放形状；不合法 job_id → ValueError；
        队列满 → QueueFullError（见模块 docstring 的通道约定）。
        """
        if job_id is not None:
            if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id) \
                    or job_id in (".", ".."):
                raise ValueError(f"invalid job_id: {job_id!r}")

        ident = _identity_key(source, params or {})
        doomed_upload = None
        async with self._lock:
            replay_job = None
            if job_id is not None and job_id in self._jobs:
                replay_job = self._jobs[job_id]          # 同 id 一律重放当前状态，绝不替换
            elif job_id is None and ident is not None:
                for job in self._jobs.values():          # 幂等命中：同 identity 活跃任务
                    if job.identity == ident and job.status in ACTIVE_STATUSES:
                        replay_job = job
                        break
            if replay_job is not None:
                out = self._view(replay_job)
                out["dedup"] = True
                doomed_upload = upload_tmp_path  # 重复 upload 的新文件立即删除（spec §4.1）
            else:
                active = sum(1 for j in self._jobs.values() if j.status in ACTIVE_STATUSES)
                if active >= self.max_queue:
                    raise QueueFullError(self.retry_after_seconds)
                job = Job(
                    job_id=job_id if job_id is not None else uuid.uuid4().hex,
                    source=source,
                    request={k: v for k, v in (params or {}).items() if k != "cookie"},
                    submitted_at=_now_iso(),
                    submitted_mono=time.monotonic(),
                    identity=ident,
                    upload_path=upload_tmp_path,
                )
                self._jobs[job.job_id] = job
                self._ready.append(job.job_id)
                if upload_tmp_path:
                    self.register_upload(upload_tmp_path)
                out = {"job_id": job.job_id, "status": job.status}
        if doomed_upload:
            _unlink_quiet(doomed_upload)
        else:
            self._wake.set()               # 新任务就绪，即刻唤醒 worker（若有）
        return out

    # ================= 查询（spec §4.2） =================

    def get(self, job_id: str):
        """GET /jobs/{id} 形状；未知 id → None（server 层转 404）。"""
        job = self._jobs.get(job_id)
        return self._view(job) if job is not None else None

    def query(self, ids) -> dict:
        """POST /jobs/query：`{"jobs": [...不含 result...], "missing": [...]}`。
        单批 ≤500 截断由调用方保证，本方法全量处理。
        """
        jobs, missing = [], []
        for jid in ids or []:
            job = self._jobs.get(jid)
            if job is None:
                missing.append(jid)
            else:
                jobs.append(self._item_view(job))
        return {"jobs": jobs, "missing": missing}

    def list_jobs(self, status: Optional[str] = None) -> list:
        """GET /jobs 列表（调试/对账）：倒序；cap 100 只作用于终态任务。"""
        def order(j):
            return (j.finished_mono if j.finished_mono is not None else j.submitted_mono,
                    j.submitted_mono, j.job_id)
        selected = [j for j in self._jobs.values()
                    if status is None or j.status == status]
        active = sorted((j for j in selected if j.status in ACTIVE_STATUSES),
                        key=order, reverse=True)
        terminal = sorted((j for j in selected if j.status not in ACTIVE_STATUSES),
                          key=order, reverse=True)[:LIST_TERMINAL_CAP]
        merged = active + terminal
        merged.sort(key=order, reverse=True)
        return [self._item_view(j) for j in merged]

    def _view(self, job: Job) -> dict:
        """GET 完整结构。进度读侧拼装两槽（spec §5：转录槽 + 分离槽整体替换，
        槽值优先于 Job 字段——Task 7 采样器/回调写槽）。"""
        tslot = job.transcript_slot or {}
        dslot = job.diarize_slot or {}
        return {
            "job_id": job.job_id,
            "request": dict(job.request),
            "status": job.status,
            "phase": tslot.get("phase", job.phase),
            "percent": tslot.get("percent", job.percent),
            "message": tslot.get("message", job.message),
            "diarize_stage": dslot.get("diarize_stage", job.diarize_stage),
            "queue_position": (self._ready.index(job.job_id) + 1)
                              if (job.status == "queued" and job.job_id in self._ready) else None,
            "submitted_at": job.submitted_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "result": job.result,
        }

    def _item_view(self, job: Job) -> dict:
        """query / 列表项：不含 result（终态后按 id 单独取，spec §4.2）。"""
        view = self._view(job)
        view.pop("result")
        return view

    # ================= 取消（spec §4.3；执行联动归 Task 7） =================

    async def cancel(self, job_id: str):
        """DELETE /jobs/{id}「已接受」语义：返回请求后即时状态 dict。
        queued → 即时 cancelled（出队）；running → cancelling + 联动钩子
        （`self._cancel_exec`，Task 7 注入）；cancelling → 幂等不重复联动；
        终态 → 幂等返回当前态；未知 id → None（server 层转 404）。

        fixwave F5：共享×无分离形态需要 capability 决定「即时 cancelled vs
        cancelling+join」——探测可能发 HTTP（≤5s），先在锁外解析（to_thread），
        绝不在持有 JobManager 锁的临界区内发起网络调用。窥探只读（asyncio 单
        线程），全部状态变更仍在锁内按当时值复查执行。
        """
        cap = None
        job = self._jobs.get(job_id)       # 锁外窥探（判定输入，锁内复查）
        if job is not None and job.status == "running":
            entry = self._lookup_entry(job)
            if entry is not None and entry.get("waiters", 0) > 1 \
                    and not bool((job.request or {}).get("diarize")):
                cap = await self._capability_async()
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in TERMINAL_STATUSES:
                return {"job_id": job_id, "status": job.status}
            if job.status == "queued":
                try:
                    self._ready.remove(job_id)
                except ValueError:
                    pass
                self._mark_finished(job, "cancelled")
                return {"job_id": job_id, "status": "cancelled"}
            if job.status == "running":
                job.status = "cancelling"
                if self._cancel_exec is not None:
                    if cap is not None:
                        self._cancel_exec(job, cap)
                    else:
                        self._cancel_exec(job)
            return {"job_id": job_id, "status": job.status}

    # ================= 状态迁移原语（Task 7 worker 消费） =================

    def _mark_running(self, job: Job) -> Job:
        job.status = "running"
        job.started_at = _now_iso()
        return job

    def _mark_finished(self, job: Job, status: str, result: Optional[dict] = None) -> Job:
        """终态迁移（status ∈ TERMINAL_STATUSES）。

        upload 登记同步注销 + 内部键文件删除（spec §6 终态清理：任何终态——
        含 queued 即时取消——都删除 tmp/jobfile_*，覆盖全路径）。
        """
        job.status = status
        job.finished_at = _now_iso()
        job.finished_mono = time.monotonic()
        job.result = result
        if job.upload_path:
            self.unregister_upload(job.upload_path)
            _unlink_quiet(job.upload_path)
            job.upload_path = None
        return job

    # ================= upload 登记维护（spec §4.1/§6） =================

    def register_upload(self, path) -> None:
        if path:
            self._active_uploads.add(path)

    def unregister_upload(self, path) -> None:
        self._active_uploads.discard(path)

    def active_upload_paths(self) -> set:
        """活跃 upload 登记路径集快照（server 层 tmp 清理豁免消费；拷贝防迭代期变更）。"""
        return set(self._active_uploads)

    # ================= 清扫（spec §6：全部走事件循环） =================

    async def _sweep_once(self):
        """单轮清扫：① TTL 到期终态记录整条移除（=missing，noteflow attempt+1 重提）；
        ② result 总字节超预算时从最旧终态整条驱逐（单条超限保留）；
        ③ upload 登记维护（已消失路径摘除）。活跃记录绝不触碰。
        """
        async with self._lock:
            ttl_seconds = self.record_ttl_minutes * 60.0
            now = time.monotonic()
            expired = [jid for jid, j in self._jobs.items()
                       if j.status in TERMINAL_STATUSES and j.finished_mono is not None
                       and now - j.finished_mono > ttl_seconds]
            for jid in expired:
                del self._jobs[jid]
                self._discard_ready(jid)

            terminal = sorted(
                (j for j in self._jobs.values() if j.status in TERMINAL_STATUSES),
                key=lambda j: (j.finished_mono or 0.0, j.job_id))
            sizes = {j.job_id: _result_size(j.result) for j in terminal}
            total = sum(sizes.values())
            remaining = len(terminal)
            for j in terminal:
                if total <= self.max_result_bytes or remaining <= 1:
                    break  # 预算内 / 剩单条（哪怕单条超限也保留）
                total -= sizes[j.job_id]
                del self._jobs[j.job_id]
                self._discard_ready(j.job_id)
                remaining -= 1

            for path in list(self._active_uploads):
                if not os.path.exists(path):
                    self._active_uploads.discard(path)

    def _discard_ready(self, job_id: str) -> None:
        """防御性清队：终态记录被清扫移除时不得残留就绪引用。"""
        try:
            self._ready.remove(job_id)
        except ValueError:
            pass

    async def _sweep_loop(self):
        while True:
            await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
            try:
                await self._sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("jobs 清扫失败（下一周期重试）")

    # ================= 生命周期 =================

    async def start(self):
        """拉起清扫任务；`_executor_factory` 已注入时同时拉起 worker 与采样器（Task 7）。幂等。"""
        if self._running:
            return
        self._running = True
        self._sweep_task = asyncio.create_task(self._sweep_loop())
        if self._executor_factory is not None:
            self._worker_task = asyncio.create_task(self._worker_loop())
            self._worker_task.add_done_callback(self._on_worker_done)
            self._sampler_task = asyncio.create_task(self._sampler_loop())

    async def stop(self):
        """停止后台任务（幂等）。

        活跃 running/cancelling job 先走取消语义收尾（复用 §4.3 既有路径：
        cancel_computation / 协程取消，视 entry 形态分派）→ awaiting → 落
        `cancelled`（message 注明 service stopped）——否则 exec_task 被孤儿化、
        job 永久停留 running 占用活跃口径，且「可重启」承诺失效（重启后新
        worker 会在旧计算持 GPU 时取下一件）。queued 任务保持排队；存储本身
        不受影响。收尾完成后可安全 `start()` 重启。
        """
        self._running = False
        await self._stop_finalize_current()
        for attr in ("_worker_task", "_sweep_task", "_sampler_task", "_heal_task"):
            task = getattr(self, attr, None)
            if task is not None:
                task.cancel()
        for attr in ("_worker_task", "_sweep_task", "_sampler_task", "_heal_task"):
            task = getattr(self, attr, None)
            if task is not None:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("jobs 后台任务停止异常")
                setattr(self, attr, None)
        self._current_job = None

    async def _stop_finalize_current(self):
        """stop 时的活跃 job 收尾：复用取消生命周期（评审修复轮 1，Important-1）。

        running → cancelling（锁内，与 cancel() 同款联动）→ awaiting（cancel_join
        内部完成 thread_future join / 协程取消 / 计算退出 join）→ 终态。cancelling
        中途被 stop 的 job 直接等其既有 join 任务；二者均兜底等 exec_task 结束，
        确保 stop 返回后无孤儿执行体持模型/GPU。
        """
        job = self._current_job
        if job is None or job.status not in ("running", "cancelling"):
            return
        cap = None
        if job.status == "running":
            # 共享×无分离形态的即时判定需要 capability——锁外解析（F5，同 cancel()）
            entry = self._lookup_entry(job)
            if entry is not None and entry.get("waiters", 0) > 1 \
                    and not bool((job.request or {}).get("diarize")):
                cap = await self._capability_async()
            job.message = "service stopped（服务停止，任务已中止）"
            async with self._lock:
                if job.status == "running":     # 锁内复查（与 cancel() 同款语义）
                    job.status = "cancelling"
                    if self._cancel_exec is not None:
                        if cap is not None:
                            self._cancel_exec(job, cap)
                        else:
                            self._cancel_exec(job)   # 即时终态（路径 C）或创建 cancel_join
        join_task = job.cancel_join_task
        if join_task is not None:
            try:
                await join_task
            except Exception:
                logger.exception("job %s 停止收尾 join 异常", job.job_id)
        task = job.exec_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except BaseException:
                pass
        if job.status == "cancelling":   # join 缺失/异常兜底：绝不留活 job
            self._mark_finished(job, "cancelled")

    # ================= worker：单消费循环（spec §6） =================

    def _gate_closed(self, hook) -> bool:
        """注入的 () -> bool 门取值；钩子异常一律视为未关闭（不阻塞队列）。"""
        if hook is None:
            return False
        try:
            return bool(hook())
        except Exception:
            logger.exception("jobs 取件门查询失败（视为未关闭）")
            return False

    async def _take_next(self):
        """锁内取件：出队即取（Task 3 提醒：_mark_finished 不清 _ready）+
        状态复查（取件竞态：已非 queued 的陈旧引用跳过）。"""
        async with self._lock:
            while self._ready:
                job_id = self._ready.popleft()
                job = self._jobs.get(job_id)
                if job is None or job.status != "queued":
                    continue
                self._mark_running(job)
                return job
            return None

    async def _worker_loop(self):
        """单 worker 消费循环：取件门（pause → 上游暂停自冷）→ 出队 → 执行。"""
        while self._running:
            try:
                if self._gate_closed(self._pause_gate) or \
                        self._gate_closed(self._upstream_pause_probe):
                    await asyncio.sleep(WORKER_IDLE_WAIT_SECONDS)
                    continue
                job = await self._take_next()
                if job is None:
                    self._wake.clear()
                    job = await self._take_next()   # clear 与 take 之间的提交补检
                    if job is None:
                        # 空闲等待：asyncio.wait 不吞取消（wait_for 在 py3.10 有
                        # bpo-42130 竞态——外部 cancel 可被 TimeoutError 顶掉，
                        # worker 将无法被取消，teardown gather 永久等待）
                        waiter = asyncio.ensure_future(self._wake.wait())
                        try:
                            await asyncio.wait({waiter}, timeout=WORKER_IDLE_WAIT_SECONDS)
                        finally:
                            waiter.cancel()
                        continue
                await self._run_job(job)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("worker 循环异常（下一轮自愈检查）")
                await asyncio.sleep(WORKER_IDLE_WAIT_SECONDS)

    async def _run_job(self, job: Job):
        """执行单个 job 并持引用；收尾时若取消 join 未完（共享×join 路径的
        「计算退出才取下一件」），等待其终态后再返回。

        shield：外部 cancel worker 时不得连带取消 exec_task（py3.10 Task.cancel
        会先取消 `_fut_waiter`（子任务）并吞掉外层取消——自愈句柄由 heal 持有，
        exec_task 的终止只归 heal/取消路径显式驱动）。"""
        task = asyncio.create_task(self._execute_job(job))
        job.exec_task = task
        self._current_job = job
        try:
            await asyncio.shield(task)
        finally:
            # 注意：外部 cancel 路径下 _current_job 必须保留给自愈（_self_heal
            # 据此落 error + 清引用），故清引用放在 try/finally 之后——取消
            # 传播时下方语句不执行。
            join_task = job.cancel_join_task
            if join_task is not None and job.status == "cancelling":
                try:
                    await join_task
                except Exception:
                    logger.exception("job %s 取消 join 收尾异常", job.job_id)
        if self._current_job is job:
            self._current_job = None

    # ================= 执行体 sinks 与终态映射 =================

    def _make_sinks(self, job: Job) -> dict:
        """执行体回调束（线程安全：槽以单 dict 整体替换为原子快照，spec §5；
        全部吞异常——progress_cb/stage_cb 契约为不得抛出）。"""
        def _progress(done, total):
            try:
                slot = dict(job.transcript_slot)
                if not slot.get("phase"):
                    slot["phase"] = job.phase or "transcribing"
                # percent 门控只作用于 ASR 段（legacy 态 percent 关，spec §6 引擎侧第 7 条）；
                # downloading 段为 B 站真实值，不受 ASR capability 降级影响
                is_asr = slot.get("phase") != "downloading"
                if total and (not is_asr or self._capability() == "progress+scoped"):
                    slot["percent"] = round(done / total * 100, 1)
                slot["message"] = f"{done}/{total}"
                job.transcript_slot = slot
            except Exception:
                pass

        def _stage(name):
            try:
                job.diarize_slot = {"diarize_stage": name}
            except Exception:
                pass

        def _phase(name, percent=None, message=None):
            try:
                slot = {"phase": name}
                if percent is not None:
                    slot["percent"] = percent
                if message is not None:
                    slot["message"] = message
                job.transcript_slot = slot
            except Exception:
                pass

        def _note_download(seconds):
            try:
                job.download_seconds = seconds
            except Exception:
                pass

        return {
            "progress": _progress,
            "stage": _stage,
            "phase": _phase,
            "cancelled": job.cancel_event.is_set,
            "cancel_event": job.cancel_event,
            "note_download_time": _note_download,
            "set_inflight_key": lambda key: setattr(job, "inflight_key", key),
            "set_backend_name": lambda name: setattr(job, "backend_name", name),
            "diarize_wait_deadline": time.monotonic() + DIARIZE_WAIT_SECONDS,
        }

    async def _execute_job(self, job: Job):
        """执行体包装：终态映射（spec §4.2/§4.3）。

        - 响应 dict.status=="error" → job error（result=该 dict）；否则 done
          （bilibili/douyin 注入 timing.download）；
        - JobCancelledError / DownloadCancelled / 协程取消 → cancelled（cancelling
          态的终态由取消 join 路径拥有，此处让位）；
        - UpstreamPausedError → error + paused/resume_at/paused_at（worker 自冷由
          取件门探询 `_upstream_pause_probe` 实现）；
        - 其余异常 → error（带类型与消息）。
        """
        sinks = self._make_sinks(job)
        try:
            response = await self._executor_factory(job, sinks)
        except (JobCancelledError, DownloadCancelled):
            # healing 窗口终态归自愈（fixwave F2：自愈 join thread_future 后拥有者
            # 经探针中止，不得抢先把 cancelled 落态，否则覆盖「worker crashed」error）
            if job.status == "running" and not self._healing:
                self._mark_finished(job, "cancelled")
            return
        except UpstreamPausedError as e:
            if job.status == "running":
                self._mark_finished(job, "error", {
                    "status": "error", "message": str(e) or "upstream paused",
                    "paused": True, "resume_at": e.resume_at, "paused_at": e.paused_at})
            return
        except asyncio.CancelledError:
            # 协程级取消（共享撤销自身等待 / 无 entry 分离段 / 自愈窗口）：
            # cancelling 态由取消路径终态；healing 窗口由自愈落 error。
            if job.status == "running" and not self._healing:
                self._mark_finished(job, "cancelled")
            return
        except BaseException as e:
            if job.status == "running":
                self._mark_finished(job, "error", {
                    "status": "error", "message": f"{type(e).__name__}: {e}"})
            return

        # 提交点语义（fixwave F3，webdav 归档）：执行体已提交不可回退副作用
        # （commit_done）且取消已受理（cancelling）→ done 胜出取消，result 照常
        # 返回（与已归档文件一致）；其余 cancelling/终态：取消路径拥有终态，迟到响应丢弃
        committed = job.status == "cancelling" and job.commit_done
        if job.status != "running" and not committed:
            return
        result = response if isinstance(response, dict) \
            else {"status": "success", "data": response}
        if result.get("status") == "error":
            self._mark_finished(job, "error", result)
            return
        if job.source in ("bilibili", "douyin") and job.download_seconds is not None:
            result = dict(result)
            timing = dict(result.get("timing") or {})
            timing.setdefault("download", round(job.download_seconds, 3))
            result["timing"] = timing
        self._mark_finished(job, "done", result)

    # ================= 取消生命周期（spec §4.3 即时状态矩阵） =================

    def _effective_phase(self, job: Job) -> Optional[str]:
        slot = job.transcript_slot or {}
        return slot.get("phase", job.phase)

    def _cancel_running(self, job: Job, cap=None):
        """running → cancelling 的执行联动（`cancel()` 于锁内调用，同步返回）。

        cap（fixwave F5）：cancel() 在锁外预解析的 capability（to_thread 探测）；
        缺省时读快照，快照也缺失才同步探测兜底（历史行为，理论窗口极窄）。

        即时状态矩阵分派：
        - 共享（waiters>1）× 无分离 × capability=progress+scoped（自串行推导）→
          即时 cancelled（不 join、不动 entry；撤销自身等待 fire-and-forget）；
        - 其余（独占 / 共享×分离 / 共享×legacy / 无 entry）→ cancelling +
          `_cancel_join` 协程按矩阵完成 join/协程取消后落 cancelled。
        """
        job.cancel_event.set()
        entry = self._lookup_entry(job)
        if entry is not None and entry.get("waiters", 0) > 1 \
                and not bool((job.request or {}).get("diarize")):
            capability = cap if cap is not None else self._cap_snapshot
            if capability is None:
                capability = self._capability()
            if capability == "progress+scoped":
                self._mark_finished(job, "cancelled")
                asyncio.create_task(self._cancel_exec_task(job))
                return
        job.cancel_join_task = asyncio.create_task(self._cancel_join(job))

    def _lookup_entry(self, job: Job):
        if job.inflight_key is None or self._entry_lookup is None:
            return None
        try:
            return self._entry_lookup(job.inflight_key)
        except Exception:
            logger.exception("job %s entry 查询失败", job.job_id)
            return None

    async def _cancel_join(self, job: Job):
        """cancelling → cancelled 的推进者（spec §4.3 取消传播顺序：
        entry 标记 → backend 信号 → 等 thread_future 完成 → 才取下一件）。

        fixwave F5：本协程不持锁，但 capability 探测与 cancel_request（scoped
        HTTP ≤10s）一律经 to_thread 离开事件循环。
        fixwave F1：独占路径的信号强度按 capability 二分——legacy 引擎的取消
        端点是全局语义，signal=False（只置标志）降级「标志 + join 自然结束」。
        """
        try:
            entry = self._lookup_entry(job)
            if entry is None:
                # 无 entry（diarize_only/merge/annotate 无 ASR/entry 未注册）：
                # 下载段依赖标志自然中止（协程 cancel 会吞掉线程侧
                # DownloadCancelled，.part 清理丢失）；分离段直接协程取消。
                # 执行体已提交不可回退副作用（commit_done，F3 归档提交点）时
                # 同样等自然收尾——done 胜出取消（_execute_job 按提交点落态）。
                if self._effective_phase(job) == "downloading" or job.commit_done:
                    await self._await_exec(job)
                else:
                    await self._cancel_exec_task(job)
                return
            if entry.get("waiters", 0) <= 1:
                # 独占：真实取消（flag + 归属化 backend 信号；legacy 引擎只置
                # 标志——其 scoped HTTP 实为全局取消，会误杀他人转录，F1）
                signal = (await self._capability_async()) == "progress+scoped"
                entry = None
                if self._cancel_computation is not None:
                    try:
                        entry = await asyncio.to_thread(
                            self._cancel_computation, job.inflight_key, signal)
                    except Exception:
                        logger.exception("cancel_computation 失败（探针兜底）")
                cf = await self._wait_thread_future(entry) if entry is not None else None
                if cf is not None:
                    # 独占主路径：等 ASR 线程收尾（thread_future 完成前绝不
                    # 取消拥有者协程——防止模型计数/租约/文件提前释放）
                    try:
                        await asyncio.wrap_future(cf)
                    except BaseException:
                        pass
                    await self._await_exec(job)   # 拥有者经探针自然中止
                else:
                    # 第五形态（annotate/ASR 未启动，thread_future 恒空）：
                    # 协程级取消，既有 except BaseException 链回收分离进程
                    await self._cancel_exec_task(job)
                return
            # 共享：仅撤销自身等待（不调 cancel_computation、不置 entry 标志）
            await self._cancel_exec_task(job)
            capability = await self._capability_async()
            if bool((job.request or {}).get("diarize")) \
                    or capability != "progress+scoped":
                # 含分离（任意后端）或非自串行（qwen3/funasr/legacy）：join 计算
                # 退出执行资源才终态（worker 取下一件前等待本任务完成）
                await self._join_computation(entry)
        finally:
            if job.status == "cancelling":
                self._mark_finished(job, "cancelled")

    async def _wait_thread_future(self, entry: dict):
        """entry.thread_future 读取（submit→登记为同步序列，但跨线程读存在
        理论窗口：asr_started 已置而 future 未及登记时短暂轮询等待）。"""
        if entry is None:
            return None
        for _ in range(200):
            cf = entry.get("thread_future")
            if cf is not None:
                return cf
            if not entry.get("asr_started"):
                return None
            await asyncio.sleep(0.005)
        return entry.get("thread_future")

    async def _cancel_exec_task(self, job: Job):
        task = job.exec_task
        if task is not None and not task.done():
            task.cancel()
        await self._await_exec(job)

    async def _await_exec(self, job: Job):
        task = job.exec_task
        if task is None:
            return
        try:
            await task
        except BaseException:
            pass

    async def _join_computation(self, entry: dict):
        """共享 join：等计算退出（thread_future 优先，兜底共享任务本身）。
        绝不取消共享任务——遗留等待者仍依赖它。"""
        cf = entry.get("thread_future")
        if cf is not None:
            try:
                await asyncio.wrap_future(cf)
            except BaseException:
                pass
            return
        target = entry.get("task")
        if target is None:
            return
        try:
            await target
        except BaseException:
            pass

    # ================= capability 消费（Task 5 probe 结果） =================

    async def _capability_async(self) -> str:
        """异步语境（取消流程）的 capability 读取（fixwave F5）：probe_capability
        可能发 HTTP（首次探测 / 404 负缓存到期重探 / 瞬时失败不缓存，≤5s）——
        一律经 to_thread 移出事件循环；结果刷新快照供锁内即时判定点免 HTTP 读取。"""
        backend = self._backend_lookup() if self._backend_lookup is not None else None
        if backend is None:
            self._cap_snapshot = "legacy"
            return "legacy"
        try:
            cap = await asyncio.to_thread(backend.probe_capability)
        except Exception:
            logger.exception("capability 探测失败（按 legacy 处理）")
            cap = "legacy"
        self._cap_snapshot = cap
        return cap

    def _capability(self) -> str:
        """backend 能力自述（'progress+scoped' | 'legacy'）。正/负缓存由 backend
        自持（Task 5：正缓存持续、404 负缓存 10min、瞬时不缓存），本层不重复建缓存；
        legacy → 共享取消降级「标志+join」、percent 关闭（spec §6 引擎侧第 7 条）。

        同步调用点仅限线程语境（ASR 线程 progress sink）与锁内兜底——异步语境
        必须走 `_capability_async`（F5）。结果写快照（`_cap_snapshot`）。"""
        backend = self._backend_lookup() if self._backend_lookup is not None else None
        if backend is None:
            self._cap_snapshot = "legacy"
            return "legacy"
        try:
            cap = backend.probe_capability()
        except Exception:
            logger.exception("capability 探测失败（按 legacy 处理）")
            cap = "legacy"
        self._cap_snapshot = cap
        return cap

    # ================= 采样器（spec §5：单一异步任务 ~1s） =================

    async def _sampler_loop(self):
        while self._running:
            await asyncio.sleep(self.sample_interval)
            try:
                await self._sample_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("jobs 进度采样失败（下一周期重试）")

    async def _sample_once(self):
        if self._progress_poll is None:
            return
        async with self._lock:
            jobs = [j for j in self._jobs.values()
                    if j.status in ("running", "cancelling")]
        for job in jobs:
            try:
                await self._sample_job(job)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("job %s 进度采样失败", job.job_id)

    async def _sample_job(self, job: Job):
        """asr-engine 远端样本合并（spec §5）：仅更新转录槽 percent/message；
        应用条件 = 响应 request_id 与当前计算归属匹配 且 应用时本地 phase 为
        transcribing（迟到样本不得拉低已推进的 phase）；inactive/不匹配/失败忽略。"""
        if job.backend_name != ASR_ENGINE_BACKEND_NAME:
            return
        entry = self._lookup_entry(job)
        request_id = entry.get("request_id") if entry else None
        if not request_id:
            return
        sample = await self._progress_poll(job, request_id)
        if not isinstance(sample, dict) or not sample.get("active"):
            return
        if sample.get("request_id") != request_id:
            return
        slot = dict(job.transcript_slot)
        if slot.get("phase", job.phase) != "transcribing":
            return
        chunk, total = sample.get("chunk"), sample.get("total")
        if isinstance(total, (int, float)) and total > 0 \
                and isinstance(chunk, (int, float)):
            slot["percent"] = round(chunk / total * 100, 1)
            slot["message"] = f"{chunk}/{total}"
            job.transcript_slot = slot

    # ================= worker 自愈（spec §6） =================

    def _on_worker_done(self, task):
        """worker 异常退出：当前 job 落 error 带原因、清理遗留引用后重拉；
        关闭（stop 置 _running=False）期间不重拉。"""
        if not self._running:
            return
        self._heal_task = asyncio.create_task(self._self_heal(task))

    async def _self_heal(self, dead):
        reason = "worker exited"
        try:
            dead.result()
        except asyncio.CancelledError:
            reason = "worker task cancelled"
        except BaseException as e:
            reason = f"{type(e).__name__}: {e}"
        self._healing = True
        try:
            job = self._current_job
            if job is not None and job.status in ("running", "cancelling"):
                # 复用取消传播顺序（fixwave F2）：置 entry 标志 → 先 join
                # thread_future（ASR 线程跑到自然结束——worker 已死无人消费探针，
                # 结果由探针/收尾按 flag 丢弃；join 与取消路径同语义，仅设灾备
                # 上限不设短超时）→ 才收尾 exec_task。在 thread_future 完成前
                # 取消拥有者协程会提前释放模型计数/租约/线程正在读的文件。
                entry = self._lookup_entry(job)
                if entry is not None:
                    try:
                        entry["cancel_flag"].set()
                    except Exception:
                        logger.exception("job %s 自愈置取消标志失败", job.job_id)
                cf = await self._wait_thread_future(entry) if entry is not None else None
                if cf is not None:
                    try:
                        await asyncio.wait_for(asyncio.wrap_future(cf),
                                               self.heal_join_timeout)
                    except BaseException:
                        pass
                task = job.exec_task
                if task is not None and not task.done():
                    task.cancel()
                if task is not None:
                    try:
                        await task
                    except BaseException:
                        pass
                if job.status in ("running", "cancelling"):
                    self._mark_finished(job, "error", {
                        "status": "error",
                        "message": f"worker 异常退出，任务中止: {reason}"})
            self._current_job = None
        finally:
            self._healing = False
        if self._running:
            self._worker_task = asyncio.create_task(self._worker_loop())
            self._worker_task.add_done_callback(self._on_worker_done)
