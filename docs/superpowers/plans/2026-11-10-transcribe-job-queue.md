# 转录任务队列 + 进度轮询 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 transcribe-service 增加统一异步任务入口（`POST /jobs` 系）与轮询进度，noteflow 改两段式提交释放 worker 槽位，asr-engine 补真实进度与归属化取消。

**Architecture:** transcribe-service 新增 `jobs.py` JobManager（单 worker 串行消费、内存队列、TTL/字节预算清扫、异步采样器）；`process_transcription` 的 in-flight entry 升级为执行控制句柄（cancel_flag/request_id/thread_future/asr_started）；进度走「分槽快照 + 采样器」；取消走「pending-cancel 两段式 + 探针前置」。asr-engine 双锁分离 + 锁内事件安装 + scoped cancel。noteflow 新 `awaiting` 状态 + 全局 poller，attempt 观察者递增。

**Review 状态:** Plan Review R5 三模块全 Approved（评审轨迹：R1 haiku 20 条 → R2 opus 8+3 → R3 fable 10+3 → R4 sonnet 3+5 → R5 全过，advisory 全部折叠）

**Tech Stack:** Python 3.10 / FastAPI / pytest（transcribe、asr-engine，conda `funasr` 环境）；TypeScript / Nuxt Nitro / Prisma(SQLite) / vitest（mryk24）。

**Spec:** `docs/superpowers/specs/2026-11-10-transcribe-job-queue-design.md`（六轮评审通过版，锚点均经核实；实现争议以 spec 为准）

## Global Constraints

- **任务内一律不提交**（出门模式覆盖默认流程；提交仅在本计划全部完成、Code Review 通过后由主会话按功能点分次执行）
- **实施顺序 = 任务编号顺序**（Phase A 内依赖已按 1→2→3→4下载器→5后端→6分离事件→7执行worker 排布，评审 R3A 采纳）
- 生产基线 = asr-engine 后端；gguf 仅兼容路径（单测覆盖即可，不做真机验收）
- 既有四个同步转录端点与 pause 机制的可观察行为**完全不变**；遗留断连 `task.cancel()` 语义不变
- 部分转录不得入缓存/归档/作为成功响应；`thread_future` 完成前禁止取消拥有者协程、禁止释放模型计数/音频租约
- 锁序单向：只允许「转录锁 → 状态锁」；`/v1/progress`、`/admin/cancel` 只取状态锁
- `DownloadCancelled` 派生自 `BaseException`（穿透三处 `except Exception`：bilibili_base.py:113/229 及重试循环）
- job_id 字符集 `^[A-Za-z0-9._:@-]{1,128}$` 且完整值非 `.`/`..`；服务端视为不透明字符串
- 队列配置：`max_queue=1000`（含 running）、`record_ttl_minutes=60`、`max_result_bytes=256MB`；pending-cancel TTL 5min/容量 64；探测负缓存 10min；Retry-After 默认 30s（头+body 双写）
- 终态映射规则：执行体响应 dict `status == "error"` → job `error`，否则 `done`
- mryk24 零 DDL（status 为 String 列；attempt/nextSubmitAt/diarizeFailed 存 sysMeta）；SQLite JSON 无过滤 → 条件迁移一律「事务内读验 + JS 判断 + `updateMany` 锁定」；远端 DELETE 一律事务外
- noteflow 异步提交受全局开关 `noteflow.asyncTranscribe` 控制，部署默认 OFF；`persist=false`（quick 调用）恒走旧同步路径

## Review Focus

1. **取消传播顺序**：thread_future 完成前不得释放模型计数/租约/删在读文件——Task 2 的传播顺序测试钉住。
2. **部分转录污染缓存**：chunk 级取消后探针（`await diar_task` 之前）必须中止于缓存写入之前——Task 2/7 探针测试 + Task 10 集成断言（评审 R5A 勘误：该断言在 Task 10 的补项清单）。
3. **asr-engine 排队请求互覆取消事件**：锁内安装 + pending-cancel 命中——Task 11 并发测试钉住（评审 R2B 勘误任务号）。
4. **attempt 代次竞态**：旧 worker 迟到回执/取消不得作用于新一代（含成功回执）——Task 14/15 条件工具测试钉住。
5. **重启丢失语义**：未知 id 404 / query missing / noteflow 重启自愈链——Task 8/10 集成 + Task 15 自愈测试钉住。

---

## Phase A：transcribe-service（conda `funasr` 环境；测试命令 `python -m pytest tests/<file> -q`，仓库根执行）

### Task 1: 共享库 progress_cb 与取消标记

**Files:**
- Modify: `qwen_asr_gguf/inference/asr.py`（`transcribe` 约行 396、`asr` 约行 420、chunk 循环约行 485-490、取消 break 处）
- Test: `tests/test_asr_progress.py`

**Interfaces:**
- Produces: `QwenASREngine.transcribe(..., progress_cb=None)`、`asr(..., progress_cb=None)`；`progress_cb(done: int, total: int)` 每 chunk 边界调用一次（顺序值）；取消 break 时 `result.performance["cancelled"] = True`
- Consumes: 无（库内自足）

- [ ] **Step 1: 写失败测试**——用 `QwenASREngine.__new__(QwenASREngine)` 构造实例并初始化 `encoder`（桩）、`aligner=None`、`verbose=False`、`_cancel_event=None`（评审 R1A fold），monkeypatch `encoder.encode`/`_build_prompt_embd`/`_safe_decode` 为假实现；直接调 `asr(..., chunk_size_sec=1)`（该方法不读 `self.config`），喂 2 个 chunk 长度的假音频。断言：① `progress_cb` 收到 `(1,2)`、`(2,2)`；② 预置 `set_cancel_event(已置位 Event)` 后调用 `asr()`，返回结果的 `performance["cancelled"] is True` 且提前终止（done < total）。
- [ ] **Step 2: 运行验证失败**：`python -m pytest tests/test_asr_progress.py -q` → FAIL（TypeError: unexpected keyword `progress_cb`）
- [ ] **Step 3: 实现**——两方法加 `progress_cb=None` 形参并透传；chunk 循环每轮完成处 `if progress_cb: progress_cb(i + 1, num_chunks)`；取消 break 后、构造返回前置 `performance["cancelled"] = True`
- [ ] **Step 4: 运行验证通过**；再跑 `python -m pytest tests/ -q` 全绿（回归）

### Task 2: entry 控制句柄与取消传播（transcribe.py）

**Files:**
- Modify: `transcribe.py`（`_shared_transcription` 约行 1000-1011 entry 结构；`process_transcription` 签名约行 1013；`_process_transcription` 约行 1052-1291；`_annotate_cached_asr` 约行 961-983；`_run_diarization` 约行 819-836）、**`tests/test_transcription_safety.py` 与 `tests/test_diarization_orchestration.py` 的桩签名同步（CountingBackend/FakeBackend 加 `progress_cb=None, request_id=None` 形参——评审 R2A：否则本任务全量回归必红）**
- Test: `tests/test_entry_cancel.py`

**Interfaces:**
- Produces:
  - entry 字段：`cancel_flag: threading.Event`、`request_id: str`（uuid4 hex）、`thread_future: concurrent.futures.Future | None`、`asr_started: bool`、`backend`、`inflight_key`
  - `TranscriptionService.cancel_computation(inflight_key) -> dict | None`：置 cancel_flag；`asr_started` 时调 `backend.cancel_request(request_id)`；返回 entry（未找到返回 None）
  - `process_transcription(..., diarize=False, progress_cb=None, cancelled=None, diarize_wait_deadline=0.0)`（新增三参，向后兼容）
  - `JobCancelledError(BaseException)`（新异常，探针命中抛出；**派生 BaseException 以穿透 `_process_transcription` 外层 `except Exception` → error dict 的既有转换，保证 job 终态为 cancelled 而非 error**）
  - `TranscriptionService.build_inflight_key(audio_file_path, audio_url, bvid, audio_id, file_path_for_cache, context, diarize, no_cache) -> tuple`（组装完整 in-flight 键，含 running loop；**JobManager 以与调用 process_transcription 完全相同的实参调用本函数，键必然一致**——遗留先创建、job 后加入的场景由此命中同一 entry）
  - `_run_diarization(audio_file_path, audio_duration, wait_deadline=0.0)`：`DiarizationBusyError` 时每 30s 重试至 deadline（`time.monotonic()`），超期按既有降级返回；deadline=0 行为与现状一致
  - `_annotate_cached_asr(..., cancelled=None, wait_deadline=0.0)`：入口探针 + **wait_deadline 透传其内部 `_run_diarization`（评审 R2A 补：队列 diarize=true 缓存命中任务在混用期遇分离忙不得直接降级）**
  - **`diarize_only(..., wait_deadline=0.0)` / `diarize_merge_subtitle(..., wait_deadline=0.0)` / `_diarize_turns_or_error(..., wait_deadline=0.0)`（评审 R4A 补：队列官方字幕任务的有界退避链；默认 0 保持遗留行为不变）**
- Consumes: Task 1 的 progress_cb；Task 5 的 `backend.cancel_request`（本任务先留 base no-op 调用点，Task 5 落地实现）

- [ ] **Step 1: 写失败测试**（mock backend + 真 TranscriptionService）：
  - `test_probe_aborts_before_cache`：cancelled() 恒 True 的探针 + 桩 backend 返回正常 result → 断言 `JobCancelledError` 上抛、`cache_manager.save_transcript_to_cache` 未被调用（monkeypatch 计数）
  - `test_probe_position_before_diar_await`：diarize=True + 探针命中 → 断言 diar mock task 的 cancel 被触发（经既有 except 链）
  - `test_thread_future_join_before_release`：桩 backend 的线程 sleep 0.3s；取消后断言 `model_manager.release` 与租约释放在 `thread_future.result()` 完成之后才发生（事件顺序记录列表断言）
  - `test_annotate_probe`：ASR 缓存命中路径 + cancelled 探针 → `JobCancelledError` 且不进入分离
  - `test_busy_wait_deadline`：`_run_diarization` stub DiarizationBusyError 两次后成功；deadline=0 → 立即降级；deadline=65s → 重试后成功（sleep 用 monkeypatch 加速）
  - `test_cancel_computation`：entry 存在时 flag 置位、asr_started=True 时 backend.cancel_request 被调、未开始时未调；键不存在返回 None
  - `test_shared_entry_cross_caller`：同 identity 两次调用（模拟遗留先创建 + job 后加入）→ `cancel_computation(build_inflight_key(...))` 命中同一 entry
  - `test_jobcancelled_passthrough`：探针命中后外层不转 error dict（响应/异常类型断言为 JobCancelledError 贯穿）
- [ ] **Step 2: 运行验证失败**
- [ ] **Step 3: 实现**要点——`_shared_transcription` entry 扩展五字段；ASR 调用改经**模块级 `_ASR_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="asr")`（边缘注记：遗留并发转录 >4 路时在专用池排队，单卡+后端锁下实际影响趋零，评审 R5A 知晓项）**：`cf = _ASR_EXECUTOR.submit(functools.partial(backend.transcribe, file, None, context, progress_cb=progress_cb, request_id=entry.request_id))` + `await asyncio.wrap_future(cf)`（不依赖 loop 的 executor 访问 API；提交前置 `asr_started=True`、`entry.backend=backend`，thread_future 登记为 concurrent Future）；探针检查置于「executor future 返回后、`await diar_task` 前」与「字幕生成/缓存写入前」两处（命中抛 `JobCancelledError`，被既有 except 链捕获后 diar_task.cancel()、上抛）；取消路径对 `entry.thread_future` 先 `await asyncio.wrap_future(cf)` 再走 finally（保证释放顺序）；`_annotate_cached_asr` 加 cancelled 探针；遗留端点不传新参 → 行为不变
- [ ] **Step 4: 运行验证通过** + 全量回归 `python -m pytest tests/ -q`

### Task 3: jobs.py JobManager——存储、提交语义、清扫

**Files:**
- Create: `jobs.py`
- Modify: `config.py`（`jobs.*` 默认值兜底）、`config.yaml`（**本任务落 `jobs:` 段**；数值校验与注释归 Task 9——评审 R2A 澄清）
- Test: `tests/test_jobs_core.py`

**Interfaces:**
- Produces:
  - `@dataclass Job`：`job_id, source, request(dict), status, phase, percent, message, diarize_stage, submitted_at/started_at/finished_at, result, upload_path, inflight_key, progress 转录槽/分离槽(dict), backend_name`
  - `class JobManager`: `__init__(config)`；`async submit(source, params, job_id=None, upload_tmp_path=None) -> dict`（202/200-replay/422/429 形状按 spec §4.1）；`get(job_id) -> dict | None`（GET §4.2 形状）；`query(ids) -> {"jobs", "missing"}`（单批 ≤500 截断由调用方保证，此处全量处理）；`async cancel(job_id) -> dict`（§4.3 语义，执行联动在 Task 7 注入 callable——评审 R4A 勘误重编号残留）；`list_jobs(status=None) -> list`；`register_upload(path)/unregister_upload(path)`；`async start()/stop()`（worker+清扫+采样任务的拉起/停止钩子，执行体注入点 `self._executor_factory` 留桩，Task 7 实现）
  - identity 代理键：`(source, bvid+page|aweme_id|path, clamp(context), diarize, diarize_only, md5(json.dumps(body)), no_cache)`（`clamp_asr_context` 复用 transcribe.py）
  - 清扫（60s asyncio 任务）：TTL 整条移除；字节预算从最旧终态整条驱逐（单条超限保留）；upload 登记维护
- Consumes: 无（Task 7 注入执行）

- [ ] **Step 1: 写失败测试**（纯逻辑，直接构造 JobManager 不 start）：job_id 校验（`#`/`/`/`..`/`./超长` → ValueError→422 路径；合法通过）/同 id 重放（活跃+终态均返回完整 GET 形状+`dedup:true`，jobs dict 长度不变）/identity 去重（无 job_id 时同 identity 活跃命中）/队列满 429（max_queue 桩为 1）/TTL 驱逐（record_ttl 桩为 0 整条消失）/字节预算驱逐（总限桩小值 → 最旧终态先没；单条超限保留）/`cancelling` 计入活跃（max_queue 计数含之）/query 命中+missing 形状
- [ ] **Step 2: 验证失败**
- [ ] **Step 3: 实现**（queue 满判定含 running 与 cancelling；`PAUSED` 门在 Task 7 worker 内——评审 R4A 勘误重编号残留）
- [ ] **Step 4: 验证通过**

### Task 4: 下载器进度与专用取消异常

**Files:**
- Modify: `downloaders/bilibili_base.py`（`download_audio` 约行 63-124、`download` 约行 125+）、`downloaders/bilibili_video.py`（**覆写 `download(id, cookie, save_dir, page=1)` 固定形参转发——不同步会静默丢 BV 主路径的 progress_cb/cancel_flag，评审 R5A 补**）、`downloaders/base.py`（`DownloadCancelled` 定义处 + **`Downloader.download` 两层转发，评审 R3A 补**；**`BilibiliDownloader.download_bilibili_audio` 实际定义在 base.py:72-83，`__init__.py` 仅 re-export——评审 R4A 勘误**：公共入口签名加 `progress_cb=None, cancel_flag=None` 并转发）
- Test: `tests/test_download_cancel.py`

**Interfaces:**
- Produces: `class DownloadCancelled(BaseException)`（定义于 `downloaders/base.py`）；`download_audio(audio_url, cookie, filename, progress_cb=None, cancel_flag=None)`、`download(id, cookie, save_dir="tmp", page=1, progress_cb=None, cancel_flag=None)`（progress_cb(downloaded, total)）、**`BilibiliDownloader.download_bilibili_audio(bvid, cookie, save_dir, page, progress_cb=None, cancel_flag=None)` 公共入口全链转发（BV/ep 两路径）**
- Consumes: 无

- [ ] **Step 1: 写失败测试**：stub `requests.get` 返回分块流 → percent 回调序列正确；`cancel_flag` 置位 → `DownloadCancelled` 上抛且 `except Exception`/重试循环/外层 catch 均不吞（直接断言异常类型穿透 `download()`）；**公共入口测试（评审 R3A 补）：经 `download_bilibili_audio` 走 BV 与 ep 两路径，回调与取消信号到达 iter_content 层**；抖音侧无改动（既有签名）
- [ ] **Step 2: 验证失败**
- [ ] **Step 3: 实现**（iter_content 循环内查 flag、算百分比；**`DownloadCancelled(BaseException)` 定义在 `downloaders/base.py` 顶部 import 语句之前**——base.py 顶部导入 bilibili_video/episode，而后者导入 bilibili_base，类若定义在导入之后会成环（评审 R2A fold）；BaseException 使三处 except Exception 天然放行——无需改动 catch 点，测试验证之；**`Downloader.download` 与 `download_bilibili_audio` 两层参数转发**）
- [ ] **Step 4: 验证通过**

### Task 5: 三后端签名统一与 asr_engine_backend 增强

**Files:**
- Modify: `backends/base.py`（签名 + `cancel_request` no-op）、`backends/gguf_backend.py`（progress_cb 透传 + pending-cancel 进程内集合 + transcribe_lock 内挂接/清除）、`backends/asr_engine_backend.py`（request_id form 字段、cancelled 解析、cancel_request、**capability 探测/负缓存/降级**——评审 R1A/B 补）、`backends/qwen3_backend.py`/`funasr_backend.py`（签名加参，body 不变）
- Test: `tests/test_backends_progress_cancel.py`

**Interfaces:**
- Produces: `ASRBackend.transcribe(self, audio_file, language=None, context=None, progress_cb=None, request_id=None)`；`ASRBackend.cancel_request(self, request_id) -> None`（base no-op；gguf：pending 集合 + 持锁挂接时命中即置位；asr-engine：POST `/admin/cancel` json `{"request_id": ...}`）；asr_engine_backend 将 verbose_json 顶层 `cancelled` 映射进 `performance["cancelled"]`；**`ASRBackend.probe_capability() -> "progress+scoped" | "legacy"`**（**能力映射明确（评审 R4A fold）：gguf → 恒 progress+scoped；asr-engine → 探测决定；qwen3/funasr → 恒 legacy**。asr_engine_backend 实现：首次 GET `/v1/progress` 探测，404 → 负缓存 10min 后重探，成功 → 正缓存持续；legacy 态 = JobManager 关 percent、取消降级「标志+join」）
- Consumes: Task 1（gguf engine progress_cb）

- [ ] **Step 1: 写失败测试**：gguf 桩 engine 断言 progress_cb 透传 + pending-cancel「登记时命中即置位、未命中不置位、TTL/容量清理」（时间用注入 clock）；asr_engine_backend 用 httpx MockTransport 断言 form 含 request_id、响应含顶层 `"cancelled": true` 时 performance 映射、cancel_request 发送 scoped body；**capability：404 → legacy 降级生效、clock 推进后重探、200 → 正缓存；legacy 态 cancel_request 不发 HTTP**（评审 R1A/B 补）；qwen3/funasr 新参默认不破坏旧调用
- [ ] **Step 2: 验证失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 验证通过**

### Task 6: diarization stage 事件

**Files:**
- Modify: `diarization/manager.py`（`diarize()` 内部：解码/分割/嵌入/聚类/重建五段边界回调）、`diarization/worker.py`（`run()` 加 `stage_cb=None`；`_main` 子进程转发 `{"event":"stage","name":...}`）、**`transcribe.py`（评审 R1A 补：`_run_diarization(..., stage_cb=None)` → `_diarize_samples(..., stage_cb=None)` → `worker.run(..., stage_cb=None)` 穿透链；diarize_only/merge 同链）**
- Test: `tests/test_diar_stages.py`

**Interfaces:**
- Produces: `DiarizationWorker.run(audio_path, timeout, stage_cb=None)`；`DiarizationManager.diarize(samples, stage_cb=None)`（复用既有 `_log_pipeline_stage` 钩子位）；`_run_diarization/_diarize_samples/diarize_only/diarize_merge_subtitle` 全链 stage_cb 透传
- Consumes: 无

- [ ] **Step 1: 写失败测试**：父进程侧——桩子进程连接依序发 decoded/stage/stage/result 事件 → stage_cb 收到序列；manager 侧——`diarize()` 用 monkeypatch 桩各阶段函数 → stage_cb 五段序列；**链路测试（评审 R1A 补 + R3A 扩展）**：`process_transcription` → `_run_diarization` → `_diarize_samples` → `worker.run` 的 stage_cb 参数**全链逐层到达**（monkeypatch 断言调用形状，不再只测 `_run_diarization` 以下）；**既有分离测试桩同步（评审 R3A fold）**：`tests/` 中 `_diarize_samples(path, timeout)` 替身与 `FakeManager.diarize(samples)` 加 `stage_cb=None` 形参
- [ ] **Step 2: 验证失败**
- [ ] **Step 3: 实现**——五段中解码（`_main`）、分割/嵌入（pyannote hook）、重建（自研子类直发）有天然回调位；**「聚类」段无现成钩子（评审 R2A 提示）**——在 reconstruct 边界或管线缝隙补发即可，五段序列测试桩已定契约
- [ ] **Step 4: 验证通过**

### Task 7: JobManager——执行 worker、取消生命周期、采样器

**Files:**
- Modify: `jobs.py`
- Test: `tests/test_jobs_exec.py`

**Interfaces:**
- Produces: `JobManager.start()` 拉起三任务：worker（单消费循环）、清扫（Task 3 已有）、采样器（1s）；`self._executor_factory(job, sinks) -> 协程` 由 `server.py` 注入（Task 8 完成真实注入，本任务测试用桩执行体）
- Consumes: Task 2 `cancel_computation(inflight_key)`/`JobCancelledError`；Task 3 全部；Task 4 下载器 progress/cancel；Task 5 backend progress_cb/request_id/cancel_request；Task 6 stage_cb

- [ ] **Step 1: 写失败测试**（桩执行体/桩下载器/桩 backend，`configure` 注入）：
  - `test_status_error_mapping`：执行体返回 `{"status": "error", ...}` → job error 且 result 为该 dict；正常 dict → done
  - `test_cancel_queued`：出队即时 cancelled；取件竞态（cancel 后 worker 取件跳过）
  - `test_cancel_running_sole_self_serial`：桩 backend 带锁语义 → DELETE 后 status=cancelling → thread_future join → cancelled；`transcription_service.cancel_computation` 被调；**加 diarize=true 变体（评审 R4A fold）：钉住「running·含分离 → join」矩阵行——ASR join 与分离进程回收顺序**
  - `test_cancel_running_shared_self_serial`：entry.waiters 桩 >1 → 即时 cancelled（不 join）
  - `test_cancel_running_shared_nonserial`：qwen3/funasr → cancelling + join 后才取下一件
  - `test_probe_blocks_cache`：独占取消 + backend 返回 `performance.cancelled` → job cancelled 且缓存写入函数未被调
  - `test_pause_gate`：pause_manager.is_paused 桩 True → 不取件；False 后继续
  - `test_upstream_paused_selfcool`：执行体抛 `UpstreamPausedError(resume_at=...)` → error + paused 三键；`_query_asr_paused` 桩 True → 不取下一件
  - `test_sampler_slots`：转录槽/分离槽分别写 → GET 输出拼装两槽；asr-engine backend_name 时走 HTTP 桩且 request_id 不匹配不采信、phase 非 transcribing 不应用
  - `test_worker_selfheal`：worker 任务被外部 cancel → 当前 job 落 error、无遗留引用后重拉
  - `test_cancel_annotate_running`（评审 R1A 补）：冷却重试路径（`_annotate_cached_asr`）分离已启动 → 取消 = **Job 自身执行协程 task.cancel + await**（JobManager 持有每 job 的 `_execute` Task 句柄）→ 分离进程回收 → cancelled
  - `test_cancel_diarize_only_running`（评审 R1A 补）：无 entry 路径同上句柄机制
  - `test_cancel_legacy_created_shared`（评审 R1A 补）：遗留先创建计算 → job 加入 → 遗留离开 → job 独占取消经 `build_inflight_key` 命中 entry
  - `test_cancel_running_sole_nonserial`（评审 R2A 补）：独占 × qwen3/funasr 放弃式 → `cancelling` → join 自然结束 → 探针按 `entry.cancel_flag` 丢弃结果 → `cancelled` 且不入缓存
  - `test_upload_tmp_cleanup`（评审 R2A fold）：upload job 终态后内部键文件删除
  - `test_probe_capability_cache`（评审 R1B 补）：asr_engine backend `/v1/progress` 404 → capability=legacy（percent 关闭、取消降级标志+join）；负缓存 10min（注入 clock）后重探恢复
  - `test_download_cancel_cleanup`（评审 R1A 补）：下载中取消 → `.part`/部分文件被清理（executor 捕获 `DownloadCancelled` 后删 tmp）→ cancelled
- [ ] **Step 2: 验证失败**
- [ ] **Step 3: 实现**（**pause 门与上游探询走注入 callable（`_pause_gate`/`_upstream_pause_probe`，Task 8 装配传入）——jobs.py 直接 import server.py 单例会成环，评审 R5A fold**；worker 循环 + `_execute` 按 source 分派复用 `diarize_only`/`process_transcription`/webdav 归档/douyin to_thread/upload 内部键；**执行协程句柄 = 每 job 的 `_execute` asyncio.Task，由 JobManager 持有**——无 entry/无 ASR 路径的取消即对该 Task cancel + await；取消矩阵按 spec §4.3，**共享取消「自串行 vs 非自串行」二分经 `probe_capability()` 值推导（legacy → join），不按 backend 名称另建判断**（评审 R2A fold）；`cancelling` 态；`webdav` 成功后 `move_to_processed`；bilibili/douyin 注入 `timing.download`；采样器独立任务；**executor 捕获 `DownloadCancelled` → 清理部分文件 → cancelled**；**capability 探测/负缓存/降级消费 Task 5 的 probe 结果**）
- [ ] **Step 4: 验证通过** + 全量回归

### Task 8: server.py 端点与拦截

**Files:**
- Modify: `server.py`（新增四个端点 + PAUSED_REJECT_PATHS 扩 `"/jobs","/jobs/upload"` + JobManager 装配与 `startup` 拉起/`shutdown` 停止 + `cleanup_stale_tmp_files(protected)` 活跃感知 + 真实执行体工厂注入）
- Test: `tests/test_jobs_api.py`

**Interfaces:**
- Produces: `POST /jobs`（Pydantic `JobSubmitRequest`：source 判别 + 各 source 字段校验复用现有模型字段）、`POST /jobs/upload`（multipart）、`POST /jobs/query`、`GET /jobs/{job_id}`、`GET /jobs`、`DELETE /jobs/{job_id}`；执行体工厂按 spec §6 复用 `diarize_only`/`process_transcription`/`move_to_processed`
- Consumes: Task 3/7 全部；Task 4/5/6 的回调链

- [ ] **Step 1: 写失败测试**（TestClient + monkeypatch mock backend，覆盖 spec §9 API 集成清单）：四源提交→终态全链路（桩执行体即时完成）/result 同构断言/404/暂停提交 503/query 命中+missing/**query 单批 >500 行为断言：静默截断，超出部分按 missing 返回（评审 R4A fold + R5A 定死语义，与终态驱逐同语义）**/GET cap 规则与 cookie 不回显/**提交 job_id 不合法 422；GET/DELETE 未知 id 404（评审 R4A 勘误：422 仅限提交格式不合法）**/DELETE 三态响应/`timing.download` 注入存在/upload 同 id 重放时新上传文件立即删除（tmp 目录断言）/**队列满 API 层断言：HTTP 429 且 `Retry-After` 响应头与 body 字段一致（评审 R1A fold）*/*/**`cleanup_stale_tmp_files(protected)` 行为测试：活跃路径与 `.job-*` 别名被跳过、未登记超龄文件被清（评审 R2A fold）**
- [ ] **Step 2: 验证失败**
- [ ] **Step 3: 实现**——`cleanup_stale_tmp_files(protected)` 活跃感知：protected = JobManager 活跃 upload 路径集，**跳过活跃登记路径及其 `.job-*` 别名；未登记的超龄 `.job-*` 照常清理**（评审 R2A 修正措辞：无条件跳过与 24h 兜底矛盾）
- [ ] **Step 4: 验证通过** + 全量回归

### Task 9: config、docs 归档与过时表述更新

**Files:**
- Modify: `config.yaml`（`jobs:` 段 + `backend.name` 注释改为「asr-engine 为生产基线，gguf 为兼容路径」）、`CLAUDE.md`（后端表格与 GGUF 段落更新为「gguf 兼容路径」；新增 `/jobs` 端点行与 docs 索引行）
- Create: `docs/2026-11-10_transcribe-job-queue.md`（评审 R1A 补：项目规范要求的归档文档，实现完成定稿后撰写——决策/坑位/验收记录）
- Test: 无独立测试（config.py 已有点号访问兜底默认值，Task 3 覆盖）

- [ ] **Step 1: 更新 config.yaml 与 CLAUDE.md（按 Global Constraints 数值逐字落地；校验 Task 3 落的 `jobs:` 段数值与注释——评审 R2A 澄清：Task 3 落段、本任务校数值；CLAUDE.md docs 索引指向归档文档，不留失效链接）**
- [ ] **Step 2: 归档文档成稿（收尾时补验收结果），并显式记录 spec/plan 裁定两笔：①ASR 提交用模块级 `_ASR_EXECUTOR` + `asyncio.wrap_future`（spec §4.3 的 `run_in_executor` 字样以本裁定为准，评审 R2A fold）；②asr-engine 版本偏斜（legacy）时共享取消走 join——比 spec §4.3 矩阵「asr-engine=自串行不 join」更保守，取 §6.7 降级口径（评审 R5A fold）**
- [ ] **Step 3: 验证**：`python -m pytest tests/ -q` 全绿（确认 config 加载不回归）

### Task 10: 集成回归与真机手动验证

**Files:**
- Modify: `tests/test_jobs_api.py`（补 §9 集成清单残余项：上游暂停 error 元数据+自冷、取消期间 503 穿透回归、**取消后转录不入缓存且 webdav 不归档的集成断言（评审 R1A 补——手动检查不能替代该回归）**）
- 手动（**推迟到 Task 11 完成后执行，评审 R4A fold——scoped cancel 依赖 Phase B**；需本机服务重启）：重启 `tmux transcribe` 与 `tmux asr` → **B站小样**（评审 R1A 修正：webdav 无下载阶段，验下载进度必须用 B站）`POST /jobs` → 轮询 downloading percent → transcribing chunk percent → done → result 与同步端点一致 → **webdav 小样**轮询 transcribing percent → done → `DELETE /jobs/{id}`（running）确认 chunk 边界取消 + `cache/` 无部分结果残留 → `tmux asr` 日志确认 scoped cancel 命中

---

## Phase B：asr-engine（仓库 `/home/zhenyi/ownprojects/asr-engine`；测试 `python -m pytest tests/<file> -q`）

### Task 11: 双锁、锁内安装、pending-cancel、scoped cancel、/v1/progress

**Files:**
- Modify: `asr_engine/engine.py`（`__init__` 加 `_state_lock`/`_active_request_id`/`_progress`/`_pending_cancels: dict[str, float]`；`transcribe(..., request_id=None)`：锁内安装（事件/归属/进度/pending 命中即置位并移除）+ 释放前清除——修复现状锁外安装缺陷；`progress_cb` 透传共享库更新 `_progress`；`cancel()` 保留旧无作用域行为）
- Modify: `asr_engine/server.py`（`/v1/audio/transcriptions` 加 `request_id: Optional[str] = Form(None)` 透传；`GET /v1/progress`（`_check_token`）；`POST /admin/cancel` 改收可选 json `{"request_id"}`（**形态 `Optional[Body(None)]`，防旧调用方无 body POST 被 422 拒，评审 R5B fold**）+ `_check_token` + 状态锁内归属校验/记集合（**engine 未加载窗口带 request_id 时仍记入 pending-cancel——正是「DELETE 早于归属注册」要覆盖的场景；not-loaded 响应形状 builder 自择并留一句，评审 R5B fold**）；pending 清理：TTL 300s/容量 64，校验时顺带 pruning）
- Test: `tests/test_progress_and_scoped_cancel.py`

**Interfaces:**
- Produces: `GET /v1/progress` → `{"active": bool, "request_id": str|null, "chunk": int|null, "total": int|null}`；`POST /admin/cancel` body `{"request_id": str?}` → `{"ok", "matched"}`；引擎面对旧共享库（无 progress_cb）不传该参即可（同树部署，部署顺序：共享库先落盘）
- Consumes: Task 1（共享库，部署顺序约束）

- [ ] **Step 1: 写失败测试**（**并发测试用真实 `ASREngineWrapper`、仅替换底层推理引擎，`threading.Barrier`/Event 控时序——评审 R3B fold：整 mock wrapper 或纯 sleep 覆盖不了锁内安装与清除**；桩 wrapper 内部循环尊重 cancel event、sleep 让转录可观察）：
  - `/v1/progress` 活跃（active/request_id/chunk/total）与空闲（active:false）两态；无 token → 401
  - pending-cancel 归属匹配 → 置位、返回 matched:true；不匹配 → 记 pending 且不动在跑事件；**随后同 request_id 的转录登记时命中 pending → 立即取消并移除条目**；**pending TTL 过期清理 / 容量 64 逐最旧——时间用注入 clock/monkeypatch，同 Task 5 模式（评审 R5B fold）**
  - pending TTL 过期清理 / 容量 64 逐最旧（**时间用注入 clock/monkeypatch，同 Task 5 模式，评审 R5B fold**）
  - **并发排队不互覆**：请求 A 在跑、请求 B 排队提交 → A 的事件引用保持有效（A 可被取消）；B 登记/清除不破坏 A
  - 旧调用方（无 request_id）：行为与现状一致（取消=无作用域）
  - `/admin/cancel` 无 token → 401（评审 R1B fold：与 /v1/progress 对齐）
  - **progress_cb 特性检测（评审 R1B fold）**：旧共享库（`QwenASREngine.transcribe` 签名无 progress_cb，经 `inspect.signature` 判定）→ 引擎不传该参、转录正常完成
  - **request_id 端点级断言（评审 R2B 补）**：经 `/v1/audio/transcriptions` multipart 带 `request_id` 提交（桩循环挂起）→ `GET /v1/progress` 返回该 request_id（防 server.py Form 接线漏传静默失效）。**TestClient 单请求会阻塞：POST 放侧线程跑、主线程轮询后放行；或简化为断言 manager.engine.transcribe 的 kwargs 含 request_id（评审 R5B fold）**。**TestClient 单请求会阻塞：POST 放侧线程跑，主线程等信号后放行；或简化为断言 manager.engine.transcribe 的 kwargs 含 request_id（评审 R5B fold）**
- [ ] **Step 2: 验证失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 验证通过** + `python -m pytest tests/ -q` 全绿（含既有 test_admin_pause/test_transcribe_endpoint 回归；**点名补跑 `tests/test_engine_dual_path.py`、`tests/test_engine_with_prompt.py`——两文件直接断言 wrapper 调用形状，若 fake engine 配显式签名需同步更新，评审 R2B fold**）

---

## Phase C：mryk24 / noteflow（仓库 `/home/zhenyi/ownprojects/mryk24`；测试 `pnpm vitest run <file>`；最终 `pnpm build`）

### Task 12: 错误类型、条件工具、scheduler 扩展

**Files:**
- Modify: `server/utils/noteflow/errors.ts`（`JobSubmittedError`/`JobQueueFullError` + `isJobSubmitted`/`isJobQueueFull`，仿 `isServicePaused` 的 name 判定）
- Modify: `server/utils/noteflow/scheduler.ts`（新增：`conditionalTransition(taskId, expectedStatus, expectedAttempt|null, patch)`——交互式事务内读验 + `updateMany` 锁定，支持传入事务客户端；`persistTaskProgress(taskId, expectedAttempt, patch)`（写 itemMeta/attempt，校验 processing+attempt）；`markTaskAwaiting(taskId, expectedAttempt)`；`bumpAttempt(taskId)`；`markTaskFailedConditional/CancelledConditional/DeferredConditional/CompletedConditional`（条件化变体，不动既有无条件函数）；**`updateTaskSysMeta` 改造为事务内读验合并（评审 R3C 补：现状事务外读+整段替换，迟到写会把旧 attempt 写回、撤销递增）**；**`fetchAndLockNextTask` 领取门控（评审 R2C 补，最严重项）：候选 JS 侧过滤 `nextSubmitAt > now`（事务内 findMany pending → 逐个解析 sysMeta → 对通过者条件 updateMany 锁定），防止 429 任务回 pending 后被立即重领形成零延迟热循环**）
- Test: `tests/unit/noteflow/scheduler-conditional.test.ts`（vitest，mock prisma 事务，沿用现有模式；**含领取过滤与空队列场景，评审 R2C 补**）

**Interfaces:**
- Produces: 上述导出名；`conditionalTransition` 语义 = 校验失败返回 null（调用方按「本代作废」处理）
- Consumes: 无

- [ ] **Step 1: 写失败测试**：attempt 匹配/不匹配、status 匹配/不匹配、事务客户端透传、JSON 整字段替换语义（patch 合并在 JS 侧完成）；**领取门控：nextSubmitAt 未到期任务被跳过（其余候选正常领取）、全部未到期 → 不领取不报错（空队列场景，评审 R2C 补）**；**updateTaskSysMeta 交错测试（评审 R3C 补）：读取旧元数据 → reset（attempt+1）→ 迟到写入 → 断言旧 attempt 未被写回**
- [ ] **Step 2: 验证失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 验证通过**（`pnpm vitest run tests/unit/noteflow/scheduler-conditional.test.ts`）

### Task 13: 提交助手与 extractor 异步分支

**Files:**
- Create: `server/utils/noteflow/job-submit.ts`（`submitTranscriptionJob(opts) -> {jobId}`：POST `/jobs`（job_id=`${taskId}-${attempt}`，getTranscribeBaseUrl 推导）；202/200-dedup → 抛 `JobSubmittedError`；429 → 条件转 pending + nextSubmitAt + 每事件独立 `setTimeout`（不复用 scheduleFallbackWake）→ 抛 `JobQueueFullError`；提交期 503 → 既有 `parsePausedResponse` → `ServicePausedError`；网络错误 → 原样上抛（不转失败，同 attempt 安全重试）；**同文件导出 `deleteRemoteJob(jobId)`**）
- Modify: `server/utils/noteflow/pipeline.ts`（PipelineContext 增 `persist: boolean`，默认 true 透传；**`ctx` 接线补齐 webdav/douyin 两条 step→extractor 传递链（评审 R3C 补：现只传 signal/taskId/itemMeta，ctx.persist 不会自动可达）**）、`server/utils/noteflow/extractors/bilibili.ts`（`transcribeAudio` 约 779/893、`diarizeSubtitle` 约 937/1171、`1052`/`1210` 调用点：`ctx.persist && config.asyncTranscribe` → 独立提交分支置于重试循环外；`bilibili.ts:829-833` 内层重包装 catch 不经独立分支即自然规避）、`extractors/webdav.ts`（约 114/194）、`extractors/douyin.ts`（约 57/130）、**`steps/bilibili-extract.ts`（约 79 forceManualTranscribe 直调路径，评审 R1C 补：与 6 透传点同规则改造；约 129 行 `extractBilibiliSubtitle` 调用的 options 需同步扩展携带 persist/attempt——否则 1052/1210/937 分支无法判，评审 R5C fold）**、**`steps/webdav-extract.ts` 与 `steps/douyin-extract.ts`（评审 R3C 补：ctx 透传接线）**——同步路径保持原样；**diarizeFailed 消费端（评审 R3C 补）：重新提取时读取 `sysMeta.diarizeFailed` → 跳过 diarize job 提交（循环闭合），用户 reset 清标记后恢复正常提交**
- Test: `tests/unit/noteflow/job-submit.test.ts` + `tests/unit/noteflow/extractor-async-branch.test.ts`

**Interfaces:**
- Produces: `submitTranscriptionJob({ taskId, attempt, source, params, context }) -> { jobId }`；`deleteRemoteJob(jobId)`；extractor 在 `ctx.persist===false` 或开关 OFF 时走旧路径（行为不变）；**diarizeFailed 跳过分支**
- Consumes: Task 12（`persistTaskProgress`——**评审 R1C 补：提交前必须先调它落库基线元数据 + attempt，条件写失败（返回 null）则本代作废、不发 POST**）

- [ ] **Step 1: 写失败测试**：202→JobSubmittedError（payload 含 jobId）/200 dedup 同等/429→JobQueueFullError 且 pending+nextSubmitAt 落库且定时器已设（fake timers）/503→ServicePausedError/网络错误原样；**基线先行：persistTaskProgress 失败 → 断言未发 POST（评审 R1C 补）**；extractor 分支：开关 ON + persist → 提交不 fetch 转录；persist=false 或 OFF → 走旧 fetch（mock 断言调用形状不变）；forceManualTranscribe 路径同分支断言；**diarizeFailed：首败后再提取不提交分离、reset 清标记后允许重提交（评审 R3C 补）**；**控制异常透传链（评审 R3C fold）：真实 extractor（B站最外层 catch 在内）→ worker 分派，JobSubmittedError 不被包装成普通失败**
- [ ] **Step 2: 验证失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 验证通过**

### Task 14: worker 分派与条件化收尾

**Files:**
- Modify: `server/api/noteflow/internal/worker.ts`（领取捕获 `expectedAttempt = sysMeta?.attempt ?? 1` 并随 task 传递；错误分派加 `JobSubmittedError → markTaskAwaiting`、`JobQueueFullError → 无操作`；收尾六处——取消/失败/deferred/**成功回执 markTaskCompleted**/安全网 processing→pending——全部改走 Task 12 条件化变体）
- Test: `tests/unit/noteflow/worker-dispatch.test.ts`

**Interfaces:**
- Produces: 分派契约（本任务）；`expectedAttempt` 经 task 对象传递给 steps（Task 13 的 extractor 消费）
- Consumes: Task 12/13

- [ ] **Step 1: 写失败测试**：JobSubmittedError → awaiting（attempt 校验）；JobQueueFullError → 状态不变、无 retryCount 变化；旧代收尾（attempt 不匹配）全部作废（含成功回执）；暂停 503 → 既有 deferred 分支不受影响
- [ ] **Step 2: 验证失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 验证通过**

### Task 15: 全局 poller 与启动编排

**Files:**
- Create: `server/utils/noteflow/job-poller.ts`（`startJobPoller()/stopJobPoller()`：3-5s 间隔、禁止重入、每请求 30s 超时、逐任务 try/catch、按 500 分批 `POST /jobs/query`；分支：done→取 result→per-source 映射写 rawData（webdav 含 formatWebdavTranscriptBody + duration 回填；diarize_only `rawData=JSON.stringify({body})`）→条件 `startStep='cleaning'+pending`→triggerWorker；error→取 result→paused→条件 Deferred+**`scheduleFallbackWake` 兜底定时（评审 R3C 补：paused-error 改由 poller 消费后不再经 worker 分支，不设兜底则恢复推送失效即永久挂起）** / 普通→条件 Failed+attempt+1→**实际转 pending 时 triggerWorker（评审 R3C 补：markTaskFailed 未达上限转 pending，异步等待期 worker 已退出，必须补触发）** / diarize_only→写 diarizeFailed+attempt+1→pending→triggerWorker；cancelled→条件 Cancelled+attempt+1；missing→pending+retryCount+1+attempt+1→triggerWorker；**统一规则：凡实际转为 pending 的结果一律 triggerWorker（评审 R3C 补）**；网络失败≠missing；取 result 404 按 missing；心跳 + 监督者（>2×间隔重启 poller）；24h awaiting 硬上限先单查确认）
- Modify: `server/utils/noteflow/config.ts`（**`asyncTranscribe` 落 DB 键值配置体系三处：`DEFAULT_CONFIG` 默认 false / `NoteFlowConfig` 接口 / `getAllConfig` 读取——评审 R2C 勘误：noteflow 配置是 NoteFlowConfig 表而非 runtimeConfig；**写入口（saveConfig 或设置页分支）builder 自择并在归档记录一笔（评审 R5C fold）**）、`server/plugins/noteflow-init.ts`（插桩：resetAllProcessingTasks 之后、pending 触发之前——①回滚迁移（OFF 时：awaiting→pending+attempt+1+尽力 DELETE remote）②启动扫描（ON）③恢复门控 pending 最早到期唤醒 ④启动 poller 立即跑一轮 ⑤既有 pending 触发）
- Test: `tests/unit/noteflow/job-poller.test.ts`

**Interfaces:**
- Produces: `startJobPoller/stopJobPoller`；config `noteflow.asyncTranscribe`
- Consumes: Task 12/13/14

- [ ] **Step 1: 写失败测试**：done/error(paused/普通/diarize_only)/cancelled/missing 五分支 + 网络失败不误判 + 分批 + 条件回写（awaiting+attempt 校验）+ 看门狗重启 + 429 定时器到期重提 + 回滚迁移 + **paused 分支 attempt+1 断言（评审 R1C 补：恢复后旧 job_id 会被重放，必须新号）+ paused 消费后 `scheduleFallbackWake` 兜底定时已设（fake timers，评审 R3C 补）** + **普通错误未达上限转 pending 后 triggerWorker 已触发（评审 R3C 补）** + **启动恢复（评审 R1C 补）：开关 ON 重启 → 启动扫描现存 awaiting 一轮 + 恢复受门控 pending 的最早 nextSubmitAt 唤醒（内存定时器重启即失，必须重排）**
- [ ] **Step 2: 验证失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 验证通过**

### Task 16: 取消/重置/删除入口扩展

**Files:**
- Modify: `server/api/noteflow/[id]/cancel.post.ts`（**实际路径非 internal/ 下，评审 R1C 勘误**；扩展 awaiting；processing 且 sysMeta.attempt 存在 → 尽力 DELETE remote）、`server/api/noteflow/batch/cancel-all.post.ts`（批量取消实际入口；**awaiting 逐任务读 sysMeta.attempt 拼 `{taskId}-{attempt}` 再 DELETE——updateMany 携带不了 per-task 远端操作，评审 R5C fold**）、`server/api/noteflow/[id].delete.ts`（删除实际入口：awaiting 先 DELETE remote）、`server/api/noteflow/[id]/reset.post.ts`（**直接更新 DB 的单任务重置，评审 R1C 补列**：attempt+1 + 清 diarizeFailed + 事务外 DELETE remote）、`server/utils/noteflow/reset.ts` 的 `prepareSysMetaForReset`（attempt+1 + 清 diarizeFailed，覆盖 ①②③）、`server/api/noteflow/create-batch.post.ts` forceReset 调用点（同前）、`server/api/noteflow/[id]/resummarize.post.ts`（extracting 型 attempt+1 + 清 diarizeFailed）——**远端 DELETE 一律事务外（经 `deleteRemoteJob`）**
- Test: `tests/unit/noteflow/reset-entries.test.ts`

**Interfaces:**
- Consumes: Task 12 条件工具；Task 13 的 `deleteRemoteJob(jobId) -> Promise<void>`（`job-submit.ts` 导出，尽力而为：404 静默、网络错误静默）
- Consumes(补): Task 15 的回滚迁移复用 `deleteRemoteJob`

- [ ] **Step 1: 写失败测试**：六入口逐一断言 attempt+1 + diarizeFailed 清除 + remote DELETE 调用（事务外时序）；cancel 对 awaiting 与 processing-已提交竞态；skip.post 不受影响
- [ ] **Step 2: 验证失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 验证通过**

### Task 17: 前端状态与构建验证

**Files:**
- Modify: `utils/noteflow-status.ts`（`awaiting` → 「等待转录」展示/色值/计数）、`pages/noteflow/tasks.vue`（评审 R1C 补：awaiting 计数、筛选/状态显示、移动端与桌面端取消按钮，沿用现有组件）、**计数接线定位与补齐（评审 R3C 勘误：`list-query.ts` 仅 13 行 `parseIdsParam`、非计数来源——定位任务列表 API 实际生成 counts 的实现，将 awaiting 计入初始表与返回值；**接口断言：零 awaiting 时仍返回 `awaiting: 0`——挂到现有 `tests/unit/noteflow/fetch-task-list-batch-ids.test.ts` 或 `progress.test.ts`（评审 R5C fold）**）、**`composables/useNoteFlow.ts`（约 155 TaskCounts 接口键集）**、**前端 `NoteFlowTask.status` 联合类型扩展（评审 R3C fold）**、`tests/unit/noteflow/status-meta.test.ts`（评审 R1C 补：覆盖新状态）
- Test: `pnpm vitest run`（全量回归）+ `pnpm build`

- [ ] **Step 1: 状态映射与按钮**
- [ ] **Step 2: `pnpm vitest run` 全绿 + `pnpm build` 通过 + typecheck（如脚本存在）**

---

## 收尾（阶段 5 完成后、阶段 6 之前）

### Task 18: 三仓库手动冒烟（真机，需重启本机服务）

- [ ] 重启 `tmux transcribe`（`bash start.sh`）与 `tmux asr`；**部署顺序：transcribe-service 仓库（共享库）先落盘 → 再重启 tmux asr（评审 R2B fold，spec §6.5）；重启用真实进程操作（`tmux kill-session` + `start.sh` 重拉），幂等自拉起不算重启（评审 R3C fold）**；确认 `/health` 与 `/status` 正常
- [ ] **B站小样**（验下载进度——webdav 无下载阶段，评审 R1A 修正同步至此）：`POST /jobs` → 轮询 downloading percent → transcribing chunk percent → done；result 与同步端点同构（timing.download 仅 bilibili/douyin 注入）
- [ ] **webdav 小样**：轮询 transcribing percent → done（预期无 downloading 阶段）
- [ ] running 中 `DELETE /jobs/{id}` → chunk 边界取消生效；`ls cache/` 与缓存读取确认无部分结果
- [ ] noteflow dev（`pnpm dev`，开关 ON + 指向本机 transcribe）：创建一条真实任务 → awaiting 展示 → 完成续跑 cleaning→save → 笔记产出；再验一条 diarize 官方字幕任务与一条手动 reset
- [ ] 开关 OFF 回滚演练：重启 noteflow → awaiting 迁移 → legacy 路径建一条任务正常；**受门控恢复验收（评审 R3C fold）：两个不同 nextSubmitAt 的门控任务重启后均能自行到期重提**
- [ ] 回归：三仓库测试套件全绿；`tmux transcribe`/`asr` 无异常日志
