# 转录任务队列 + 进度轮询 设计文档

日期：2026-11-10
状态：Spec Review 通过（R6 全模块 Approved），进入 planning
评审记录：builder-check（代码锚点核实）→ Fable 独立评审（9 项并入）→ Spec Review R1（haiku×3，17 条并入）→ R2（opus×3：A Approved；B 3 条、C 4 条并入）→ R3（fable×3：A 2 条、B 7 条、C 5 条并入）→ R4（sonnet×3：A 2 条并入；B/C Approved）→ R5（sonnet×3：B/C Approved；A 2 条并入）→ **R6（sonnet×3：A/B/C 全部 Approved——Spec Review 通过**；三模块 advisory 14 条已全部折叠入文）

## 1. 意图与背景

noteflow（mryk24）目前长阻塞调用四个转录端点（`/transcribe`、`/transcribe_url`、`/transcribe_file`、`/transcribe_douyin`），请求跨 WG 隧道从 mryk VPS 挂到 home WSL，长任务几分钟到几十分钟：

- **长连接超时/断连丢任务**：HTTP 连接贯穿下载+转录全程，任何一跳网络抖动即失败。
- **并发提交抢 GPU**：多个请求同时进 `backend.transcribe`，单卡（RTX 2080 Ti）争用。
- **diarization 忙拒绝**：分离 worker 单飞，忙时 `DiarizationBusyError` 直接失败不排队。

方案：新增统一异步任务入口 `POST /jobs`，提交即返 `job_id`，客户端轮询获取状态、阶段、进度百分比与最终结果。

### 现状可复用的进度锚点（已核实）

| 环节 | 代码位置 | 进度可行性 |
|------|---------|-----------|
| B站/抖音下载 | `downloaders/bilibili_base.py:88` 已是 `requests` 流式 `iter_content`，integrity 有 content-length | 百分比现成，加回调即可 |
| ASR 转录 | `qwen_asr_gguf/inference/asr.py:485` 按 chunk 循环（默认 40s/chunk） | `i/num_chunks` 真实百分比 |
| ASR 取消 | `asr.py:132-137` `set_cancel_event`，chunk 边界检查；**当前无调用方**（本次接线） | 机制已建好 |
| 说话人分离 | `diarization/worker.py` 事件式 IPC（`decoded`/`loaded`/`result`），父进程 50ms 轮询 | 管道已建，补 stage 事件 |
| 在途去重 | `transcribe.py:1013` `process_transcription` 按缓存键合并重复请求 | 提交去重直接复用，并升级为执行控制句柄（§4.3） |

## 2. 目标 / 非目标 / 反目标

### 目标
1. 统一异步任务入口，四源一步到位：`bilibili`（含 `diarize_only` 变体）、`webdav`、`douyin`、`upload`。
2. 串行执行（单 worker）：消除 GPU 并发争用；稳态下（遗留端点迁移完成后）队列路径不再出现 `DiarizationBusyError`；混用期忙冲突按 §6 有界退避。
3. 进度：阶段（phase）+ 阶段内百分比（percent），映射见 §5。**生产基线 asr-engine 本期即获得真实 percent**；gguf 兼容路径同样实现。
4. 取消：queued 直接出队；running 按独占/共享双路径处理（§4.3）；**取消绝不污染缓存、绝不截断他人正在等待的转录、绝不引发资源提前释放**。
5. 引擎侧进度与归属化取消（asr-engine 仓库，本期范围）：详见 §6「引擎侧改造」。
6. 既有四个同步端点行为完全不变；pause 机制（503/挂起/唤醒）完全不变。

### 非目标
- SSE/WebSocket 推送（本次不做；进度存储按可增量加 `GET /jobs/{id}/events` 的方式设计）。
- 任务持久化/重启恢复：队列与状态纯内存，重启即清。noteflow 侧有自愈与重提语义（§8）。
- 多 worker 并行转录（单卡 GPU，串行即正确）。
- funasr/qwen3 后端的细粒度进度（签名预留 `progress_cb`，phase-only）。

### 反目标（失败模式）
- 引入新的丢任务方式：重启丢队列必须有契约化的 missing 表达，noteflow 能可靠重提。
- 改变现有同步端点或 pause 的任何可观察行为；**部分转录不得进入缓存、不得作为成功响应返回**；**推理线程退出前不得释放模型计数/音频租约/删除其正在读取的文件**。

## 3. 约束

- uvicorn 单进程（`config.yaml` `reload: false`），内存态一致。
- ASR 与分离在同一任务内仍并行；队列串行化粒度是**任务级**。
- 暂停语义：暂停不中断在途任务（现有），队列必须遵守。
- 鉴权：现有 token 中间件全局生效，`/jobs` 系自动受保护。
- **锁序单向约束**（评审 R3 确认）：实现中只允许「转录锁 → 状态锁」单向获取；进度与取消端点只取状态锁，绝不反向。

## 4. API 契约

### 4.1 提交

**`POST /jobs`**（`application/json`，三源）→ `202`：

```json
{"source": "bilibili", "bvid": "...", "cookie": "...", "page": 1,
 "context": null, "diarize": false, "diarize_only": false, "body": null, "no_cache": false}
{"source": "webdav", "path": "inbox/xxx.mp3", "context": null, "diarize": false, "no_cache": false}
{"source": "douyin", "aweme_id": "1234567890123456789", "context": null, "diarize": false, "no_cache": false}
```

字段校验与现有 Pydantic 模型同构；各 source body 均可附**可选 `job_id`**（见下）。

**`POST /jobs/upload`**（`multipart/form-data`）→ `202`：`file` + `diarize` + `context` + 可选 `job_id`。

**可选 `job_id`（客户端幂等键，含尝试号）**：

- **服务端把 job_id 当不透明字符串**，格式校验：`^[A-Za-z0-9._:@-]{1,128}$` 且**完整值不得为 `.` 或 `..`**（URL 归一化会吞掉，实测确认；禁 `#`——fragment 截断；禁 `/` 与控制字符）。不合法 → 422。
- noteflow 格式 = **`{任务cuid}-{attempt}`**（cuid 仅字母数字，客户端 `rsplit('-', 1)` 解析无歧义）。服务端不解析 attempt。
- 不传（用户手动场景）服务端 uuid4 生成；手动重跑 = 直接再提交（新 uuid）。
- **同 id 一律重放当前状态**，**绝不删除/替换已有记录**；重放响应 = `GET /jobs/{id}` 完整结构 + `"dedup": true`。
- **attempt 决策表——「观察者递增」**（递增由观察/消费远端终态或发起重置的一方执行；提交方只读）：
  - **递增的观察者**：poller 的 error 分支（普通错误）、missing 分支、cancelled 分支、**paused-error 分支**（远端终态被消费转入 deferred 时同样 +1——恢复后必须以新 attempt 重提，否则服务端重放旧暂停错误；评审 R3 修复）、reset 全家族（§8.1）、重新提取型 resummarize（§8.1）；
  - **只读者**：extractor 提交时读 `sysMeta.attempt`（无则 1），从不修改；
  - **不递增**：429 / 提交期直接 503 / 提交响应不确定（网络超时、连接失败）——无远端终态被消费，同 attempt 安全。

幂等命中（无 `job_id` 时）：同 identity 活跃任务。identity = `(source, bvid+page | aweme_id | path, clamp(context), diarize, diarize_only, body 摘要, no_cache)`（**body 摘要 = JSON 序列化后 md5**，评审 R6 定死）——`diarize_only`/`body`/`no_cache` 必须入键；`audio_url`/`audio_id` 下载后才存在、不进提交键（执行层缓存/在途合并兜底）。upload 无稳定 identity，仅按 `job_id` 去重。

响应：
- 新任务：`202 {"job_id": "...", "status": "queued"}`
- 重放/命中：`200 {GET /jobs/{id} 完整结构, "dedup": true}`
- 队列满：`429 {"detail": "queue full"}` + **`Retry-After` 响应头与 body 字段双写**（上限 `jobs.max_queue`，默认 1000，含 running；Retry-After 默认 30s）
- 暂停中：中间件 503（`PAUSED_REJECT_PATHS` 增加两提交端点）
- job_id 不合法：`422`

**上传文件存储**：内部键落盘 `tmp/jobfile_{uuid}{ext}`（job_id 永不进文件名；**ext 沿用 `generate_safe_filename` 的扩展名白名单**，非法 ext → `.tmp`）；JobManager 维护活跃 upload 路径登记（**含 `_audio_lease` 派生的 `.job-*` 硬链接别名**——硬链接继承旧 mtime，长排队会被 24h 清理误删，评审 R3 补）。重复 POST 同 id upload：重放响应，新上传文件立即删除。

### 4.2 查询

**`GET /jobs/{id}`** → `200`：

```json
{"job_id": "...",
 "request": {"source": "bilibili", "bvid": "...", "page": 1, "diarize": false, "diarize_only": false},
 "status": "queued | running | cancelling | done | error | cancelled",
 "phase": "downloading | transcribing | formatting | null",
 "percent": 0-100 | null,
 "message": "可选人类可读说明",
 "diarize_stage": "decoding | segmenting | embedding | clustering | reconstructing | null",
 "queue_position": 3 | null,
 "submitted_at": "...", "started_at": "...", "finished_at": "...",
 "result": { ...与现有端点完全同构的响应 dict（含 timing.download 注入）... } | null}
```

- `request` 为提交参数摘要（cookie 永不回显）。
- percent 是阶段内百分比；无法给出时 null。
- `done` 的 result 即现有端点响应体；`error` 的 result 为现有 error dict（上游暂停竞态时附加 `paused/resume_at/paused_at`）。**error 与 done 一样取 result**。
- **终态映射规则（评审 R5 修复）**：执行体返回的响应 dict `status == "error"`（下载失败/模型加载失败/webdav 文件不存在/diarize_only `_err` 等均属此列）→ job `error`；否则 → `done`。poller 分支依赖此判定，不得把错误体判成 done 写入 rawData。
- `cancelling` 为中间态（§4.3）。
- **diarize_only 任务的 phase 序列**：`downloading → formatting → done`；无 transcribing、percent 恒 null；diarize_stage 有值。
- **终态记录保留与驱逐**：目标保留 `jobs.record_ttl_minutes`（默认 60）；result 总字节超 `jobs.max_result_bytes` 时从最旧终态整条驱逐（单条超限保留）。驱逐与 TTL 同语义：消失 = missing = attempt+1 重提。
- 未知 id：`404 {"detail": "job not found (可能因服务重启丢失)"}`。

**`POST /jobs/query`** → `200 {"jobs": [...], "missing": [...]}`：请求体 `{"ids": [...]}`，单次上限 500；每项不含 result，终态后按 id 单独取；`missing` 标准动作 attempt+1 重提。

**`GET /jobs`**：列表（调试/对账用），`?status=` 过滤，倒序 cap 100；cap 只作用于终态任务；不回显 cookie。

### 4.3 取消

**`DELETE /jobs/{id}`** → `200 {"job_id", "status"}`——「已接受」语义：返回请求后即时状态。**即时状态矩阵（评审 R6 修订：补 diarize_only 行 + 含分离列归属钉死）**：

| 执行体 × 等待者 | queued | running·无分离（自串行后端） | running·无分离（qwen3/funasr） | running·含分离（任意后端） |
|---|---|---|---|---|
| 独占（唯一等待者） | 即时 `cancelled` | `cancelling` → join → `cancelled` | `cancelling` → join → `cancelled` | `cancelling` → join → `cancelled` |
| 共享（waiters > 1） | 即时 `cancelled` | 即时 `cancelled`（不 join） | `cancelling` + join → `cancelled` | `cancelling` + join → `cancelled` |
| diarize_only/merge（无 entry） | 即时 `cancelled` | —（下载段：标志+`DownloadCancelled`；分离段：协程取消；均 `cancelling` → join → `cancelled`） | 同左 | 同左 |

注：**含分离的行统一走 join 语义**；后端二分（自串行 vs 非自串行）只决定 ASR 段信号强度（chunk 级 vs 标志+自然结束）。终态确认靠轮询。终态 id 幂等返回当前态；**未知 id → 404（与 GET 一致）**。`cancelling` **计入「活跃」口径**（identity 去重与 max_queue 计数一致）。

**执行控制句柄（评审 R3 核心修复：控制信息上移到 in-flight entry）**

现有 `process_transcription` 的 in-flight entry 只存 `{task, waiters}`，且创建者可先退出、跟随者继续等待——**控制信息若挂在 Job 上，会覆盖不了「遗留先创建、job 后加入」的计算**。修复：

- **entry 升级为执行控制句柄**（所有计算都经 `process_transcription` 创建，无论由谁发起，一律带全套控制信息）：entry 新增 `cancel_flag`（threading.Event，创建即存在）、`request_id`（每次计算生成，唯一）、`thread_future` 槽位、backend 引用。
- `_process_transcription`：挂接 backend 取消事件（带 entry.request_id）、探针读 entry.cancel_flag、登记 thread_future、向 asr-engine 传 entry.request_id。
- `TranscriptionService` 新增条目级 API：`cancel_computation(inflight_key)`——**按提交时的完整 in-flight 键 `(loop, identity, context, diarize, no_cache)` 定位**（队列执行体持有该键）；置 entry.cancel_flag + `backend.cancel_request(entry.request_id)`；返回 entry 供 join thread_future。**后加入的 job 与创建者拥有同一份控制句柄**。
- **第五形态：分离冷却重试路径（评审 R5 修复）**——diarize=true 且 ASR 基线缓存命中时走 `_annotate_cached_asr`（transcribe.py:1082-1084）：**有 entry 但全程无 ASR**（thread_future 恒空、backend 归属永不登记、「转录返回后」探针位不存在）。处置：entry 增加 `asr_started` 标记（backend.transcribe 调用时置位）；未开始 ASR 的取消**跳过 cancel_request**（不向 pending-cancel 集合写永不消费的条目）且 thread_future 为空 → **协程级取消（操作语义 = 对 entry 拥有者 task 调 `cancel()` 并由取消方 await 其收尾**；既有 except BaseException 链回收分离进程）；`_annotate_cached_asr` 启动分离前检查 entry.cancel_flag（探针）。
- 既有遗留断连路径的 `task.cancel()` 行为不变（本 API 为增量，仅队列取消使用）。

**取消传播顺序（评审 R3 修复：防资源提前释放）**：entry 标记 → backend 信号（后端支持则）→ **等待 thread_future 完成**（拥有者协程在 future 完成后才收尾引发取消/返回，finally 释放模型计数/租约/文件）→ worker 才继续。**禁止在 thread_future 完成前取消拥有者协程**——否则 asyncio future 取消而线程继续运行，finally 提前释放会出现在推理中卸载模型、删线程正在读的文件（评审经最小实验确认该行为存在）。

**`running` 取消路径**：

- **独占路径（job 是唯一等待者）→ 调 `cancel_computation(inflight_key)` 真实取消**：
  - 信号两段式（pending-cancel）：`cancel_request(request_id)` 先查活跃归属（状态锁内），匹配即置位其 per-call 取消事件；不匹配则记入 pending-cancel 集合；执行线程锁内登记归属时**命中集合即置位并移除**。「DELETE 早于归属注册」由此覆盖。
  - gguf：事件置位 → 下一 chunk 边界停止（最坏多算一个 chunk）。**backend 未登记/模型加载窗口的兜底：挂接/登记归属时同步消费 entry.cancel_flag，命中即置位 backend 事件**（否则该窗口 chunk 级取消失效，退化为探针兜底）。
  - asr-engine：scoped `/admin/cancel` 同规则。
  - 分离子任务走既有 `diar_task.cancel()` 链路（进程立即回收）。
  - **取消探针（位置在 `await diar_task` 之前）**：转录返回后立即检查 entry.cancel_flag 与 `performance.cancelled`，命中即中止（异常路径顺带取消分离子任务）——部分结果绝不入缓存、不返回。
  - 终态：`cancelling` → 等计算协程按传播顺序收尾（thread_future 已完成）→ `cancelled` → worker 取下一件。ASR 调用经 `loop.run_in_executor` 以获得可 await 的 future（`asyncio.to_thread` 不暴露底层 future，登记槽位需 run_in_executor）。
- **diarize_only/merge 路径的 running 取消（评审 R4 修复：无 entry 落点）**：diarize_only/merge 执行体不经过 `process_transcription`、无 entry——**下载段**复用 job 级标志 + `DownloadCancelled`（与主路径下载段同机制，.part 随取消清理）；**分离段 = 直接取消自身执行协程**（不持模型计数/音频租约；`_run_diarization` 既有语义「取消不降级、必须上抛」，分离进程经 IPC await 取消立即回收）→ `cancelling` → join → `cancelled`。
- **qwen3/funasr 的独占取消（放弃式，评审 R5 补正文定义）**：`cancel_computation` 置 entry.cancel_flag（backend 无 chunk 钩子不消费）→ `cancelling` → 等 ASR 线程自然跑完（thread_future join）→ 探针/收尾检查 flag 丢弃结果 → `cancelled`。**最坏阻塞 = 整次转录时长**（与 §6 版本降级同款契约，契约明示）。
- **共享路径（waiters > 1）→ 仅撤销自身等待**：不调 cancel_computation、不置事件——计算完整跑完给遗留等待者。终态语义按后端二分（与 DELETE 响应括号一致）：**自串行后端**（gguf `transcribe_lock` / asr-engine 引擎锁）→ 本 job 立即终态 `cancelled`、无需 join，worker 可立即取下一件（GPU 串行由后端锁保证）；**qwen3/funasr（无内锁）→ `cancelling` + join，待计算退出执行资源才取下一件**（否则并发进同一模型）。
- **pending-cancel 集合生命周期（评审 R3 修复）**：统一规则「先查活跃归属，不匹配才入集合」；消费点 = 归属登记命中；条目带时间戳，TTL（默认 5min）过期清理 + 容量上限（默认 64，超限逐最旧）；request_id 每计算唯一生成，无复用误伤。**集合每 backend 只落一处：gguf 进程内、asr-engine 远端引擎侧；编排层不重复建集合**（评审 R6 澄清）。
- **取件竞态**：worker 取件后同锁内复查状态，已 cancelled 跳过。
- 已知罕见竞态：独占取消置位后、转录结束前，新遗留请求并入同一共享任务 → 按既有错误路径失败（窗口极窄，接受）。

## 5. 进度映射（实现锚点）

**进度 = per-job 分槽快照 + 单一异步采样器**：

- **两个独立槽位，各自整体替换**（单不可变 dict 一次成快照）：
  - **转录槽**（phase/percent/message）：gguf 路径由 `progress_cb` 回调写（worker 线程）；下载循环逐块写。
  - **分离槽**（diarize_stage）：`worker.run()` 异步循环直写。
  - 读取侧（采样器）**拼装两槽**，杜绝双写者互抹。
- **远端样本合并规则（评审 R3 修复）**：asr-engine 路径采样器轮询 `GET /v1/progress`，响应**仅更新转录槽的 percent/message**（远端无下载/分离/formatting 信息）；应用条件 = **应用响应时**本地 phase 为 `transcribing` 且响应 request_id 与当前 running job 匹配（在应用时校验，迟到样本不得拉低已推进的 phase）；inactive / 不匹配 / 请求失败 → 本轮忽略。**phase 永远由本地执行体驱动**，远端样本不得引发阶段回退或覆盖本地快照。
- `progress_cb` 参数统一存在于后端签名，**仅本地路径（gguf）实际承载**；asr-engine 后端不使用，进度经采样器轮询。**混用期归属注记（评审 R6 确认接受）**：progress_cb 归属在计算创建时即定，队列 job 加入遗留已启动计算期间，该 job 的 percent 恒 null（正确性无碍）。

| phase | percent 来源 | 实现点 |
|-------|-------------|--------|
| `downloading` | `downloaded_bytes / content_length`（B站真实值；抖音 null） | iter_content 循环写转录槽 |
| `transcribing` | `(i+1)/num_chunks`（gguf 本地回调；asr-engine 经采样；funasr/qwen3 null） | 共享库 `asr.py` chunk 循环调 `progress_cb(i+1, num_chunks)`（`QwenASREngine.transcribe` 加参，两仓库共用） |
| `formatting` | 直接 100 | `_process_transcription` 字幕生成前置位（**diarize_only 任务例外：percent 恒 null**，见 §4.2） |
| （并行分离） | 无百分比，`diarize_stage` 标签（分离槽） | **`diarization/manager.py` 的 `diarize()` 内部**发 stage 事件（子进程经 IPC 传出），父进程直写分离槽 |

## 6. 执行模型

新模块 **`jobs.py`**（JobManager 单例）：

- **存储**：`dict[job_id, Job]` + 就绪 `deque`；`Job` 含状态/双进度槽/结果/TTL/request_id/upload 路径等字段（thread_future 在 entry 上，见 §4.3）。
- **worker**：单 `asyncio.Task` 消费循环；取件前查 `pause_manager.is_paused()`，暂停保持排队；取件后同锁内复查状态（已 cancelled 跳过）。自愈：worker 异常退出时 done_callback 先将当前 running job 落 `error`、确认无遗留引用后才重拉；关闭/崩溃期间禁止取新件。
- **运行中撞上上游暂停（worker 自冷，评审 R4 修复：恢复正文定义）**：asr-engine 被独立暂停（本地未暂停）时，`UpstreamPausedError` 穿透 → 该任务终态 `error`、result 附加 `paused/resume_at/paused_at`（noteflow 走既有 ServicePausedError 解析与 deferred 语义）；**自冷 = 后续取件前探询 asr-engine paused 态，上游暂停期间不启动新任务**（本地 `/pause` 场景由既有 is_paused 取件门覆盖，不热循环烧穿队列）。
- **阻塞环节全部离线程**：下载、ffmpeg、ASR 均经 `asyncio.to_thread`（ASR 例外：为暴露可 await 的 thread_future 经 `loop.run_in_executor`，见 §4.3）。下载取消经**专用异常 `DownloadCancelled`，派生自 BaseException**（三处 `except Exception` 均不会吞它；若实现选 Exception 派生则必须逐处显式 re-raise——评审 R5 定死默认）穿透 `download_audio` 的 `except Exception`、3 次重试循环**及 `download()` 外层 catch（bilibili_base.py:229）**；命中 → 清理 tmp → `cancelled`。抖音 ffmpeg 无逐块取消点 →「标志 + join 自然结束」。
- **混用期分离忙冲突**：`_run_diarization`（`transcribe.py:819-836`）现于**内部**捕获 `DiarizationBusyError` 转降级——JobManager 外层捕不到。修复：队列路径经**策略参数**（如 `diarize_wait_deadline`）透传至 `_run_diarization`：busy 时按策略等待重试至 deadline（每 30s，上限 10min），超时按既有降级语义；**遗留端点默认 deadline=0 行为不变**（评审 R3 修复：退避落点）。
- **执行体复用现有入口**：bilibili（download → diarize_only/merge 或 process_transcription）、webdav（路径回退 + process + `move_to_processed`，取消/失败不归档）、douyin、upload。走 `process_transcription`（含 entry 控制句柄）；探针如 §4.3。**响应同构**：bilibili/douyin 队列执行体同样注入外层 `timing.download`。
- **终态清理**：upload 内部键文件终态时删除；`cleanup_stale_tmp_files` 感知活跃队列（排除登记的在用 upload 路径**及其 `_audio_lease` 硬链接别名**）；webdav/B站/抖音缓存清理逻辑不变。
- **清扫全部走事件循环**：JobManager 单 asyncio 清扫任务（~60s）：TTL 整条移除、字节预算驱逐（单条超限保留）、upload 登记维护。`monitor_loop` 只管模型卸载，不触碰 job 存储。
- **缓存命中**：快速完成（formatting→done），不做特殊态。

### 引擎侧改造（asr-engine 仓库，本期范围）

1. **双锁结构**：转录串行锁（`self._lock`，整次转录持有）+ **短时状态锁**（`self._state_lock`，毫秒级）。进度/归属/取消事件/pending-cancel 集合读写、scoped cancel 校验+置位全部在状态锁内；锁序单向：只允许「转录锁 → 状态锁」。
2. **事件与归属的锁内安装**：执行线程获得转录锁后，状态锁内安装 per-call 取消事件 + 登记 `request_id` + 初始化进度快照 + **检查 pending-cancel 集合（命中即置位并移除）**；转录锁释放前清除三者。修复现有锁外挂接被排队请求覆盖的缺陷（`engine.py:119-131`）。
3. **`request_id` 的 HTTP 闭环**：`/v1/audio/transcriptions` 新增可选 multipart form 字段 `request_id`；`GET /v1/progress` 返回 `{"active", "request_id", "chunk", "total"}`（纳入 `_check_token` 鉴权，采样器带 Authorization）；`POST /admin/cancel` 加可选 request_id，状态锁内原子校验：活跃归属匹配 → 置位；不匹配 → 记入 pending-cancel 集合（**HTTP 往返不持引擎状态锁——校验+置位在引擎侧状态锁内原子完成，编排层不持本地锁发起 HTTP**，评审 R6 澄清；**并补 `_check_token` 鉴权**——现状该端点无鉴权，评审 R6 要求统一）。旧调用方行为不变（FastAPI 忽略未声明字段）。gguf 同构：backend 维护 pending 取消标志集合（进程内）。
4. **pending-cancel 集合生命周期**：条目带时间戳，TTL（默认 5min）+ 容量上限（默认 64，超限逐最旧）；消费即移除；request_id 每计算唯一，无复用误伤。
5. **共享库 progress_cb + 取消标记**：`QwenASREngine.transcribe/asr()` 加可选 `progress_cb`；chunk 循环取消时置 `performance["cancelled"]=True`。**部署顺序约束（评审 R3 补）**：共享库与引擎同树（PYTHONPATH 指向 transcribe-service 仓库），含 progress_cb 的库必须先落盘、引擎后重启——写进部署清单。
6. **transcribe 侧 `asr_engine_backend`**：`transcribe(..., progress_cb=None, request_id=None)`；**解析 verbose_json 顶层 `cancelled`（format.py:48 已透出）映射进 `performance["cancelled"]`**（在现有成功响应解析处补入，不改 503/连接重试分支——暂停信号仍原样穿透）；新增 `cancel_request(request_id)`（两段式语义见 §4.3）。`ASRBackend` 统一接口加 `cancel_request(request_id)`。
7. **版本偏斜降级**：transcribe 首次使用时探测 `/v1/progress`（**正缓存持续；负缓存带 TTL（默认 10min）自动重探**——修复升级方向永久缓存的矛盾）；404 期间 percent 关闭 + 取消降级「标志 + join 自然结束」（最坏阻塞 = 整次转录，契约明示）。引擎回滚旧版建议同步重启 transcribe。
8. **已知边界**：asr-engine 崩溃自动重启窗口由现有 ConnectError 重试覆盖，进度采样失败静默 null。

## 7. 配置

`config.yaml` 新增段：

```yaml
jobs:
  max_queue: 1000          # 含 running；满则 429 + Retry-After
  record_ttl_minutes: 60   # 终态记录目标保留期；预算压力可能提前整条驱逐（同 missing 语义）
  max_result_bytes: 268435456  # 终态 result 总字节预算（256MB），超限从最旧终态整条驱逐（单条超限保留）
```

## 8. noteflow 侧实施方案（mryk24 仓库，两段式改造）

现状要点：worker 并发默认 2；双层占坑 = `activeWorkers` + `funasr pLimit(1)`；转录透传调用点 6 处（§8.2）；暂停机制已有 deferred + wake + 兜底 setTimeout + 重启自愈。

### 8.0 数据库能力边界与通用工具（builder 必读）

datasource 为 **SQLite**，Prisma 对 JSON 字段不支持比较/路径过滤，`updateMany` 无法合并 JSON（只能整字段替换）。规定：

- **共用工具函数**（评审 R3 建议：统一形态防各写各的）：`conditionalTransition(taskId, expectedStatus, expectedAttempt?, patch)`——交互式事务内读 → JS 校验 status/attempt → 整字段替换写回；可传入现有事务客户端（事务内复用连接）。**校验失败 = 本代作废：调用方不再提交远端、不再回写**（防旧代继续 POST /jobs 产生无主孤儿 job，评审 R4 补）。
- **禁止嵌套事务、禁止在持有数据库事务时等待远端 DELETE**：forceReset 位于批量事务内 → 本地状态先行落库，远端 DELETE 一律事务外尽力而为。
- 领取排除 `nextSubmitAt > now`：交互式事务内 JS 侧过滤候选 + 条件锁定。
- **builder 须知（评审 R6）**：pipeline.ts 的 `persistStepOutput` 与 `updateTaskSysMeta`（currentStep）无代次校验——forceReset 命中在途 processing 任务时，旧代中途产物可能短暂覆盖新代 currentStep（仅展示层短暂脏读；状态机关键写入已受条件工具保护、最终被 baseline/poller 覆写自愈），实现时知晓即可，不必加锁。

### 8.1 两段式任务流

**第一段（提交，worker 内、秒级）**：

1. worker 领取任务时捕获 **`expectedAttempt = sysMeta.attempt`（无则 1）**——本代标记（评审 R3 修复：代次防护扩展到第一段与收尾）。
2. **基线落盘**：`persistTaskProgress`（经 §8.0 条件工具，校验 `status='processing'` 且 **attempt == expectedAttempt**）写入 `itemMeta`/`transcribeType` 等 cleaning/summarize 所需元数据，**并顺带持久化 `sysMeta.attempt = expectedAttempt`**（attempt 首次落库，支撑「processing 已提交」取消竞态检测）。官方字幕 body 不落盘（missing 场景重提取 → attempt+1 重提）。
3. `POST /jobs`（`job_id` = `{cuid}-{attempt}`，attempt 只读）：
   - `202` → 抛 `JobSubmittedError`（name 判定，不烧 retryCount）；
   - `200 dedup` → 同等视为已提交，沿用响应内 job_id，同样抛 `JobSubmittedError`；
   - `429` → **抛 `JobQueueFullError`**（命名对齐 ServicePausedError 的 name 判定惯例，评审 R6 定名）→ 读 `Retry-After`（响应头或 body 均可，双写已保证）→ **条件转 `pending`**（校验 expectedAttempt，防复活已取消任务）+ `sysMeta.nextSubmitAt`（不烧 retryCount）；**每事件独立 `setTimeout` 到点兜底唤醒（评审 R4 定死：非单槽覆盖，防后到期任务失去唤醒；不复用会调 `markResumeSignal()` 的暂停唤醒函数）**；
   - 提交期 503 → ServicePausedError → markTaskDeferred（既有，attempt 不动——无远端终态被消费）。
   - 全部现有 catch 与 ServicePausedError 同样透传 JobSubmittedError——**异步提交实现为重试循环之外的独立分支**（评审 R4 修复：点名两处内层重包装 catch——`bilibili.ts:829-833`、`douyin.ts:89-92` 会把异常重包装成普通 `Error` 丢失 name，不绕开即被重试循环烧满后转 failed）。
4. worker 错误分派：`JobSubmittedError` → `markTaskAwaiting`（§8.0 条件工具，校验 status=processing 且 attempt==expectedAttempt）→ runWorker 正常收尾，槽位释放、自动补位；**`JobQueueFullError` → 无操作收尾**（状态已由 submit 分支条件迁移，评审 R5 补点名、R6 定名，防落 markTaskFailed 烧重试）。**funasr limiter 仅旧同步路径使用**。
5. **worker 收尾（取消/失败/deferred/**成功回执 markTaskCompleted**/安全网 processing→pending）一律经 §8.0 条件工具校验 expectedAttempt**（评审 R3 修复 + R6 补枚举：旧 worker 的任何迟到回执——含成功——不得作用于新一代）。

**等待态（新状态 `awaiting`，String 新值，零 DDL）**：
- 不能留 `processing`（zombie cleaner 与启动重置会误杀并重复提交 job）；不复用 `deferred`；前端状态映射加「等待转录」+ 取消按钮 + 计数。

**第二段（结果消费，全局 poller 驱动）**：单例 poller（3-5s，禁止重入、每请求带超时、逐任务异常隔离），按 500 个一批 `POST /jobs/query`。**全部分支状态迁移走 §8.0 条件工具（校验 status='awaiting'）**：
- `done` → 取 result → 按源写 `rawData`——**per-source 映射为 poller 侧独立单元，与同步路径逐点一致**（如 webdav：`formatWebdavTranscriptBody` 格式化 + `result.audio_duration → itemMeta.duration` 回填；diarize_only 路径：job result 即 diarize_merge 响应、与现端点同构，直接 `rawData = JSON.stringify({body: 标注结果})`——**rawData 为 String 列，写入前 JSON 序列化**（评审 R6 澄清），clean 只读解析后的 `.body`）→ `startStep='cleaning'` + `pending` → triggerWorker。
- `error` → 取 result 读暂停元数据：paused → 条件化 `markTaskDeferred` + **attempt+1**（评审 R3 修复：远端终态被消费，恢复后必须新号重提，否则重放旧暂停错误）；普通 error → 条件化 `markTaskFailed` + attempt+1；diarize_only error → 写 `sysMeta.diarizeFailed=true` + 条件转 pending + attempt+1 → triggerWorker（重提取读标记跳过 diarize job 提交，循环有界；**标记清除边界**：自动重试保留，用户主动全量 reset/重新提取时清除）。
- `cancelled` → 条件化 `markTaskCancelled` + **attempt+1**（观察者递增，评审 R6 内联补明）。
- `missing` → 条件转 pending + retryCount+1（封顶防死循环）+ attempt+1 → triggerWorker。
- 查询网络失败 ≠ missing；取 result 404 按 missing。
- **凡转回 pending 一律 triggerWorker**；poller 之外另设独立 `setTimeout` 到 `nextSubmitAt` 兜底唤醒。

**迟到回写防护**：poller 一切回写经 §8.0 条件工具（status='awaiting' + attempt 校验）。

**取消/重置/删除全覆盖（重提入口六处，评审 R3 修复：原「四入口」前提不成立）**：
- **单任务 cancel**：扩展 `awaiting` → 尽力 DELETE remote + 条件转 `cancelled`；竞态覆盖：processing 但 sysMeta.attempt 已存在（job 已提交未转 awaiting）→ 同样尽力 DELETE remote 再取消。
- **批量 cancel**：同规则。
- **单任务 delete**：`awaiting` 先 DELETE remote 再删记录。
- **reset/重提入口（attempt+1 落点，编号①-⑥，评审 R4 对齐计数）**：① `[id]/reset.post.ts`、② `batch/reset-failed`、③ `batch/reset-cancelled`——三者共用 `reset.ts` 的 `prepareSysMetaForReset`，**在该函数内 attempt+1 并清除 diarizeFailed**（评审 R5 钉死落点，防重置后永远跳过分离）；④ **create-batch forceReset**（直接合并 sysMeta、不经该函数——**调用点改造**补 attempt+1，**并清除 diarizeFailed**——merge 语义不会自然覆盖旧标记）；⑤ **详情页「重新执行」resummarize(startStep='extracting')**（重新提取型，同样 attempt+1，并清除 diarizeFailed）；⑥ **启动回滚迁移**（开关关闭的排空迁移，见启动顺序①）。统一原则：**任何把任务转回「将再次提交转录」状态的入口都必须 attempt+1**。各入口尽力 DELETE remote（§8.0：事务外）。
- **前端**：awaiting 标签/计数/取消按钮。

**快速调用排除（评审 R3 修复）**：quick-subtitle / quick-summarize 与 worker 共享 extractor，但构造 `id:""`、`persist:false` 的临时任务——**异步模式仅适用于持久化 worker 任务**；快速调用保留同步路径。**persist 标志经 step 层 PipelineContext 传入，extractor 不读全局开关**（评审 R4 定死感知机制，开关判断不落在 extractor）。

**看门狗（监督者模式）**：poller 心跳 > 2×间隔未完成 → 重启 poller（任务不因 poller 卡死被重置）；`awaiting` 存续 > 24h → 先单查确认 missing → attempt+1 重提。

**自愈与启动顺序**（插桩：`noteflow-init.ts` 的 `resetAllProcessingTasks` 之后、既有 pending 触发之前）：① 回滚迁移（开关关闭：awaiting → pending + attempt+1 + 尽力 DELETE remote）② 启动扫描（开关开启：awaiting query 一轮）③ **恢复受门控 pending 的最早到期唤醒**（评审 R3 修复：setTimeout 仅内存态，重启后需重排，否则系统唯一任务永久滞留）④ 启动 poller 并立即跑一轮 ⑤ 既有 pending 触发 worker。

**功能开关与回滚**：全局开关 `noteflow.asyncTranscribe`，部署默认 OFF（旧同步路径含 funasr limiter 完整保留）；**快速调用不受开关影响**（始终同步）。回滚 = 关开关 + 重启排空迁移。

### 8.2 透传点改造映射（6 处）

| # | 位置 | 改造 |
|---|------|------|
| 1 | `steps/bilibili-extract.ts:79` forceManualTranscribe 直调 | 基线落盘 → `POST /jobs`(bilibili) + JobSubmittedError |
| 2 | `extractors/bilibili.ts:1052` transcribeAudio（ep 模式） | 同上 |
| 3 | `extractors/bilibili.ts:1210` 无官方字幕回退 | 同上 |
| 4 | `extractors/bilibili.ts:937` diarizeSubtitle（diarize_only+body） | body 不落盘直接提交；error 走 skip-diarize 降级 |
| 5 | `extractors/webdav.ts:114` | `POST /jobs`(webdav) + awaiting |
| 6 | `extractors/douyin.ts:57` | `POST /jobs`(douyin) + awaiting |

各处现有 `isServicePaused` 透传点（bilibili.ts:893/1171/1262、webdav.ts:194、douyin.ts:130）即 JobSubmittedError 透传落位参照。

### 8.3 契约要点

- `job_id` = `{cuid}-{attempt}`；attempt 递增归观察者（poller 四分支含 paused、reset 全家族、重新提取 resummarize）；expectedAttempt 代次防护贯穿 worker 全程；任务表零新增列（attempt/nextSubmitAt/diarizeFailed 存 sysMeta）。
- 暂停 503 语义不变。
- **联调演练须知（评审 R5 记录）**：取消竞态的有界残留——baseline 落库后、POST /jobs 在途时收到取消 → 远端 DELETE 404（job 尚未注册）→ 孤儿 job 跑完即弃（TTL 驱逐，不回流 noteflow）。分布式固有窗口，「尽力而为」已隐含接受，演练时知晓即可。
- noteflow 侧测试：JobSubmittedError 全 catch 透传、awaiting 状态机、poller 全分支 + 条件回写、attempt 观察者（含 paused 分支与 reset 六入口）、expectedAttempt 代次防护（旧 worker 迟到回执/取消 vs 新一代）、429 门控（领取过滤 + 每事件独立定时器 + 重启恢复 + 空队列场景 + 429 控制错误无操作收尾）、skip-diarize 降级有界性 + 标记清除边界（含 prepareSysMetaForReset）、取消/重置/delete/forceReset/重新提取全覆盖、看门狗、启动顺序、开关回滚演练、quick 调用不受开关影响、与 zombie cleaner/启动重置/暂停机制回归。

## 9. 测试

- **单元（pytest）**：JobManager 提交/identity 去重/job_id 校验（字符集 422、`#`、`.`、`..` 拒绝）/同 id 重放/429+Retry-After/**条目级取消**（遗留创建的计算 job 后加入再取消可命中；取消传播顺序：thread_future 完成前不释放模型计数/租约、不删在读文件——评审实验回归）/独占（chunk 边界 + 探针位置在 diar_task await 前）/共享（自串行后端立即终态；qwen3/funasr 强制 join）/qwen3 放弃式/**qwen3-funasr 独占取消 join 自然结束后探针丢弃、不入缓存（评审 R5 补）**/**diarize_only/merge running 取消（无 entry 路径：下载段标志+DownloadCancelled、分离段协程取消 → 分离进程回收 → join → cancelled，评审 R4 补）**/**分离冷却重试路径（_annotate_cached_asr：无 ASR 取消不污染 pending-cancel、协程级回收、探针生效，评审 R5 补）**/DELETE「已接受」语义/取件竞态复查/pending-cancel 集合（命中消费、TTL、容量、无复用误伤）/部分结果不入缓存不入归档/TTL 与字节预算驱逐/暂停期 worker 不取件/上游独立暂停自冷/worker 自愈/分离忙退避经策略参数（遗留默认行为不变）/下载专用取消异常不被重试循环吞/上传内部键与重复提交清理/tmp 清理排除活跃 upload 及 `.job-*` 硬链接别名/timing.download 注入。
- **API 集成（TestClient + mock backend）**：四源全链路；result 同构；404；暂停提交 503；query 命中/missing/500 上限；cap 规则与 cookie 不回显；上游暂停 → error 元数据 + worker 自冷；**取消期间收到 503/连接重试的暂停信号穿透回归**（评审 R3 建议）。
- **asr-engine 侧（该仓库 tests/）**：`/v1/progress` 两态 + 双锁分离 + 鉴权；`/admin/cancel` 归属匹配/不匹配（pending-cancel 记录/命中/过期）/锁内安装与清除（并发排队不互覆）；request_id form 字段；旧调用方行为不变。
- **transcribe 侧补充**：采样器合并规则（远端样本仅更新转录槽 percent、phase 不回退、request_id 归属校验）、`cancel_request` 两段式、`/v1/progress` 404 降级与负缓存 TTL 重探、`performance.cancelled` 传递（含 verbose_json 解析）。
- **手动真机（asr-engine 生产基线）**：全链路轮询进度；DELETE running 确认 chunk 边界取消 + 缓存无部分结果残留。gguf 仅单测。
- 回归：现有同步端点行为不变（现 tests/ 全绿）。

## 10. 实施阶段（builder 参考切分）

1. `process_transcription` entry 控制句柄升级（cancel_flag/request_id/thread_future/探针/取消传播顺序）+ `cancel_computation` API + 单元测试（含评审实验回归：资源不提前释放）。
2. `jobs.py` JobManager（cancelling 生命周期、pending-cancel、异步清扫、字节预算、upload 内部键、策略参数透传分离忙退避）+ 单元测试。
3. 进度分槽快照穿链：下载器（to_thread + 专用取消异常）+ 三后端签名 + 共享库 chunk 循环（两仓库共用，含 performance.cancelled）+ 分离槽 + 异步采样器（含远端合并规则与归属校验）。
4. asr-engine 仓库改造（§6）：双锁 + 锁内安装 + pending-cancel 集合 + `/v1/progress` + scoped cancel + 引擎侧测试（可与 1-3 并行；注意共享库先落盘）。
5. `server.py` 端点 + `PAUSED_REJECT_PATHS` + `cleanup_stale_tmp_files` 活跃感知（含硬链接别名）。
6. API 集成测试 + 手动真机验证 + docs 归档 + 过时表述更新（config.yaml 注释、CLAUDE.md）。
7. noteflow 仓库两段式改造（§8，独立排期可并行，联调在 6 之后；含开关与回滚演练）。

## 11. 留给 builder 的决定（已授权自行裁量）

- `Job`/entry 字段命名、模块内部组织。
- percent 取整与 message 文案。
- Retry-After 实现形态（常量或配置）。
- GET /jobs 列表字段裁剪。
- 看门狗/退避/pending-cancel TTL 与容量等参数取值（文档给默认）。
- noteflow 侧 `persistTaskProgress`/条件工具的函数签名细节（须符合 §8.0 约束）。
- 测试文件组织（各仓库现有风格）。
