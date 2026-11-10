# 转录任务队列与进度轮询

日期：2026-11-10

noteflow 原先同步调用转录端点，HTTP 连接需跨网等待下载、转录与说话人分离，遇到网络抖动容易断连，并发任务也可能争用单卡 GPU。本次新增内存态任务队列：提交后立即返回 `job_id`，客户端按需轮询进度和结果；单 worker 串行执行，既有同步端点与暂停语义保持不变。

## API 契约速览

| 端点 | 用途与约定 |
|---|---|
| `POST /jobs` | 提交 B 站、WebDAV 或抖音任务；新任务返回 `202` 与 `job_id`，可用 `job_id` 幂等重放。队列满返回 `429`，Retry-After 同时写入响应头和 body。 |
| `POST /jobs/upload` | multipart 上传音频并提交任务；上传临时文件使用内部随机键，不将客户端 `job_id` 写入文件名。 |
| `POST /jobs/query` | 按最多 500 个 id 批量读取轻量状态；返回 `jobs` 与 `missing`，终态结果再按 id 单独获取。 |
| `GET /jobs/{id}` | 读取任务状态、阶段、进度、队列位置及终态 `result`；未知 id 返回 `404`。 |
| `GET /jobs` | 调试/对账列表，可按状态过滤，最多返回 100 条，不回显 cookie。 |
| `DELETE /jobs/{id}` | 请求取消并返回即时状态；终态 idempotent 返回当前状态，未知 id 返回 `404`。 |

服务重启会清空内存队列与记录；查询到 `missing` 或 `404` 时，客户端按契约将任务作为缺失处理后重提。终态记录默认保留 60 分钟；结果总字节预算超限时，从最旧终态开始整条驱逐，单条超限记录保留。

## 取消矩阵

取消遵循“先标记、后确认收尾”：`cancelling` 是中间态；只有确认计算线程/分离工作已收尾，才转为 `cancelled`，以免提前释放模型、音频租约或删除仍在读取的文件。

| 形态 | 取消行为 |
|---|---|
| 排队中（queued） | 立即从待执行队列取消，状态变为 `cancelled`。 |
| 运行中、唯一等待者、无分离、自串行后端 | 发送后端取消信号；GGUF 在 chunk 边界停止，支持 scoped cancel 的 asr-engine 按 `request_id` 取消；进入 `cancelling`，等待计算线程结束后转 `cancelled`。 |
| 运行中、唯一等待者、无分离、非自串行能力 | 置取消标志并等待 ASR 线程自然结束；结果由取消探针丢弃，不入缓存；确认 join 后转 `cancelled`，最坏等待整次转录。 |
| 运行中、多个等待者、无分离、自串行后端 | 仅撤销当前 job 的等待，不取消共享计算；当前 job 可立即转 `cancelled`，其余等待者继续使用计算结果。 |
| 运行中、多个等待者、无分离、非自串行能力 | 不取消共享计算，但进入 `cancelling` 并 join；计算退出后才转 `cancelled`、允许 worker 继续，避免同一模型并发执行。 |
| 运行中含分离，或 `diarize_only`/merge 无 in-flight entry | 含分离任务统一等待 join；无 ASR 的缓存分离路径不发无效 ASR 取消，而取消本 job 执行协程并等待分离回收。`diarize_only`/merge 下载阶段用取消标志与 `DownloadCancelled` 清理部分文件，分离阶段取消执行协程并等待收尾。 |

**asr-engine 能力降级：**若远端版本偏斜、`/v1/progress` 能力探测为 legacy，则共享取消按非自串行能力处理，必须 join；不采用“asr-engine 自带串行锁，所以共享取消可立即返回”的乐观假设。这样更保守，但避免 worker 提前取下一件而与仍在运行的计算并发。

取消后的部分转录不得作为成功结果返回或写入缓存；WebDAV 任务只有成功才归档。

## 进度分槽

每个 job 有两个独立快照槽，避免转录回调与分离事件互相覆盖：

- **转录槽**：`phase`、`percent`、`message`。下载阶段按已下载字节计算百分比；转录阶段 GGUF 使用 chunk 回调，asr-engine 由采样器轮询远端进度；FunASR/Qwen3 无细粒度进度时只报告阶段。格式化阶段为 100%。
- **分离槽**：独立报告 `diarize_stage`（解码、分割、嵌入、聚类、重建），不伪造百分比。

读取时由单一采样器拼装两槽。远端 asr-engine 样本只更新当前匹配 job 的转录百分比/说明，阶段仍由本地执行体推进，迟到样本不会让阶段倒退。`diarize_only` 任务没有转录阶段，percent 保持 `null`。

## noteflow 两段式对接

1. **提交阶段：**worker 捕获本代 `expectedAttempt`，先条件化落盘任务基线与 attempt，再提交 `POST /jobs`，幂等键为 `{taskCuid}-{attempt}`。收到 `202` 或 dedup `200` 后将任务转为 `awaiting` 并释放 worker 槽位；提取器只读取 attempt，不负责递增。
2. **观察与消费：**全局 poller 通过 `POST /jobs/query` 批量取状态；终态结果再逐个读取，并按来源写入 noteflow 结果后续跑。attempt 由消费远端终态的观察者递增（普通错误、缺失、取消、暂停错误等）以及 reset/重新提取入口递增；提交期 `429`、`503` 或网络结果不确定时不递增，避免同一 job 被错误地当成新尝试。
3. **429 门控：**根据 `Retry-After` 条件转回 `pending`，设置 `nextSubmitAt`，不消耗普通重试次数；领取时跳过未到期任务。每个被门控任务有独立到期唤醒，进程重启时重新安排唤醒，避免任务因定时器只存在于内存而滞留。
4. **并发回写保护：**worker/poller 的状态迁移校验任务状态与 `expectedAttempt`；旧代迟到回执不能覆盖新一代。暂停期间仍沿用既有 deferred/唤醒语义。

## 部署与评审记录

- 共享 ASR 库与 asr-engine 同树共用；先让共享库代码落盘，再重启 `tmux asr`，使引擎加载带进度回调的库。部署时也重启转录服务以加载队列代码。
- Spec 经六轮评审通过，Plan 经五轮评审通过。
- 实施裁定：ASR 线程提交使用模块级 `_ASR_EXECUTOR`，并以 `asyncio.wrap_future` 包装并等待底层 concurrent future；Spec §4.3 中 `run_in_executor` 的表述以此裁定为准。
- 版本偏斜时，共享取消按 legacy 能力降级并 join，采用 Spec §6.7 的保守降级口径，而不是 §4.3 矩阵中“asr-engine 自串行、共享取消不 join”的假设。

**验收：见完成报告**（Task 10/18 后由主会话补充）。
