# transcribe-service 暂停功能设计(GPU 释放 + noteflow 联动)

日期:2026-09-30 · 状态:待用户 review

## 1. 背景与目标

偶尔需要暂停转录服务释放 GPU(RTX 2080 Ti 22G)做别的事。触发方式:Windows 桌面 bat(两个:暂停/恢复,不合并),暂停脚本输入小时数(支持小数,直接回车默认 2)。暂停 N 小时到时自动恢复;可提前手动恢复。

**noteflow 联动要求**:暂停期间任务挂起、不烧 retryCount;恢复后自动重跑;提前手动恢复也要及时唤醒。暂停范围包括说话人业务(diarize_only)——开启了说话人相关的任务一并暂停,不降级继续。

**历史教训**(上一轮实施回退,2026-09-30 上午):noteflow 错误传播链有三处包装点会丢失错误类型;Prisma client 字段映射在 generate 时固化进构建产物;生产 DB 迁移历史分叉不可 migrate;`--reload` 运行模式有假死史。本轮设计逐一规避。

## 2. 架构总览

```
Windows bat ──HTTP──> transcribe-service(31080,状态与通知中枢)
                          │  暂停时: 拒新请求 + 卸载 diarization + 转发暂停
                          ▼
                      asr-engine(31090,暂停语义下沉层:拒新+卸载)
                          
transcribe-service ──恢复通知(推送)──> noteflow/mryk24 唤醒端点
noteflow(挂起任务 deferred) <──兜底: 自身 setTimeout 到点唤醒──┘
```

暂停时长的时间状态只有一个 owner:transcribe-service(`pause_state.json` 落盘)。noteflow 挂起时从 503 body 获得固定的 `resume_at`,用于兜底定时与用户展示,不做探测、不做轮询。

## 3. transcribe-service

### 3.1 端点(均走既有 Bearer 中间件)

| 端点 | 行为 |
|---|---|
| `POST /pause {hours: 0<h≤48}` | 置暂停态(覆盖语义)→ 卸载链 → 设进程内 `setTimeout(到点恢复通知)` → 返回各环节真实状态 |
| `POST /resume` | 清暂停态 → `clearTimeout` → 立即发恢复通知 → 返回 |
| `GET /status` | `{paused, resume_at(带时区ISO), remaining_seconds, backend, model_loaded, in_use, diarization_loaded, asr_engine_paused}` |

暂停期间三个转录端点(`/transcribe`、`/transcribe_url`、`/transcribe_file`,**含 diarize_only 变体**)在中间件层(token 校验后)返回:

```
503 + Retry-After: <剩余秒>
{"detail": "服务暂停中，预计 HH:MM 恢复", "paused": true, "resume_at": "<+08:00 ISO>", "paused_at": "<+08:00 ISO>"}
```

`paused_at` 是本次暂停的**起始时刻**——noteflow 用它识别迟到的旧暂停事件(见 §5.1)。

中间件层拦截先于缓存检查/下载/临时文件落盘,缓存命中同样拒绝(语义统一)。`resume_at` 必须带时区偏移(noteflow 在 VPS,naive ISO 会被 JS 按本地时区解析偏移)。

### 3.2 暂停状态

- `pause_manager.py`:`paused_until`(epoch)、原子落盘 `pause_state.json`(tmp + os.replace)、损坏/过期视为未暂停(仅告警)、重复 pause 覆盖
- 恢复判定:惰性时间戳比较,无定时器
- 启动/`__main__` 预加载在暂停态跳过;重启读回状态文件时按剩余时长重设通知 setTimeout

### 3.3 GPU 释放链(pause 时)

1. 本地 backend unload(当前 asr-engine 后端为 no-op,保留通用逻辑)
2. diarization 卸载:新增 `unload()`。**竞态修正**:`diarize()` 的 `_load()` 移入 `_infer_lock` 内(先拿推理锁再懒加载),使"加载→推理"成为受同一锁保护的完整生命周期;unload 按 `_infer_lock → _load_lock` 顺序清空 + `gc.collect()` + `torch.cuda.empty_cache()`。模块级 `unload_global()`(卸载并丢弃单例)/ `is_loaded()`
3. asr-engine 转发暂停(见 §4)

`in_use` 时 pause 立即拒新请求;**释放链一律异步后台执行**(`asyncio.to_thread` 触发后不等待,`/pause` 立即返回),monitor 线程按 `should_unload` 标志周期补做(幂等)。理由:① diarization 的 `unload()` 要等 `_infer_lock`(在跑的长推理可达分钟级),同步执行会阻塞 `/pause` 响应;② `diarize_only` 路径不经过 ModelManager 的 in_use 计数,"完全空闲才同步释放"的判定不可靠——交给后台线程等锁本身就是排空语义(推理自然跑完才卸载)。

**世代令牌防交错**:`pause_manager` 维护单调递增 `_generation`,`/pause` 与 `/resume` 均自增;后台释放链开始时捕获当前世代,**每个耗时步骤前(等 diarization 锁后、转发 asr `/admin/pause` 前)检查世代未变,变了即中止**——防止"后台链等锁期间用户已 resume,旧链随后把已恢复的 asr-engine 又暂停"的交错。**`/resume` 同时清除 `should_unload` 标志**(否则 monitor 在恢复后持续卸载,退化为每次请求重加载);后台链对本地 backend 卸载沿用 in_use 守卫(当前 asr-engine 后端 no-op 无碍,防将来切回本地 GPU 后端时在推理中卸载)。响应字段(扁平,如实):`{new_requests: "rejected", backend: "triggered", diarization: "triggered", asr_engine: "triggered", note: "释放已在后台执行,在途任务跑完后完成(通常 ≤90s,长视频说话人分离可能更久);最终状态可查 /status"}`——异步化后 `/pause` 返回的是"已触发"而非结果,`/status` 的 `asr_engine_paused` 可查询最终态;bat 文案据实显示,不笼统宣称"GPU 已释放"。

### 3.3.1 asr-engine 暂停的上行传播(窄竞态窗口)

已在途请求(过了 transcribe 暂停检查)转发到 asr-engine 时恰逢其暂停:`ASREngineClientBackend` 收到 503 + `{paused:true}` body 时**不再包成 RuntimeError**,而是抛专用 `UpstreamPausedError`;**`transcribe.py process_transcription` 的大 `except Exception`(现为 ~1116 行)与 `/transcribe_file` 端点外层 catch 遇 `UpstreamPausedError` 必须原样重抛,不得转成 200 的 `{status:"error"}` dict**(Python 侧的第五个包装点,漏掉则 noteflow 收到 200 error 烧重试,用例 9a 必失败);最终由 `server.py` 三个端点捕获并返回与中间件同格式的 `503 {paused, resume_at, detail}`(优先取 UpstreamPausedError 携带的 asr body 值(asr 与 transcribe 暂停窗口同步,提前恢复后 transcribe 态已清空,asr body 的值才能正确支持 noteflow 迟到事件判定);缺失时回退当前时间)。noteflow 收到的仍是标准暂停 503 → 挂起,不烧重试。

### 3.4 恢复通知(主动推送)

- 新增配置 `pause.notify_url`(默认指向 mryk 公网的 noteflow 内部唤醒端点)与 `pause.notify_token`
- `/pause` 设 `setTimeout(paused_until - now)`:到点(= 惰性恢复时刻)POST 通知,失败重试 3 次(短退避)
- `/resume` 立即 POST 同一通知
- 重启恢复暂停态时按剩余时长重设
- 通知失败不阻塞 pause/resume 返回;最终兜底是 noteflow 自身定时与手动 reset

## 4. asr-engine(暂停语义下沉层)

- `POST /admin/pause {hours}` / `POST /admin/resume`(无 token,回环惯例同 `/admin/cancel`)
- 暂停态:置 `paused_until`(惰性判定)+ 立即卸载(in_use 时由其 monitor 补做)+ `/v1/audio/transcriptions` 拒新请求返回 `503 {paused: true, resume_at}`;`/health` 响应增加 `paused` 字段(transcribe `/status` 的 `asr_engine_paused` 据此查询,不加新端点)
- 当前无其他调用方,但暂停做在 asr 层,将来任何直连方都被管住
- transcribe-service `/pause` 转发 hours、`/resume` 转发恢复;asr-engine 不可达时容错(其 idle 600s 卸载兜底)

## 5. noteflow(mryk24)

### 5.1 挂起机制:新状态 `deferred`(零 DDL)

- `status` 为 String 列,加值不改表;挂起信息(error 文案)已够展示,**无新列、无 JSON 字段、生产 DB 完全不动**
- `ServicePausedError`(name 判定)从 503 body 解析(`paused === true` 才识别;携带 `resume_at` 与 `paused_at`;普通 503/网络错误走原路径)
- worker catch 识别 → `markTaskDeferred(taskId, error)`:**旧事件判定——`paused_at <= 最近恢复信号时刻`(该次暂停开始于最近一次唤醒之前,是迟到的事件)或 `resume_at` 已过期,均落回 `markTaskFailed` 常规路径**(烧一次 retryCount 属可接受的自收敛,不静默置 pending)。注意判定必须用**暂停起始时刻**而非截止时刻:提前恢复场景(截止 18:00、17:00 恢复、17:01 迟到 503 带 resume_at=18:00)中截止时间晚于信号,比较截止拦不住;比较起始(paused_at=暂停开始 ≤ 17:00)才正确。`paused_at` 缺失时退化为仅时间过期判定(保守挂起,兜底定时救)。通过判定才挂起:status='deferred' + error="System: 转录服务暂停中,预计 {恢复时间} 恢复,恢复后自动重跑"(恢复时间用 `resume_at` 按 `Asia/Shanghai` 时区格式化——VPS 本地时区是 UTC,直接 toLocaleString 会差 8 小时),**不碰 retryCount**
- 领取查询 `status='pending'` 天然排除 deferred:无 CAS 复活竞态、扫描器建任务跳过(unique 冲突)、僵尸清理不涉及

### 5.2 错误透传(识别与透传并重,漏一处即失效)

**识别位置**:503 的识别发生在 `!response.ok` 分支内**主动解析 body**(`paused === true` 才抛 `ServicePausedError`,带 `resume_at`);不是等异常从 catch 里捞。网络错误(fetch failed/超时)不识别为暂停——不凭空挂起,走原有重试;只有带 `paused:true` 的 503 才挂起。

**透传点**(四处 TS + 一处 Python,每处都是"包装/吞错前先判型重抛"):

1. `bilibili.ts transcribeAudio`:`!response.ok` 分支识别;catch 对 `ServicePausedError` 直接重抛(**不走退避重试循环**——fatal 同款短路)
2. `bilibili.ts extractBilibiliSubtitle` 总 catch(转 `{success:false}` 返回前透传)
3. `webdav.ts extractWebdavSubtitle`:`!response.ok` 分支识别;外层 catch(转 `{success:false}` 返回前透传)
4. **`bilibili.ts` 官方字幕说话人标注路径的上层降级 catch**(`diarizeSubtitle` 的 503 识别 + 抛出后,调用方 catch 记警告并降级返回无标注字幕处)——必须先判 `isServicePaused` 透传,否则任务带着无标注字幕"成功"结束,违反挂起语义
5. (Python 侧见 §3.3.1)`transcribe.py` catch-all 与 `/transcribe_file` 外层 catch 对 `UpstreamPausedError` 重抛

`diarizeSubtitle`(官方字幕说话人标注)收到 503 同样抛 `ServicePausedError` 挂起任务,不再降级为无标注继续。

### 5.3 唤醒(双层,均为固定时间调度,无轮询无探测)

- **被动主路径**:内部唤醒端点 `POST /api/noteflow/internal/transcribe-wake`(token 校验:token 从 Nuxt runtimeConfig 环境变量 `NUXT_NOTEFLOW_INTERNAL_TOKEN` 读取,transcribe 侧 `pause.notify_token` 配同一值;现有 `/internal/worker` 无鉴权是历史遗留,本端点带头):所有 `deferred` → `pending`(清 error 中的挂起文案)+ `clearTimeout` 兜底定时 + 触发 worker。幂等,重复调用无害
- **自身兜底**:`markTaskDeferred` 内创建/覆盖 `setTimeout(到点自唤醒)`(模块级句柄;新的挂起取更晚的 resume_at 时重设,取更早时也重设——始终对齐最近一次挂起的 resume_at),与唤醒端点走同一段唤醒代码;被端点唤醒后 `clearTimeout`。**无常驻定时器注册**——定时只在有挂起任务时存在;内存态,重启丢失由自愈覆盖
- **重启自愈**:启动时发现 `deferred` 存在(定时已丢)→ 直接置回 pending 并触发;若服务仍暂停,任务跑到转录步再次挂起、重建定时与通知链,自愈
- 手动 reset API 必须支持 `deferred` 状态:置回 `pending` + 触发调度。当前实现对部分状态有白名单限制,实施时把 `deferred` 加入白名单(若返回错误则改);这是"点了重试却不动"的预防
- **手动 cancel API 同样把 `deferred` 加入状态白名单**(当前 `cancel.post.ts` 仅允许 `pending`/`processing`,会拒绝取消挂起任务);取消后的任务不再被唤醒端点复活(端点只动 `deferred`)

### 5.4 前端

tasks.vue:`statusLabels` 加 `deferred: "挂起"`(蓝色系)、筛选选项、计数;挂起任务显示 error 文案中的预计恢复时间;领取后清除失效挂起文案

## 6. Windows bat

桌面真实路径 `D:\OneDrive\Desktop_home`(OneDrive 重定向)。两个文件,UTF-8 无 BOM + `chcp 65001` + 显式 `curl.exe`(Win10 自带,WSL2 localhost forwarding 已验证连通):

- `暂停转录.bat`:`set /p` 输入小时数(回车默认 2)→ 数字校验 → POST /pause → 显示响应(据 §3.3 的扁平释放状态字段如实呈现)
- `恢复转录.bat`:POST /resume → GET /status 展示

## 7. 验证计划(本地全链路,用户已确认策略)

直接用生产 transcribe(本机服务,失败手动重置可接受)。本地 noteflow 容器 `funasr.apiUrl` 改指 `http://<宿主>:31080/transcribe_url`(容器内可达性实测;容器→宿主走网关 IP,不是 localhost)。测试任务选短视频。

前置:本地 `prisma generate` → `pnpm build` → 重启本地容器;验证 `.output` 内 client 与 schema 一致。

用例清单:

1. 暂停 → 三端点 503(Retry-After/paused body/带时区 resume_at);缓存命中同样 503
2. 无字幕短视频任务(走转录)在暂停窗口入队 → 挂起:deferred、retryCount 不变、error 含恢复时间、任务页显示"挂起"
3. 官方字幕+说话人标注任务在暂停窗口 → diarizeSubtitle 503 → 同样挂起(不降级)
4. 到期自动恢复(短窗口如 0.02h)→ transcribe 推送通知到达 → deferred → pending → 任务自动跑完
5. 手动 /resume 提前恢复 → 通知即时唤醒 → 任务重跑
6. 通知失败模拟(临时改错 notify_url)→ noteflow 自身兜底定时到点唤醒
7. noteflow 重启(挂起中)→ 自愈唤醒或重新挂起
8. 暂停中重启 transcribe → 状态文件恢复、通知 setTimeout 重设
9. in_use 时 pause → 拒新请求立即、响应不阻塞、卸载由后台补做(diarize_only 在途同理)
9a. 暂停瞬间在途请求到达 asr-engine → UpstreamPausedError → 端点返回 503 paused → noteflow 挂起(不烧重试)
9b. 提前恢复通知完成后,迟到的在途 503 才到 worker → resume_at 已过期 → 不挂起直接重试
9c. 后台释放链等锁期间 /resume → 世代令牌中止旧链,asr-engine 不会被再次暂停
9d. 取消挂起任务 → 成功;后续唤醒通知不影响已取消任务
10. 手动 reset 挂起任务 → 放行重跑
11. 连续 pause 覆盖 / pause 后立即 resume / hours 越界(0/负/49/非数字)→ 422
12. 无 Authorization → 401 优先于 503

生产验证(部署后):真实短视频任务重置一轮,观察挂起→恢复→完成;失败可手动重置(用户已确认此策略)。

## 8. 部署顺序与回滚

顺序:

1. asr-engine(新管理端点,旧接口兼容,不影响现有流量)
2. transcribe-service(新端点;**生产切换 `start.sh` 为 `--no-reload` 并实测进程参数**——历史假死根源)
3. noteflow(generate → build → 本地验证 → sync.sh;**无 DB 变更**)
4. 桌面 bat

回滚:

- 代码回退后必须重新 `prisma generate` 再 build(上轮教训:client 固化)
- transcribe 若处暂停态,先恢复服务再回滚 noteflow(旧 noteflow 不认识 deferred,挂起任务需手动置回 pending)
- 回滚 transcribe 后,noteflow 新代码的 deferred 任务靠手动 reset 兜底

## 9. 边界条件清单

| 场景 | 行为 |
|---|---|
| pause 时任务在跑 | 拒新立即生效;释放链异步后台等锁排空(diarize_only 在途同样被 _infer_lock 排空);≤90s 内补做完成 |
| 暂停中重启 transcribe | pause_state.json 恢复;通知 setTimeout 按剩余重设 |
| 暂停中重启 noteflow | deferred 在 DB;启动自愈置回 pending,若仍暂停则重新挂起 |
| 通知推送失败 | transcribe 重试 3 次;noteflow 自身兜底定时;最终手动 reset |
| asr-engine 不可达 | 转发容错(其 idle 600s 卸载兜底);/pause 不因此失败 |
| 状态文件损坏/过期 | 视为未暂停 + 告警;过期顺手删文件 |
| 恢复瞬间的在途 503(通知先到、503 后到) | 503 的 `paused_at` 早于最近恢复信号(或 resume_at 已过期)→ `markTaskDeferred` 不挂起,落回常规重试(自收敛) |
| 后台释放链执行中用户 /resume | 世代令牌使旧链在每个耗时步骤前中止,不会把已恢复的 asr-engine 再暂停 |
| 取消挂起任务后再收到唤醒通知 | cancel 置 `cancelled`(唤醒端点只动 deferred),互不影响 |
| 网络错误(非 503) | 不识别为暂停,走原有重试链(不凭空挂起) |
| 连续 pause | 覆盖;noteflow 兜底定时随最新 resume_at 重设 |
| 多任务同时挂起/唤醒 | 唤醒为批量 updateMany,幂等 |
| 手动取消挂起任务 | 现有取消逻辑对 deferred 生效(status 直改 cancelled),唤醒端点只动 deferred 不影响 |
| 暂停期其他调用方(visual-split-mark、repair 脚本) | 收到 503 + 明确 detail;人工暂停期属预期失败,不改代码 |
| 时区 | resume_at 带时区 ISO;挂起文案用本地化时间展示 |
| 自动扫描器暂停期建任务 | 任务跑到转录步即挂起,恢复后重跑 |

## 10. 非目标(YAGNI)

- 不引入队列/Redis/分布式锁/跨服务事务
- 不做周期探测、不做轮询对账(固定时间调度 + 推送 + 自愈已覆盖)
- 不暂停整个 noteflow pipeline(字幕提取/AI 总结/存 git 照常;只有依赖转录/说话人的任务挂起)
- 不给 asr-engine 加 token 鉴权(回环,既有惯例)
- asr-engine 不做暂停状态落盘(它崩溃重启即未暂停;时间权威在 transcribe,通知链自愈)
