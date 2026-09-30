# 暂停端点与 GPU 释放链(/pause /resume /status + noteflow 挂起联动)

日期:2026-09-30 定稿(superpowers 流程:brainstorm → spec → plan → 逐 task 实施 + review → 本地/生产双重验证)

## 需求

偶尔需要暂停转录服务释放 GPU 做别的事。Windows 桌面 bat(两个,不合并)触发:输入小时数(支持小数,回车默认 2)暂停 N 小时,到时自动恢复,可提前恢复。noteflow 联动:暂停期间任务挂起不烧 retryCount、恢复后自动重跑、提前恢复即时唤醒;说话人业务(diarize_only)一并挂起不降级。

## 架构

```
bat ──> transcribe-service(31080,时间权威:pause_state.json 落盘)
          ├─ 暂停:中间件拒三端点(503+Retry-After+{detail,paused,resume_at,paused_at})
          │        + 同步转发 asr-engine /admin/pause + 本地释放链后台(diarization 排空卸载)
          ├─ 恢复:到期 setTimeout 推送通知 / /resume 立即推送 ──> noteflow 唤醒端点
          └─ asr-engine(31090):暂停语义下沉层(拒新+卸载+惰性到期)
noteflow(mryk24):deferred 状态挂起(零 DDL) + 兜底 setTimeout + 重启自愈
```

## 三服务职责

| 服务 | 改动 |
|---|---|
| transcribe-service | `pause_manager.py`(状态落盘/惰性恢复/到点通知/世代令牌);`server.py`(/pause /resume /status、中间件拦截、`_release_gpu_for_pause` 后台释放链);`diarization/manager.py`(**diarize 的 _load 移入 _infer_lock 消除卸载竞态** + unload);`backends/asr_engine_backend.py`(UpstreamPausedError 上行穿透);config `pause.notify_url/notify_token` |
| asr-engine | ModelManager.pause/resume/is_paused;`/admin/pause` `/admin/resume`(回环无鉴权);`/health` 加 paused;transcribe 拦截(扁平 503 带 paused_at) |
| noteflow(mryk24) | `deferred` 状态(String 列新值,零 DDL);`errors.ts`(ServicePausedError+parsePausedResponse);`pause-state.ts`(恢复信号时刻,中立防循环);`pause-wake.ts`(wakeDeferredTasks 包装/兜底 setTimeout/自愈);scheduler.markTaskDeferred(旧事件判定:pausedAt≤恢复信号 或 resumeAt 过期 → 常规失败);worker 分支;唤醒端点 `transcribe-wake`(token=runtimeConfig NUXT_NOTEFLOW_INTERNAL_TOKEN,空放行);extractors **5 处透传点**(transcribeAudio×2、extractBilibiliSubtitle 总 catch、diarizeSubtitle+上层降级 catch、webdav×2);reset/cancel 白名单放行;tasks.vue 挂起展示与操作;compose 传 token env |

## 关键设计决策(踩坑沉淀)

1. **时间权威唯一在 transcribe**(pause_state.json);noteflow 不轮询不探测:恢复 = transcribe 主动推送(到期 setTimeout / resume 立即)+ noteflow 兜底 setTimeout(挂起时已知固定 resume_at)+ 重启自愈,三层各自独立失效也不丢唤醒
2. **迟到 503 判定用 paused_at(暂停起始)而非 resume_at(截止)**:提前恢复场景中截止晚于信号,比较截止拦不住;比较起始才正确
3. **auto-import 歧义**:pause-wake 包装版刻意命名 `wakeDeferredTasks`(≠scheduler 的 `resumeDeferredTasks`),Nuxt auto-import 重名会静默拿错版本
4. **startupSelfHeal 不记恢复信号**(启动≠服务恢复,记了会把跨重启旧暂停判 stale 烧重试)
5. **asr 转发与本地释放解耦**:asr 转发在 /pause 同步完成(快速 HTTP);本地释放链后台(等 diarization 推理锁是排空语义),世代令牌防"等锁期间已 resume、旧链把 asr 再暂停"
6. **错误传播链 5 处包装点全透传**(1 Python + 4 TS),漏一处挂起机制整体失效——第二轮才找全 diarizeSubtitle 的上层降级 catch
7. resume_at/paused_at 一律带时区 ISO(notateflow 在 VPS,naive 串差 8 小时)
8. **生产 DB 零变更**;Prisma client 固化于 generate(build 前必须 generate)
9. bat 的 token 用占位入库、部署时 sed 到桌面副本(真实 token 绝不入库)

## 验证记录

- 单测:transcribe 111 / asr 18 / noteflow 513+1skip,三仓库 typecheck/构建绿
- **本地全链路 13/13**(本地 noteflow 容器直连本机 transcribe):503 拦截/缓存拒绝/两路任务挂起(retryCount=0)/到期通知精确送达+自动完成/resume 89ms 唤醒/通知三连失败后兜底定时/双服务重启自愈/in_use 不阻塞+排空卸载/UpstreamPausedError 端到端/越界 422/401 优先
- **生产冒烟**:公网 503 全字段 → 生产任务 deferred(retryCount=0,文案正确)→ /resume → 公网推送通知送达 → 唤醒 processing → completed;asr_engine paused/resumed 转发同步生效

## 部署与运维

- 生产 notify_url:`https://076200.xyz/api/noteflow/internal/transcribe-wake`;token 三处一致(transcribe config.yaml pause.notify_token / mryk24 .env.prod 与本地 .env 的 NUXT_NOTEFLOW_INTERNAL_TOKEN / funasr.apiToken 是另一个值=转录 API token)
- 本地服务已切 `--no-reload`(start.sh 固化,根除假死史)
- 桌面 bat:`D:\OneDrive\Desktop_home\暂停转录.bat`/`恢复转录.bat`(OneDrive 重定向;仓库 scripts/ 存占位版)
- 回滚:git revert 即可;deferred 是新状态值,回滚 noteflow 前先把 deferred 任务手动置 pending(旧代码不认识)

## 文件索引

- spec:`docs/superpowers/specs/2026-09-30-transcribe-pause-design.md`
- plan:`docs/superpowers/plans/2026-09-30-transcribe-pause.md`
- 上一轮回退教训(为何走完整流程):见 git 历史 2026-09-30 上午
