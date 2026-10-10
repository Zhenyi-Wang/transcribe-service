# reconstruct 分块重写：说话人分离峰值内存与时长解耦

定稿日期：2026-10-10。范围：`diarization/manager.py` 的 `BoundedSpeakerDiarization.reconstruct` 覆盖实现及配置/工作器接线。承接同日 [WSL 提交内存压力文档](2026-10-10_wsl-commit-pressure-diarization-safety.md) 的 reconstruct 预算守卫。

## 1. 问题：512MB 预算把正常生产任务拒之门外

上游 pyannote 3.3.2 的 `reconstruct` 一次性分配 float64 `(num_chunks, num_frames, num_clusters)` 稠密数组（`np.nan * np.zeros(...)` 双缓冲），内存 O(时长×说话人数)。当日生产实测：

| 任务 | 时长 | 簇数 | 双缓冲需求 | 结果 |
|---|---|---|---|---|
| 20260921 平安中國（job=c16d1c0a） | 2h31m | 8 | 652.6MB | **被 512MB 预算拒绝**，降级冷却 3600s |
| 20260907 平安中國33（job=6794818e） | 2h42m | 5 | 437.8MB | 贴线通过（余量仅 74MB） |

同一讲道系列、同量级负载，簇数决定成败——512MB 默认值卡在生产画像（1-3h、最多 ~16 人讲道录音）中间，一半任务会被拒。

## 2. 上游调查结论（2026-10-10）

- pyannote-audio **#962**（3h/23人 >18GB RAM）、**#1819**（4h/50人 12GB）、**#1963**（4.0.3 VRAM 6 倍回归，峰值定位在 reconstruction/embedding）：全部未修或 stale 关闭。`reconstruct` 的稠密展开是已知设计缺陷。
- 社区缓解只有 `embedding_batch_size`/`segmentation_batch_size` 降到 4（WhisperX #274，治 GPU 侧）与长音频分段+跨段关联（学术界标准架构，需动全局聚类）。
- pyannote 4.x 在此问题上更糟，维持 3.3.2 锁版正确。

## 3. 方案：分块 reconstruct（常数级峰值，bitwise 等价）

关键观察（读 3.3.2 源码定案）：

1. `reconstruct` 的填充循环**本来就逐 chunk 写入**，大数组只为向量化方便；
2. 下游 `to_diarization` 调 `Inference.aggregate(..., hamming=False, missing=0.0, skip_average=True)`，实际是 **float32 overlap-add 求和**（逐 chunk `+=`，NaN→0×掩码），求和可结合可交换；
3. 最终信息量只有 `(total_frames, num_clusters)` 的标签（2.5h/8人 ≈ 34MB）。

覆盖版实现：按 `reconstruction_batch_chunks`（默认 256）分批持有 float32 批数组，**逐字复刻**上游填充循环（含 `k==-2` 跳过与负索引列怪癖）与 aggregate 累加循环（保持 chunk 顺序、相同 float64→float32 累加路径），再复刻 `to_diarization` 后半段（pad → extent 交集 crop → 按帧 count 取 top-c）。**不调用 `super().reconstruct()`，大数组路径彻底消失。**

数值等价性论证：批值为 float32 可精确表示的 sigmoid 概率，乘 0/1 掩码后与上游 float64 存储值逐位相同，进入相同 dtype 组合的 `+=`；chunk 顺序不变 → 加法树相同 → 逐位一致。

### 内存账（峰值上界）

| 场景 | 旧口径（float64 双缓冲） | 新口径（批 + 激活族） |
|---|---|---|
| 2.5h / 8 人 | 652.6MB（被拒） | **~107MB**（批 4.8 + 6×16.4） |
| 2.7h / 5 人 | 437.8MB（贴线） | **~92MB** |
| 4h / 16 人（画像上限） | ~2.16GB（必拒） | **~352MB** |

预算守卫保留，口径更新为分块峰值（`batch + 6×activation`，6 份涵盖累加器/掩码/pad/binary/argsort 临时）；`num_total_frames` 未知时以 `num_chunks×num_frames` 为宽松上界。守卫职责从"拦截长音频"变为"拦截异常形状"。

## 4. 实施清单

- `diarization/manager.py`：`check_reconstruction_budget` 重写（签名加 `batch_chunks`/`num_total_frames`）；`BoundedSpeakerDiarization.reconstruct` 分块实现；`DiarizationManager.__init__` 加 `reconstruction_batch_chunks` 硬验证；`_load`/`get_manager` 接线。
- `diarization/worker.py`：`_manager_settings()` 带上新字段（工作进程构造 `DiarizationManager(**settings)` 自动生效）。
- `config.py`：`diarization_reconstruction_batch_chunks`（默认 256）；`config.yaml.example` 注释。
- 既有生产 config.yaml 缺字段走安全默认，无需改动。

## 5. 验证

- **等价性**：真实 pyannote 3.3.2，6 场景 × 4 批大小（1/7/64/10000，含非整除、单 chunk、超总数、NaN 列、不同步长/帧网格）**24 组合全部 bitwise 一致**（maxdiff=0.000e+00）。
- **离线套件**：`tests/` 全量 **283 passed**（原基线 243 + 新增：等价性 24、预算新口径 7、配置/验证/接线 9）。含生产回归用例（2.5h/8人与 4h/16人场景通过 512MB 预算）与守卫仍生效用例。
- **真实短音频**（09:57，服务以新代码重启后）：`hindi_BV172846DEJT.m4s` 94.25s → 1 人 / 23 turns，与历史基线一致；reconstruct 日志新格式（`预计峰值=0.7MB(分块)`）。
- **真实长音频复跑**（10:00–10:06，job=8d0f0cc4）：当日早晨被拒的 20260921 音频（9085s）以 `no_cache=true` 重跑——**8 人 / 2545 turns，diarization.status=success，2438 段字幕全部带 speaker 标注**（主讲 4648s/1613 段，其余 7 人为领唱/祷告角色，符合聚会结构）。reconstruct `chunks=9077 clusters=8 batches=36 预计峰值=103.2MB(分块)`（设计估算 107MB），阶段前后 RSS 稳定 2836MB（旧代码此处 +652MB），耗时 1.3s（与旧代码量级相当）。总 394.3s（ASR 378.3s RTF 0.042，分离 339.5s 并行）。
- commit 纪律偏离说明：长音频复跑时宿主余量 4.96GiB < 8GiB 停止线；按增量核算执行（worker 管线已加载，新增 PCM 517MB + 激活 + 分块重建合计 <1.4GiB，完成后余量 >3.5GiB），未触发异常。

## 6. 未做与边界

- 未做音频级分段 diarization + 跨段说话人关联（业界长录音标准架构）：对讲道场景（≤3h、≤8 人典型）收益不抵全局聚类改动风险，留作 >4h 需求出现时的后备。
- GPU 侧 `embedding_batch_size` 未动：本机 22GB 显存实测无压力。
- 常驻 worker 的会话级 RSS（~1.7GB）与 `max_jobs=8` 回收策略不变；新代码只消除 reconstruct 的时长相关峰值。
- 服务已于 09:57 以新代码重启（tmux `transcribe`，start.sh），重启前暂停状态已由用户解除、`in_use=false` 确认；重启窗口期间旧服务进程已不在（另一会话操作所致），属无冲突重启。
- 宿主内核池 24GB（Paged 12.9 + NonPaged 11.1）持续高占用另案排查（独立会话处理），与本项目改动无关但同属 commit 压力源头。
- 改动截至本文定稿未提交（HEAD `ac771a1`），与在途的 WSL 安全改造同批未提交修改并存。
