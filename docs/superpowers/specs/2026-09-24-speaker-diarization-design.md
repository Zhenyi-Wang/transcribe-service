# 说话人分离集成设计（transcribe-service）

日期：2026-09-24（r2，经 spec review 修订）。调查与实测数据见 `docs/2026-09-23_speaker-diarization-feasibility.md`（本文只写实施设计，不重复论证）。

## 1. 目标

转录接口新增可选说话人分离：默认不分离，请求显式传 `diarize=true` 时，body 段带 `speaker` 标注；转录与分离并行执行，分离失败/超时降级为无标注正常转录且**不污染缓存**。

## 2. 范围

**做**：pyannote-hybrid 管线（唯一实现）；`diarization/` 模块；并行编排；词级后端"先按说话人分块再分句" + funasr 段级对齐退化路径；三端点 `diarize` 参数；响应字段（`body[].speaker` / `speakers` / `timing.diarization`）；缓存 key；配置组；pytest。

**不做**（预留接口，后续版本）：`sherpa-onnx` 第二实现（配置项 `backend` 保留枚举，非法值按禁用处理，见 §7）；声纹注册；B 站 prefix 输出模式；请求级 `num_speakers` 透传（仅配置级）；`segmentation` 模型固定为 `pyannote/segmentation-3.0`，不暴露配置项。

## 3. 模块设计

### 3.1 `diarization/`（新目录）

```
diarization/
├── __init__.py        # 仅导出 DiarizationManager, SpeakerTurn；顶层零 pyannote/torch 导入
└── manager.py
```

- `SpeakerTurn`：`dataclass(start: float, end: float, speaker: int)`。
- `DiarizationManager`：
  - **延迟导入**：pyannote / torch 的 import 全部位于 `_load()` 函数体内；模块顶层（`__init__.py` / `manager.py`）不出现任何重依赖 import——未安装 pyannote 时服务启动与 `enabled=false` 路径完全无感。
  - 懒加载 + `threading.Lock` 双检锁（照 `server.py` ModelManager 模式）；加载后进程内常驻（模型小，不做闲置释放）。另持一把 `_infer_lock`：`diarize()` 推理段串行化（pyannote 管线非并发安全；当前架构单请求，此锁为防御性约束）。
  - `diarize(samples: np.ndarray, sample_rate: int = 16000) -> list[SpeakerTurn]`：同步执行（由调用方放入 `to_thread`）。
  - 内部管线：`SpeakerDiarization(segmentation="pyannote/segmentation-3.0", embedding=<本地 ONNX 路径>, clustering="AgglomerativeClustering", segmentation_batch_size=32, embedding_batch_size=32)`，`instantiate` 超参（`clustering: {method: centroid, threshold, min_cluster_size}`、`segmentation: {min_duration_off: 0.0}`），`to(cuda)`。embedding 传本地 ONNX 路径即走 `ONNXWeSpeakerPretrainedSpeakerEmbedding`（pyannote.audio 3.3.2 `pipelines/speaker_verification.py:385`）。
  - `hf_token` 注入：配置非空时在 `_load()` 中 `os.environ.setdefault("HF_TOKEN", token)`（huggingface_hub 凭据链读取，不覆盖既有环境）。
  - 配置校验：`backend` 值 ≠ `"pyannote-hybrid"` → `_load()` 抛 `ValueError`（走 §7 禁用路径，ERROR 日志），不做静默回退。
  - 配置驱动：`cluster_threshold`（默认 0.7046）、`min_cluster_size`（默认 12）、`num_speakers`（默认 -1，>0 时作为 `pipeline(audio, num_speakers=n)` 透传）、`embedding_model` 路径。
  - 分离后把 turn 标签重映射为紧凑 0..N-1（pyannote 标签是 SPEAKER_XX 字符串，按首次出现顺序映射）。

### 3.2 音频解码

复用 `qwen_asr_gguf/inference/audio.py:load_audio(path)`（16kHz mono float32）。diarization 与 ASR 各自解码一次，不改 `ASRBackend` 接口。**load_audio 须在分离线程函数体内延迟导入**（`qwen_asr_gguf/inference/__init__.py` 会连带执行引擎导入，顶层 import 削弱 §3.1 的零依赖声明）。

## 4. TranscriptionService 集成

### 4.1 编排（`process_transcription`，transcribe.py:570）

- 签名加 `diarize: bool = False`；三个端点透传（见 §5）。
- `diarize=true` 且 `config.diarization.enabled` 时：

```
diarize_task = asyncio.wait_for(
    to_thread(diarize_thread_fn), timeout=max(60, audio_duration * 0.5)
)
asr_result, turns = await asyncio.gather(
    to_thread(backend.transcribe, path, None, context),   # 现有逻辑不动
    diarize_task,
)
```

> 实现提示：`wait_for` 超时抛 `TimeoutError` 会从 gather 传播——diarize 侧需先以 `wait_for` 包装隔离（或 `gather(..., return_exceptions=True)` 后自行处理），确保超时只影响 `turns`（置 None 降级），不中断 ASR 分支。

- 分离线程函数：`load_audio(path)` → `manager.diarize(samples)`，全程 try/except：任何异常 → WARNING 日志、返回 `turns=None`（降级），**不中断转录**。
- **超时**：`wait_for` 超时 → `turns=None` 降级；`to_thread` 线程无法强杀，超时后主流程继续，线程自然跑完即弃（可接受的短暂后台泄漏）。
- **降级不写缓存**：`turns=None`（异常或超时）时跳过 `save_transcript`，本次响应正常返回但不缓存，下次请求自然重试分离。其余情况（含单人退化、funasr `-1` 段）正常写缓存。
- `processing_time` 语义不变（= ASR transcription 耗时）；并行墙钟体现在 `timing.total`。
- 计时：`timing.diarization` **仅在 `diarize=true` 请求中存在**，值为分离分支墙钟耗时（成功、失败、超时均记录实际值；`enabled=false` 等"请求被判定不执行分离"的路径为 `0.0`）。`diarize=false` 响应不含该键（与现版本逐字节一致）。

### 4.2 字幕格式化分派（`generate_subtitle_segments_from_timestamps`，transcribe.py:268）

输入 `turns` 与 `asr_result.timestamps`：

1. **无效路径**（`turns=None`、聚类结果仅 1 个说话人、`timestamps` 为 None/空）→ 现有逻辑原样，body 不含 speaker 字段（单人退化；无时间戳时无法可靠归属，不做插值归属）。
2. **词级路径**（进入 `_merge_char_timestamps` 与 `_segment_by_punctuation` 的 item 级流，含英文词级 avg_len>2 直达 `_segment_by_punctuation` 的情况）→
   a. **speaker 标注以元数据附着**：为每个 timestamps 项附加 `speaker` 键（不改 `TranscribeResult` 结构，格式化函数内部构造扩展列表）。归属三级规则：与某 turn 有时间重叠 → 归重叠面积最大者（字主体在 A、仅尾巴漂进换人间隙时仍判 A）；与所有 turn 均无重叠（项完全落在换人间隙——ASR 不会在纯静音出字，落间隙即字边界漂移）→ 归距离最近的 turn（度量：项中点到 turn 区间 [start,end] 的距离，区间内为 0）；项整体超出 turns 覆盖范围 → 同最近规则。空格语言路径 `_regroup_fragments_by_space`（transcribe.py:309）重组新建 word 时，word 继承其成员碎片的 speaker（成员不一致时取与 word 时间范围重叠最大者），不丢弃该键。
   b. **CJK 落点（`_merge_char_timestamps`）——保持"文本优先"结构，段定位后做段内边界拆分**：该函数现有流程为"按标点把 text 切成段 → 短段并入 → 每段去 ts_chars find 定位对应 timestamps 范围 [i,j) → 按字数规则合并输出"。新增逻辑插在**段定位之后、段输出之前**：检查 [i,j) 范围内 timestamps 项的 speaker 序列，存在变化时在该项边界把当前段**局部拆分**为两段（text 按对应字符位置切、时间范围按项界切），对每段重复检查直至段内 speaker 恒一；拆分子段直接走既有的时间提取/输出逻辑，**不再参与短段并入**（避免跨界短段并回不同 speaker 的前段）。**局部拆分不破坏全局对齐**：两段 clean 文本拼接仍等于原 clean_seg，其他段的 find 匹配互不影响；除该拆分外，标点分句、字数钳制、短段并入、估算兜底逻辑零改动。时间戳耗尽走估算兜底的段无 [i,j) 项范围，speaker 检查自然为空、不拆分，**继承前一段的 speaker**（无前段则该段不带 speaker 键）。
   c. **空格语言路径（`_segment_by_punctuation`）**：该函数为按 timestamps 项累加结构，在累加循环中新增"相邻项 speaker 不同即 flush 当前段"条件（与句末标点 flush 同级），text 原样传递。
   d. 词级路径产出的所有段继承其成员项的 speaker（段内 speaker 恒一致，**此路径不产生 -1**）。拆分产生的块首短段无前段可并时保留为独立段（说话人切换点的短句独立成条，符合字幕规范）。
3. **句级退化路径**（funasr；仅 `_segment_simple` 分支即无可用 item 时间戳的内容）→ 现有逻辑生成段后，每段 `[from,to]` 与 turns 求最大重叠，重叠 ≥ 段时长 50% 才赋标签，否则 `-1`。

### 4.3 响应组装（`diarize=true` 且非无效路径时）

- `body[]` 段追加 `"speaker": int`（词级路径全部 ≥0；funasr 路径可能 -1）。
- 顶层 `speakers`: `[{"id": 0, "duration": 120.5, "segments": 40}, ...]`——**仅统计 speaker ≥ 0 的段**（`-1` 段不计入任何 id），按 id 升序；`duration` = Σ(to-from)，`segments` = 段数。funasr 路径全部段为 `-1` 时 `speakers` 为空数组 `[]`（键仍存在）。
- `timing.diarization`：见 §4.1。

三种典型响应形态：

| 场景 | `body[].speaker` | 顶层 `speakers` | `timing.diarization` |
|------|------------------|-----------------|---------------------|
| 多说话人成功 | 段级 int（≥0） | 存在 | 实际耗时 |
| 单人视频（退化） | 不存在该键 | 不存在该键 | 实际耗时 |
| 分离失败/超时降级 | 不存在该键 | 不存在该键 | 实际耗时 |
| `diarize=false` / `enabled=false` | 不存在该键 | 不存在该键 | 键不存在（false）/ 0.0（enabled=false） |

## 5. API 与缓存

- `POST /transcribe`：加 `diarize: bool = Form(False)`（server.py:217）。
- `POST /transcribe_url`：`BilibiliTranscribeRequest` 加 `diarize: bool = False`（server.py:132）。
- `POST /transcribe_file`：`WebdavTranscribeRequest` 加 `diarize: bool = False`（server.py:143）。
- 缓存 key：`diarize=true` 时源串追加 `#diar:1` 后再 md5（照 context 的 `#ctx:` 范式，cache_manager.py:46-49）；**拼接顺序固定**：`源串 + "#ctx:{sha1[:10]}"（context 非空时） + "#diar:1"（diarize=true 时）`——保证 key 稳定可复现。`diarize=false` 的 key 与现状完全一致（旧缓存兼容）。缓存命中原样返回缓存 JSON，响应层不追加任何键。
- **降级不缓存**：见 §4.1。

## 6. 配置

`config.yaml` / `config.yaml.example` 新增组（照 `asr_engine` 组范式，config.py:98）：

```yaml
diarization:
  enabled: false
  backend: pyannote-hybrid      # 唯一实现；其他值按禁用处理（§7）
  cluster_threshold: 0.7046
  min_cluster_size: 12
  num_speakers: -1              # -1 自动
  embedding_model: "~/models/diarization/wespeaker_cnceleb_resnet34_LM.onnx"
  hf_token: ""                  # 空 = huggingface_hub 默认凭据链；非空注入 HF_TOKEN 环境变量
```

`config.py` 加对应 property（`get('diarization.xxx', default)`）。模型说明：`pyannote/segmentation-3.0` 首次加载经 HF 下载缓存（gated，需 token）；embedding 本地无门。

## 7. 错误处理

| 故障 | 行为 | 缓存 |
|------|------|------|
| 分离线程异常（模型加载失败/推理失败/onnxruntime 缺失/ONNX 文件缺失） | WARNING 日志，`turns=None`，正常返回无 speaker 转录，`timing.diarization`=实际耗时 | **不写** |
| 分离超时（`wait_for`） | 同上，超时时长 `max(60, audio_duration*0.5)` | **不写** |
| `backend` 配置非法 | `_load()` 抛 ValueError，经 catch-all 记 WARNING，等同禁用 | — |
| `diarize=true` 但 `enabled=false` | 按未启用处理（无 speaker 字段），日志提示 | 正常写；key 仍含 `#diar:1`，缓存内容为无 speaker 结果 |
| 聚类仅 1 人 | 单人退化：无 speaker 字段 | 正常写 |
| `timestamps` 为空 | 无可靠归属 → 不启用 speaker | 正常写 |
| `diarize=false` | 现状路径完全不变 | 正常写 |

> 注：`enabled=false` 但 `diarize=true` 的请求缓存的是无 speaker 结果且 key 含 `#diar:1`——用户后来改 `enabled=true` 需等待 TTL 或手动清缓存才生效。可接受（个人服务，配置变更是低频操作），记入已知限制。

## 8. 测试（tests/，pytest，mock turns 不依赖真实模型）

- 字级归属：重叠面积最大；完全落换人间隙取最近 turn（中点到 turn 区间距离）；超出范围取最近。
- CJK 段内边界拆分：speaker 变化落在标点段内部时正确拆段，**两段 clean 文本拼接 == 原 clean_seg**（对齐不变式）；无 speaker 变化的输入与现有 `_merge_char_timestamps` 测试基线逐字节一致。
- 空格语言：`_regroup_fragments_by_space` 重组后 word 携带 speaker（跨界碎片取重叠最大者）；`_segment_by_punctuation` 的 speaker flush 正确断句。
- 既有断句行为回归：无 speaker 变化的输入下，标点/字数/短段并入行为与 `tests/test_merge_char_timestamps.py` 基线完全一致。
- funasr 退化：重叠过半/不过半 → 赋值 / -1。
- 单人退化：单 turn → 无 speaker 键、无 speakers 键、`timing.diarization` 有值。
- 降级：diarize 抛异常 / `wait_for` 超时 → 正常转录响应、`timing.diarization`=实际耗时、**无缓存写入**。
- 缓存：`#diar:1` 追加与 `diarize=false` key 不变；context 与 diarize 同时非空/开启时拼接顺序为 `#ctx:{sha1[:10]}#diar:1`（照 `tests/test_context_cache_key.py` 模式）。
- speakers 汇总：多 id 聚合正确、`-1` 段被排除；funasr 全 `-1` 时 `speakers == []`（键存在）。
- 端点参数：三接口 `diarize` 透传。
- 配置：`backend` 非法值走禁用路径；`enabled=false` 行为。

## 9. 依赖与环境

- conda `funasr` 环境：`pyannote.audio==3.3.2`（已装；**部署重装时必须锁 `numpy==1.26.4`**，pyannote 依赖链会拉 2.x 破坏 funasr-onnx）、onnxruntime 包可导入（本机为 onnxruntime-gpu 1.23.2，实现时验证 `import onnxruntime` 满足 pyannote 的 `ONNX_IS_AVAILABLE` 检查）。
- 模型：`~/models/diarization/wespeaker_cnceleb_resnet34_LM.onnx`（已就位）；`pyannote/segmentation-3.0` 经 HF 缓存（token 已配 `~/.cache/huggingface/token`）。
- 显存：分离 ~0.4GB，与 ASR 实际加载 5~7GB 共存，22GB 充裕。

## 10. 验收

1. `python -m pytest tests/ -q` 全绿。
2. 实发 `diarize=true`：官方 4 人音频 → body speaker 聚成 4 组、`speakers` 汇总正确；单人视频 → 无 speaker 字段；世相样本 → 分组结构与 bench 一致。
3. 并行生效：`timing.diarization` 与 `timing.transcription` 量级重叠，`timing.total` ≈ max 而非求和。
4. `diarize=false` 响应与现版本逐字节一致（缓存命中路径复验）。

## 11. 下游消费者（另见 mryk24 spec）

mryk24 NoteFlow 为首个消费方（`diarize` 开关 + 【说话人N】分组格式化 + summarize prompt），独立 spec：`mryk24/docs/superpowers/specs/2026-09-24-noteflow-speaker-diarization-design.md`。实施顺序：本 spec 先行。
