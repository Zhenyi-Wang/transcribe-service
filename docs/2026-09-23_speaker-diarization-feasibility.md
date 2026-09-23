# 说话人识别集成可行性调查

> 调查日期：2026-09-23（含本机全量实测）。
> **最终结论：推荐「混合管线」= pyannote GPU 分割/批量推理 + wespeaker CN-Celeb 中文 embedding（ONNX）**：17.4 分钟音频仅 27.7s（RTF 0.027）、显存 412MB、官方 4 人基准全对、阈值 0.5~0.7 全程稳定。sherpa-onnx CPU 作零显存备选；sherpa-onnx GPU 经实测排除（比 CPU 更慢，见 §5.3）。

## 1. 问题定义

"说话人识别"在字幕场景实际包含两层需求，建议分两期做：

1. **说话人分离（diarization）**：解决"谁在什么时候说话"，产出 speaker turn 时间线并映射到字幕段。这是 B 站字幕/多人对话视频（访谈、播客、会议）的核心需求，本调查主体。
2. **声纹识别（speaker identification）**：对着已注册的声纹库做身份匹配（把 SPEAKER_00 变成"张三"）。二期扩展，复用同一套 embedding 模型（CAM++/wespeaker ONNX 均可），成本很低。

## 2. 现状与集成点（代码库事实）

- `TranscribeResult` 统一输出 `timestamps: [{"text","start","end"}, ...]`（秒），四种后端共用（`backends/base.py:7-13`）。时间戳粒度：funasr 句级、gguf/qwen3/asr-engine 词级。
- 转录主流程（`transcribe.py:570-771`）：clamp context → 查缓存 → 懒加载后端 → `backend.transcribe(path)` → **字幕格式化（字级合并为句段，`_merge_char_timestamps` transcribe.py:358-393，`max_segment_length=20`）** → 组装响应 → 存缓存。
- Bilibili 字幕 body 段字段：`{"from", "to", "sid", "location", "content", "music"}`（transcribe.py:238-245）。**speaker 字段是自然的外挂点**。
- 三个端点参数模式：`/transcribe` 为纯 UploadFile（server.py:217-239）；`/transcribe_url`、`/transcribe_file` 为 pydantic JSON 模型，`context` 是"加可选参数"的现成参照（server.py:132-149）。
- 缓存 key：context 以 `#ctx:{sha1[:10]}` 后缀进 md5（cache_manager.py:46-49），diarize 开关可照抄此范式。
- 配置新增组的现成范式 = `asr_engine` 组（yaml 顶层 key + config.py 两个 property，config.py:98-107）。
- 音频链路：downloaders 不转码；后端内部 `load_audio` 解码为 16kHz mono float32（`qwen_asr_gguf/inference/audio.py:155-174`），**全程无中间 wav 文件** → diarization 需自行解码一次（ffmpeg 秒级开销）。
- 运行架构：transcribe-service（conda `funasr`）经 `backend.name: asr-engine` HTTP 调用 asr-engine（同环境，tmux `asr`）。**显存按实际加载算：ASR ≈5~7GB**（`nvidia-smi` 的 2.1GB 是闲置卸载后的假象）。

## 3. 候选方案对比（实测后）

| 方案 | 实测表现 | 结论 |
|------|----------|------|
| A. FunASR `spk_model` | 未实测（社区：34min/11 人 CPU 全程 587s） | funasr 后端顺带增强，非主线 |
| B1. sherpa-onnx CPU（CAM++） | RTF 0.14；官方基准 4/4 对；零显存 | **备选**（速度敏感场景之外的稳妥选择） |
| B2. sherpa-onnx GPU（官方 CUDA wheel） | **RTF 0.42~0.49，比 CPU 慢 3~5 倍**（§5.3） | **排除** |
| B3. pyannote 原版 GPU（wespeaker-voxceleb 英文 embedding） | RTF 0.018 极快；但中文内容归属过粗（SPEAKER_01 全片仅 0.37s），调参救不了（§5.4） | 排除（embedding 换中文后复活 → B4） |
| **B4. 混合管线：pyannote GPU + wespeaker CN-Celeb 中文 ONNX embedding** | **RTF 0.027、显存 412MB、官方 4/4 对、阈值全程稳定** | **推荐** |
| C. 端到端多说话人 ASR | 推翻现有引擎 | 排除 |

## 4. 推荐集成设计（B4 混合管线）

> **改动范围：仅 transcribe-service**（diarization/ 模块 + 编排层）。asr-engine 零改动零感知，继续纯 ASR；分离在 transcribe-service 进程内执行，故四种后端切换均保留说话人能力。部署：transcribe-service 重启生效，旧缓存兼容（仅 `diarize=true` 用新 key）。

### 4.1 架构

新增独立 `diarization/` 模块，**与 ASR 后端完全解耦**，插在转录完成后、字幕格式化之后：

```
backend.transcribe(path) ─┐
                          ├─ asyncio.gather 并行
DiarizationManager.diarize(path) ┘
        ↓
① 字级 speaker 标注（词级后端）：每个字按时间与 turn 的重叠归属（边界字取重叠更大者/中点判定）
        ↓
② 连续同 speaker 的字聚成「说话人块」——块边界 = 说话人边界，优先级高于标点
        ↓
③ 块内跑现有分句逻辑（_merge_char_timestamps 原样复用，输入从全片字流变为块内字流）
        ↓
④ 段继承块的 speaker → body（天然全单说话人）→ 缓存 → 响应
```

**分句与说话人的关系（定稿：先按说话人分、再按字数分）**：说话人边界作为分句第一道边界，跨 speaker 段从根源消除。性质：
- `_merge_char_timestamps` 逻辑零改动，仅调用粒度变为块内；
- `-1` 基本消失（对齐误差 ±0.1~0.3s 只影响边界一两个字的归属）；短插话（如"嗯"）独立成段——符合字幕规范，且 diarization 过碎不再致命（错块最多多分段，不会错标）；
- **funasr 后端退化路径**：句级时间戳无字可分 → 段级对齐（body 段 [from,to] 与 turns 最大重叠，过半才赋标签，否则 `-1`），"宁缺勿错"；
- 无时间戳估算兜底段：按比例插值后同样走字级归属。

**转录与分离并行执行**（2026-09-23 已验证）：两者无数据依赖、资源异构——ASR 推理在 asr-engine 远端 GPU 进程（本机仅 HTTP 等待，不吃 CPU），分离在本地 CPU（sherpa 档）或 GPU（混合档，412MB 与 ASR 共存无压力）。实现为 `asyncio.gather(to_thread(backend.transcribe, ...), to_thread(diarizer.diarize, ...))`，总耗时从"相加"变"取最大"：

| 档位 | 串行 | 并行 | 实测依据 |
|------|------|------|----------|
| sherpa CPU | ×3.5 | **×2.5**（max(59, 146)≈146s） | 双向实测：分离满载期间 ASR RTF 0.058（基线 0.057）；ASR 并发期间分离 RTF 0.146（基线 0.140，+4.7s 为 ffmpeg 解码轻微竞争），聚类结果一致 |
| 混合 GPU | ×1.47 | **≈×1.0**（max(59, 28)≈60s） | 分离仅 28s，被 ASR 完全掩盖 |

工程要点：
- **降级策略**：diarization 失败/超时 → 返回不带 speaker 的正常转录结果（gather 里单独捕获，不整体失败）；
- 解码各做一次（两次 ffmpeg，秒级，可忽略）；gguf 后端的 llama.cpp 互斥锁只锁 GPU 推理，不阻塞 CPU 分离线程；
- 服务本身单请求串行（引用计数），diarization 管线无并发调用问题；首次请求含模型加载（混合档 ~2s，sherpa ~0.4s）。

- 对齐放在 body 句段级（四种后端统一汇成 body 段）；词级按说话人切开留作后续优化。
- 解码：复用 `qwen_asr_gguf/inference/audio.py:load_audio`，diarization 与 ASR 各自消费一次，不改 `ASRBackend` 接口。
- 生命周期：混合管线显存仅 ~0.5GB，懒加载常驻即可；照 ModelManager 模式做引用计数（可选）。
- **`diarization.backend` 保留两个实现**：`pyannote-hybrid`（默认）与 `sherpa-onnx`（零显存备选），接口统一为 `diarize(samples_16k_f32) -> turns`。

### 4.2 配置（新增 `diarization` 组，照 `asr_engine` 组范式）

```yaml
diarization:
  enabled: false
  backend: pyannote-hybrid     # pyannote-hybrid（默认）| sherpa-onnx
  num_speakers: -1             # -1 自动；已知人数时指定更稳（pyannote 原生支持）
  cluster_threshold: 0.7046    # pyannote 默认值；实测 0.5~0.7 稳定
  segmentation_model: "pyannote/segmentation-3.0"   # gated，需 hf_token
  embedding_model: "~/models/diarization/wespeaker_cnceleb_resnet34_LM.onnx"
  min_cluster_size: 12
  hf_token: ""                 # 或读 ~/.cache/huggingface/token
```

模型落位（`~/models/diarization/`，均已就位）：
- `wespeaker_cnceleb_resnet34_LM.onnx`（26MB，中文 CN-Celeb 训练，输入 `feats B×T×80` / 输出 `embs B×256`，wespeaker 官方 ONNX 导出格式，pyannote 的 `ONNXWeSpeakerPretrainedSpeakerEmbedding` 原生支持本地路径 + CUDA EP）
- `sherpa-onnx-pyannote-segmentation-3-0/model.onnx` + `3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx`（sherpa-onnx 备选档用）

### 4.3 API 与输出

三接口共用 `process_transcription` 响应结构（顶层 Bilibili 字幕头 + `body` 段 + `device_used/audio_duration/processing_time/rtf/timing/status`），改动如下：

- 参数（默认不分离，显式传入才分离）：
  - `/transcribe`：`diarize: bool = Form(False)`（multipart 与 Form 共存）
  - `/transcribe_url`：`BilibiliTranscribeRequest` 加 `diarize: bool = False`
  - `/transcribe_file`：`WebdavTranscribeRequest` 加 `diarize: bool = False`
- `diarize=true` 时响应新增三处：
  - `body[].speaker`：int，聚类簇编号（0 起）；段跨说话人/重叠不过半 → `-1`（字段保持存在，schema 稳定）
  - 顶层 `speakers`：汇总 `[{"id": 0, "duration": 120.5, "segments": 40}, ...]`，下游据此决定是否渲染
  - `timing.diarization`：分离阶段单独计时
- **单人退化**：仅识别出 1 人 → body 不加 speaker 字段、顶层不加 speakers，响应与 `diarize=false` 一致（单人视频不加噪声标注）
- B 站兼容：body 未知字段在 B 站字幕上传时的表现未验证（大概率忽略），阶段 2 加 prefix 输出模式（content 前缀 `【说话人1】`）兜底

### 4.4 缓存

diarize 开关参与缓存 key：追加 `#diar:1` 再 md5（照 `#ctx:` 范式），回归测试进 `tests/`。

### 4.5 下游消费者：mryk24 NoteFlow（首个真实消费方）

mryk24 的 NoteFlow 流水线经两条路径消费转录：B 站无字幕回退 `POST /transcribe_url`（`extractors/bilibili.ts:778-901`，body 原样 JSON 存 rawData）与 WebDAV `POST /transcribe_file`（`extractors/webdav.ts:70-206`）。下游 `clean.ts:19-27` 把 body 格式化为 `[mm:ss] content` 行（现只读 from+content）→ AI 总结 → `### ORIG` 入 Markdown。

结合方式（约半天）：`NoteFlowConfig.funasr.diarize` 开关（默认 false）→ 两个 extractor body 加 `diarize: true` → 两个格式化点读 `item.speaker`（`>= 0` 时行内插 ` [说话人N]`）。增益自动传导：AI 总结获得对话结构、ORIG 可读性提升；TTS 只读 summary 不受影响；单人/`-1` 段自动退化为现状格式；B 站官方字幕路径不走转录、不受影响。

### 4.6 二期扩展：声纹注册

用同一 embedding ONNX：`/voices` 注册（enroll 音频 → embedding 存 JSON）→ 转录时余弦匹配命名。

## 5. 本机实测（2026-09-23，RTX 2080 Ti 22GB / WSL2 / conda funasr）

测试音频：sherpa-onnx 官方 4 人中文音频（56.9s，有标注）+ B 站《世相访谈×陈昊宇》（BV1SZeB62EqK，17.4 分钟，无标注，用 ASR 转录文本对照裁决）。脚本归档：`test/diarization_bench_{sherpa,pyannote,hybrid}.py`、`test/diarization_tune_pyannote.py`。

### 5.1 性能与资源总表

| 实现 | 官方 57s | 世相 17.4min | 峰值资源 | 官方聚类 | 备注 |
|------|---------|-------------|----------|----------|------|
| sherpa-onnx CPU | RTF 0.10 | **RTF 0.14**（146s） | RSS 400MB，**零显存** | 4/4 ✅（thr=0.7） | 4 簇结构自洽 |
| sherpa-onnx CUDA | RTF 0.17 | RTF 0.42（439s） | RSS 1.3GB + <1GB VRAM | 4/4（CPU 模式） | **比 CPU 慢，排除** |
| pyannote 原版 GPU | RTF 0.046 | **RTF 0.018**（19s） | **VRAM 1.6GB** | ❌ 5 人 | 世相欠分裂（§5.4） |
| **混合管线（推荐）** | RTF 0.043（2.4s） | **RTF 0.027**（27.7s） | **VRAM 412MB** | **4/4 ✅** | 与 sherpa CPU 边界一致 |

对总请求耗时的影响（17.4min 音频，ASR 59.4s / RTF 0.057 为基线）：串行时混合管线总耗时 ×1.47、sherpa CPU ×3.5；**并行执行后（§4.1）分别为 ≈×1.0 与 ×2.5**。

### 5.2 分离效果（与转录文本对照）

- 官方 4 人音频（有真值）：**混合管线与 sherpa CPU（thr=0.7）均 4/4 正确**，边界几乎一致；pyannote 原版分 5 人。
- 世相访谈（无真值，文本对照）：sherpa CPU 4 簇与文本自洽（混剪自白簇 / 嘉宾正声簇 / 123-136s 疑似第二人）；pyannote（含混合管线）2 人保守合并。**该样本真值本身模糊**（混剪与主体很可能都是嘉宾本人声音，差异只在混响/配乐）；"欠分裂"对字幕场景比"过碎"安全（错标 vs 少标），且可用 `num_speakers` 先验纠正。

### 5.3 sherpa-onnx GPU 为什么更慢（负结果记录）

官方确有 CUDA 预编译 wheel（`pip install sherpa_onnx-1.13.8+cuda12...whl --no-index -f https://k2-fsa.github.io/sherpa/onnx/cuda.html`，存 HF csukuangfj2/sherpa-onnx-wheels），provider 在 `OfflineSpeakerSegmentationModelConfig` 和 `SpeakerEmbeddingExtractorConfig` 各自设置。但 diarization 是**小模型 × 大量滑窗 chunk**（segmentation 每 1s 一个 10s 窗 + 逐段 embedding），每次 GPU 调用的 launch 开销超过计算收益 → RTF 反升 3~5 倍；且 CUDA/CPU 浮点差异使聚类边界漂移。pyannote 快是因为 segmentation/embedding 均按 batch=32 批量推理。

部署 sherpa CUDA 的坑（若未来需要）：其捆绑 onnxruntime 1.28 要求 **CUDA 12.9 运行库**（系统 12.4/12.6 的 libcudart 缺 `cudaLibraryGetKernel` 版本化符号），需 `pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-curand-cu12` 并把 site-packages/nvidia/*/lib 前置到 LD_LIBRARY_PATH。

### 5.4 pyannote 原版欠分裂的根因与修复

默认 embedding（wespeaker-voxceleb-resnet34-LM，英文 VoxCeleb 训练）对中文内容区分度不足：threshold 0.6/0.5 完全无感（2 人，823s+0.4s），0.4+min_cluster_size=6 直接炸成 15~48 簇——无健康中间态。**换成 wespeaker CN-Celeb 中文 ONNX embedding 后**：官方基准 4/4、世相行为稳定、threshold 0.5~0.7046 全程一致。pyannote 的 `SpeakerDiarization` 构造参数 `embedding` 直接接受本地 ONNX 路径（`ONNXWeSpeakerPretrainedSpeakerEmbedding` 分支）。

## 6. 风险与坑（含已踩）

1. **numpy 锁版**：pip 装 pyannote.audio 会拉 numpy 2.2.6，破坏 funasr-onnx（要求 <=1.26.4）。已踩已回滚；部署时**必须显式锁 `numpy==1.26.4`**（pyannote-metrics 的 numpy>=2.2 声明只影响评估工具，推理不受影响）。
2. **nvidia cu12 组件已升级至 12.9**（pip 元数据与 torch 2.5.1 的固定版本声明冲突仅为警告）：CUDA minor 向后兼容，已验证 torch CUDA matmul、funasr import、ASR 服务转录冒烟（RTF 0.083）均正常。若日后 torch 异常，回滚 `nvidia-cuda-runtime-cu12==12.1.105` 等固定版本即可。
3. 单样本（世相）真值模糊，中文效果结论以官方标注音频 + 文本对照综合判断；实施后应积累更多 B 站样本验收，保留 backend 可切换。
4. HF token：已建 `diarization-test`（read），存 `~/.cache/huggingface/token`（600）；只需 gated 的 `pyannote/segmentation-3.0`，embedding 本地 ONNX 无门。不用时可在 HF 设置页吊销。
5. B 站前端对 body 未知字段的兼容性未验证（prefix 模式兜底）。

## 7. 实施步骤与工作量

| 阶段 | 内容 | 工作量 |
|------|------|--------|
| ~~0. 验证实验~~ | ~~四方案实测~~ | 已完成（§5） |
| 1. 基础版 | diarization/ 模块（pyannote-hybrid 默认 + sherpa-onnx 备选，统一接口）+ config 组 + 三端点 `diarize` 参数 + **先按说话人分块再分句**（词级后端）/ 段级对齐（funasr 退化路径）+ body speaker 字段 + 缓存 key + 单人退化 + pytest | 2~3 天 |
| 2. 打磨 | prefix 输出模式、num_speakers 透传、更多 B 站样本验收 | ~1 天 |
| 3. 二期（可选） | 声纹注册命名 | ~1 天 |

## 8. 参考

- pyannote：`pyannote/speaker-diarization-3.1`（HF，gated）；VRAM 基准 issue #1963；`ONNXWeSpeakerPretrainedSpeakerEmbedding`（pyannote.audio 3.3.2 `pipelines/speaker_verification.py:385`）
- 中文 embedding：`Wespeaker/wespeaker-cnceleb-resnet34-LM`（HF，ONNX 导出）
- sherpa-onnx：CUDA wheel 索引 `k2-fsa.github.io/sherpa/onnx/cuda.html`；分离示例 `python-api-examples/offline-speaker-diarization.py`
- FunASR spk_model：`iic/speech_campplus_sv_zh-cn_16k-common`（ModelScope）
