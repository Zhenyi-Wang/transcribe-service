# diarization ONNX Conv Fallback 警告刷屏与提速修复

**日期**：2026-10-09　**类型**：根因分析 + 修复　**状态**：已实施

## 现象

tmux `transcribe` 面板每次跑说话人分离刷出 70+ 条警告：

```
[W:onnxruntime:Default, conv.cc:425 UpdateState] OP Conv(Conv_xx) running in Fallback mode. May be extremely slow.
```

项目 `logs/*.log` 里一条都没有——警告由 ORT C++ 层直写 stderr，不走 Python logging。

## 根因

- 警告全部来自 diarization 的 wespeaker 声纹模型 `wespeaker_cnceleb_resnet34_LM.onnx`（ResNet34，~40 个 Conv 算子）
- `diarization/manager.py` 把 pyannote 管线 `.to("cuda")` → pyannote 用 CUDA EP 跑该 ONNX
- **pyannote.audio 3.3.2 在 `speaker_verification.py` 硬编码 `cudnn_conv_algo_search: "DEFAULT"`**
- 本机组合（onnxruntime-gpu 1.23.2 + 系统 cuDNN 9.8.0 + RTX 2080 Ti Turing）下，DEFAULT 搜索模式给卷积找不到可用 cuDNN 算法 → ORT 全部 Conv 回退内置慢速核，每个 Conv × 每个新输入形状（每 batch 形状不同）各刷一条
- 三组件版本自 2026-09-23 部署 diarization 起从未变过——不是回归，是一直如此，当天才注意到

## 实测数据（/tmp 隔离复现，同模型同输入）

| CUDA EP 配置 | 每 batch 耗时 | Fallback 警告 |
|---|---|---|
| `cudnn_conv_algo_search: DEFAULT`（pyannote 现状） | 433 / 210 / 128 ms | 72 条 |
| 不带该选项（ORT 默认） | 42 / 43 / 41 ms | 0 条 |

embedding 推理慢 3~10 倍。对 50 分钟视频（BV1xipb6DEcv）：diarize 步骤 9.4s，修复后预期 ~4s；整体转录请求无感（ASR 是大头）。

另：ORT 实际加载的是**系统** cuDNN 9.8.0（`/usr/lib/x86_64-linux-gnu/`，2025-02 装），不是 conda 环境 pip 版 nvidia-cudnn 9.1.0.70——run.sh 未设 LD_LIBRARY_PATH 时系统库优先。

## 修复

`diarization/manager.py` 新增 `_strip_cudnn_conv_algo_search_default()` contextmanager：在 `_load()` 构建管线期间（SpeakerDiarization 构造 + instantiate + `.to()`，ONNX 会话即在此窗口创建，已实测验证）monkey-patch `ort.InferenceSession`，把 CUDA EP 选项里的 `cudnn_conv_algo_search: "DEFAULT"` 剥掉，作用域结束恢复。

- 仅匹配「CUDAExecutionProvider 且该选项值为 DEFAULT」的会话，其余原样透传——GGUF encoder（裸 `CUDAExecutionProvider`，本就无警告）不受影响
- pyannote 若未来移除该硬编码，patch 自动变 no-op
- pytest 129 passed；本地全流程复验 0 警告、推理正常

## 教训

- 「一堆警告」先看它走哪条通道：stderr 直写 vs logging，决定去哪找历史
- pyannote 的 ONNX 后端不暴露 provider options，只能在其上游包 `InferenceSession`——patch 要精准匹配目标配置并留恢复路径
