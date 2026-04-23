# Qwen3-ASR GGUF 加速：模型量化与精度对比

## 动机

原版 Transformers 后端 RTF ~0.17，目标通过 GGUF + llama.cpp 将 RTF 降到 0.05 以下。

## 方案

架构：ONNX Encoder (CUDA) → audio_embedding → llama.cpp Decoder (CUDA) → text → ONNX Aligner (CUDA) → timestamps

模型来源：
- Encoder/Aligner ONNX：从 [Qwen3-ASR-GGUF](https://github.com/HaujetZhao/Qwen3-ASR-GGUF) Releases 下载
- ASR Decoder GGUF：从 HuggingFace 原版 safetensors 导出，再量化
- Aligner Decoder GGUF：仅 release 版 Q4_K 可用（自导出版 output.weight shape 与 llama.cpp 不兼容）

## 导出流程

```
HF safetensors → 提取 thinker.model.* 权重 → 保存 HF 格式 → convert_hf_to_gguf → F16 GGUF → llama-quantize → Q8_0/Q4_K/Q4_K_M
```

关键：HF 模型文件需复制到简单路径（如 `/tmp/qwen3-asr-1.7b/`），直接从 `~/.cache/huggingface/hub/.../snapshots/` 加载会触发 `HFValidationError`。

## 对比结果

测试音频：`test/test.mp3`（112.85s 中文祷告），每精度跑 5 轮取平均。

| ASR 1.7B 精度 | 平均 RTF | RTF 波动 | 平均文本长度 | 质量 |
|---|---|---|---|---|
| F16 | 0.0420 | ±4.5% | 471.8 | 基准 |
| Q8_0 | 0.0362 | ±1.4% | 473.0 | 偶现"死风" |
| Q4_K | **0.0340** | ±1.0% | 471.4 | 最稳定 |
| Q4_K_M | 0.0339 | ±1.5% | 470.8 | 与 Q4_K 相当 |

**选型：Q4_K** — 速度最快、RTF 最稳定、质量与 F16 无明显差异。

## 模型文件清单

```
~/models/qwen3-asr-gguf/
├── qwen3_asr_encoder_frontend.int4.onnx    # 20M  (共享)
├── qwen3_asr_encoder_backend.int4.onnx     # 158M (共享)
├── qwen3_asr_llm.f16.gguf                 # 3.8G (基准参考)
├── qwen3_asr_llm.q8_0.gguf                # 2.1G
├── qwen3_asr_llm.q4_k.gguf                # 1.2G ★ 推荐
├── qwen3_asr_llm.q4_k_m.gguf              # 1.2G
├── qwen3_asr_0.6b_llm.f16.gguf            # 1.5G (需配套 0.6B encoder，暂不可用)
├── qwen3_asr_0.6b_llm.q8_0.gguf           # 768M
├── qwen3_asr_0.6b_llm.q4_k.gguf           # 462M
├── qwen3_asr_0.6b_llm.q4_k_m.gguf         # 462M
├── qwen3_aligner_encoder_frontend.int4.onnx
├── qwen3_aligner_encoder_backend.int4.onnx
└── qwen3_aligner_llm.q4_k.gguf            # 462M (仅此精度可用)
```

## 踩坑记录

1. **HFValidationError**：`from_pretrained` 校验路径格式，snapshot 长路径被拒绝。复制到 `/tmp/` 简化路径解决。
2. **Aligner GGUF 导出失败**：Aligner 的 `lm_head` 是 5000 类分类头（不是 vocab embedding），导出后 `output.weight` shape `[1024, 5000]` 与 `qwen3vl` 架构期望的 `[1024, 152064]` 冲突。Release 版 Q4_K 做了特殊处理（替换为 token_embd），自导出无法复现。
3. **ASR 0.6B 不可用**：ONNX encoder 输出维度 2048，与 0.6B decoder 的 embedding 维度 1024 不匹配。需要单独导出 0.6B encoder。
4. **soundfile 读 MP3 挂起**：`soundfile.read()` 对 MP3 支持不稳定，改为路由到 ffmpeg。
5. **Segfault 排查**：GPU 残留进程导致，`kill` 后恢复正常。
