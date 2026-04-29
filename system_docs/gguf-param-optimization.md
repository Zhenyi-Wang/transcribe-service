# GGUF 后端参数封装优化

## 背景

当前 GGUF 后端需要手动计算 `n_batch`/`n_ubatch`/`n_ctx`，容易出错。原因是 llama.cpp 采用静态预分配策略，要求调用者提前确定这些参数，而 Qwen3-ASR 的多平面位置编码和 non-causal attention 进一步增加了复杂度。

## 参数科普

| 参数 | 类比 | 说明 |
|------|------|------|
| `n_ctx` | 最大并发连接数 | KV Cache 容量，决定能处理多长的上下文 |
| `n_batch` | 批处理缓冲区 | 单次 decode 的最大 token 数，Qwen3 需要 ×4（多平面位置编码） |
| `n_ubatch` | 微批次大小 | GPU 单次计算量，non-causal attention 要求 `n_ubatch >= total_tokens` |
| `attention_type` | 注意力类型 | -1=UNSPECIFIED（从模型读取），0=CAUSAL，1=NON_CAUSAL |

## Qwen3-ASR 核心公式

```
音频秒数 × 13 = embedding frames 数量（固定帧率）
total_len = frames + prefix_tokens + suffix_tokens
n_batch ≥ total_len × 4
n_ubatch ≥ total_len
```

## 关键发现：causal_attn 导致 n_batch 被限制

**问题**：Qwen3-ASR GGUF 模型被错误映射到 `qwen3vl` 架构，元数据中 `causal_attn=true`。llama.cpp 在初始化 Context 时：

```cpp
cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;
```

当 `causal_attn=true` 时，`n_batch` 被限制到 `n_ctx=2048`，导致长音频处理崩溃。

**根因**：
1. GGUF 转换脚本没有 Qwen3-ASR 的专用处理
2. 原始 HF 模型配置中没有 `causal` 字段，需要根据用途推断
3. Qwen3-ASR 用于音频转录，需要 bidirectional attention（non-causal）

**修复**：创建 Context 时传入 `attention_type=1`（NON_CAUSAL），强制关闭 causal attention。

## 当前实现

### ASR 引擎 (`asr.py`)

```python
frames_per_chunk = int(config.chunk_size * 13)
total_max_frames = frames_per_chunk * (config.memory_num + 1) + 500
n_batch = max(4096, ((total_max_frames * 4 + 4095) // 4096) * 4096)
n_ubatch = max(512, ((total_max_frames + 511) // 512) * 512)
self.ctx = llama.LlamaContext(self.model, n_ctx=config.n_ctx, n_batch=n_batch, n_ubatch=n_ubatch, 
                                embeddings=False, attention_type=1)
```

### Aligner 对齐器 (`aligner.py`)

```python
n_batch = 32768   # 固定安全上限
n_ubatch = 8192
self.ctx = llama.LlamaContext(self.model, n_ctx=config.n_ctx, n_batch=n_batch, n_ubatch=n_ubatch,
                                embeddings=False, attention_type=1)
```

### Aligner 防御性截断

当 ASR 遇到噪声产生大量垃圾文本时，Aligner 会截断：

```python
MAX_ALIGN_WORDS = 600
if len(words) > MAX_ALIGN_WORDS:
    logger.warning(f"[Aligner] 文本过长 ({len(words)} 词)，截断至 {MAX_ALIGN_WORDS}")
    words = words[:MAX_ALIGN_WORDS]
```

截断后被丢弃的词仍保留在转录文本中，仅缺少字级时间戳。

## 显存成本

| n_ctx | KV Cache 显存估算 (Qwen3-ASR-1.7B) | 说明 |
|-------|--------------------------------------|------|
| 2048 | ~400 MB | 当前默认值 |
| 8192 | ~1.5 GB | 推荐安全上限 |
| 差值 | ~1 GB | RTX 2080 Ti (11 GB) 完全可接受 |

## 后续优化：GGUF 转换脚本修复

需要在 `convert_hf_to_gguf.py` 中为 Qwen3-ASR 添加专用处理：

1. 识别 `model_type = "qwen3_asr"` 的模型
2. 调用 `self.gguf_writer.add_causal_attention(False)`
3. 正确映射架构名称

## 封装原则

| 层级 | 暴露什么 | 隐藏什么 |
|------|---------|---------|
| 用户层 | `chunk_size`, `memory_chunks`, `use_gpu`, `system_prompt` | `n_batch`, `n_ubatch`, `n_ctx`, `attention_type`, 帧率 13 |
| 引擎层 | 计算逻辑 | llama.cpp 细节 |
| llama.cpp 层 | 全部参数 | — |