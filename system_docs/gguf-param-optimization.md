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

## 关键发现：causal attention 是必需的

**事实**：Qwen3-ASR GGUF 模型被映射到 `qwen3vl` 架构，元数据中 `causal_attn=true`。**这是正确的**——实测验证 causal attention 才能正常转录，non-causal attention 会导致输出完全乱码或空。

llama.cpp 在 causal 模式下的限制：

```cpp
cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;
```

当 `causal_attn=true` 时，`n_batch` 被限制到 `n_ctx`。因此 **必须自动调大 `n_ctx`** 以容纳 mrope 四平面位置编码（`total_len × 4`）。

**错误的历史结论**：之前认为"Qwen3-ASR 需要 non-causal attention"是基于错误的推理。实际上"长音频崩溃"是 `n_ctx` 太小导致的，而非 causal attention 本身的问题。向 llama.cpp 提交的 PR #22511 也被维护者正确拒绝。

**修复**：使用 `attention_type=0`（CAUSAL），并自动将 `n_ctx` 调大到 `n_batch + 4096`。

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

**已提交 PR**: https://github.com/ggml-org/llama.cpp/pull/22511（已关闭）

**关联 Issue**:
- https://github.com/ggml-org/llama.cpp/issues/22357 — 转录重复/乱码
- https://github.com/ggml-org/llama.cpp/issues/21847 — 长音频空转录

**根因**: Qwen3-ASR 的 HuggingFace config 不包含 `is_causal=False` 字段，导致转换后的 GGUF 模型继承 `causal_attn=true`，llama.cpp 据此限制 `n_batch = min(n_ctx, n_batch)`，长音频处理崩溃。

**临时方案**: Python 代码中传入 `attention_type=1` 强制使用 non-causal attention（已实现）。

**PR 结果**: 被维护者 ngxson 拒绝，理由：
1. PR #20746 是给 embedding 模型用的，不是 ASR
2. Qwen3-ASR config.json 中没有信息表明需要 non-causal attention
3. 音频输入不应该用 non-causal attention（除了 encoder-decoder 架构如 Whisper）
4. 没有测试结果和官方参考代码

**后续验证**: 维护者的判断是正确的。实测证明 causal attention 才是正确的，non-causal attention 导致完全乱码。之前的 workaround（`attention_type=1`）解决了"长音频崩溃"但引入了"所有音频乱码"的严重问题。正确的做法是保持 causal attention 并增大 `n_ctx`。

## 封装原则

| 层级 | 暴露什么 | 隐藏什么 |
|------|---------|---------|
| 用户层 | `chunk_size`, `memory_chunks`, `use_gpu`, `system_prompt` | `n_batch`, `n_ubatch`, `n_ctx`, `attention_type`, 帧率 13 |
| 引擎层 | 计算逻辑 | llama.cpp 细节 |
| llama.cpp 层 | 全部参数 | — |

## C 库（llama.cpp 预编译二进制）

GGUF 后端依赖 llama.cpp 的 C 动态库，位于 `lib/` 目录（约 510M），已通过 `.gitignore` 排除，**不在 git 中**。

**重要**：该目录丢失后服务无法启动，且无法从 git 恢复。需要重新获取预编译库。

**相关文件**:
- `lib/libllama.so` — 主推理库
- `lib/libggml.so` / `lib/libggml-base.so` — 底层张量运算
- `lib/libggml-cpu.so` / `lib/libggml-cuda.so` — CPU/CUDA 后端
- `lib/libllama-common.so` — 公共工具

**重新获取方式**: 从 llama.cpp 官方 release 下载或本地编译（需要 CUDA 工具链）。

**上游仓库**: https://github.com/ggml-org/llama.cpp
**本项目 fork**: https://github.com/Zhenyi-Wang/llama.cpp（用于提交 PR）