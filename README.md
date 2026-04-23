# Transcribe Service

基于 Qwen3-ASR 的音频转录服务，提供 REST API 将音频文件和 B 站视频转为 Bilibili 字幕格式 JSON。

支持三种 ASR 后端，通过配置文件一键切换：

| 后端 | 说明 | 时间戳精度 | RTF (参考) |
|------|------|-----------|------------|
| GGUF | llama.cpp + ONNX encoder，量化推理 | 字级 | ~0.034 |
| Qwen3-ASR | HuggingFace Transformers + flash_attention_2 | 字级 | ~0.19 |
| FunASR | 阿里 Paraformer + VAD + 标点 | 句级 | ~0.02 |

## 快速开始

```bash
# 激活 conda 环境
conda activate funasr

# 复制并编辑配置
cp config.yaml.example config.yaml

# 启动服务
bash run.sh
```

## API

| Endpoint | 说明 |
|----------|------|
| `POST /transcribe` | 上传音频文件转录 |
| `POST /transcribe_url` | B 站视频转录（BV号 + cookie） |
| `POST /transcribe_file` | 网盘文件转录（路径相对 webdav.base_path） |

```bash
# 上传音频文件
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.mp3" \
  -H "Authorization: Bearer YOUR_TOKEN"

# B 站视频
curl -X POST "http://localhost:8000/transcribe_url" \
  -d '{"bvid": "BV1xx...", "cookie": "your_cookie"}'
```

## 配置

编辑 `config.yaml`，参考 `config.yaml.example`。

### 后端切换

```yaml
backend:
  name: "gguf"  # gguf / qwen3-asr / funasr
```

### GGUF 量化版本

```yaml
backend:
  gguf:
    asr_precision: "q4_k"  # f16 / q8_0 / q4_k / q4_k_m
    model_dir: "~/models/qwen3-asr-gguf"
    use_cuda: true
```

推荐 `q4_k`（RTF 0.034，精度损失可忽略）。详见 `docs/gguf-benchmark/`。

### 旧配置迁移

`model.*` 顶层字段已移除，需改为 `backend` 分组：

```yaml
# 旧（已不支持）
model:
  asr_model: "Qwen/Qwen3-ASR-1.7B"

# 新
backend:
  name: "qwen3-asr"
  qwen3-asr:
    asr_model: "Qwen/Qwen3-ASR-1.7B"
```

## 架构

```
请求 → FastAPI → TranscriptionService → ModelManager → BackendFactory → ASRBackend
```

- `backends/` — 后端抽象层（ASRBackend ABC + BackendFactory + 三后端实现）
- `server.py` — FastAPI 入口，ModelManager 懒加载 + 闲置自动释放
- `transcribe.py` — 转录编排：缓存 → 转录 → 字幕格式化
- `cache_manager.py` — MD5 键 + TTL 过期缓存
- `downloaders/` — B 站音频下载（BV号 / ep号）
- `qwen_asr_gguf/` — GGUF 推理引擎（ONNX encoder + llama.cpp decoder + ForcedAligner）

## 环境

- Python 3.10+，conda 环境 `funasr`
- PyTorch + CUDA（或 CPU）
- GGUF 后端：模型文件需预先下载到 `~/models/qwen3-asr-gguf/`
- Transformers 后端：首次运行自动从 HuggingFace 下载
