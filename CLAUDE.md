# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Qwen3-ASR 音频转录服务，提供 REST API 将音频文件和B站视频转为 Bilibili 字幕格式 JSON。支持三种 ASR 后端，通过配置文件切换。

## Commands

```bash
conda activate funasr

# 启动服务（自动重载）
bash run.sh

# 启动服务（无自动重载）
bash run.sh --no-reload

# 测试 API
curl -X POST "http://localhost:31080/transcribe" -F "file=@test/test.mp3" -H "Authorization: Bearer ACG3_3hgbvsf"
```

无 pytest/unittest 框架，测试为 `test/` 下的独立脚本。无 linting 配置。

## Architecture

请求流程：`HTTP → server.py (FastAPI) → TranscriptionService → ModelManager → BackendFactory → ASRBackend`

### 后端抽象层 (`backends/`)

工厂模式 + 注册表。`config.yaml` 的 `backend.name` 决定使用哪个后端：

| 后端 | name | 说明 | 时间戳精度 |
|------|------|------|-----------|
| GGUF | `gguf` | llama.cpp + ONNX encoder，量化推理 | 字级 (ForcedAligner) |
| Qwen3-ASR | `qwen3-asr` | HuggingFace transformers + flash_attention_2 | 字级 (ForcedAligner) |
| FunASR | `funasr` | 阿里 FunASR Paraformer + VAD + 标点 | 句级 |

`BackendFactory.create(config)` 返回 `ASRBackend` 实例，统一接口：`load()` / `unload()` / `transcribe(file, lang?) → TranscribeResult`。

GGUF 后端在 `backends/gguf_backend.py` 之上还有 `qwen_asr_gguf/` 包（ONNX encoder + llama.cpp decoder + ForcedAligner 实现）。

### 关键类

- `ModelManager` (server.py) — 后端生命周期管理，懒加载 + 闲置自动释放
- `TranscriptionService` (transcribe.py) — 转录编排：缓存检查 → 转录 → 字幕格式化 → 缓存保存
- `CacheManager` (cache_manager.py) — MD5 键 + TTL 过期，启动时清理
- `BilibiliDownloader` (downloaders/) — 按 BV/ep 分发到 VideoDownloader 或 EpisodeDownloader

### 配置

`config.py` 的 `Config` 类加载 `config.yaml`，支持点号访问 `config.get('backend.gguf.asr_precision')`。`config.yaml.example` 是模板（注意：它的结构可能落后于 `config.yaml`，新增字段以实际运行配置为准）。

GGUF 量化版本通过 `backend.gguf.asr_precision` 控制：`f16` / `q8_0` / `q4_k` / `q4_k_m`。模型文件位于 `~/models/qwen3-asr-gguf/`。

### API Endpoints

| Endpoint | 说明 |
|----------|------|
| `POST /transcribe` | 上传音频文件转录 |
| `POST /transcribe_url` | B站视频转录（BV号 + cookie） |
| `POST /transcribe_file` | 网盘文件转录（路径相对 webdav.base_path） |

## Environment

- conda 环境 `funasr`，Python 3.10（`~/miniconda3/envs/funasr/bin/python`）
- GPU: RTX 2080 Ti (Turing, compute capability 7.5, 无 BF16 支持)
- Qwen3-ASR 的 `temperature` 警告来自 `qwen_asr` 库内部，不影响功能
