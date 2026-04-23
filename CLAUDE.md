# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Qwen3-ASR-based audio transcription service** that provides REST APIs for transcribing audio files and Bilibili videos. It returns subtitle data in Bilibili-compatible JSON format.

## Commands

```bash
# Activate conda environment
conda activate funasr

# Run the service (default port 8000, with auto-reload)
bash run.sh

# Or run directly
python server.py

# Test the API
curl -X POST "http://localhost:8000/transcribe" -F "file=@test/test.mp3"

# With authentication token
curl -X POST "http://localhost:8000/transcribe" -F "file=@test/test.mp3" -H "Authorization: Bearer YOUR_TOKEN"
```

## Architecture

```
transcribe-service/
├── server.py           # FastAPI entry point, defines 3 endpoints
├── transcribe.py       # Core transcription logic (TranscriptionService class)
├── config.py           # Configuration management
├── cache_manager.py    # Transcription caching
├── downloaders/        # Audio downloader modules
│   ├── bilibili_video.py   # Download by bvid
│   └── bilibili_episode.py # Download by episode ID
├── config.yaml         # Runtime configuration (not in git)
└── run.sh             # Startup script
```

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /transcribe` | Upload audio file for transcription |
| `POST /transcribe_url` | Transcribe Bilibili video by BV号+CID |
| `POST /transcribe_file` | Transcribe local file (path relative to webdav.base_path) |

### Configuration (config.yaml)

- `server.idle_timeout`: Model unload timeout (seconds)
- `server.check_interval`: Model health check interval
- `model.asr_model`: Qwen3-ASR model (default: Qwen/Qwen3-ASR-1.7B)
- `model.forced_aligner`: Timestamp aligner (default: Qwen/Qwen3-ForcedAligner-0.6B)
- `api.host`, `api.port`: Server listen address
- `api.token`: Bearer token for authentication (empty = no auth)
- `webdav.base_path`: Base path for /transcribe_file endpoint (default: /mnt/webdav)

### Key Classes

- `TranscriptionService` (transcribe.py): Core transcription orchestration
- `ModelManager` (server.py): Qwen3-ASR model loading/unloading
- `BilibiliDownloader`, `BilibiliVideoDownloader`, `BilibiliEpisodeDownloader` (downloaders/): Audio download from Bilibili

## Environment

- Python 3.12+ with conda environment named `funasr`
- PyTorch with CUDA support (or CPU version)
- Qwen3-ASR models downloaded from HuggingFace on first run
