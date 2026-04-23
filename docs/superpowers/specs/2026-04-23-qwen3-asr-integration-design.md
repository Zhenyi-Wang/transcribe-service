# Qwen3-ASR 集成设计

## 目标

用 Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B 完全替换 FunASR (paraformer-zh)，提升中文识别准确率和时间戳精度。

## 环境

- Conda 环境升级：Python 3.11 → 3.12
- 新增依赖：`qwen-asr`，移除 `funasr`

## 配置变更

### model 段

```yaml
model:
  asr_model: "Qwen/Qwen3-ASR-1.7B"
  forced_aligner: "Qwen/Qwen3-ForcedAligner-0.6B"
  dtype: "bfloat16"
  max_new_tokens: 4096
```

移除：`name`、`vad_model`、`punc_model`、`disable_update`。

### processing 段

```yaml
processing:
  max_segment_length: 20                # 字幕最大长度（字符数）
  duration_per_segment: 3.0             # 每段字幕持续时间（秒，仅无时间戳回退时使用）
```

移除：
- `batch_size_s` — FunASR 专属参数，Qwen3-ASR 不需要
- `chinese_ratio_threshold` — `detect_language_from_result()` 被删除，不再需要
- `enable_timestamp` — Qwen3-ASR + ForcedAligner 始终返回时间戳，无需开关。不加载 ForcedAligner 时退化为无时间戳模式，由 model 配置中的 `forced_aligner` 是否为空控制

## 改动文件

| 文件 | 改动 |
|------|------|
| server.py | ModelManager 重写；环境变量替换（`MODELSCOPE_CACHE` → `HF_HOME`）；移除 FunASR 相关注释 |
| transcribe.py | 适配新 API 调用；重写 `generate_subtitle_segments()`；删除 `detect_language_from_result()`；保留 `format_duration()`、`split_text_into_segments()`、`get_audio_duration()` |
| config.py | 删除旧属性，新增新属性 |
| config.yaml.example | 更新 model 和 processing 段 |
| run.sh | 更新环境描述 |
| CLAUDE.md | 更新项目描述 |

## server.py 详细改动

### 环境变量

```python
# 删除
os.environ['MODELSCOPE_CACHE'] = str(Path.home() / ".cache/modelscope")

# 替换为
os.environ['HF_HOME'] = str(Path.home() / ".cache/huggingface")
# 保留
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
```

### ModelManager

用 `qwen_asr.Qwen3ASRModel.from_pretrained()` 替换 `funasr.AutoModel()`：

```python
from qwen_asr import Qwen3ASRModel

model_kwargs = {
    "asr_model": config.asr_model,  # "Qwen/Qwen3-ASR-1.7B"
    "dtype": torch.bfloat16,
    "device_map": target_device,     # "cuda:0" 或 "cpu"
    "max_new_tokens": config.max_new_tokens,
}

if config.forced_aligner:
    model_kwargs["forced_aligner"] = config.forced_aligner
    model_kwargs["forced_aligner_kwargs"] = dict(
        dtype=torch.bfloat16,
        device_map=target_device,
    )
```

保留机制：
- 懒加载：双检锁模式不变（`if self.model is None` + `with self.lock`）
- 自动卸载：`unload_model()` + `monitor_loop()` 完全保留
- GPU/CPU 自动切换：`load_model_if_needed()` 中 OOM 回退保留
- 运行时 OOM：`transcribe.py` 中 `if "out of memory" in str(e).lower()` 保留（PyTorch OOM 消息格式一致）
- 线程锁：`self.lock` 完全保留

### Qwen3-ASR API 调用方式

```python
# 加载模型
model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", ...)

# 转录（接受文件路径、URL、base64、(ndarray, sr) 元组）
return_time_stamps = bool(config.forced_aligner)  # 有 aligner 才传 True

results = model.transcribe(
    audio=audio_file_path,       # 文件路径
    language=None,               # None = 自动检测，或 "Chinese"/"English"
    return_time_stamps=return_time_stamps,
)

# 返回值结构
results[0].language   # str: "Chinese", "English", "Japanese" 等
results[0].text       # str: 转录文本
results[0].time_stamps  # list: 时间戳对象列表，每个对象有 .text, .start_time, .end_time
```

音频输入直接接受文件路径，无需额外预处理（qwen-asr 包内部处理采样率转换）。

## transcribe.py 详细改动

### 删除

- `detect_language_from_result()` — Qwen3-ASR 直接返回 `.language` 字段

### 保留

- `get_audio_duration()` — 不依赖 ASR 模型
- `format_duration()` — 纯格式化函数
- `split_text_into_segments()` — 作为无时间戳时的回退路径保留

### 重写 `generate_subtitle_segments()`

用 Qwen3-ASR 的时间戳格式替换 FunASR 的多种格式解析逻辑。新逻辑：

1. 从 `results[0].time_stamps` 获取字/词级别时间戳列表
2. 按标点符号分组为句子级时间戳（取每组首个 start_time 和末尾 end_time）
3. 回退路径：如果 `time_stamps` 为空或不存在，调用 `split_text_into_segments()` 使用均匀分配时间

### 语言检测

```python
# 语言映射：Qwen3-ASR 返回全名 → 项目使用的短代码
LANG_MAP = {
    "Chinese": "zh", "English": "en", "Japanese": "ja",
    "Korean": "ko", "French": "fr", "German": "de",
    "Spanish": "es", "Portuguese": "pt", "Russian": "ru",
    "Arabic": "ar", "Thai": "th", "Vietnamese": "vi",
    "Indonesian": "id", "Italian": "it", "Cantonese": "yue",
    "Turkish": "tr", "Hindi": "hi", "Malay": "ms",
}
detected_lang = LANG_MAP.get(results[0].language, "zh")
```

## 缓存处理

升级后需清除旧缓存，避免返回 FunASR 的旧转录结果。迁移步骤：
1. 删除 `cache/transcripts/` 目录下所有 JSON 文件
2. 或者在缓存 key 中增加模型版本标识（推荐，避免每次升级都清缓存）

## 不变部分

API 接口（`/transcribe`、`/transcribe_url`、`/transcribe_file`）、缓存系统框架、下载器、字幕样式配置、响应 JSON 格式 — 调用方无感知。
