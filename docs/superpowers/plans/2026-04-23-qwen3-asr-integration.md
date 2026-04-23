# Qwen3-ASR 集成实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B 完全替换 FunASR，保留自动加载/卸载机制。

**Architecture:** 重写 ModelManager 使用 qwen_asr 包加载模型，适配 transcribe.py 的 API 调用和结果格式，更新配置结构。

**Tech Stack:** Python 3.12, qwen-asr, FastAPI, PyTorch

---

## 文件结构

| 文件 | 责任 |
|------|------|
| `config.py` | 配置属性定义，删除旧属性，新增新属性 |
| `config.yaml.example` | 配置示例，更新 model 和 processing 段 |
| `server.py` | ModelManager 重写，环境变量替换 |
| `transcribe.py` | API 调用适配，时间戳处理重写，删除 detect_language_from_result |
| `run.sh` | 启动脚本，更新环境名 |
| `CLAUDE.md` | 项目文档，更新描述 |

---

### Task 1: 更新 config.py 配置属性

**Files:**
- Modify: `config.py:49-92`（非连续，需逐个删除指定属性）

- [ ] **Step 1: 删除旧属性，添加新属性**

删除旧的模型属性：`model_name`、`vad_model`、`punc_model`、`disable_update`。删除旧的处理属性：`batch_size_s`、`enable_timestamp`、`chinese_ratio_threshold`。注意保留 `max_segment_length` 和 `duration_per_segment`。

添加新属性：

```python
@property
def asr_model(self) -> str:
    """ASR 模型名称"""
    return self.get('model.asr_model', 'Qwen/Qwen3-ASR-1.7B')

@property
def forced_aligner(self) -> str:
    """时间戳对齐模型"""
    return self.get('model.forced_aligner', 'Qwen/Qwen3-ForcedAligner-0.6B')

@property
def dtype(self) -> str:
    """模型精度"""
    return self.get('model.dtype', 'bfloat16')

@property
def max_new_tokens(self) -> int:
    """最大生成 token 数"""
    return self.get('model.max_new_tokens', 4096)
```

- [ ] **Step 2: 验证配置文件语法正确**

Run: `python -c "from config import config; print(config.asr_model, config.forced_aligner, config.dtype, config.max_new_tokens)"`
Expected: 输出默认值（因 config.yaml 尚未更新，使用默认值）

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "refactor: 更新配置属性，移除 FunASR 相关属性，添加 Qwen3-ASR 属性"
```

---

### Task 2: 更新 config.yaml.example

**Files:**
- Modify: `config.yaml.example:9-21`

- [ ] **Step 1: 替换 model 和 processing 配置段**

将旧的 model 和 processing 段替换为新结构：

```yaml
# 模型配置
model:
  asr_model: "Qwen/Qwen3-ASR-1.7B"           # ASR 模型
  forced_aligner: "Qwen/Qwen3-ForcedAligner-0.6B"  # 时间戳对齐模型（可选，留空则无时间戳）
  dtype: "bfloat16"                          # 模型精度
  max_new_tokens: 4096                       # 最大生成长度

# 处理配置
processing:
  max_segment_length: 20                    # 字幕最大长度（字符数）
  duration_per_segment: 3.0                 # 每段字幕持续时间（秒，仅无时间戳回退时使用）
```

- [ ] **Step 2: 验证 YAML 格式正确**

Run: `python -c "import yaml; yaml.safe_load(open('config.yaml.example'))"`
Expected: 无报错

- [ ] **Step 3: Commit**

```bash
git add config.yaml.example
git commit -m "docs: 更新配置示例，适配 Qwen3-ASR"
```

---

### Task 3: 重写 server.py 的 ModelManager

**Files:**
- Modify: `server.py:23-101, 368`

- [ ] **Step 1: 替换环境变量**

将第 23-25 行的环境变量设置替换为：

```python
# 设置 HuggingFace 缓存目录和日志
os.environ['HF_HOME'] = str(Path.home() / ".cache/huggingface")
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
```

删除 `MODELSCOPE_CACHE` 相关行。注意：保留第 6 行的 `import torch`，仅替换环境变量相关代码。

- [ ] **Step 2: 删除 `_build_model_kwargs` 方法**

删除第 37-61 行的 `_build_model_kwargs` 方法（FunASR 专属，不再需要）。

- [ ] **Step 3: 重写 `load_model_if_needed` 方法**

将第 63-101 行的 `load_model_if_needed` 方法替换为。注意：`unload_model` 方法（第 103-111 行）保持不变。

```python
def load_model_if_needed(self):
    self.last_active_time = time.time()
    
    if self.model is None:
        with self.lock:
            if self.model is None:
                # 优先尝试 GPU
                target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
                logger.info(f"正在加载模型 (Target: {target_device})...")
                logger.info("如果是第一次运行，正在自动从 HuggingFace 下载模型，请耐心等待...")

                try:
                    from qwen_asr import Qwen3ASRModel

                    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
                    
                    # 构建模型参数
                    forced_aligner_kwargs = None
                    if config.forced_aligner:
                        forced_aligner_kwargs = dict(
                            dtype=dtype,
                            device_map=target_device,
                        )
                        logger.info("已启用时间戳对齐模型")

                    self.model = Qwen3ASRModel.from_pretrained(
                        config.asr_model,  # 第一个参数是模型路径（位置参数）
                        dtype=dtype,
                        device_map=target_device,
                        max_new_tokens=config.max_new_tokens,
                        forced_aligner=config.forced_aligner if config.forced_aligner else None,
                        forced_aligner_kwargs=forced_aligner_kwargs,
                    )
                    self.device = target_device
                    logger.info(f"模型加载成功！运行在: {self.device}")
                    
                except Exception as e:
                    # 如果是显存炸了(OOM)，切回 CPU 重试
                    if "out of memory" in str(e).lower() and target_device.startswith("cuda"):
                        logger.warning("显存不足，正在切换回 CPU 模式...")
                        torch.cuda.empty_cache()

                        from qwen_asr import Qwen3ASRModel
                        
                        dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
                        
                        forced_aligner_kwargs = None
                        if config.forced_aligner:
                            forced_aligner_kwargs = dict(
                                dtype=dtype,
                                device_map="cpu",
                            )

                        self.model = Qwen3ASRModel.from_pretrained(
                            config.asr_model,
                            dtype=dtype,
                            device_map="cpu",
                            max_new_tokens=config.max_new_tokens,
                            forced_aligner=config.forced_aligner if config.forced_aligner else None,
                            forced_aligner_kwargs=forced_aligner_kwargs,
                        )
                        self.device = "cpu"
                        logger.info("CPU 模式加载成功。")
                    else:
                        # 其他错误直接抛出
                        raise e
    return self.model
```

- [ ] **Step 4: 更新 `__main__` 块中的 ModelScope 引用**

将第 368 行的日志消息从 "ModelScope" 改为 "HuggingFace"：

```python
logger.info("注意：第一次运行时仍需要从 HuggingFace 下载模型，请耐心等待...")
```

- [ ] **Step 5: 验证语法正确**

Run: `python -m py_compile server.py`
Expected: 无报错

- [ ] **Step 6: Commit**

```bash
git add server.py
git commit -m "refactor: 重写 ModelManager，使用 qwen_asr 替换 FunASR"
```

---

### Task 4: 重写 transcribe.py 的转录逻辑

**Files:**
- Modify: `transcribe.py:72-114, 171-310, 389-445`

- [ ] **Step 1: 删除 `detect_language_from_result` 函数**

删除第 72-114 行的 `detect_language_from_result` 函数。

- [ ] **Step 2: 添加语言映射常量**

在文件顶部（import 之后）添加：

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
```

- [ ] **Step 3: 重写 `generate_subtitle_segments` 函数**

将第 171-310 行的 `generate_subtitle_segments` 函数替换为：

```python
def generate_subtitle_segments(text, asr_result=None):
    """生成带时间戳的字幕段落

    Args:
        text: 转录文本
        asr_result: Qwen3-ASR 结果对象（包含 time_stamps 属性）
    """
    import re

    body = []

    # 如果有 ASR 结果且包含时间戳
    if asr_result is not None:
        time_stamps = getattr(asr_result, 'time_stamps', None)
        
        if time_stamps and len(time_stamps) > 0:
            # time_stamps 是字/词级别列表，每个元素有 .text, .start_time, .end_time
            # 按标点符号分组为句子级时间戳
            
            current_sentence = []
            sentence_start = None
            
            for ts in time_stamps:
                word_text = ts.text if hasattr(ts, 'text') else str(ts[0])
                word_start = ts.start_time if hasattr(ts, 'start_time') else ts[1]
                word_end = ts.end_time if hasattr(ts, 'end_time') else ts[2]
                
                if sentence_start is None:
                    sentence_start = word_start
                
                current_sentence.append(word_text)
                
                # 检查是否是句子结束标点
                if word_text in ['，', '。', '！', '？', '；', '、'] or word_text.strip() == '':
                    if current_sentence:
                        sentence_text = ''.join(current_sentence).strip()
                        if sentence_text:
                            body.append({
                                "from": round(sentence_start, 2),
                                "to": round(word_end, 2),
                                "sid": len(body) + 1,
                                "location": 2,
                                "content": sentence_text,
                                "music": 0
                            })
                        current_sentence = []
                        sentence_start = None
            
            # 处理最后一个未结束的句子
            if current_sentence:
                sentence_text = ''.join(current_sentence).strip()
                if sentence_text and sentence_start is not None:
                    last_end = time_stamps[-1].end_time if hasattr(time_stamps[-1], 'end_time') else time_stamps[-1][2]
                    body.append({
                        "from": round(sentence_start, 2),
                        "to": round(last_end, 2),
                        "sid": len(body) + 1,
                        "location": 2,
                        "content": sentence_text,
                        "music": 0
                    })
            
            if body:
                return body

    # 回退路径：无时间戳时使用均匀分配
    segments = split_text_into_segments(text)
    for i, segment in enumerate(segments):
        start_time = i * config.duration_per_segment
        end_time = (i + 1) * config.duration_per_segment

        body.append({
            "from": round(start_time, 2),
            "to": round(end_time, 2),
            "sid": i + 1,
            "location": 2,
            "content": segment,
            "music": 0
        })

    return body
```

- [ ] **Step 4: 重写 `process_transcription` 方法中的转录调用**

将第 389-404 行的 FunASR 调用替换为 Qwen3-ASR 调用：

```python
# 根据配置决定是否启用时间戳
return_time_stamps = bool(config.forced_aligner)

res = asr_model.transcribe(
    audio=audio_file_path,
    language=None,  # 自动检测
    return_time_stamps=return_time_stamps,
)

# 调试：打印返回结果结构
if res:
    logger.info(f"Qwen3-ASR 返回: language={res[0].language}, text 长度={len(res[0].text)}")
    if return_time_stamps and hasattr(res[0], 'time_stamps'):
        logger.info(f"时间戳数量: {len(res[0].time_stamps) if res[0].time_stamps else 0}")
```

- [ ] **Step 5: 更新转录文本和语言提取**

将第 412-445 行整体替换为（此范围内包含：转录文本提取、RTF 计算、日志输出、FunASR 时间戳日志 `config.enable_timestamp`、语言检测、字幕生成调用——全部由新代码覆盖）：

```python
# 获取转录文本
transcript_text = res[0].text if res else ""

# 计算RTF比值
rtf_ratio = 0.0
if audio_duration > 0:
    rtf_ratio = processing_time / audio_duration

# 控制台输出关键指标
logger.info(f"\n{'='*50}")
logger.info(f"转录完成!")
logger.info(f"{'='*50}")
logger.info(f"音频时长:     {format_duration(audio_duration)} ({audio_duration:.2f}秒)")
logger.info(f"处理时长:     {format_duration(processing_time)} ({processing_time:.2f}秒)")
logger.info(f"RTF比值:      {rtf_ratio:.3f}")
if rtf_ratio < 1:
    logger.info(f"状态:         实时处理 ✅ (RTF < 1)")
else:
    logger.info(f"状态:         非实时处理 ⏱️ (RTF ≥ 1)")
logger.info(f"{'='*50}\n")

# 检测语言（从 Qwen3-ASR 结果直接获取）
detected_lang = LANG_MAP.get(res[0].language, "zh") if res else "zh"
logger.info(f"检测到语言: {detected_lang}")

# 生成字幕格式（传递 ASR 结果对象以获取时间戳）
subtitle_body = generate_subtitle_segments(transcript_text, res[0] if res else None)
```

- [ ] **Step 6: 验证语法正确**

Run: `python -m py_compile transcribe.py`
Expected: 无报错

- [ ] **Step 7: Commit**

```bash
git add transcribe.py
git commit -m "refactor: 重写转录逻辑，适配 Qwen3-ASR API 和时间戳格式"
```

---

### Task 5: 更新 run.sh 启动脚本

**Files:**
- Modify: `run.sh:9`

- [ ] **Step 1: 更新 conda 环境名注释**

将第 9 行的注释更新为：

```bash
# 激活环境（注意：Qwen3-ASR 需要 Python 3.12+）
conda activate funasr
```

- [ ] **Step 2: Commit**

```bash
git add run.sh
git commit -m "docs: 更新启动脚本注释"
```

---

### Task 6: 更新 CLAUDE.md 项目文档

**Files:**
- Modify: `CLAUDE.md:7, 55, 63, 68-70`

- [ ] **Step 1: 更新项目描述**

将第 7 行替换为：

```markdown
This is a **Qwen3-ASR-based audio transcription service** that provides REST APIs for transcribing audio files and Bilibili videos. It returns subtitle data in Bilibili-compatible JSON format.
```

- [ ] **Step 2: 更新环境描述**

将第 68-70 行替换为：

```markdown
## Environment

- Python 3.12+ with conda environment named `funasr`
- PyTorch with CUDA support (or CPU version)
- Qwen3-ASR models downloaded from HuggingFace on first run
```

- [ ] **Step 3: 更新配置描述**

将第 55 行替换为：

```markdown
- `model.asr_model`: Qwen3-ASR model (default: Qwen/Qwen3-ASR-1.7B)
- `model.forced_aligner`: Timestamp aligner (default: Qwen/Qwen3-ForcedAligner-0.6B)
```

- [ ] **Step 4: 更新类描述**

将第 63 行替换为：

```markdown
- `ModelManager` (server.py): Qwen3-ASR model loading/unloading
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: 更新项目文档，反映 Qwen3-ASR 集成"
```

---

### Task 7: 清理旧缓存

**Files:**
- N/A（手动操作）

- [ ] **Step 1: 删除旧的 FunASR 转录缓存**

Run: `rm -rf cache/transcripts/*.json`
Expected: 清除所有旧缓存文件（如果存在）

- [ ] **Step 2: Commit（记录清理操作）**

```bash
git add -A
git commit -m "chore: 清理旧 FunASR 转录缓存"
```

---

### Task 8: 验证集成

**Files:**
- N/A（测试运行）

- [ ] **Step 1: 安装 qwen-asr 包**

Run: `pip install qwen-asr`
Expected: 成功安装

- [ ] **Step 2: 启动服务测试**

Run: `bash run.sh`
Expected: 服务启动，模型下载并加载成功

- [ ] **Step 3: 用测试音频验证转录功能**

在一个终端启动服务：

```bash
bash run.sh
```

等待模型加载完成后，在另一个终端发送测试请求：

```bash
curl -s -X POST "http://localhost:31080/transcribe" -F "file=@test/test.mp3" | python -m json.tool
```

Expected: 返回 JSON 包含 `"status": "success"`、`"body"` 数组（带 `from`/`to` 时间戳和 `content` 字段）、`"lang"` 字段

验证要点：
- `status` 为 `"success"`
- `body` 不为空数组
- 每个字幕段都有 `from`、`to`、`content` 字段
- `from` < `to`（时间戳递增）

- [ ] **Step 4: 最终 Commit**

```bash
git add -A
git commit -m "feat: 完成 Qwen3-ASR 集成"
```