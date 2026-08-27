# 实施计划：修复闲置超时竞争条件

**Spec**: `docs/superpowers/specs/2026-05-31-idle-timeout-race-condition.md`

## 阶段审查建议处理

| 建议 | 处理 |
|------|------|
| acquire/release 线程安全 | CPython GIL 保证 `int += 1` 原子性，不需要 Lock。用 `threading.Lock` 反而可能引入死锁（如果 acquire 持锁期间 transcribe 阻塞） |
| OOM 时 unload_model 与 in_use 冲突 | OOM 是致命错误，需要强制释放。`unload_model` 在 `in_use > 0` 时也应允许执行（OOM 特殊路径） |
| 连续请求导致模型永不释放 | 这是预期行为：有请求在处理就保持模型加载。idle timeout 仅在真正闲置时生效 |

## Task 1: ModelManager 添加 in_use 计数器

**文件**: `server.py`

### Step 1: 在 `__init__` 中添加 `_in_use` 属性

```python
# server.py:32-35
def __init__(self):
    self._backend = None
    self.lock = threading.Lock()
    self.last_active_time = 0
    self._in_use = 0            # 新增：转录进行中的计数器
```

### Step 2: 添加 `acquire()`、`release()`、`in_use` 属性

在 `unload_model()` 方法后面添加：

```python
# server.py，在 unload_model() 之后
def acquire(self):
    """标记模型正在使用"""
    self._in_use += 1

def release(self):
    """标记模型使用完毕"""
    self._in_use = max(0, self._in_use - 1)

@property
def in_use(self):
    """是否有转录任务正在使用模型"""
    return self._in_use > 0
```

### Step 3: monitor_loop 添加 in_use 检查

```python
# server.py:131-136
def monitor_loop():
    while True:
        time.sleep(config.check_interval)
        if manager._backend is not None and not manager.in_use:   # 新增 and not manager.in_use
            if time.time() - manager.last_active_time > config.idle_timeout:
                manager.unload_model()
```

## Task 2: transcribe.py 添加 acquire/release + asyncio.to_thread

**文件**: `transcribe.py`

### Step 1: 在模型加载后 acquire

```python
# transcribe.py:504-508，现有代码
try:
    # 1. 触发懒加载
    model_load_start = time.time()
    backend = self.model_manager.load_model_if_needed()
    timing["model_load"] = time.time() - model_load_start
    self.model_manager.acquire()    # 新增
```

### Step 2: 用 finally 包裹第二个 try 块，确保 release

将现有的第二个 `try:`（从 `# 2. 获取音频时长` 开始）改为 `try...except...finally` 结构，在 `finally` 中调用 `release()`：

```python
# transcribe.py，在第二个 try 块的 except 末尾（return 之后）添加 finally
        except Exception as e:
            ...（现有错误处理不变）...
            return {
                "status": "error",
                ...
            }

        finally:
            self.model_manager.release()
```

### Step 3: 将 backend.transcribe() 包装进 asyncio.to_thread

```python
# transcribe.py:538-541，现有代码
            # 3. 调用后端转录
            transcription_start_time = time.time()
            result = await asyncio.to_thread(backend.transcribe, audio_file_path)   # 改为异步
            processing_time = time.time() - transcription_start_time
```

需要在文件顶部添加 `import asyncio`。

## Task 3: 验证

### Step 1: 检查语法正确性

```bash
python -c "import py_compile; py_compile.compile('server.py'); py_compile.compile('transcribe.py')"
```

### Step 2: 启动服务，确认模型正常加载

通过 `bash run.sh` 启动，确认日志中出现 "后端加载成功" 和 "预加载完成"。

### Step 3: 发送一个转录请求，确认功能正常

```bash
curl -X POST "http://localhost:31080/transcribe_url" \
  -H "Authorization: Bearer ACG3_3hgbvsf" \
  -H "Content-Type: application/json" \
  -d '{"bvid": "BV1g8Gy6yEq7", "cookie": ""}'
```

确认返回正常结果。
