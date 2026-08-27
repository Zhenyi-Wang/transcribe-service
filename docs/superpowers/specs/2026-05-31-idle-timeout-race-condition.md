# 修复闲置超时在转录期间释放模型导致服务永久挂死

## 问题背景

2026-05-27 18:05，一个 472 秒音频的转录任务开始执行（`backend.transcribe()` 在事件循环线程同步阻塞调用）。10 分钟后（18:15），后台 `monitor_loop` 线程因闲置超时将模型卸载，导致 ctypes FFI 调用进入不可恢复状态。事件循环被永久阻塞，端口完全不可达。直到 2026-05-31 用户手动 `^C` 杀进程，服务才恢复。此后被调用方的重试风暴淹没。

## 根因分析：3 个连锁 bug

### Bug 1（触发器）：闲置超时在转录进行中释放模型

**代码位置**：`server.py:131-136`（monitor_loop）、`transcribe.py:545`（last_active_time 更新）

**现状**：`monitor_loop` 每 60 秒检查 `time.time() - manager.last_active_time > idle_timeout(600s)`。`last_active_time` 仅在两处更新：
- `load_model_if_needed()` 调用时（server.py:39）
- 转录**完成后**（transcribe.py:545）

转录进行期间没有任何机制阻止超时释放。当转录耗时超过 600 秒，后台线程调用 `unload_model()`：
1. `gguf_backend.unload()` 调用 `shutdown()`（空操作，仅打印日志）
2. 设置 `self._engine = None`
3. `ModelManager` 设置 `self._backend = None`

此时正在事件循环线程执行的 `QwenASREngine.transcribe()` → `asr()` → `_decode()` 正在做 ctypes FFI 调用（`llama_decode(self.ctx.ptr, ...)`）到 GPU 推理。模型卸载后，该 FFI 调用进入不可恢复状态。

**修复目标**：转录进行期间，`monitor_loop` 不得释放模型。

### Bug 2（放大器）：同步阻塞调用卡死事件循环

**代码位置**：`transcribe.py:468`（async def process_transcription）、`transcribe.py:540`（同步 backend.transcribe() 调用）

**现状**：`process_transcription` 是 `async def`，但内部直接调用同步阻塞的 `backend.transcribe()`。没有 `asyncio.to_thread` 或 `run_in_executor`。

后果：
1. `transcribe()` 阻塞 uvicorn 单 worker 事件循环
2. 事件循环阻塞 → 所有新请求（包括健康检查）无法被处理
3. TCP listen backlog 满后，端口完全不可达
4. 即使没有 Bug 1，一个长音频转录也会阻塞所有其他请求

**修复目标**：转录在独立线程执行，不阻塞事件循环。

### Bug 3（低优先级）：QwenASREngine.shutdown() 是空操作

**代码位置**：`qwen_asr_gguf/inference/asr.py:82-83`

**现状**：`shutdown()` 只打印日志，不释放 `self.ctx` 或 `self.model`。实际资源释放依赖 `__del__` GC。Bug 1 修复后此问题不再触发（不会在转录中 shutdown），暂不修改避免引入新风险。

## 涉及文件

| 文件 | 修改内容 |
|------|----------|
| `server.py` | `ModelManager` 添加 `_in_use` 计数器 + `acquire()`/`release()` 方法；`monitor_loop` 检查 `in_use` |
| `transcribe.py` | 转录前 `acquire()`，`finally` 中 `release()`；`backend.transcribe()` 包装进 `asyncio.to_thread()` |

## 不涉及

- 不修改 `QwenASREngine.shutdown()`（Bug 3，低优先级）
- 不修改 `gguf_backend.py`
- 不修改配置文件
- 不修改 API 接口签名
