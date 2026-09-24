# diarize_only：仅说话人分离模式（2026-09-24）

## 背景

noteflow 的「自动字幕」路径（B站官方字幕，`transcribeType: "auto"`）自带精确的 `from/to` 时间轴，只缺"谁在说"。为它单独跑一次完整 ASR 是浪费——pyannote 分离不依赖 ASR（RTF 0.027），因此提供仅分离模式。同时为**避免对齐逻辑在 Python/TS 写两遍**，noteflow 直接把拉到的字幕 body 发过来，由本服务标注后返回。

## 接口

`POST /transcribe_url`，请求体新增（均默认关闭，互不影响正常转录）：

| 字段 | 类型 | 语义 |
|---|---|---|
| `diarize_only` | bool | 只跑说话人分离，跳过 ASR |
| `body` | list | diarize_only=true 时携带官方字幕 body → 分离+标注拼接模式 |

### 模式一：`diarize_only=true`（不带 body）→ 返回说话人时间轴

```json
{
  "status": "success",
  "video_id": "BV1xx",
  "turns":   [{"speaker": 0, "start": 0.0, "end": 4.2}],
  "speakers": [{"id": 0, "duration": 620.3, "turns": 48}],
  "timing": {"download": 2.1, "diarization": 12.4, "total": 14.8}
}
```

单人也原样返回 turns（是否采用由调用方判断）。

### 模式二：`diarize_only=true` + `body` → 返回标注后的字幕 body

```json
{
  "status": "success",
  "video_id": "BV1xx",
  "body": [{"from": 0.0, "to": 3.0, "content": "甲说", "speaker": 0}],
  "speakers": [{"id": 0, "duration": 5.5, "segments": 2}],
  "timing": {"download": 2.1, "diarization": 12.4, "total": 14.8}
}
```

对齐复用 funasr 退化路径的 `_posthoc_align_speakers`：每段取重叠面积最大的说话人，重叠 ≥ 段时长 50% 才赋标签，否则 -1（跨界段，分组输出中延续当前组）——**与 funasr 路径完全同一份代码、同一语义**。

## 失败语义（区别于转录路径的静默降级——调用方需要显式失败信号决定是否拼接）

| 场景 | 响应 |
|---|---|
| `diarization.enabled=false` | error，message `diarization disabled` |
| 分离超时（`max(60, duration×0.5)s`，同转录路径） | error，message `diarization timeout` |
| 管线异常 | error，message `diarization failed: <原因>` |

拼接模式的成功响应带 `annotated: bool` 与 `reason`：**单人是"成功识别但无需标注"而非失败**——
返回 `{"status": "success", "annotated": false, "reason": "single_speaker", "body": <原样>}`（与转录路径
单人退化为无标注的语义对齐）；多说话人时 `annotated: true, reason: null`。

- 不写转录缓存（没跑 ASR）；音频下载缓存照常生效，重跑只花推理时间
- `page` 照常透传，多 P 视频字幕（对应 cid）与分离时间轴同页同轴

## 调用方（mryk24 noteflow）

`extractors/bilibili.ts` 的 `diarizeSubtitle()`：官方字幕分支在 `funasr.autoSubtitleDiarize` 开启时，把 `fetchSubtitle` 得到的 body POST 给 `funasr.apiUrl`（共用 funasr 限流队列，超时 10 分钟），按 `annotated` 回写标注 body 并记 info；error 才 warn 降级为无标注字幕。下游 clean/summarize 的【说话人N】分组与提示词注入全自动复用。

## 测试

`tests/test_diarize_only.py`（15 个）：公共执行体错误路径（disabled/异常/超时且响应不被卡死）、turns 模式组装与单人原样返回、拼接模式标注/跨界 -1/单人 error、端点分流（带 body 走拼接、不带走 turns、默认 false 走转录、模型默认值）。
