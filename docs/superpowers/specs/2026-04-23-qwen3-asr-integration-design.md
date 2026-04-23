# Qwen3-ASR 集成设计

## 目标

用 Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B 完全替换 FunASR (paraformer-zh)，提升中文识别准确率和时间戳精度。

## 环境

- Conda 环境升级：Python 3.11 → 3.12
- 新增依赖：`qwen-asr`，移除 `funasr`

## 配置变更

config.yaml 模型部分：

```yaml
model:
  asr_model: "Qwen/Qwen3-ASR-1.7B"
  forced_aligner: "Qwen/Qwen3-ForcedAligner-0.6B"
  dtype: "bfloat16"
  max_new_tokens: 4096
```

移除：`name`、`vad_model`、`punc_model`、`disable_update`。

## 改动文件

| 文件 | 改动 |
|------|------|
| server.py | ModelManager 重写：`funasr.AutoModel` → `qwen_asr.Qwen3ASRModel`；移除 FunASR 环境变量 |
| transcribe.py | 适配新 API（`model.transcribe()`）；重写 `generate_subtitle_segments()` 处理新时间戳格式；删除 `detect_language_from_result()`（Qwen3-ASR 直接返回语言） |
| config.py | 删除旧模型属性，新增 `asr_model`、`forced_aligner`、`dtype`、`max_new_tokens` |
| config.yaml.example | 更新模型配置 |
| run.sh | 更新环境依赖 |

## 不变部分

API 接口、缓存系统、下载器、字幕样式、响应 JSON 格式 — 调用方无感知。

## 保留机制

- 懒加载：首次请求时加载模型
- 自动卸载：idle timeout 后释放 GPU 资源
- GPU/CPU 自动切换：OOM 时回退 CPU
- 线程锁：防止并发加载
