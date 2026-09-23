# 说话人分离集成 Implementation Plan（transcribe-service）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 转录接口新增可选说话人分离（`diarize=true` 时 body 段带 `speaker`），pyannote-hybrid 管线与 ASR 并行执行，失败/超时降级且不污染缓存。

**Architecture:** 新增 `diarization/` 模块（懒加载 pyannote 管线 + 中文 ONNX embedding）；`TranscriptionService` 用 `asyncio.gather` 并行 ASR 与分离；字幕格式化函数族按"timestamps 项附 speaker 元数据 → 段内边界拆分 / 累加 flush / 句级 post-hoc 对齐"三条路径注入 speaker。

**Tech Stack:** Python 3.10（conda funasr）、pyannote.audio 3.3.2、onnxruntime（GPU 包）、FastAPI、pytest。

**Spec:** `docs/superpowers/specs/2026-09-24-speaker-diarization-design.md`（r3 Approved）

## Global Constraints

- 运行环境：conda `funasr`（`~/miniconda3/envs/funasr/bin/python`）；测试命令 `~/miniconda3/envs/funasr/bin/python -m pytest tests/ -q`（仓库根可直连导入）
- **numpy 必须保持 1.26.4**（pyannote 依赖链若拉高需立即回滚，funasr-onnx 要求 ≤1.26.4）；每个 Task 完成后 `pip list | grep numpy` 抽查
- `diarize=false` 的请求响应与现版本逐字节一致（body 无 speaker 键、顶层无 speakers 键、timing 无 diarization 键、缓存 key 不变）
- 单说话人聚类 → 不加任何 speaker 标注（单人退化）
- 分离任何失败/超时 → `turns=None` 降级，**跳过缓存写入**，不影响转录
- pyannote/torch/qwen_asr_gguf 的 import 一律延迟（函数体内），模块/服务顶层零重依赖
- 缓存 key 拼接顺序固定：`源串 + "#ctx:{sha1[:10]}"（context 非空时） + "#diar:1"（diarize=true 时）`
- 模型路径：`~/models/diarization/wespeaker_cnceleb_resnet34_LM.onnx`（已存在）；segmentation 固定 `pyannote/segmentation-3.0`
- 禁止 git commit —— 完成后统一等待用户指示（出门模式收尾时由主会话按功能点提交）

---

### Task 1: diarization/ 模块与配置组

**Files:**
- Create: `diarization/__init__.py`
- Create: `diarization/manager.py`
- Modify: `config.py`（在 asr_engine property 组之后，约 :108 前插入新组）
- Modify: `config.yaml.example`（`asr_engine` 或 `webdav` 组之后新增组）
- Modify: `config.yaml`（同样新增组，`enabled: false`）
- Test: `tests/test_diarization_manager.py`

**Interfaces:**
- Produces: `diarization.manager.SpeakerTurn`（dataclass：`start: float, end: float, speaker: int`）；`diarization.manager.get_manager() -> DiarizationManager`（单例，配置来自 config.py）；`DiarizationManager.diarize(samples: np.ndarray, sample_rate: int = 16000) -> list[SpeakerTurn]`
- Produces: `config.diarization_enabled / diarization_backend / diarization_cluster_threshold / diarization_min_cluster_size / diarization_num_speakers / diarization_embedding_model / diarization_hf_token`（property）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_diarization_manager.py
import numpy as np
import pytest

from diarization.manager import DiarizationManager, SpeakerTurn


def test_speaker_turn_fields():
    t = SpeakerTurn(start=1.0, end=2.5, speaker=0)
    assert (t.start, t.end, t.speaker) == (1.0, 2.5, 0)


def test_backend_validation_rejects_unknown():
    """backend 非法值 → diarize 抛 ValueError（走降级路径，不静默回退）"""
    m = DiarizationManager(
        backend="sherpa-onnx", cluster_threshold=0.7046, min_cluster_size=12,
        num_speakers=-1, embedding_model="~/nonexistent.onnx", hf_token="",
    )
    with pytest.raises(ValueError):
        m.diarize(np.zeros(16000, dtype=np.float32))


def test_turns_compact_relabel():
    """pyannote 标签重映射为紧凑 0..N-1（按首次出现顺序）——通过注入假管线测重映射逻辑"""
    m = DiarizationManager.__new__(DiarizationManager)  # 跳过 __init__
    raw = [("SPEAKER_05", 0.0, 1.0), ("SPEAKER_02", 1.0, 2.0), ("SPEAKER_05", 2.0, 3.0)]
    turns = m._compact_turns(raw)
    assert [(t.speaker, t.start, t.end) for t in turns] == [
        (0, 0.0, 1.0), (1, 1.0, 2.0), (0, 2.0, 3.0)
    ]
```

- [ ] **Step 2: 运行确认失败**

Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/test_diarization_manager.py -v`
Expected: FAIL（`ModuleNotFoundError: diarization`）

- [ ] **Step 3: 实现 diarization/ 模块**

```python
# diarization/__init__.py
"""说话人分离模块。顶层零重依赖：pyannote/torch 仅在 manager._load() 内延迟导入。"""
from .manager import DiarizationManager, SpeakerTurn, get_manager

__all__ = ["DiarizationManager", "SpeakerTurn", "get_manager"]
```

```python
# diarization/manager.py
"""说话人分离管理器：pyannote-hybrid 管线的懒加载与推理。

延迟导入约定：pyannote/torch 的 import 全部在 _load()/diarize() 函数体内，
本模块顶层（含 __init__.py）不允许出现重依赖 import。
"""
import logging
import os
import threading
from dataclasses import dataclass

from config import config

logger = logging.getLogger("diarization")


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: int


class DiarizationManager:
    """懒加载 + 常驻的分离管线。加载失败抛异常（调用方降级），不静默回退。"""

    def __init__(self, backend: str, cluster_threshold: float, min_cluster_size: int,
                 num_speakers: int, embedding_model: str, hf_token: str):
        self.backend = backend
        self.cluster_threshold = cluster_threshold
        self.min_cluster_size = min_cluster_size
        self.num_speakers = num_speakers
        self.embedding_model = embedding_model
        self.hf_token = hf_token
        self._pipeline = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()  # pyannote 管线非并发安全，推理串行化

    def _load(self):
        with self._load_lock:
            if self._pipeline is not None:
                return
            if self.backend != "pyannote-hybrid":
                raise ValueError(f"未实现的 diarization.backend: {self.backend!r}")
            if self.hf_token:
                # huggingface_hub 凭据链读取；setdefault 不覆盖用户既有环境
                os.environ.setdefault("HF_TOKEN", self.hf_token)
            # 以下均为延迟导入（见模块 docstring）
            import torch
            from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

            pipeline = SpeakerDiarization(
                segmentation="pyannote/segmentation-3.0",
                embedding=os.path.expanduser(self.embedding_model),  # 本地 ONNX（~ 展开）→ ONNXWeSpeakerPretrainedSpeakerEmbedding
                clustering="AgglomerativeClustering",
                segmentation_batch_size=32,
                embedding_batch_size=32,
            )
            pipeline.instantiate({
                "segmentation": {"min_duration_off": 0.0},
                "clustering": {"method": "centroid",
                               "threshold": self.cluster_threshold,
                               "min_cluster_size": self.min_cluster_size},
            })
            pipeline.to(torch.device("cuda"))
            self._pipeline = pipeline
            logger.info("说话人分离管线加载完成（pyannote-hybrid, cuda）")

    @staticmethod
    def _compact_turns(raw) -> list:
        """[(label, start, end), ...] → 紧凑编号的 SpeakerTurn 列表（按 label 首次出现顺序）"""
        label_to_id, turns = {}, []
        for label, start, end in raw:
            if label not in label_to_id:
                label_to_id[label] = len(label_to_id)
            turns.append(SpeakerTurn(start=start, end=end, speaker=label_to_id[label]))
        return turns

    def diarize(self, samples, sample_rate: int = 16000) -> list:
        self._load()
        with self._infer_lock:
            import torch
            waveform = torch.from_numpy(samples).unsqueeze(0)  # (1, T)
            audio = {"waveform": waveform, "sample_rate": sample_rate}
            kwargs = {"num_speakers": self.num_speakers} if self.num_speakers > 0 else {}
            diarization = self._pipeline(audio, **kwargs)
        raw = [(label, turn.start, turn.end) for turn, _, label in diarization.itertracks(yield_label=True)]
        turns = self._compact_turns(raw)
        logger.info(f"说话人分离完成: {len(set(t.speaker for t in turns))} 人 / {len(turns)} turns")
        return turns


_manager = None
_manager_lock = threading.Lock()


def get_manager() -> DiarizationManager:
    """进程级单例；首次调用时按 config 构造（懒加载，模型在首次 diarize 时才加载）"""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = DiarizationManager(
                    backend=config.diarization_backend,
                    cluster_threshold=config.diarization_cluster_threshold,
                    min_cluster_size=config.diarization_min_cluster_size,
                    num_speakers=config.diarization_num_speakers,
                    embedding_model=config.diarization_embedding_model,
                    hf_token=config.diarization_hf_token,
                )
    return _manager
```

`config.py` 在 asr_engine 组（:107）后插入。（若 config.py 顶部无 `import os` 则补；manager 侧 `os` 已在 manager.py 顶部 import。）

```python
    # ========== 说话人分离配置 ==========
    @property
    def diarization_enabled(self) -> bool:
        """是否启用说话人分离"""
        return self.get('diarization.enabled', False)

    @property
    def diarization_backend(self) -> str:
        """分离后端（当前唯一实现 pyannote-hybrid；非法值走禁用降级）"""
        return self.get('diarization.backend', 'pyannote-hybrid')

    @property
    def diarization_cluster_threshold(self) -> float:
        """聚类距离阈值（pyannote 默认 0.7046）"""
        return self.get('diarization.cluster_threshold', 0.7046)

    @property
    def diarization_min_cluster_size(self) -> int:
        """聚类最小簇大小（帧）"""
        return self.get('diarization.min_cluster_size', 12)

    @property
    def diarization_num_speakers(self) -> int:
        """说话人人数先验（-1 自动）"""
        return self.get('diarization.num_speakers', -1)

    @property
    def diarization_embedding_model(self) -> str:
        """声纹 embedding ONNX 路径（wespeaker CN-Celeb 中文）"""
        return self.get('diarization.embedding_model', '~/models/diarization/wespeaker_cnceleb_resnet34_LM.onnx')

    @property
    def diarization_hf_token(self) -> str:
        """HuggingFace token（gated segmentation 模型；空 = 默认凭据链）"""
        return self.get('diarization.hf_token', '')
```

`config.yaml.example` 与 `config.yaml` 在 `asr_engine` 组后新增（example 带注释）：

```yaml
# 说话人分离（可选，diarize=true 请求参数触发）
diarization:
  enabled: false
  backend: pyannote-hybrid      # 唯一实现；其他值走禁用降级
  cluster_threshold: 0.7046     # 聚类距离阈值
  min_cluster_size: 12
  num_speakers: -1              # -1 自动
  embedding_model: "~/models/diarization/wespeaker_cnceleb_resnet34_LM.onnx"
  hf_token: ""                  # 空 = 默认凭据链（~/.cache/huggingface/token）
```

- [ ] **Step 4: 运行测试通过 + numpy 抽查**

Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/test_diarization_manager.py -v && ~/miniconda3/envs/funasr/bin/pip list 2>/dev/null | grep numpy`
Expected: 3 passed；numpy 1.26.4

- [ ] **Step 5: 冒烟验证真实管线可加载（GPU，一次性）**

Run: `~/miniconda3/envs/funasr/bin/python -c "
from diarization.manager import get_manager
import numpy as np
turns = get_manager().diarize(np.zeros(16000 * 3, dtype=np.float32))
print('smoke ok, turns:', len(turns))
"`
Expected: 打印 `smoke ok`（全静音音频 turns 可能为空，不报错即通过；首次运行会下载 segmentation 模型）

---

### Task 2: 字级 speaker 归属（三级规则）

**Files:**
- Modify: `transcribe.py`（`generate_subtitle_segments_from_timestamps` 之前，约 :267 处插入）
- Test: `tests/test_diarization_speaker.py`（新建，后续 Task 复用此文件追加）

**Interfaces:**
- Consumes: `SpeakerTurn`（Task 1）
- Produces: `_assign_speakers_to_timestamps(timestamps: list, turns: list) -> list`（每项为 `{**ts, "speaker": int}` 新 dict）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_diarization_speaker.py
from transcribe import _assign_speakers_to_timestamps
from diarization.manager import SpeakerTurn


TS = lambda text, s, e: {"text": text, "start": s, "end": e}
TURNS = [SpeakerTurn(0.0, 10.0, 0), SpeakerTurn(10.0, 20.0, 1)]


def test_assign_by_overlap_majority():
    """字主体在 turn0、尾巴漂进 turn1 → 重叠面积最大者胜出"""
    ts = [TS("我", 9.0, 10.8)]
    out = _assign_speakers_to_timestamps(ts, TURNS)
    assert out[0]["speaker"] == 0


def test_assign_in_gap_nearest_turn():
    """完全落间隙的字（ASR 边界漂移）→ 归时间最近 turn"""
    ts = [TS("啊", 10.0, 10.05)]
    # 10.0-10.05 与 turn0 重叠 0.05、与 turn1 重叠 0.05 → 平手走重叠；构造纯间隙用例：
    turns = [SpeakerTurn(0.0, 9.0, 0), SpeakerTurn(11.0, 20.0, 1)]
    out = _assign_speakers_to_timestamps([TS("啊", 9.8, 9.9)], turns)
    assert out[0]["speaker"] == 0  # 中点 9.85 距 turn0 尾 0.85 < 距 turn1 头 1.15


def test_assign_out_of_range_nearest():
    ts = [TS("字", 25.0, 25.5)]
    out = _assign_speakers_to_timestamps(ts, TURNS)
    assert out[0]["speaker"] == 1


def test_original_dicts_not_mutated():
    ts = [TS("我", 1.0, 2.0)]
    out = _assign_speakers_to_timestamps(ts, TURNS)
    assert "speaker" not in ts[0] and out[0]["speaker"] == 0
```

- [ ] **Step 2: 运行确认失败**

Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/test_diarization_speaker.py -v`
Expected: FAIL（ImportError: `_assign_speakers_to_timestamps`）

- [ ] **Step 3: 实现**

transcribe.py 在 `generate_subtitle_segments_from_timestamps`（:268）之前插入：

```python
def _assign_speakers_to_timestamps(timestamps: list, turns: list) -> list:
    """给每个时间戳项附加 speaker 键（spec 三级归属规则，不改输入 dict）

    1. 与某 turn 有重叠 → 归重叠面积最大者（字主体归属优先）
    2. 与所有 turn 无重叠（换人间隙/超范围）→ 归最近 turn（项中点到 turn 区间的距离）
    """
    def _nearest_speaker(t: float) -> int:
        best_speaker, best_dist = None, float("inf")
        for turn in turns:
            d = max(turn.start - t, 0.0, t - turn.end)
            if d < best_dist:
                best_dist, best_speaker = d, turn.speaker
        return best_speaker

    out = []
    for ts in timestamps:
        s, e = ts.get("start", 0.0), ts.get("end", 0.0)
        best_speaker, best_overlap = None, 0.0
        for turn in turns:
            overlap = min(e, turn.end) - max(s, turn.start)
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, turn.speaker
        if best_speaker is None:
            best_speaker = _nearest_speaker((s + e) / 2)
        out.append({**ts, "speaker": best_speaker})
    return out
```

- [ ] **Step 4: 运行测试通过**

Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/test_diarization_speaker.py -v`
Expected: 4 passed

---

### Task 3: `_merge_char_timestamps` 段内拆分（CJK 主路径）

**Files:**
- Modify: `transcribe.py:350-453`（`_merge_char_timestamps`）
- Test: `tests/test_diarization_speaker.py`（追加）

**Interfaces:**
- Consumes: 带 `speaker` 键的 timestamps（Task 2）
- Produces: `_merge_char_timestamps(text, timestamps, audio_duration=None)` 签名不变；speaker 键存在时 body 段带 `"speaker": int`；无 speaker 键的输入输出与现基线**逐字节一致**（回归红线）

- [ ] **Step 1: 写失败测试（追加到 tests/test_diarization_speaker.py）**

```python
from transcribe import _merge_char_timestamps


def _char_ts(chars_spk):
    """[(char, speaker), ...] → 字级 timestamps（间隔 0.5s）"""
    return [TS(c, i * 0.5, i * 0.5 + 0.4) for i, (c, spk) in enumerate(chars_spk)
            for c in [c] for spk in [spk]] if False else [
        {"text": c, "start": i * 0.5, "end": i * 0.5 + 0.4, "speaker": spk}
        for i, (c, spk) in enumerate(chars_spk)
    ]


def test_merge_splits_at_speaker_change_without_punct():
    """无标点长句跨说话人 → 段内边界拆分，两段各自单 speaker"""
    text = "今天天气不错我们出去玩吧好吧那就这样决定"  # 20 字，A 前 10 字 + B 后 10 字，无标点
    ts = _char_ts([(c, 0 if i < 10 else 1) for i, c in enumerate(text)])
    body = _merge_char_timestamps(text, ts)
    assert len(body) == 2
    assert body[0]["speaker"] == 0 and body[1]["speaker"] == 1
    assert body[0]["content"] == "今天天气不错我们出去" and body[1]["content"] == "玩吧好吧那就这样决定"
    # 对齐不变式：两段内容拼接（去标点后）== 原 clean 文本
    import re
    assert "".join(re.sub(r"[^\w]", "", b["content"]) for b in body) == text


def test_merge_fallback_segment_inherits_previous_speaker():
    """估算兜底段（时间戳耗尽）继承前一段 speaker——句号切两段，时间戳只覆盖第一段"""
    text = "前半句是甲说的。后半句超出时间戳范围走估算兜底。"  # 句号保证第一段独立成段
    ts = _char_ts([(c, 0) for c in "前半句是甲说的"])  # 时间戳只覆盖第一段（6 字）
    body = _merge_char_timestamps(text, ts)
    assert body[0].get("speaker") == 0
    assert len(body) >= 2
    assert all(seg.get("speaker") == 0 for seg in body[1:])  # 兜底段继承 last_speaker


def test_merge_baseline_unchanged_without_speaker_keys():
    """无 speaker 键的输入 → 输出与 speaker 键不存在时完全一致（回归红线）"""
    text = "你好。今天天气怎么样？挺好的，谢谢。"
    ts_plain = [{"text": c, "start": i * 0.3, "end": i * 0.3 + 0.25}
                for i, c in enumerate(text.replace("。", "").replace("？", "").replace("，", ""))]
    import copy
    ts_copy = copy.deepcopy(ts_plain)
    body = _merge_char_timestamps(text, ts_plain)
    assert all("speaker" not in seg for seg in body)
    # 再跑一次确认输入未被改动
    assert ts_plain == ts_copy
```

- [ ] **Step 2: 运行确认失败**

Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/test_diarization_speaker.py -v -k "merge"`
Expected: 新增 3 个用例 FAIL（speaker 键未出现在 body）

- [ ] **Step 3: 实现（修改 transcribe.py:395-451 的映射循环）**

在 `_merge_char_timestamps` 中，`clean_seg`/`seg_len` 计算与 find 定位之后（:414-429 区域），将"单段输出"改为"按 speaker 序列拆分子段输出"。替换 `if 0 <= match_pos < len(timestamps):` 到 `body.append(...)` 的整块为：

```python
        if 0 <= match_pos < len(timestamps):
            # 有真实时间戳：取首尾字时间
            start_idx = match_pos
            end_idx = min(match_pos + seg_len - 1, len(timestamps) - 1)
            ts_offset = end_idx + 1

            # speaker 模式：项带 speaker 键且段内存在变化 → 段内边界拆分。
            # 拆分不破坏全局对齐：各子段 clean 文本拼接 == 原 clean_seg，
            # 子段时间范围取各自首尾项，ts_offset 推进不变。
            range_speakers = [timestamps[i].get("speaker") for i in range(start_idx, end_idx + 1)]
            if any(sp is not None for sp in range_speakers) and len(set(range_speakers)) > 1:
                # 组边界 = speaker 变化的项索引
                cuts = [start_idx]
                for k in range(1, len(range_speakers)):
                    if range_speakers[k] != range_speakers[k - 1]:
                        cuts.append(start_idx + k)
                cuts.append(end_idx + 1)
                # 把 seg_text 按 clean 字符切分位置切成含标点子串（标点跟随前一个 clean 字符）
                cut_clean_pos = {c - start_idx for c in cuts[1:-1]}
                text_parts = []
                buf, ci = [], 0
                for ch in seg_text:
                    if re.match(r'[\w]', ch, flags=re.UNICODE):
                        if ci in cut_clean_pos and buf:
                            text_parts.append("".join(buf))
                            buf = []
                        ci += 1
                    buf.append(ch)
                text_parts.append("".join(buf))
                text_parts = [p for p in text_parts if p.strip()]

                for gi in range(len(cuts) - 1):
                    g_start, g_end = cuts[gi], cuts[gi + 1] - 1  # 项索引闭区间
                    sub_speaker = range_speakers[g_start - start_idx]
                    body.append({
                        "from": round(timestamps[g_start].get("start", 0), 2),
                        "to": round(timestamps[g_end].get("end", 0), 2),
                        "sid": len(body) + 1,
                        "location": 2,
                        "content": text_parts[gi] if gi < len(text_parts) else "",
                        "music": 0,
                        "speaker": sub_speaker,
                    })
                last_end_time = max(last_end_time, timestamps[end_idx].get("end", 0))
                last_speaker = range_speakers[-1]  # 拆分后更新，供后续估算兜底段继承
                continue

            seg_from = timestamps[start_idx].get("start", 0)
            seg_to = timestamps[end_idx].get("end", 0)
            if seg_to <= seg_from:
                seg_to = seg_from + 0.5
            # speaker 模式下，正常段附加段内（恒一的）speaker
            range_speakers = [timestamps[i].get("speaker") for i in range(start_idx, end_idx + 1)]
            known = [sp for sp in range_speakers if sp is not None]
            body_extra = {"speaker": known[0]} if known else {}
        else:
            # 时间戳耗尽/失配：用上一段结束时间 + 按语速估算的时长兜底。
            # （原注释保留……）
            seg_from = last_end_time
            seg_to = last_end_time + max(seg_len * 0.3, 1.0)
            if audio_duration and audio_duration > 0:
                if seg_from >= audio_duration:
                    seg_from = seg_to = audio_duration
                elif seg_to > audio_duration:
                    seg_to = audio_duration
            body_extra = {"speaker": last_speaker} if last_speaker is not None else {}

        body.append({
            "from": round(seg_from, 2),
            "to": round(seg_to, 2),
            "sid": len(body) + 1,
            "location": 2,
            "content": seg_text,
            "music": 0,
            **body_extra,
        })
        last_end_time = max(last_end_time, seg_to)
        if body_extra.get("speaker") is not None:
            last_speaker = body_extra["speaker"]
```

同时：函数开头（`last_end_time = 0.0` 之后，:403）加 `last_speaker = None`；`ts_offset = end_idx + 1` 移入时间戳分支开头（原 :429 行删除——已在上面新代码中体现）。注意保留原 :430-441 的全部注释。

- [ ] **Step 4: 运行新测试 + 基线回归**

Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/test_diarization_speaker.py tests/test_merge_char_timestamps.py -v`
Expected: 全部 passed（基线测试必须全绿——无 speaker 键路径回归红线）

- [ ] **Step 5: 全量抽查 numpy**

Run: `~/miniconda3/envs/funasr/bin/pip list 2>/dev/null | grep numpy`
Expected: 1.26.4

---

### Task 4: 空格语言路径 + funasr post-hoc 对齐

**Files:**
- Modify: `transcribe.py:309-347`（`_regroup_fragments_by_space`）
- Modify: `transcribe.py:481-561`（`_segment_by_punctuation`）
- Modify: `transcribe.py:456-478`（`_segment_simple`）
- Test: `tests/test_diarization_speaker.py`（追加）

**Interfaces:**
- Consumes: 带 speaker 键的 item（词级或碎片级）
- Produces: `_segment_simple(timestamps, text)` 签名不变但 body 段带 `"speaker"`（post-hoc，可能 -1）——仅当传入 turns 时；`_posthoc_align_speakers(body, turns) -> list`（新函数）

- [ ] **Step 1: 写失败测试（追加）**

```python
from transcribe import (_segment_by_punctuation, _regroup_fragments_by_space,
                        _segment_simple, _posthoc_align_speakers)


def test_regroup_preserves_speaker():
    frags = [
        {"text": "नम", "start": 0.0, "end": 0.3, "speaker": 0},
        {"text": "स्ते ", "start": 0.3, "end": 0.6, "speaker": 0},
        {"text": "हाल ", "start": 5.0, "end": 5.4, "speaker": 1},
    ]
    words = _regroup_fragments_by_space(frags)
    assert [w.get("speaker") for w in words] == [0, 1]


def test_regroup_cross_speaker_word_takes_major_overlap():
    frags = [
        {"text": "ab", "start": 0.0, "end": 0.8, "speaker": 0},
        {"text": "cd ", "start": 0.8, "end": 1.0, "speaker": 1},
    ]
    words = _regroup_fragments_by_space(frags)
    assert words[0]["speaker"] == 0  # 0.0-1.0 词与 turn0 重叠 0.8 > turn1 0.2


def test_segment_by_punct_flushes_at_speaker_change():
    ts = [
        {"text": "hello", "start": 0.0, "end": 0.5, "speaker": 0},
        {"text": "world", "start": 0.5, "end": 1.0, "speaker": 0},
        {"text": "bonjour", "start": 2.0, "end": 2.5, "speaker": 1},
    ]
    body = _segment_by_punctuation(ts, "hello world bonjour")
    assert len(body) == 2
    assert body[0]["speaker"] == 0 and body[1]["speaker"] == 1


def test_posthoc_align_threshold():
    turns = [SpeakerTurn(0.0, 5.0, 0), SpeakerTurn(5.0, 10.0, 1)]
    body = [
        {"from": 1.0, "to": 4.0, "sid": 1, "location": 2, "content": "a", "music": 0},   # 全在 turn0
        {"from": 9.5, "to": 11.0, "sid": 2, "location": 2, "content": "b", "music": 0},  # 重叠 0.5 < 时长1.5 的一半 → -1
        {"from": 8.0, "to": 9.0, "sid": 3, "location": 2, "content": "c", "music": 0},   # 全在 turn1
    ]
    out = _posthoc_align_speakers(body, turns)
    assert out[0]["speaker"] == 0
    assert out[1]["speaker"] == -1  # 大范围落在 turns 覆盖外，最大重叠 0.5s < 1.5s*50%
    assert out[2]["speaker"] == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/test_diarization_speaker.py -v -k "regroup or punct or posthoc"`
Expected: 新增 4 用例 FAIL

- [ ] **Step 3: 实现**

`_regroup_fragments_by_space`：`current` 元组扩为 `(frag_text, start, end, speaker)`，`words.append` 时 speaker 取成员碎片与词区间重叠最大者：

```python
    words = []
    current = []  # [(frag_text, start, end, speaker)]

    def _flush_word():
        if not current:
            return
        word_text = "".join(w[0] for w in current).rstrip()
        if word_text:
            w_start, w_end = current[0][1], current[-1][2]
            spk = _majority_speaker([(s, e, sp) for (_, s, e, sp) in current], w_start, w_end)
            entry = {"text": word_text + " ", "start": w_start, "end": w_end}
            if spk is not None:
                entry["speaker"] = spk
            words.append(entry)
        current.clear()
    # 循环内：current.append((frag, ts.get("start", 0), ts.get("end", 0), ts.get("speaker")))
```

模块级加共用小函数（`_regroup_fragments_by_space` 之前）：

```python
def _majority_speaker(intervals, w_start: float, w_end: float):
    """[(start, end, speaker), ...] 中与 [w_start, w_end] 重叠最大者；全无 speaker 返回 None"""
    best, best_ov = None, 0.0
    for s, e, sp in intervals:
        if sp is None:
            continue
        ov = min(w_end, e) - max(w_start, s)
        if ov > best_ov:
            best_ov, best = ov, sp
    return best
```

`_segment_by_punctuation`：入口加 `has_speaker = any("speaker" in ts for ts in timestamps)`；`current_words` 元组扩为四元组 `(word_text, start, end, speaker)`（speaker 取 `ts.get("speaker")`）——**两处 append 都要改**：:526 的单独标点追加处与 :539 的普通词追加处（若标点处保留三元组，speaker 切换检查访问 `current_words[-1][3]` 会 IndexError）；普通词追加前检查 `if has_speaker and current_words and current_words[-1][3] is not None and ts.get("speaker") != current_words[-1][3]: _flush()`（在"构建词文本"之前）；`_flush()` 输出时 `spk = current_words[0][3]`，`entry = {...现状...}`，`if has_speaker and spk is not None: entry["speaker"] = spk`。

`_segment_simple`：签名不变；新增模块级函数并在 `_segment_simple` 末尾 `return` 前调用（需把 turns 传进来——`_segment_simple(timestamps, text)` 无 turns 参数，由调用方 `generate_subtitle_segments_from_timestamps` 在 speaker_mode 且该分支时对返回 body 做 post-hoc，**不改 _segment_simple 签名**）：

```python
def _posthoc_align_speakers(body: list, turns: list) -> list:
    """句级段的后置说话人对齐：与 turn 最大重叠 ≥ 段时长 50% 才赋标签，否则 -1"""
    for seg in body:
        s, e = seg.get("from", 0.0), seg.get("to", 0.0)
        best, best_ov = None, 0.0
        for turn in turns:
            ov = min(e, turn.end) - max(s, turn.start)
            if ov > best_ov:
                best_ov, best = ov, turn.speaker
        seg["speaker"] = best if (best is not None and best_ov >= (e - s) * 0.5) else -1
    return body
```

（即 Task 5 在分派层调用 `_posthoc_align_speakers`，Task 4 只交付该函数与两处词级路径改动。）

- [ ] **Step 4: 运行测试**

Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/test_diarization_speaker.py -v`
Expected: 全部 passed

---

### Task 5: 分派集成与 speakers 聚合

**Files:**
- Modify: `transcribe.py:268-306`（`generate_subtitle_segments_from_timestamps`）
- Modify: `transcribe.py`（`_aggregate_speakers` 新函数，放 `_assign_speakers_to_timestamps` 旁）
- Test: `tests/test_diarization_speaker.py`（追加）

**Interfaces:**
- Consumes: Task 2-4 全部
- Produces: `generate_subtitle_segments_from_timestamps(text, timestamps, lang="zh", audio_duration=None, turns=None)`（新增 turns 形参）；`_aggregate_speakers(body) -> list[dict]`（`[{"id","duration","segments"}]`，排除 -1/缺失，id 升序，duration round 1 位）

- [ ] **Step 1: 写失败测试（追加）**

```python
import pytest
from transcribe import generate_subtitle_segments_from_timestamps, _aggregate_speakers


def test_dispatch_word_level_two_speakers():
    text = "甲说这里是第一句话乙说这里是第二句话"  # 16 字无标点，8+8
    ts = [{"text": c, "start": i * 0.5, "end": i * 0.5 + 0.4,
           "speaker": 0 if i < 8 else 1} for i, c in enumerate(text)]
    turns = [SpeakerTurn(0.0, 4.0, 0), SpeakerTurn(4.0, 8.0, 1)]
    body = generate_subtitle_segments_from_timestamps(text, ts, "zh", turns=turns)
    assert [seg["speaker"] for seg in body] == [0, 1]


def test_dispatch_single_speaker_degrades_to_plain():
    """单 turn（单人）→ body 不含 speaker 键"""
    text = "只有一个人在说话的一段话没有标点符号"
    ts = [{"text": c, "start": i * 0.3, "end": i * 0.3 + 0.25, "speaker": 0}
          for i, c in enumerate(text)]
    turns = [SpeakerTurn(0.0, len(text) * 0.3 + 1.0, 0)]
    body = generate_subtitle_segments_from_timestamps(text, ts, "zh", turns=turns)
    assert all("speaker" not in seg for seg in body)


def test_dispatch_turns_none_keeps_baseline():
    text = "你好。世界。"
    ts = [{"text": c, "start": i * 0.3, "end": i * 0.3 + 0.25}
          for i, c in enumerate("你好世界")]
    body = generate_subtitle_segments_from_timestamps(text, ts, "zh")
    assert all("speaker" not in seg for seg in body)


def test_dispatch_funasr_posthoc():
    """句级 item（funasr）→ post-hoc 对齐"""
    ts = [{"text": "甲说的一句话", "start": 0.0, "end": 3.0, "speaker": 0},
          {"text": "乙说的一句话", "start": 5.0, "end": 8.0, "speaker": 1}]
    text = "甲说的一句话，乙说的一句话。"
    turns = [SpeakerTurn(0.0, 4.0, 0), SpeakerTurn(4.0, 10.0, 1)]
    body = generate_subtitle_segments_from_timestamps(text, ts, "zh", turns=turns)
    assert [seg["speaker"] for seg in body] == [0, 1]


def test_aggregate_speakers_excludes_negative():
    body = [
        {"from": 0.0, "to": 2.0, "speaker": 0},
        {"from": 2.0, "to": 5.0, "speaker": 1},
        {"from": 5.0, "to": 6.0, "speaker": -1},
        {"from": 6.0, "to": 7.0},  # 无键
    ]
    assert _aggregate_speakers(body) == [
        {"id": 0, "duration": 2.0, "segments": 1},
        {"id": 1, "duration": 3.0, "segments": 1},
    ]
```

- [ ] **Step 2: 运行确认失败**

Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/test_diarization_speaker.py -v -k "dispatch or aggregate"`
Expected: FAIL（turns 参数不存在 / `_aggregate_speakers` 不存在）

- [ ] **Step 3: 实现**

`generate_subtitle_segments_from_timestamps` 改造（:268-306）：

```python
def generate_subtitle_segments_from_timestamps(text: str, timestamps: list, lang: str = "zh",
                                                audio_duration: float = None, turns: list = None) -> list:
    """（原 docstring 保留，末尾追加一行）
    turns: SpeakerTurn 列表（diarize 结果）；None/单说话人时不注入 speaker，行为与历史版本一致。
    """
    import re

    if not timestamps:
        return generate_subtitle_segments(text)

    # speaker 模式判定：turns 有效且聚类 ≥2 人（单人退化），否则保持 timestamps 原样
    speaker_mode = False
    if turns and len({t.speaker for t in turns}) >= 2:
        timestamps = _assign_speakers_to_timestamps(timestamps, turns)
        speaker_mode = True

    _CJK_LANGS = {"zh", "ja", "ko", "yue", ""}
    avg_len = sum(len(ts.get("text", "")) for ts in timestamps[:10]) / min(len(timestamps), 10)
    if avg_len <= 2:
        if lang in _CJK_LANGS:
            body = _merge_char_timestamps(text, timestamps, audio_duration)
            return body if speaker_mode else _strip_speaker(body)
        regrouped = _regroup_fragments_by_space(timestamps)
        if regrouped:
            timestamps = regrouped

    use_spaces = lang not in _CJK_LANGS

    if use_spaces:
        body = _segment_by_punctuation(timestamps, text)
    else:
        # 非空格语言（日/韩）/ 句级（funasr）：简单按 min_len 合并
        body = _segment_simple(timestamps, text)
        if speaker_mode:
            body = _posthoc_align_speakers(body, turns)
    if speaker_mode:
        return body
    return _strip_speaker(body)


def _strip_speaker(body: list) -> list:
    """非 speaker 模式兜底：剥掉任何可能混入的 speaker 键（回归红线保障）"""
    for seg in body:
        seg.pop("speaker", None)
    return body
```

`_aggregate_speakers` 放 `_assign_speakers_to_timestamps` 之后：

```python
def _aggregate_speakers(body: list) -> list:
    """从 body 段聚合说话人汇总；-1/缺失不计入，id 升序"""
    stat = {}
    for seg in body:
        spk = seg.get("speaker")
        if spk is None or spk < 0:
            continue
        d = stat.setdefault(spk, {"id": spk, "duration": 0.0, "segments": 0})
        d["duration"] += seg.get("to", 0.0) - seg.get("from", 0.0)
        d["segments"] += 1
    return [{"id": d["id"], "duration": round(d["duration"], 1), "segments": d["segments"]}
            for d in sorted(stat.values(), key=lambda x: x["id"])]
```

- [ ] **Step 4: 运行测试**

Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/test_diarization_speaker.py tests/test_merge_char_timestamps.py -q`
Expected: 全部 passed

---

### Task 6: 缓存 key `#diar:1`

**Files:**
- Modify: `cache_manager.py:29-49`（`_get_cache_key`）、`:107`（`get_cached_transcript`）、`save_transcript_to_cache`（:141 附近）
- Test: `tests/test_diarization_cache_key.py`（新建）

**Interfaces:**
- Produces: `_get_cache_key(..., diarize: bool = False)`；`get_cached_transcript(..., diarize: bool = False)`；`save_transcript_to_cache(..., diarize: bool = False)`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_diarization_cache_key.py
from cache_manager import CacheManager


def test_diarize_flag_changes_key():
    cm = CacheManager.__new__(CacheManager)  # 跳过 __init__（不碰目录/配置）
    k_off = cm._get_cache_key(file_path="/tmp/a.wav")
    k_on = cm._get_cache_key(file_path="/tmp/a.wav", diarize=True)
    assert k_off != k_on
    # diarize=False 与不传完全一致（旧缓存兼容）
    assert cm._get_cache_key(file_path="/tmp/a.wav", diarize=False) == k_off


def test_context_then_diar_order_fixed():
    cm = CacheManager.__new__(CacheManager)
    import hashlib
    ctx_hash = hashlib.sha1("偏置".encode()).hexdigest()[:10]
    expect_src = f"/tmp/a.wav#ctx:{ctx_hash}#diar:1"
    assert cm._get_cache_key(file_path="/tmp/a.wav", context="偏置", diarize=True) == \
           hashlib.md5(expect_src.encode()).hexdigest()
```

- [ ] **Step 2: 运行确认失败**

Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/test_diarization_cache_key.py -v`
Expected: FAIL（`diarize` 参数不存在）

- [ ] **Step 3: 实现**

`_get_cache_key` 签名加 `diarize: bool = False`，:48 行 `#ctx` 块后追加：

```python
        if diarize:
            content = f"{content}#diar:1"
```

`get_cached_transcript`（:107）与 `save_transcript_to_cache`（:141 附近，签名同理）各加 `diarize: bool = False` 形参，内部三处 `self._get_cache_key(...)` 调用透传 `diarize=diarize`。

- [ ] **Step 4: 运行测试**

Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/test_diarization_cache_key.py tests/test_context_cache_key.py -v`
Expected: 全部 passed（context 基线不受影响）

---

### Task 7: TranscriptionService 编排

**Files:**
- Modify: `transcribe.py:570-716`（`process_transcription` 及 `TranscriptionService`）
- Test: `tests/test_diarization_orchestration.py`（新建，mock 后端与 manager，不依赖真实模型）

**Interfaces:**
- Consumes: Task 1（`get_manager`）、Task 5（`generate_subtitle_segments_from_timestamps(turns=...)`、`_aggregate_speakers`）、Task 6（cache diarize 形参）
- Produces: `TranscriptionService.process_transcription(..., diarize: bool = False)`（新增最后一个关键字参数）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_diarization_orchestration.py
import asyncio
import pytest
from unittest.mock import patch, MagicMock

import transcribe as T
from backends.base import TranscribeResult
from diarization.manager import SpeakerTurn


class FakeBackend:
    name = "fake"
    device = "cpu"

    def transcribe(self, path, lang=None, context=None):
        chars = "甲乙甲乙二人对话内容持续输出一直到十六字整"  # 21 字（断言不依赖字数）
        return TranscribeResult(
            text=chars,
            language="zh",
            timestamps=[{"text": c, "start": i * 0.5, "end": i * 0.5 + 0.4}
                        for i, c in enumerate(chars)],
            performance={"rtf": 0.1},
        )


def _make_service():
    svc = T.TranscriptionService(MagicMock())
    svc.model_manager.load_model_if_needed.return_value = FakeBackend()
    return svc


def _turns():
    return [SpeakerTurn(0.0, 4.0, 0), SpeakerTurn(4.0, 16.0, 1)]


def _enable_diarization(monkeypatch):
    """config.diarization_enabled 是类 property，monkeypatch 用 property 覆盖"""
    monkeypatch.setattr(type(T.config), "diarization_enabled",
                        property(lambda self: True))


@pytest.mark.asyncio
async def test_diarize_true_attaches_speakers(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 8.0)
    _enable_diarization(monkeypatch)
    svc = _make_service()
    with patch.object(T, "_diarize_samples", lambda path: _turns()):
        resp = await svc.process_transcription(str(wav), "a.wav", no_cache=True, diarize=True)
    speakers = {seg["speaker"] for seg in resp["body"]}
    assert speakers == {0, 1}
    assert "speakers" in resp and resp["speakers"][0]["id"] == 0
    assert "diarization" in resp["timing"]


@pytest.mark.asyncio
async def test_diarize_failure_degrades_and_skips_cache(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 8.0)
    _enable_diarization(monkeypatch)
    svc = _make_service()

    def boom(path):
        raise RuntimeError("model gone")
    with patch.object(T, "_diarize_samples", boom), \
         patch.object(T.cache_manager, "save_transcript_to_cache") as save_mock:
        resp = await svc.process_transcription(str(wav), "a.wav", no_cache=False,
                                               file_path_for_cache=str(wav), diarize=True)
    assert all("speaker" not in seg for seg in resp["body"])
    assert "speakers" not in resp
    save_mock.assert_not_called()  # 有 file_path_for_cache 且被跳过 → 证明降级不写缓存


@pytest.mark.asyncio
async def test_diarize_single_speaker_degrades_but_caches(tmp_path, monkeypatch):
    """单人（分离成功）→ 无 speaker 标注但正常写缓存（区别于失败降级）"""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 8.0)
    _enable_diarization(monkeypatch)
    svc = _make_service()
    with patch.object(T, "_diarize_samples", lambda path: [SpeakerTurn(0.0, 16.0, 0)]), \
         patch.object(T.cache_manager, "save_transcript_to_cache") as save_mock:
        resp = await svc.process_transcription(str(wav), "a.wav", no_cache=False,
                                               file_path_for_cache=str(wav), diarize=True)
    assert all("speaker" not in seg for seg in resp["body"])
    save_mock.assert_called_once()


@pytest.mark.asyncio
async def test_diarize_false_baseline(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 8.0)
    svc = _make_service()
    resp = await svc.process_transcription(str(wav), "a.wav", no_cache=True, diarize=False)
    assert all("speaker" not in seg for seg in resp["body"])
    assert "speakers" not in resp
    assert "diarization" not in resp["timing"]


@pytest.mark.asyncio
async def test_diarize_timeout_degrades(tmp_path, monkeypatch):
    """分离超时 → 降级且不写缓存（patch 超时函数为极小值触发真超时）"""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 8.0)
    monkeypatch.setattr(T, "_diarize_timeout", lambda d: 0.05)
    _enable_diarization(monkeypatch)
    svc = _make_service()

    def slow(path):
        import time as _t
        _t.sleep(1.0)  # 超过 0.05s 超时
        return _turns()
    with patch.object(T, "_diarize_samples", slow), \
         patch.object(T.cache_manager, "save_transcript_to_cache") as save_mock:
        resp = await svc.process_transcription(str(wav), "a.wav", no_cache=False,
                                               file_path_for_cache=str(wav), diarize=True)
    assert all("speaker" not in seg for seg in resp["body"])
    assert resp["status"] == "success"
    save_mock.assert_not_called()


@pytest.mark.asyncio
async def test_diarize_true_but_disabled(tmp_path, monkeypatch):
    """diarize=true 但 enabled=false（默认配置）：按未启用处理，timing.diarization==0.0"""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(T, "get_audio_duration", lambda p: 8.0)
    svc = _make_service()  # 不 patch enabled → 默认 False
    with patch.object(T, "_diarize_samples") as should_not_call:
        resp = await svc.process_transcription(str(wav), "a.wav", no_cache=True, diarize=True)
    assert all("speaker" not in seg for seg in resp["body"])
    assert resp["timing"]["diarization"] == 0.0
    should_not_call.assert_not_called()
```

（pytest-asyncio 可用性：实现时先跑一次确认收集正常；若环境缺 `pytest-asyncio` 则 `pip install pytest-asyncio` 并在 `pytest.ini`/`pyproject.toml` 设 `asyncio_mode = "auto"` 或保留显式 `@pytest.mark.asyncio` 标记。）

- [ ] **Step 2: 运行确认失败**

Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/test_diarization_orchestration.py -v`
Expected: FAIL（`process_transcription` 不接受 `diarize` / `_diarize_samples` 不存在）

- [ ] **Step 3: 实现**

3a. 模块级新函数（`TranscriptionService` 类之前）：

```python
def _diarize_samples(audio_file_path: str):
    """分离线程函数体：解码 + 推理（全部重依赖延迟导入；任何异常向上抛由编排层降级）"""
    from qwen_asr_gguf.inference.audio import load_audio
    from diarization.manager import get_manager
    samples = load_audio(audio_file_path)
    return get_manager().diarize(samples)
```

3b. `process_transcription` 签名（:570）追加 `diarize: bool = False`；函数开头 `context = clamp_asr_context(...)` 后加：

```python
        diarize_enabled = bool(diarize and config.diarization_enabled)
        turns = None              # 分离结果（None = 未启用/降级/单人，见各分支）
        diarization_failed = False  # 区分「分离失败不写缓存」与「单人成功正常写缓存」
        if diarize and not config.diarization_enabled:
            logger.warning("收到 diarize=true 请求，但 diarization.enabled=false，按未启用处理")
            timing["diarization"] = 0.0
```

3c. 替换 :643-646（"3. 调用后端转录"块）为（模块级新增 `_diarize_timeout` 小函数供测试 patch）：

```python
def _diarize_timeout(audio_duration: float) -> float:
    """分离超时上限（秒）：下限 60s，随音频时长放宽"""
    return max(60.0, audio_duration * 0.5)
```

```python
            # 3. 调用后端转录 + 可选说话人分离（并行，独立线程）
            transcription_start_time = time.time()
            if diarize_enabled:
                diarize_timeout = _diarize_timeout(audio_duration)
                diarize_start = time.time()

                async def _diarize_job():
                    t0 = time.time()
                    try:
                        turns = await asyncio.to_thread(_diarize_samples, audio_file_path)
                        return turns, time.time() - t0
                    except Exception:
                        logger.warning("说话人分离失败，本次降级为无 speaker 输出（不写缓存）", exc_info=True)
                        return None, time.time() - t0

                asr_task = asyncio.create_task(
                    asyncio.to_thread(backend.transcribe, audio_file_path, None, context))
                diar_task = asyncio.create_task(
                    asyncio.wait_for(_diarize_job(), timeout=diarize_timeout))
                result = await asr_task  # ASR 异常照旧冒泡给外层 except
                processing_time = time.time() - transcription_start_time  # ASR 耗时快照（语义不变，不受分离耗时影响）
                try:
                    turns, diar_elapsed = await diar_task
                except asyncio.TimeoutError:
                    diar_elapsed = time.time() - diarize_start
                    logger.warning(f"说话人分离超时（>{diarize_timeout:.0f}s），降级且不写缓存")
                    turns, diarization_failed = None, True
                except Exception:
                    diar_elapsed = time.time() - diarize_start
                    logger.warning("说话人分离任务异常，降级且不写缓存", exc_info=True)
                    turns, diarization_failed = None, True
                if turns is None:
                    diarization_failed = True  # job 内部异常降级同样标记
                timing["diarization"] = diar_elapsed
            else:
                result = await asyncio.to_thread(backend.transcribe, audio_file_path, None, context)
                processing_time = time.time() - transcription_start_time
            timing["transcription"] = processing_time
```

3d. 单人退化（`timestamps = result.timestamps` 后）——**只影响标注，不改变缓存决策**：

```python
            # 单说话人退化：聚类仅 1 人视为未启用（分离成功，缓存照常写入）
            if turns is not None and len({t.speaker for t in turns}) <= 1:
                logger.info("说话人分离结果仅 1 人，按单人视频处理（不标注）")
                turns = None
```

3e. 字幕生成（:677-680）：

```python
            if timestamps:
                subtitle_body = generate_subtitle_segments_from_timestamps(
                    transcript_text, timestamps, detected_lang,
                    audio_duration=audio_duration, turns=turns)
            else:
                subtitle_body = generate_subtitle_segments(transcript_text)
```

3f. 响应组装（:686-702）`response = {...}` 之后追加：

```python
            if subtitle_body and "speaker" in subtitle_body[0]:
                response["speakers"] = _aggregate_speakers(subtitle_body)
```

3g. 缓存写入（:704-712）：三处 `cache_manager.save_transcript_to_cache(...)` 各加 `diarize=diarize` 形参透传；**仅失败降级跳过**（单人成功正常写）：

```python
            cache_save_start = time.time()
            if diarization_failed:
                logger.info("分离降级：本次结果不写入缓存（下次请求将重试分离）")
            elif file_path_for_cache:
                cache_manager.save_transcript_to_cache(file_path=file_path_for_cache, transcript_data=response, context=context, diarize=diarize)
            elif audio_id and bvid:
                cache_manager.save_transcript_to_cache(None, response, bvid, audio_id, context=context, diarize=diarize)
            elif audio_url or bvid:
                cache_manager.save_transcript_to_cache(audio_url, response, bvid, context=context, diarize=diarize)
            timing["cache_save"] = time.time() - cache_save_start
```

3h. 缓存读取（:586-604）：三处 `get_cached_transcript(...)` 各加 `diarize=diarize` 透传。

3i. `enabled=false` 且 `diarize=true` 的路径：`diarize_enabled=False` → 走 else 分支正常转录；`timing["diarization"]` 补 0.0——在 `diarize and not config.diarization_enabled` 的 warning 之后加 `timing["diarization"] = 0.0`（spec §4.1：请求被判定不执行分离时为 0.0）。注意 `diarize=False` 时**不加**该键（timing dict 初始化不含 diarization 键，保持现状）。

- [ ] **Step 4: 运行测试**

Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/test_diarization_orchestration.py tests/test_diarization_speaker.py -q`
Expected: 全部 passed

---

### Task 8: server.py 三端点参数

**Files:**
- Modify: `server.py:132-149`（两个 pydantic 模型）、`:217-239`（/transcribe）、`/transcribe_url` 与 `/transcribe_file` 的 `process_transcription` 调用点（:278、:362 附近）
- Test: `tests/test_diarization_endpoints.py`（新建，TestClient 级参数验证）

**Interfaces:**
- Consumes: Task 7 的 `diarize` 形参
- Produces: 三端点接受 `diarize`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_diarization_endpoints.py
from fastapi.testclient import TestClient
import server


def test_models_accept_diarize():
    req = server.BilibiliTranscribeRequest(bvid="BV1", cookie="c", diarize=True)
    assert req.diarize is True
    req2 = server.WebdavTranscribeRequest(path="x", diarize=True)
    assert req2.diarize is True
    # 默认 False
    assert server.BilibiliTranscribeRequest(bvid="BV1", cookie="c").diarize is False


def test_transcribe_endpoint_accepts_form_field(tmp_path):
    client = TestClient(server.app)
    wav = tmp_path / "t.wav"
    wav.write_bytes(b"RIFF0000")
    async def fake_process(*args, **kwargs):
        assert kwargs.get("diarize") is True
        return {"status": "success", "body": []}
    import asyncio
    monkey = server.transcription_service.process_transcription
    server.transcription_service.process_transcription = fake_process
    try:
        r = client.post("/transcribe", files={"file": ("t.wav", wav.read_bytes(), "audio/wav")},
                        data={"diarize": "true"}, headers={"Authorization": "Bearer " + server.config.api_token})
        assert r.status_code == 200
    finally:
        server.transcription_service.process_transcription = monkey
```

（若 `config.api_token` 为空则 headers 不需要鉴权头——按 server.py:183 `if config.api_token:` 的实际值决定；实现时读取 `server.config.api_token` 动态构造。）

- [ ] **Step 2: 运行确认失败**

Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/test_diarization_endpoints.py -v`
Expected: FAIL（模型无 diarize 字段）

- [ ] **Step 3: 实现**

- `server.py` 顶部：确认 `from fastapi import FastAPI, File, UploadFile, ...` 中有 `Form`（现有 import 里若没有则补）。
- `BilibiliTranscribeRequest`（:137 后）与 `WebdavTranscribeRequest`（:146 后）各加：

```python
    diarize: bool = False  # 说话人分离（显式传入才启用）
```

- `/transcribe`（:218）签名改 `async def transcribe_audio(file: UploadFile = File(...), diarize: bool = Form(False)):`，:229 调用改 `process_transcription(str(temp_filename), file.filename, diarize=diarize)`。
- `/transcribe_url` 内 `process_transcription(...)` 调用（探索报告 :278）追加 `diarize=request.diarize`。
- `/transcribe_file` 内调用（:362 附近）追加 `diarize=request.diarize`。

- [ ] **Step 4: 运行测试**

Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/test_diarization_endpoints.py -q`
Expected: passed

---

### Task 9: 全量回归 + 实测验收

**Files:**
- 无新改动（验证任务）；发现问题回改对应 Task 文件

- [ ] **Step 1: 全量 pytest**

Run: `~/miniconda3/envs/funasr/bin/python -m pytest tests/ -q`
Expected: 全部 passed，零失败

- [ ] **Step 2: numpy 最终抽查**

Run: `~/miniconda3/envs/funasr/bin/pip list 2>/dev/null | grep numpy && ~/miniconda3/envs/funasr/bin/python -c "import funasr; import pyannote.audio; print('imports ok')"`
Expected: numpy 1.26.4；imports ok

- [ ] **Step 3: 服务重启 + 真实请求验收**

先确认测试样本就位：`ls /tmp/diar_test/0-four-speakers-zh.wav /tmp/diar_test/BV1SZeB62EqK.wav`（缺失则按 `test/diarization_bench_sherpa.py` 文档头与 yt-dlp 重新获取）。

重启 tmux `transcribe` 会话（重跑 `bash start.sh` 或 kill 进程后 `bash start.sh`），确认健康后：

```bash
TOKEN="ACG3_3hgbvsf"  # 本地测试 token（CLAUDE.md 记录；勿用 grep 提取——config.yaml 有多个 token 键）

# 官方 4 人音频：期望 body speaker 聚成 4 组
curl -s -X POST "http://localhost:31080/transcribe" \
  -F "file=@/tmp/diar_test/0-four-speakers-zh.wav" -F "diarize=true" \
  -H "Authorization: Bearer $TOKEN" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('speakers:', d.get('speakers')); print('timing.diarization:', d['timing'].get('diarization'))"

# diarize=false 回归：期望响应无 speaker/speakers/diarization 键
curl -s -X POST "http://localhost:31080/transcribe" \
  -F "file=@/tmp/diar_test/0-four-speakers-zh.wav" \
  -H "Authorization: Bearer $TOKEN" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('no speaker keys:', all('speaker' not in s for s in d['body']), 'no speakers:', 'speakers' not in d, 'no diar timing:', 'diarization' not in d['timing'])"

# 世相样本（17.4min）：期望与 bench 分组一致、total ≈ max(transcription, diarization)
curl -s -X POST "http://localhost:31080/transcribe" \
  -F "file=@/tmp/diar_test/BV1SZeB62EqK.wav" -F "diarize=true" \
  -H "Authorization: Bearer $TOKEN" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['speakers']); print(d['timing'])"
```

Expected: ① 4 人音频 `speakers` 长度 4；② 回归三项全 True；③ 世相 speakers 结构与 `test/diarization_bench_hybrid.py` 一致（2 人）、timing.total 明显小于 transcription+diarization 之和。

- [ ] **Step 4: 更新调查文档状态**

`docs/2026-09-23_speaker-diarization-feasibility.md` §7 表格中阶段 1 标注"已实施（见 plans/2026-09-24-speaker-diarization.md）"。

---

## Plan Self-Review 记录

- Spec 覆盖：§3.1→Task1、§3.2→Task7(3a)、§4.1→Task7、§4.2→Task2/3/4/5、§4.3→Task5/7、§5→Task6/8、§6→Task1、§7→Task7(降级/超时/缓存跳过)、§8→各 Task 测试、§10→Task9 ✓
- 类型一致性：`SpeakerTurn(start,end,speaker)` 全文一致；`_assign_speakers_to_timestamps`/`_posthoc_align_speakers`/`_aggregate_speakers`/`_diarize_samples` 命名前后一致 ✓
- 已知取舍：`_merge_char_timestamps` 内 `ts_offset = end_idx + 1` 移入时间戳分支（原 :429）；`body_extra` 通过 `last_speaker` 传递兜底继承——实现时保持原注释完整
