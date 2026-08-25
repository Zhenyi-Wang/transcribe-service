"""aligner 末尾时间戳超界 clamp 回归测试（2026-08 印地语转录事故次生问题）。

背景：对齐 LLM 的 TS 帧索引是概率预测（argmax(logits[:4000])），尾部注意力
塌缩 + 末块 ASR 在 np.pad 音频上跑导致时间尺度偏移，末词 end 可超出音频实际
时长数秒（94.25s 音频出现 101.04/98.72）。fix_timestamps 只修单调性无上界。

修复目标：align() 输出前把起止时间 clamp 到本块音频真实末尾，保证
end_time <= offset_sec + len(audio)/16000，且不破坏 start <= end。
"""
from qwen_asr_gguf.inference.schema import ForcedAlignItem
from qwen_asr_gguf.inference.utils import clamp_to_audio_end


def _item(text, start, end):
    return ForcedAlignItem(text=text, start_time=start, end_time=end)


def test_normal_items_untouched():
    """未超界的时间戳原样保留（同一对象，不重建）。"""
    items = [_item("मैं", 0.4, 0.9), _item("वहाँ", 0.9, 1.3)]
    out = clamp_to_audio_end(items, 94.25)
    assert out == items
    assert out[0] is items[0]


def test_end_over_bound_clamped():
    """end 超界压到音频末尾，start 不动。"""
    items = [_item("क", 93.8, 98.72)]
    out = clamp_to_audio_end(items, 94.25)
    assert out[0].start_time == 93.8
    assert out[0].end_time == 94.25


def test_both_over_bound_clamped_ordered():
    """start 与 end 均超界：都压到末尾且保持 start <= end。"""
    items = [_item("क", 99.1, 101.04)]
    out = clamp_to_audio_end(items, 94.25)
    assert out[0].start_time == 94.25
    assert out[0].end_time == 94.25
    assert out[0].start_time <= out[0].end_time


def test_boundary_equal_not_rebuilt():
    """恰好等于边界视为未超界。"""
    items = [_item("अंत", 93.0, 94.25)]
    out = clamp_to_audio_end(items, 94.25)
    assert out[0] is items[0]


def test_mixed_batch_order_preserved():
    """混合批次：只修超界项，顺序与文本不变。"""
    items = [
        _item("a", 1.0, 2.0),
        _item("b", 50.0, 101.04),   # end 超界
        _item("c", 99.0, 100.0),    # 双超界
        _item("d", 10.0, 11.0),
    ]
    out = clamp_to_audio_end(items, 94.25)
    assert [it.text for it in out] == ["a", "b", "c", "d"]
    assert out[0] is items[0]
    assert out[1].end_time == 94.25
    assert out[2].start_time == 94.25 and out[2].end_time == 94.25
    assert out[3] is items[3]
