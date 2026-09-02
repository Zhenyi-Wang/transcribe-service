"""_detect_tail_loop 的单元测试：词组级循环检测（2026-09 祷告会唱诗事故）。

背景：末尾音乐段 ASR 循环幻觉输出歌词"你们的喜乐，你们的荣耀，"共 570 词，
_decode 内的 token 熔断（尾部 15 token <= 3 种）对词组级循环无效。检测判据：
最常见的 5-gram 出现 >= min_repeat 次且尾部出现位置严格等距（循环周期固定）。
阈值校准：真实内容的密集重复（敬拜歌词副歌）实测最多连续 12 次，幻觉循环 60+。
"""
from qwen_asr_gguf.inference.asr import _detect_tail_loop


def test_real_case_loop_detected_with_normal_prefix():
    """真实事故场景：正常开头 + 词组级循环，截断点应精确落在正常内容末尾。"""
    normal = "我们一同借着主祷文来结束今天上午的信息。主耶稣教导我们说："
    loop = "你们的喜乐，你们的荣耀，" * 60
    cut = _detect_tail_loop(normal + loop)
    assert cut == len(normal), f"截断点应等于正常开头长度 {len(normal)}，实际 {cut}"
    assert (normal + loop)[:cut] == normal


def test_entire_loop_text_cuts_to_zero():
    """整段都是循环文本：截断点为 0（全丢），不留循环内容。"""
    cut = _detect_tail_loop("你们的喜乐，你们的荣耀，" * 60)
    assert cut == 0


def test_normal_text_not_detected():
    """正常多样性文本（真实讲道风格）不触发检测。"""
    text = (
        "我们一同借着主祷文来结束今天上午的信息。主耶稣教导我们说，"
        "所以你们祷告要这样说，我们在天上的父，愿人都尊你的名为圣。"
        "愿你的国降临，愿你的旨意行在地上，如同行在天上。"
        "我们日用的饮食，今日赐给我们。免我们的债，如同我们免了人的债。"
        "不叫我们遇见试探，救我们脱离凶恶。因为国度权柄荣耀全是你的。"
        "直到永远，阿门。感谢主的恩典，愿神祝福大家。"
    )
    assert _detect_tail_loop(text) is None


def test_real_song_chorus_repeat_12x_not_detected():
    """防误伤关键场景：真实歌词副歌连续重复（实测 0056 敬拜段 12 次）不触发。"""
    chorus = "这条路不能一个人走"
    other = "无论高山或低谷你要陪伴我，"
    text = other + (chorus + "，") * 12 + other
    assert _detect_tail_loop(text) is None


def test_below_threshold_repeat_not_detected():
    """重复次数低于阈值（19 次 < 20）：不检测。"""
    loop = "你们的喜乐，你们的荣耀，" * 19
    assert _detect_tail_loop(loop) is None


def test_threshold_boundary_20x_detected():
    """重复次数达到阈值（20 次）：检测到。"""
    loop = "你们的喜乐，你们的荣耀，" * 20
    assert _detect_tail_loop(loop) == 0


def test_non_equidistant_repetition_not_detected():
    """高频重复但尾部不等距（每次重复间隔变化）：不截断，避免误伤。"""
    text = ""
    for gap in range(1, 25):
        # 填充段每次内容不同且不自身构成 >=20 次等距循环
        text += f"第{gap}段间隔填充语料各不相同" + "你们的荣耀，"
    assert _detect_tail_loop(text) is None


def test_short_text_not_detected():
    """文本短于 ngram*min_repeat 不可能构成循环，直接返回 None。"""
    assert _detect_tail_loop("你们的喜乐，你们的荣耀，" * 5) is None


def test_min_repeat_zero_disables():
    """min_repeat=0（环境变量 QWEN_ASR_LOOP_MIN_REPEAT=0）关闭检测。"""
    loop = "你们的喜乐，你们的荣耀，" * 60
    assert _detect_tail_loop(loop, min_repeat=0) is None


def test_min_repeat_override():
    """显式传 min_repeat 覆盖默认值。"""
    loop = "你们的喜乐，你们的荣耀，" * 8
    assert _detect_tail_loop(loop, min_repeat=8) == 0
    assert _detect_tail_loop(loop, min_repeat=9) is None
