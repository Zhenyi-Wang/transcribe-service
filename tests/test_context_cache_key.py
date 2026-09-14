"""缓存 key 纳入 context 的行为测试。

只测 _get_cache_key 的派生规则（不触模型）：
- 无 context：key 与旧版完全一致（md5(源串)），存量缓存兼容
- 有 context：源串追加 #ctx:{sha1[:10]} 后 md5
- 音频文件缓存方法（get_cached_file/save_to_cache）不接受 context，行为不变
"""
import hashlib

from cache_manager import CacheManager


def _legacy_key(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()


def test_key_without_context_matches_legacy():
    cm = CacheManager.__new__(CacheManager)  # 跳过 __init__，不建目录
    assert cm._get_cache_key(file_path="/a/b.mp3") == _legacy_key("/a/b.mp3")
    assert cm._get_cache_key(bvid="BV1", audio_id=9) == _legacy_key("BV1_p1_9")
    assert cm._get_cache_key(url="http://x", bvid="BV1") == _legacy_key("http://x" + "BV1")


def test_key_with_context_differs_and_is_stable():
    cm = CacheManager.__new__(CacheManager)
    k1 = cm._get_cache_key(file_path="/a/b.mp3", context="热词")
    k2 = cm._get_cache_key(file_path="/a/b.mp3", context="热词")
    k3 = cm._get_cache_key(file_path="/a/b.mp3", context="别的")
    assert k1 == k2
    assert k1 != _legacy_key("/a/b.mp3")
    assert k1 != k3
    # 派生规则可复现：md5(源串 + #ctx:sha1[:10])
    expect = hashlib.md5(("/a/b.mp3" + "#ctx:" + hashlib.sha1("热词".encode()).hexdigest()[:10]).encode()).hexdigest()
    assert k1 == expect


def test_key_with_context_and_page():
    """page≠1 分支与 context 叠加：BV1_p2_<id>#ctx:... 路径。"""
    cm = CacheManager.__new__(CacheManager)
    k = cm._get_cache_key(bvid="BV1", audio_id=9, page=2, context="热词")
    expect = hashlib.md5(("BV1_p2_9" + "#ctx:" + hashlib.sha1("热词".encode()).hexdigest()[:10]).encode()).hexdigest()
    assert k == expect
