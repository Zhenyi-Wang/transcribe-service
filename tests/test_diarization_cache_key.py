import hashlib

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
    ctx_hash = hashlib.sha1("偏置".encode()).hexdigest()[:10]
    expect_src = f"/tmp/a.wav#ctx:{ctx_hash}#diar:1"
    assert cm._get_cache_key(file_path="/tmp/a.wav", context="偏置", diarize=True) == \
           hashlib.md5(expect_src.encode()).hexdigest()
