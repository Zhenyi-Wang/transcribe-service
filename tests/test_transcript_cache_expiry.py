"""转录结果缓存 TTL 与原子写入的行为测试。

只测 CacheManager 的转录缓存读写（不触模型、不碰生产缓存目录，用 tmp_path 隔离）：
- save_transcript_to_cache 新增末尾可选 ttl_seconds：条目在 cache_days 之外额外受显式过期约束
- 读取同时遵守 cache_days（mtime）与显式过期时间，过期即 miss（文件删除）
- 返回给调用方的 dict 不含 cached_at / 私有过期标记
- 私有 ASR 基线字段 _asr 默认随返回剥除；get_cached_transcript(include_asr=True) 显式保留（供冷却后仅重试分离）
- 写入复制入参 dict（不污染调用方），临时文件 + os.replace 原子落盘，失败只清理自己创建的临时文件
- ttl_seconds 为 0/负数/NaN/Inf/非数值时明确抛 ValueError，禁止静默降级为永久缓存
- 默认（ttl_seconds=None）行为兼容旧缓存：无标记的旧条目照常按 cache_days 命中
"""
import copy
import json
import os
import time

import pytest

from cache_manager import CacheManager

# 私有过期标记的键名（与实现约定一致，仅供测试直接构造磁盘上的到期条目）
_EXPIRES_KEY = "_cache_expires_at"


def _make_cm(tmp_path, cache_days=7):
    """构造隔离的 CacheManager：跳过 __init__（不读 config、不碰生产缓存目录）"""
    cm = CacheManager.__new__(CacheManager)
    cm.cache_dir = tmp_path
    cm.cache_enabled = True
    cm.cache_days = cache_days
    cm.transcript_dir = tmp_path / "transcripts"
    cm.transcript_dir.mkdir(exist_ok=True)
    return cm


def _key_path(cm, **kw):
    return cm.transcript_dir / f"{cm._get_cache_key(**kw)}.json"


def _save_kwargs():
    return dict(file_path="/fake/audio/a.mp3")


def _sample_data():
    return {
        "subtitle": [{"t1": 0.0, "t2": 1.0, "line": "你好"}],
        "audio_duration": 12.5,
        "timing": {"cache_check": 0.0},
    }


def test_roundtrip_default_and_return_dict_clean(tmp_path):
    """默认保存→命中，返回 dict 不含 cached_at/私有标记；磁盘仍写 cached_at（旧格式兼容）"""
    cm = _make_cm(tmp_path)
    data = _sample_data()

    cm.save_transcript_to_cache(transcript_data=data, **_save_kwargs())

    payload = cm.get_cached_transcript(**_save_kwargs())
    assert payload == data  # 内容一致，且不含 cached_at / _cache_expires_at
    assert "cached_at" not in payload
    assert _EXPIRES_KEY not in payload

    raw = json.loads(_key_path(cm, **_save_kwargs()).read_text(encoding="utf-8"))
    assert "cached_at" in raw          # 磁盘格式保持带时间戳
    assert _EXPIRES_KEY not in raw     # 默认不写私有标记


def test_ttl_entry_roundtrip_and_disk_marker(tmp_path):
    """带 TTL 保存→命中；磁盘写入 _cache_expires_at = cached_at + ttl"""
    cm = _make_cm(tmp_path)
    cm.save_transcript_to_cache(transcript_data=_sample_data(), ttl_seconds=3600, **_save_kwargs())

    payload = cm.get_cached_transcript(**_save_kwargs())
    assert payload == _sample_data()
    assert "cached_at" not in payload
    assert _EXPIRES_KEY not in payload

    raw = json.loads(_key_path(cm, **_save_kwargs()).read_text(encoding="utf-8"))
    assert raw[_EXPIRES_KEY] == raw["cached_at"] + 3600


def test_ttl_expiry_is_miss(tmp_path):
    """显式过期时间已过 → miss 且文件删除；重新写入后（mtime 仍新鲜）恢复命中"""
    cm = _make_cm(tmp_path)
    cm.save_transcript_to_cache(transcript_data=_sample_data(), ttl_seconds=3600, **_save_kwargs())
    assert cm.get_cached_transcript(**_save_kwargs()) is not None

    # 将磁盘条目的过期时间拨到过去（模拟冷却期已过）
    path = _key_path(cm, **_save_kwargs())
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw[_EXPIRES_KEY] = time.time() - 1
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    assert cm.get_cached_transcript(**_save_kwargs()) is None
    assert not path.exists()


def test_cache_days_mtime_still_enforced(tmp_path):
    """cache_days（mtime）TTL 继续生效：即使无显式标记，mtime 过旧也 miss 并删除"""
    cm = _make_cm(tmp_path)
    cm.save_transcript_to_cache(transcript_data=_sample_data(), **_save_kwargs())

    path = _key_path(cm, **_save_kwargs())
    old = time.time() - (cm.cache_days * 24 * 3600) - 60
    os.utime(path, (old, old))

    assert cm.get_cached_transcript(**_save_kwargs()) is None
    assert not path.exists()


def test_restart_new_manager_reads_ttl_entry(tmp_path):
    """重建 CacheManager（模拟服务重启）后，带 TTL 条目在冷却期内仍可读"""
    cm = _make_cm(tmp_path)
    cm.save_transcript_to_cache(transcript_data=_sample_data(), ttl_seconds=3600, **_save_kwargs())

    cm2 = _make_cm(tmp_path)
    assert cm2.get_cached_transcript(**_save_kwargs()) == _sample_data()


def test_legacy_cache_without_ttl_compatible(tmp_path):
    """旧格式兼容：磁盘条目只有 cached_at（旧代码写入）甚至裸数据也能命中，且返回值被清理"""
    cm = _make_cm(tmp_path)
    path = _key_path(cm, **_save_kwargs())

    # 旧代码格式：cached_at + 数据
    legacy = dict(_sample_data())
    legacy["cached_at"] = time.time()
    path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
    payload = cm.get_cached_transcript(**_save_kwargs())
    assert payload == _sample_data()
    assert "cached_at" not in payload

    # 更早的裸格式：无任何缓存元数据
    path.write_text(json.dumps(_sample_data(), ensure_ascii=False), encoding="utf-8")
    assert cm.get_cached_transcript(**_save_kwargs()) == _sample_data()


@pytest.mark.parametrize("bad_ttl", [0, -1, -0.5, float("nan"), float("inf"), float("-inf"), "3600"])
def test_invalid_ttl_rejected(tmp_path, bad_ttl):
    """0/负数/NaN/Inf/非数值 TTL 必须明确拒绝，且不落盘"""
    cm = _make_cm(tmp_path)
    with pytest.raises(ValueError):
        cm.save_transcript_to_cache(transcript_data=_sample_data(), ttl_seconds=bad_ttl, **_save_kwargs())
    assert not _key_path(cm, **_save_kwargs()).exists()


def test_save_does_not_mutate_caller_dict(tmp_path):
    """写入只操作副本：调用方传入的 transcript_data 不被加入 cached_at 等键"""
    cm = _make_cm(tmp_path)
    data = _sample_data()
    snapshot = copy.deepcopy(data)

    cm.save_transcript_to_cache(transcript_data=data, ttl_seconds=3600, **_save_kwargs())

    assert data == snapshot
    assert "cached_at" not in data
    assert _EXPIRES_KEY not in data


def test_corrupted_json_safe_miss(tmp_path):
    """读到损坏 JSON → 安全 miss（不抛异常），损坏文件被清理"""
    cm = _make_cm(tmp_path)
    path = _key_path(cm, **_save_kwargs())
    path.write_text('{"subtitle": [broken', encoding="utf-8")

    assert cm.get_cached_transcript(**_save_kwargs()) is None
    assert not path.exists()


def test_failed_write_cleans_only_own_temp(tmp_path):
    """序列化失败：不留自己的临时文件、不产生半截缓存，也不动别人的临时文件"""
    cm = _make_cm(tmp_path)

    foreign_tmp = cm.transcript_dir / ".deadbeef.other-writer.tmp"
    foreign_tmp.write_text("other writer's data", encoding="utf-8")

    unserializable = {"subtitle": {1, 2, 3}}  # set 无法 json.dump
    cm.save_transcript_to_cache(transcript_data=unserializable, ttl_seconds=3600, **_save_kwargs())

    assert not _key_path(cm, **_save_kwargs()).exists()  # 没有半截/空缓存文件
    left_temps = [p for p in cm.transcript_dir.iterdir() if p.suffix == ".tmp"]
    assert left_temps == [foreign_tmp]  # 只剩别人创建的临时文件
    assert foreign_tmp.read_text(encoding="utf-8") == "other writer's data"


_ASR_BASELINE = {
    "text": "你好世界",
    "language": "zh",
    "timestamps": [[0.0, 0.5, "你"], [0.5, 1.0, "好"], [1.0, 1.6, "世"], [1.6, 2.0, "界"]],
}


def test_private_asr_field_stripped_by_default(tmp_path):
    """默认返回剥除私有 _asr 基线字段：正常路径 / API 响应不泄露词级内部数据"""
    cm = _make_cm(tmp_path)
    data = _sample_data()
    data["_asr"] = copy.deepcopy(_ASR_BASELINE)

    cm.save_transcript_to_cache(transcript_data=data, **_save_kwargs())

    payload = cm.get_cached_transcript(**_save_kwargs())
    assert payload == _sample_data()  # 除 _asr 外内容一致
    assert "_asr" not in payload
    assert "cached_at" not in payload


def test_include_asr_keeps_private_field(tmp_path):
    """include_asr=True 显式保留 _asr（词级时间戳完整），cached_at 仍剥除；默认参数行为不受影响"""
    cm = _make_cm(tmp_path)
    cm.save_transcript_to_cache(transcript_data={**_sample_data(), "_asr": _ASR_BASELINE}, **_save_kwargs())

    payload = cm.get_cached_transcript(include_asr=True, **_save_kwargs())
    assert payload["_asr"] == _ASR_BASELINE  # 词级时间戳原样保留，冷却后仅重试分离可用
    assert "cached_at" not in payload

    # 不带参数（默认 False）仍剥除
    assert "_asr" not in cm.get_cached_transcript(**_save_kwargs())


def test_external_delete_while_reading_is_a_cache_miss(tmp_path, monkeypatch):
    cm = _make_cm(tmp_path)
    cm.save_transcript_to_cache(transcript_data=_sample_data(), **_save_kwargs())
    cache_path = _key_path(cm, **_save_kwargs())
    load = json.load

    def delete_after_open(stream):
        payload = load(stream)
        cache_path.unlink()
        payload[_EXPIRES_KEY] = 0  # 此次读到的快照也已到期
        return payload

    monkeypatch.setattr(json, "load", delete_after_open)
    assert cm.get_cached_transcript(**_save_kwargs()) is None


def test_delete_between_exists_and_stat_is_a_cache_miss(tmp_path, monkeypatch):
    cm = _make_cm(tmp_path)
    cm.save_transcript_to_cache(transcript_data=_sample_data(), **_save_kwargs())
    cache_path = _key_path(cm, **_save_kwargs())
    path_type = type(cache_path)
    exists = path_type.exists

    def exists_then_delete(path):
        found = exists(path)
        if path == cache_path and found:
            path.unlink()
        return found

    monkeypatch.setattr(path_type, "exists", exists_then_delete)
    assert cm.get_cached_transcript(**_save_kwargs()) is None

