"""diarization unload 语义与锁结构测试。"""
import sys
import threading
import time


def test_unload_clears_pipeline_and_flags():
    from diarization.manager import DiarizationManager
    m = DiarizationManager.__new__(DiarizationManager)
    m._pipeline = object()
    m._load_lock = threading.Lock()
    m._infer_lock = threading.Lock()

    import gc as gc_mod
    emptied = {}

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            emptied["cache"] = True

    import sys
    fake = type(sys)("fake_torch")
    fake.cuda = FakeCuda
    import pytest
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(gc_mod, "collect", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", fake)
    try:
        assert m.unload() is True
        assert m._pipeline is None and m.is_loaded is False
        assert emptied.get("cache") is True
        assert m.unload() is False  # 未加载再卸返回 False
    finally:
        monkeypatch.undo()


def test_diarize_loads_inside_infer_lock():
    """锁修正验证:diarize 源码中 _load 调用必须出现在 with self._infer_lock 之后"""
    import inspect
    from diarization import manager as dm
    src = inspect.getsource(dm.DiarizationManager.diarize)
    infer_pos = src.index("with self._infer_lock")
    load_pos = src.index("self._load()")
    assert load_pos > infer_pos, "_load() 必须在 _infer_lock 内(防卸载竞态)"


def _bare_manager(pipeline):
    """构造走 __new__ 的裸 manager(假 pipeline + 两把锁,不触真实依赖)"""
    from diarization.manager import DiarizationManager
    m = DiarizationManager.__new__(DiarizationManager)
    m._pipeline = pipeline
    m._load_lock = threading.Lock()
    m._infer_lock = threading.Lock()
    return m


def test_unload_should_abort_after_lock_wait():
    """世代检查在等锁之后:等锁期间 should_abort 变 True → 锁内中止,返回 False 且管线不清空。

    修复前(if should_abort 在等锁前)此用例会在未等锁时即返回,无法防
    "等长推理锁期间用户 resume、旧释放链随后仍卸载"的交错。
    """
    m = _bare_manager(pipeline=object())

    m._infer_lock.acquire()  # 本线程持推理锁,模拟在跑的长推理
    result = {}

    def worker():
        result["released"] = m.unload(should_abort=lambda: True)  # 恒中止=模拟等锁期间世代已变

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.1)
    assert t.is_alive()  # 锁仍被持有,worker 必然阻塞在等锁上(若实现把检查放等锁前则早已返回)
    m._infer_lock.release()
    t.join(timeout=5)
    assert not t.is_alive()
    assert result["released"] is False   # 锁内中止而非卸载
    assert m._pipeline is not None       # 管线保持原状


def test_unload_should_abort_false_unloads_normally():
    """should_abort 恒 False(世代未变)→ 走正常卸载路径,与不传回调行为一致"""
    import gc as gc_mod
    import pytest

    m = _bare_manager(pipeline=object())
    emptied = {}

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            emptied["cache"] = True

    fake = type(sys)("fake_torch")
    fake.cuda = FakeCuda
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(gc_mod, "collect", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", fake)
    try:
        assert m.unload(should_abort=lambda: False) is True
        assert m._pipeline is None and m.is_loaded is False
        assert emptied.get("cache") is True
    finally:
        monkeypatch.undo()
