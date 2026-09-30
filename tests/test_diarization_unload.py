"""diarization unload 语义与锁结构测试。"""
import threading


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
