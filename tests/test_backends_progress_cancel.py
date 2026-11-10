"""Task 5：三后端签名统一与 asr_engine_backend 增强测试。

覆盖（spec §5 进度映射 transcribing 行 + §6 引擎侧第 3/5/6 条）：
- gguf：progress_cb 透传共享库 engine.transcribe；进程内 pending-cancel 集合
  「登记时命中即置位并移除、未命中不置位、TTL/容量清理」（注入 clock）；
  probe_capability 恒 progress+scoped。
- asr-engine：form 含 request_id；verbose_json 顶层 cancelled 映射 performance；
  cancel_request 发 scoped body（best-effort 不抛）；capability 探测/
  负缓存 10min 重探/正缓存持续；legacy 态 cancel_request 照常发 scoped 语义
  （取消降级是消费方 Task 7 的职责，backend 只如实报告 capability）。
- qwen3/funasr：新参默认不破坏旧调用（含遗留三参位置调用）。
- base：cancel_request no-op、probe_capability 默认 legacy。
"""
import json
import threading
import time
import types
from types import SimpleNamespace

import httpx
import pytest
import respx

import backends.asr_engine_backend as asr_engine_mod
from backends.asr_engine_backend import ASREngineClientBackend
from backends.base import TranscribeResult
from backends.funasr_backend import FunASRBackend
from backends.gguf_backend import GGUFBackend
from backends.qwen3_backend import Qwen3Backend


# ---------------------------------------------------------------------------
# gguf：桩 engine（不加载真实模型）
# ---------------------------------------------------------------------------

class _StubEngine:
    """QwenASREngine 桩：记录 set_cancel_event 调用序列与 transcribe 实参。"""

    def __init__(self):
        self.events = []                    # set_cancel_event 序列（收尾应为 None）
        self.transcribe_kwargs = None
        self.event_during_transcribe = None

    def set_cancel_event(self, event):
        self.events.append(event)

    def transcribe(self, audio_file, language=None, context=None, progress_cb=None):
        self.transcribe_kwargs = {"progress_cb": progress_cb}
        self.event_during_transcribe = self.events[-1]
        if progress_cb:
            progress_cb(1, 1)
        return types.SimpleNamespace(
            text="x", language=None, alignment=None,
            performance={"audio_duration": 1.0},
        )


class _FakeClock:
    """可控单调时钟，注入 GGUFBackend（TTL/容量清理的时间来源）。"""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, sec):
        self.now += sec


def make_gguf(clock=None):
    backend = GGUFBackend(SimpleNamespace(), clock=clock)
    backend._engine = _StubEngine()
    return backend


def test_gguf_progress_cb_passthrough_to_engine():
    """transcribe 把 progress_cb 透传给共享库 engine.transcribe。"""
    backend = make_gguf()
    calls = []
    backend.transcribe("/tmp/x.wav", None, None,
                       progress_cb=lambda d, t: calls.append((d, t)), request_id="r1")
    assert backend._engine.transcribe_kwargs["progress_cb"] is not None
    assert calls == [(1, 1)]
    # 转录收尾清除挂接：最后一条 set_cancel_event 为 None
    assert backend._engine.events[-1] is None


def test_gguf_legacy_three_positional_call_still_works():
    """遗留三参位置调用零破坏（不传新参 = 不挂取消语义）。"""
    backend = make_gguf()
    result = backend.transcribe("/tmp/x.wav", "zh", "ctx")
    assert isinstance(result, TranscribeResult)
    assert backend._engine.event_during_transcribe is not None   # 仍挂接事件
    assert not backend._engine.event_during_transcribe.is_set()  # 但未置位


def test_gguf_pending_cancel_hit_sets_event_and_consumed():
    """登记先于转录开始 → 挂接时命中即置位并移除（spec §6 引擎侧第 2 条同构）。"""
    backend = make_gguf()
    backend.cancel_request("r1")
    backend.transcribe("/tmp/x.wav", request_id="r1")
    assert backend._engine.event_during_transcribe.is_set()
    assert backend._engine.events[-1] is None
    # 消费即移除：同 request_id 再转录不再置位
    backend.transcribe("/tmp/x.wav", request_id="r1")
    assert not backend._engine.event_during_transcribe.is_set()


def test_gguf_pending_cancel_miss_keeps_unset_and_entry():
    """未命中：事件不置位，登记条目保留等待真正同名请求。"""
    backend = make_gguf()
    backend.cancel_request("other")
    backend.transcribe("/tmp/x.wav", request_id="r1")
    assert not backend._engine.event_during_transcribe.is_set()
    assert "other" in backend._pending_cancels


def test_gguf_pending_cancel_ttl_expiry():
    """TTL 300s 过期清理（校验时 pruning，注入 clock）。"""
    clock = _FakeClock()
    backend = make_gguf(clock)
    backend.cancel_request("r1")
    clock.advance(301)                     # 越过 TTL
    backend.transcribe("/tmp/x.wav", request_id="r1")
    assert not backend._engine.event_during_transcribe.is_set()
    assert backend._pending_cancels == {}
    # 过期条目不复活：同 id 可重新登记并命中
    backend.cancel_request("r1")
    backend.transcribe("/tmp/x.wav", request_id="r1")
    assert backend._engine.event_during_transcribe.is_set()


def test_gguf_pending_cancel_capacity_eviction():
    """容量上限 64，超限逐最旧。"""
    clock = _FakeClock()
    backend = make_gguf(clock)
    backend.cancel_request("oldest")
    for i in range(64):                    # 共 65 条 → 超容量
        backend.cancel_request(f"r{i}")
    assert len(backend._pending_cancels) == 64
    assert "oldest" not in backend._pending_cancels
    backend.transcribe("/tmp/x.wav", request_id="oldest")
    assert not backend._engine.event_during_transcribe.is_set()
    backend.transcribe("/tmp/x.wav", request_id="r63")   # 最新条目仍命中
    assert backend._engine.event_during_transcribe.is_set()


def test_gguf_cancel_request_ignores_empty_id():
    backend = make_gguf()
    backend.cancel_request(None)
    backend.cancel_request("")
    assert backend._pending_cancels == {}


def test_gguf_cancel_while_running_hits_active_event():
    """转录进行中 cancel_request → 活跃归属命中直接置位（spec §6 引擎侧第 3 条
    「活跃归属匹配 → 置位」gguf 同构），不入 pending 集合。"""
    gate = threading.Event()
    started = threading.Event()
    events_seen = []

    class _BlockingEngine(_StubEngine):
        def transcribe(self, audio_file, language=None, context=None, progress_cb=None):
            self.transcribe_kwargs = {"progress_cb": progress_cb}
            ev = self.events[-1]
            events_seen.append(ev)
            started.set()
            gate.wait(5)                       # 模拟长转录，等待主线程取消
            self.event_during_transcribe = ev
            return types.SimpleNamespace(text="x", language=None, alignment=None,
                                         performance={"audio_duration": 1.0})

    backend = GGUFBackend(SimpleNamespace(), clock=_FakeClock())
    backend._engine = _BlockingEngine()

    worker = threading.Thread(target=backend.transcribe, kwargs={
        "audio_file": "/tmp/x.wav", "request_id": "r-run"}, daemon=True)
    worker.start()
    assert started.wait(5)
    backend.cancel_request("r-run")            # 此时转录正在进行
    gate.set()
    worker.join(5)
    assert not worker.is_alive()
    assert events_seen[-1].is_set()            # 活跃事件被置位 → 共享库 chunk 边界将中止
    assert backend._active_cancels == {}       # 转录收尾清除活跃归属
    assert backend._pending_cancels == {}      # 未误入 pending 集合


def test_gguf_engine_unloaded_midflight_still_releases_lock():
    """评审 Important：unload() 不持 transcribe_lock，窗口内 engine 被置 None 时，
    收尾清除段不得因解引用 self._engine 抛 AttributeError 致锁永不释放（死锁）。
    桩 engine 在转录期间模拟并发 unload()；请求正常完成，锁必须释放。"""
    backend = make_gguf()

    class _UnloadMidflightEngine(_StubEngine):
        def transcribe(self, audio_file, language=None, context=None, progress_cb=None):
            backend._engine = None            # 模拟并发 unload() 置 None
            self.event_during_transcribe = self.events[-1]
            return types.SimpleNamespace(text="x", language=None, alignment=None,
                                         performance={"audio_duration": 1.0})

    backend._engine = _UnloadMidflightEngine()
    result = backend.transcribe("/tmp/x.wav", request_id="r1")   # 不得抛 AttributeError
    assert isinstance(result, TranscribeResult)
    assert backend._transcribe_lock.acquire(blocking=True, timeout=2), "锁未释放（死锁）"
    backend._transcribe_lock.release()


def test_gguf_engine_unloaded_while_queued_releases_lock_and_raises():
    """排队等待期间被 unload：获得锁后发现 engine 已为 None → 立即释放锁并报
    「模型未加载」，不得吞锁也不得 AttributeError 泄漏锁。"""
    backend = make_gguf()
    assert backend._transcribe_lock.acquire(timeout=2)     # 模拟前一请求持锁
    results = {}

    def run():
        try:
            results["ret"] = backend.transcribe("/tmp/x.wav", request_id="r1")
        except BaseException as e:                         # noqa: BLE001 —— 测试需捕获任意异常
            results["err"] = e

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    time.sleep(0.2)                            # 让 worker 通过前置检查并阻塞在 acquire
    backend._engine = None                     # 排队期间并发 unload()
    backend._transcribe_lock.release()
    worker.join(5)
    assert not worker.is_alive()
    assert isinstance(results.get("err"), RuntimeError)
    assert "模型未加载" in str(results["err"])
    assert backend._transcribe_lock.acquire(blocking=True, timeout=2), "锁未释放（死锁）"
    backend._transcribe_lock.release()


def test_gguf_capability_constant_progress_scoped():
    assert make_gguf().probe_capability() == "progress+scoped"


# ---------------------------------------------------------------------------
# asr-engine：MockTransport（respx）断言真实 form/body
# ---------------------------------------------------------------------------

def _make_engine_config(url="http://127.0.0.1:31090", token="tok"):
    return SimpleNamespace(asr_engine_url=url, asr_engine_token=token)


@respx.mock
def test_asr_engine_form_includes_request_id(tmp_path):
    """有 request_id 则 form data 携带（真实断言 multipart body）。"""
    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"fake-audio")
    verbose = {"task": "transcribe", "language": "chinese", "duration": 1.0,
               "text": "你", "words": [], "segments": []}
    route = respx.post("http://127.0.0.1:31090/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json=verbose))

    backend = ASREngineClientBackend(_make_engine_config())
    cb_calls = []
    result = backend.transcribe(str(audio), None, None,
                                progress_cb=lambda d, t: cb_calls.append((d, t)),
                                request_id="req-abc")

    assert isinstance(result, TranscribeResult)
    assert route.called
    body = route.calls.last.request.content
    assert b'name="request_id"' in body
    assert b"req-abc" in body
    assert b'name="context"' not in body          # 无 context 不发该字段
    assert cb_calls == []                          # progress_cb 接受但不使用


@respx.mock
def test_asr_engine_form_omits_request_id_when_absent(tmp_path):
    """无 request_id（遗留调用）form 不携带该字段。"""
    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"fake-audio")
    verbose = {"task": "transcribe", "language": "chinese", "duration": 1.0,
               "text": "你", "words": [], "segments": []}
    route = respx.post("http://127.0.0.1:31090/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json=verbose))

    backend = ASREngineClientBackend(_make_engine_config())
    backend.transcribe(str(audio))                # 遗留单参调用
    assert route.called
    assert b'name="request_id"' not in route.calls.last.request.content


@respx.mock
def test_asr_engine_cancelled_true_maps_to_performance(tmp_path):
    """verbose_json 顶层 "cancelled": true → performance["cancelled"] is True。"""
    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"fake-audio")
    verbose = {"task": "transcribe", "language": "chinese", "duration": 1.0,
               "text": "你", "words": [], "segments": [], "cancelled": True}
    respx.post("http://127.0.0.1:31090/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json=verbose))

    backend = ASREngineClientBackend(_make_engine_config())
    result = backend.transcribe(str(audio))
    assert result.performance["cancelled"] is True


@respx.mock
def test_asr_engine_no_cancelled_field_performance_falsy(tmp_path):
    """无 cancelled 字段（正常完成）→ performance 不带取消标记。"""
    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"fake-audio")
    verbose = {"task": "transcribe", "language": "chinese", "duration": 1.0,
               "text": "你", "words": [], "segments": []}
    respx.post("http://127.0.0.1:31090/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json=verbose))

    backend = ASREngineClientBackend(_make_engine_config())
    result = backend.transcribe(str(audio))
    assert not result.performance.get("cancelled")


@respx.mock
def test_asr_engine_cancel_request_posts_scoped_body():
    """cancel_request → POST /admin/cancel，json {"request_id": ...}，带 token header。"""
    route = respx.post("http://127.0.0.1:31090/admin/cancel").mock(
        return_value=httpx.Response(200, json={"ok": True}))

    backend = ASREngineClientBackend(_make_engine_config(token="tok"))
    backend.cancel_request("req-9")

    assert route.called
    req = route.calls.last.request
    assert json.loads(req.content) == {"request_id": "req-9"}
    assert req.headers.get("Authorization") == "Bearer tok"


@respx.mock
def test_asr_engine_cancel_request_empty_id_no_http():
    route = respx.post("http://127.0.0.1:31090/admin/cancel").mock(
        return_value=httpx.Response(200, json={"ok": True}))
    backend = ASREngineClientBackend(_make_engine_config())
    assert backend.cancel_request(None) is None
    assert backend.cancel_request("") is None
    assert not route.called


@respx.mock
def test_asr_engine_cancel_request_best_effort_no_raise():
    """取消传输失败不影响调用方（best-effort，记日志）。"""
    respx.post("http://127.0.0.1:31090/admin/cancel").mock(
        side_effect=httpx.ConnectError("down"))
    backend = ASREngineClientBackend(_make_engine_config())
    backend.cancel_request("req-x")               # 不抛


@respx.mock
def test_asr_engine_cancel_request_404_best_effort_no_raise():
    """旧引擎无 /admin/cancel 端点（404，legacy 最常见形态）→ best-effort 不抛。"""
    respx.post("http://127.0.0.1:31090/admin/cancel").mock(
        return_value=httpx.Response(404))
    backend = ASREngineClientBackend(_make_engine_config())
    backend.cancel_request("req-old")             # 不抛


# ---------------------------------------------------------------------------
# asr-engine：capability 探测（负缓存/正缓存/时钟推进重探）
# ---------------------------------------------------------------------------

class _MonotonicFake:
    """monkeypatch mod.time.monotonic 用的可控时钟。"""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, sec):
        self.now += sec


@respx.mock
def test_capability_404_negative_cache_then_reprobe(monkeypatch):
    """404 → legacy 且负缓存 10min；时钟推进后重探；200 → 正缓存持续。"""
    clock = _MonotonicFake()
    monkeypatch.setattr(asr_engine_mod.time, "monotonic", clock)
    probe_route = respx.get("http://127.0.0.1:31090/v1/progress").mock(
        return_value=httpx.Response(404))
    backend = ASREngineClientBackend(_make_engine_config())

    assert backend.probe_capability() == "legacy"
    assert backend.probe_capability() == "legacy"
    assert probe_route.call_count == 1            # 负缓存生效，未重探

    clock.advance(601)                            # 越过 10min 负缓存
    probe_route.mock(return_value=httpx.Response(200, json={"active": False}))
    assert backend.probe_capability() == "progress+scoped"
    assert probe_route.call_count == 2
    assert backend.probe_capability() == "progress+scoped"
    assert probe_route.call_count == 2            # 正缓存持续，不再探测


@respx.mock
def test_capability_200_positive_cache_immediately(monkeypatch):
    clock = _MonotonicFake()
    monkeypatch.setattr(asr_engine_mod.time, "monotonic", clock)
    probe_route = respx.get("http://127.0.0.1:31090/v1/progress").mock(
        return_value=httpx.Response(200, json={"active": True, "request_id": None,
                                               "chunk": 0, "total": 0}))
    backend = ASREngineClientBackend(_make_engine_config())

    assert backend.probe_capability() == "progress+scoped"
    assert backend.probe_capability() == "progress+scoped"
    assert probe_route.call_count == 1            # 正缓存
    # 探测请求带 token header
    assert probe_route.calls.last.request.headers.get("Authorization") == "Bearer tok"


@respx.mock
def test_capability_probe_transient_error_not_cached(monkeypatch):
    """连接失败等瞬时错误 → legacy 但不缓存，下次再探。"""
    monkeypatch.setattr(asr_engine_mod.time, "monotonic", _MonotonicFake())
    probe_route = respx.get("http://127.0.0.1:31090/v1/progress").mock(
        side_effect=httpx.ConnectError("down"))
    backend = ASREngineClientBackend(_make_engine_config())

    assert backend.probe_capability() == "legacy"
    assert backend.probe_capability() == "legacy"
    assert probe_route.call_count == 2            # 未缓存，每次重探


@respx.mock
def test_legacy_capability_cancel_request_still_sends_scoped(monkeypatch):
    """legacy 态 cancel_request 照常发 scoped 语义：降级（标志+join）是消费方
    Task 7 的职责，backend 只如实报告 capability，不自行吞请求。"""
    monkeypatch.setattr(asr_engine_mod.time, "monotonic", _MonotonicFake())
    respx.get("http://127.0.0.1:31090/v1/progress").mock(
        return_value=httpx.Response(404))
    cancel_route = respx.post("http://127.0.0.1:31090/admin/cancel").mock(
        return_value=httpx.Response(200, json={"ok": True}))
    backend = ASREngineClientBackend(_make_engine_config())

    assert backend.probe_capability() == "legacy"
    backend.cancel_request("req-legacy")
    assert cancel_route.called
    assert json.loads(cancel_route.calls.last.request.content) == {
        "request_id": "req-legacy"}


# ---------------------------------------------------------------------------
# qwen3 / funasr：签名加参，旧调用不破坏；base no-op 与默认 capability
# ---------------------------------------------------------------------------

def test_qwen3_funasr_signature_accepts_new_params_without_breaking(tmp_path):
    """三参位置调用与全量新参调用均到达「模型未加载」检查（非 TypeError）。"""
    audio = str(tmp_path / "a.wav")
    q3 = Qwen3Backend(SimpleNamespace())
    fn = FunASRBackend(SimpleNamespace())
    for backend in (q3, fn):
        with pytest.raises(RuntimeError, match="模型未加载"):
            backend.transcribe(audio, None, None)                    # 遗留三参位置调用
        with pytest.raises(RuntimeError, match="模型未加载"):
            backend.transcribe(audio, language=None, context=None,
                               progress_cb=None, request_id="r1")   # 新参全给
        # base 继承：cancel_request no-op、probe_capability 恒 legacy
        assert backend.cancel_request("r1") is None
        assert backend.probe_capability() == "legacy"
