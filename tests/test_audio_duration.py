"""时长探测只能读取元数据，不能在请求进程中完整解码长音频。"""
import subprocess
import sys
from types import SimpleNamespace

import transcribe as T


def test_duration_fallback_uses_metadata_not_full_torchaudio_decode(monkeypatch):
    monkeypatch.setattr(T.os, "system", lambda cmd: 0)
    monkeypatch.setattr(T.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))
    monkeypatch.setitem(sys.modules, "mutagen", SimpleNamespace(File=lambda path: None))
    decoded = []

    def full_decode(*args):
        decoded.append(True)
        raise AssertionError("时长探测不应加载整段波形")

    monkeypatch.setitem(sys.modules, "torchaudio", SimpleNamespace(load=full_decode))
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(
        info=lambda path: SimpleNamespace(frames=1440000, samplerate=16000)))
    assert T.get_audio_duration("test.mp3") == 90.0
    assert decoded == []


def test_ffprobe_timeout_keeps_existing_retry(monkeypatch):
    monkeypatch.setattr(T.os, "system", lambda cmd: 0)
    calls = []

    def probe(*args, **kwargs):
        calls.append(kwargs["timeout"])
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("ffprobe", 10)
        return SimpleNamespace(returncode=0, stdout="120.0")

    monkeypatch.setattr(T.subprocess, "run", probe)
    assert T.get_audio_duration("test.mp3") == 120.0
    assert calls == [10, 10]
