from types import SimpleNamespace

import httpx
import respx

from backends.asr_engine_backend import ASREngineClientBackend
from backends.base import TranscribeResult


def _make_config(url="http://127.0.0.1:31090", token=""):
    """假 config：ASREngineClientBackend 只读 asr_engine_url / asr_engine_token。"""
    return SimpleNamespace(asr_engine_url=url, asr_engine_token=token)


@respx.mock
def test_transcribe_parses_verbose_json_to_base_result(tmp_path):
    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"fake-audio")
    verbose = {
        "task": "transcribe",
        "language": "chinese",
        "duration": 1.2,
        "text": "你好世界",
        "words": [
            {"word": "你", "start": 0.0, "end": 0.3},
            {"word": "好", "start": 0.3, "end": 0.6},
        ],
        "segments": [{"id": 0, "start": 0.0, "end": 0.6, "text": "你好世界"}],
    }
    route = respx.post("http://127.0.0.1:31090/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json=verbose)
    )

    backend = ASREngineClientBackend(_make_config())
    result = backend.transcribe(str(audio))

    assert route.called
    assert isinstance(result, TranscribeResult)
    assert result.text == "你好世界"
    assert result.language == "zh"                      # base 用短代码
    assert result.timestamps == [
        {"text": "你", "start": 0.0, "end": 0.3},
        {"text": "好", "start": 0.3, "end": 0.6},
    ]


@respx.mock
def test_transcribe_no_words_gives_empty_timestamps(tmp_path):
    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"fake-audio")
    verbose = {"task": "transcribe", "language": "english", "duration": 5.0,
               "text": "hello", "words": [], "segments": []}
    respx.post("http://127.0.0.1:31090/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json=verbose))

    backend = ASREngineClientBackend(_make_config())
    result = backend.transcribe(str(audio))
    assert result.text == "hello"
    assert result.language == "en"
    assert result.timestamps == []


def test_name_and_device():
    backend = ASREngineClientBackend(_make_config())
    assert backend.name == "asr-engine"
    assert backend.device == "remote"                  # 不占本地 GPU
