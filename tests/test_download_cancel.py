"""下载器进度回调与专用取消异常测试（Task 4）。

spec：docs/superpowers/specs/2026-11-10-transcribe-job-queue-design.md
§5 downloading 行（percent = downloaded_bytes / content_length，iter_content 循环写）；
§4.3/§6 下载段取消（DownloadCancelled 派生 BaseException，穿透 download_audio 的
except Exception、3 次重试循环、download() 外层 catch bilibili_base.py:229）。

覆盖：
1. download_audio percent 回调序列（downloaded/total，total 来自 content-length）；
2. 无 content-length 时回调不触发（total 不可得 → percent 恒 null 语义）；
3. cancel_flag 置位 → DownloadCancelled 上抛，且是 BaseException 而非 Exception；
4. DownloadCancelled 穿透重试循环（失败重试途中抛出不被吞）；
5. 公共入口 download_bilibili_audio：BV 与 ep 两路径，回调/取消信号到达 iter_content 层
   （底层 requests.get 捕获形参，含 bilibili_video.download 覆写的转发验证）；
6. 默认参数下遗留行为完全不变（不带新参可正常完成全链）。
"""
import json
import threading
from pathlib import Path

import pytest

import downloaders.bilibili_base as bb
from downloaders import BilibiliDownloader
from downloaders.base import DownloadCancelled
from downloaders.bilibili_base import BilibiliDownloaderBase

AUDIO_URL = "https://fake.site/audio.m4s"
EP_AUDIO_URL = "https://fake.site/ep.m4s"
AUDIO_ID = 30216
TOTAL = 10000
CHUNKS = [b"a" * 3000, b"b" * 4000, b"c" * 3000]


# ---------- 测试桩 ----------

class FakeStreamResponse:
    """音频流响应桩：content-length + 分块 iter_content。"""

    def __init__(self, chunks, total=TOTAL):
        self._chunks = chunks
        self.status_code = 200
        self.headers = {} if total is None else {"content-length": str(total)}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=8192):
        for c in self._chunks:
            yield c


class FakePageResponse:
    """页面响应桩：.text + raise_for_status。"""

    def __init__(self, text):
        self.text = text
        self.status_code = 200

    def raise_for_status(self):
        pass


def bv_page():
    """BV 视频页 __playinfo__（dash 分离音轨）。"""
    playinfo = {"data": {
        "dash": {"audio": [{"baseUrl": AUDIO_URL, "id": AUDIO_ID,
                            "bandwidth": 65536, "codecs": "mp4a.40.2"}]},
        "timelength": 60000,
    }}
    return f'<html><script>window.__playinfo__={json.dumps(playinfo)}</script></html>'


def ep_page():
    """番剧页 playurlSSRData（dash 音轨）。"""
    ssr = {"status": 200, "data": {"result": {
        "video_info": {"dash": {"audio": [{"base_url": EP_AUDIO_URL, "id": AUDIO_ID,
                                           "bandwidth": 65536, "codecs": "mp4a.40.2"}]}},
        "timelength": 60000,
    }}}
    return f'<html><script>const playurlSSRData = {json.dumps(ssr)};</script></html>'


def install_fake_net(monkeypatch, pages, streams):
    """按 URL 路由的 requests.get 桩：页面走 pages，音频流走 streams。"""
    def fake_get(url, headers=None, stream=False, timeout=None, **kw):
        if url in pages:
            return FakePageResponse(pages[url])
        if url in streams:
            return streams[url]
        raise AssertionError(f"unexpected GET: {url}")
    monkeypatch.setattr(bb.requests, "get", fake_get)


def stub_pipeline(monkeypatch):
    """全链测试的辅助桩：缓存不命中/不写真实缓存、校验直接通过。"""
    monkeypatch.setattr(bb.cache_manager, "get_cached_file", lambda *a, **k: None)
    monkeypatch.setattr(bb.cache_manager, "save_to_cache",
                        lambda url, file_path, *a, **k: file_path)
    monkeypatch.setattr(bb, "verify_audio_file", lambda *a, **k: (True, "ok (stubbed)"))


# ---------- download_audio 层：进度 ----------

def test_progress_callback_sequence(tmp_path, monkeypatch):
    """percent 回调序列：每块累计 downloaded，total 恒为 content-length，末点 (total, total)。"""
    calls = []
    target = str(tmp_path / "out.m4s")
    install_fake_net(monkeypatch, pages={}, streams={AUDIO_URL: FakeStreamResponse(CHUNKS)})
    ok, result = BilibiliDownloaderBase().download_audio(
        AUDIO_URL, "cookie", target, progress_cb=lambda d, t: calls.append((d, t)))
    assert ok is True
    assert result == str(Path(target).resolve())
    assert calls == [(3000, TOTAL), (7000, TOTAL), (10000, TOTAL)]
    assert Path(target).read_bytes() == b"".join(CHUNKS)


def test_progress_skips_empty_chunks(tmp_path, monkeypatch):
    """空块（keep-alive）：不写不回调，累计序列不受影响。"""
    calls = []
    target = str(tmp_path / "out.m4s")
    install_fake_net(monkeypatch, pages={},
                     streams={AUDIO_URL: FakeStreamResponse([CHUNKS[0], b"", CHUNKS[1], CHUNKS[2]])})
    ok, _ = BilibiliDownloaderBase().download_audio(
        AUDIO_URL, "cookie", target, progress_cb=lambda d, t: calls.append((d, t)))
    assert ok is True
    assert calls == [(3000, TOTAL), (7000, TOTAL), (10000, TOTAL)]


def test_no_content_length_no_callback(tmp_path, monkeypatch):
    """无 content-length：total 不可得，回调不触发（percent 恒 null 语义），下载仍成功。"""
    calls = []
    target = str(tmp_path / "out.m4s")
    install_fake_net(monkeypatch, pages={},
                     streams={AUDIO_URL: FakeStreamResponse(CHUNKS, total=None)})
    ok, _ = BilibiliDownloaderBase().download_audio(
        AUDIO_URL, "cookie", target, progress_cb=lambda d, t: calls.append((d, t)))
    assert ok is True
    assert calls == []


def test_legacy_three_arg_call_unchanged(tmp_path, monkeypatch):
    """遗留 3 参签名不变：不带新参行为与旧版完全一致（成功、文件落盘、无回调）。"""
    target = str(tmp_path / "out.m4s")
    install_fake_net(monkeypatch, pages={}, streams={AUDIO_URL: FakeStreamResponse(CHUNKS)})
    ok, result = BilibiliDownloaderBase().download_audio(AUDIO_URL, "cookie", target)
    assert ok is True
    assert result == str(Path(target).resolve())
    assert Path(target).read_bytes() == b"".join(CHUNKS)


# ---------- download_audio 层：取消 ----------

def test_cancel_flag_raises_base_exception(tmp_path, monkeypatch):
    """flag 预置位：首块边界即抛 DownloadCancelled，且它是 BaseException 而非 Exception。"""
    flag = threading.Event()
    flag.set()
    calls = []
    install_fake_net(monkeypatch, pages={}, streams={AUDIO_URL: FakeStreamResponse(CHUNKS)})
    with pytest.raises(DownloadCancelled):
        BilibiliDownloaderBase().download_audio(
            AUDIO_URL, "cookie", str(tmp_path / "out.m4s"),
            progress_cb=lambda d, t: calls.append((d, t)), cancel_flag=flag)
    assert calls == []  # 首块写入前已中止


def test_cancel_mid_stream_via_callback(tmp_path, monkeypatch):
    """流中途取消：回调里置位 flag → 下一块边界抛出（每块查 flag 的响应性）。"""
    flag = threading.Event()
    calls = []

    def cb(downloaded, total):
        calls.append((downloaded, total))
        flag.set()

    install_fake_net(monkeypatch, pages={}, streams={AUDIO_URL: FakeStreamResponse(CHUNKS)})
    with pytest.raises(DownloadCancelled):
        BilibiliDownloaderBase().download_audio(
            AUDIO_URL, "cookie", str(tmp_path / "out.m4s"),
            progress_cb=cb, cancel_flag=flag)
    assert calls == [(3000, TOTAL)]  # 首块后即中止，后续块不再处理


def test_no_flag_no_cb_null_defaults(tmp_path, monkeypatch):
    """cancel_flag=None / progress_cb=None（默认）：行为与遗留完全一致。"""
    target = str(tmp_path / "out.m4s")
    install_fake_net(monkeypatch, pages={}, streams={AUDIO_URL: FakeStreamResponse(CHUNKS)})
    ok, _ = BilibiliDownloaderBase().download_audio(
        AUDIO_URL, "cookie", target, progress_cb=None, cancel_flag=None)
    assert ok is True


# ---------- 重试循环 / 外层 catch 不吞 ----------

def test_retry_loop_does_not_swallow_cancel(tmp_path, monkeypatch):
    """spec §6：DownloadCancelled 穿透 3 次重试循环——attempt 1 正常失败重试，
    attempt 2 抛 DownloadCancelled 必须逃出 download()（不被 continue/外层 catch 吞）。"""
    stub_pipeline(monkeypatch)
    install_fake_net(monkeypatch, pages={k: bv_page() for k in
                                         ["https://www.bilibili.com/video/BV1xx411c7mD/"]},
                     streams={})
    dl = BilibiliDownloader()
    calls = []

    def fake_download_audio(url, cookie, filename, progress_cb=None, cancel_flag=None):
        calls.append(filename)
        if len(calls) == 1:
            return False, "network error"  # attempt 1 走正常失败 → continue 重试
        raise DownloadCancelled("attempt 2 cancelled")  # attempt 2 必须穿透

    monkeypatch.setattr(dl.video_downloader, "download_audio", fake_download_audio)
    with pytest.raises(DownloadCancelled):
        dl.download_bilibili_audio("BV1xx411c7mD", "cookie", save_dir=str(tmp_path))
    assert len(calls) == 2  # 确实经过了失败重试再抛出


# ---------- 公共入口全链：BV / ep 两路径 ----------

BV_PAGE_URL = "https://www.bilibili.com/video/BV1xx411c7mD/"
EP_PAGE_URL = "https://www.bilibili.com/bangumi/play/ep2289525"


def test_public_entry_bv_path_progress(tmp_path, monkeypatch):
    """BV 主路径（经 bilibili_video.download 覆写转发）：回调到达 iter_content 层。"""
    stub_pipeline(monkeypatch)
    streams = {AUDIO_URL: FakeStreamResponse(CHUNKS)}
    install_fake_net(monkeypatch, pages={BV_PAGE_URL: bv_page()}, streams=streams)
    calls = []
    result = BilibiliDownloader().download_bilibili_audio(
        "BV1xx411c7mD", "cookie", save_dir=str(tmp_path),
        progress_cb=lambda d, t: calls.append((d, t)))
    assert result[0] is True
    assert calls == [(3000, TOTAL), (7000, TOTAL), (10000, TOTAL)]


def test_public_entry_bv_path_cancel(tmp_path, monkeypatch):
    """BV 主路径取消：预置 flag → DownloadCancelled 穿透全链（覆写 → 基类 → 重试 → 外层 catch）。"""
    stub_pipeline(monkeypatch)
    install_fake_net(monkeypatch, pages={BV_PAGE_URL: bv_page()},
                     streams={AUDIO_URL: FakeStreamResponse(CHUNKS)})
    flag = threading.Event()
    flag.set()
    with pytest.raises(DownloadCancelled) as exc_info:
        BilibiliDownloader().download_bilibili_audio(
            "BV1xx411c7mD", "cookie", save_dir=str(tmp_path), cancel_flag=flag)
    assert not isinstance(exc_info.value, Exception)


def test_public_entry_ep_path_progress(tmp_path, monkeypatch):
    """ep 路径（episode 继承基类 download）：回调到达 iter_content 层。"""
    stub_pipeline(monkeypatch)
    streams = {EP_AUDIO_URL: FakeStreamResponse(CHUNKS)}
    install_fake_net(monkeypatch, pages={EP_PAGE_URL: ep_page()}, streams=streams)
    calls = []
    result = BilibiliDownloader().download_bilibili_audio(
        "ep2289525", "cookie", save_dir=str(tmp_path),
        progress_cb=lambda d, t: calls.append((d, t)))
    assert result[0] is True
    assert calls == [(3000, TOTAL), (7000, TOTAL), (10000, TOTAL)]


def test_public_entry_ep_path_cancel(tmp_path, monkeypatch):
    """ep 路径取消：预置 flag → DownloadCancelled 穿透全链。"""
    stub_pipeline(monkeypatch)
    install_fake_net(monkeypatch, pages={EP_PAGE_URL: ep_page()},
                     streams={EP_AUDIO_URL: FakeStreamResponse(CHUNKS)})
    flag = threading.Event()
    flag.set()
    with pytest.raises(DownloadCancelled):
        BilibiliDownloader().download_bilibili_audio(
            "ep2289525", "cookie", save_dir=str(tmp_path), cancel_flag=flag)


def test_public_entry_legacy_defaults(tmp_path, monkeypatch):
    """公共入口不带新参：遗留行为不变（成功 + 结果 dict 形状）。"""
    stub_pipeline(monkeypatch)
    install_fake_net(monkeypatch, pages={BV_PAGE_URL: bv_page()},
                     streams={AUDIO_URL: FakeStreamResponse(CHUNKS)})
    ok, result = BilibiliDownloader().download_bilibili_audio(
        "BV1xx411c7mD", "cookie", save_dir=str(tmp_path))
    assert ok is True
    assert set(result) == {"file_path", "audio_url", "audio_id"}
    assert result["audio_url"] == AUDIO_URL


def test_download_cancelled_hierarchy():
    """spec 定死：DownloadCancelled 派生 BaseException，三处 except Exception 天然放行。"""
    assert issubclass(DownloadCancelled, BaseException)
    assert not issubclass(DownloadCancelled, Exception)
