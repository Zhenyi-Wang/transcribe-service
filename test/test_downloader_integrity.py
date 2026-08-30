"""下载器完整性防护测试

- download_audio：content-length 比对、timeout 参数
- download：验证失败重试（共 3 次尝试）、坏缓存拒绝、timelength 提取
网络与下载用 mock，控制流与缓存行为走真实代码。
"""
import os
import sys
from unittest.mock import patch, MagicMock, call

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downloaders.bilibili_video import BilibiliVideoDownloader
from downloaders.bilibili_episode import BilibiliEpisodeDownloader

MAX_ATTEMPTS = 3  # 首次 + 2 次重试


# ---------- helpers ----------

def make_response(chunks, content_length=None):
    resp = MagicMock()
    if content_length is not None:
        resp.headers = {"content-length": str(content_length)}
    else:
        resp.headers = {}
    resp.iter_content = lambda chunk_size: iter(chunks)
    resp.raise_for_status = lambda: None
    return resp


VIDEO_HTML = (
    '<html><script>window.__playinfo__={"data":{"timelength":123456,'
    '"dash":{"audio":[{"id":30216,"bandwidth":66560,"codecs":"mp4a.40.2",'
    '"baseUrl":"http://audio.example/a.m4s"}]}}}</script></html>'
)

EPISODE_HTML = (
    '<html><script>const playurlSSRData = {"status":200,"data":{"result":{'
    '"timelength":234567,"video_info":{"dash":{"audio":['
    '{"id":30280,"bandwidth":66560,"codecs":"mp4a.40.2",'
    '"base_url":"http://audio.example/b.m4s"}]}}}}}</script></html>'
)


@pytest.fixture
def video_dl():
    return BilibiliVideoDownloader()


@pytest.fixture
def episode_dl():
    return BilibiliEpisodeDownloader()


# ---------- download_audio: size 比对 ----------

class TestDownloadAudioSizeCheck:
    def test_size_mismatch_fails_and_removes_file(self, video_dl, tmp_path):
        target = tmp_path / "a.m4s"
        chunks = [b"x" * 1000] * 5  # 实际 5000 字节
        with patch("downloaders.bilibili_video.requests.get",
                   return_value=make_response(chunks, content_length=10000)):
            ok, detail = video_dl.download_audio("http://x", "ck", str(target))
        assert not ok
        assert "不完整" in detail
        assert not target.exists()

    def test_size_match_succeeds(self, video_dl, tmp_path):
        target = tmp_path / "a.m4s"
        chunks = [b"x" * 1000] * 5
        with patch("downloaders.bilibili_video.requests.get",
                   return_value=make_response(chunks, content_length=5000)):
            ok, result = video_dl.download_audio("http://x", "ck", str(target))
        assert ok, result
        assert target.exists()

    def test_missing_content_length_passes_size_check(self, video_dl, tmp_path):
        # 无 content-length 时此层不拦（兜底靠 ffprobe 时长校验）
        target = tmp_path / "a.m4s"
        chunks = [b"x" * 1000] * 3
        with patch("downloaders.bilibili_video.requests.get",
                   return_value=make_response(chunks)):
            ok, result = video_dl.download_audio("http://x", "ck", str(target))
        assert ok, result

    def test_passes_timeout_to_requests(self, video_dl, tmp_path):
        target = tmp_path / "a.m4s"
        with patch("downloaders.bilibili_video.requests.get",
                   return_value=make_response([b"x"], content_length=1)) as m:
            video_dl.download_audio("http://x", "ck", str(target))
        assert m.call_args.kwargs.get("timeout") is not None

    def test_duplicated_content_length_header_tolerated(self, video_dl, tmp_path):
        # 重复头会被 requests 合并为 "n, n"，应取首段而非崩溃
        target = tmp_path / "a.m4s"
        chunks = [b"x" * 1000] * 5
        with patch("downloaders.bilibili_video.requests.get",
                   return_value=make_response(chunks, content_length="5000, 5000")):
            ok, result = video_dl.download_audio("http://x", "ck", str(target))
        assert ok, result


# ---------- download: 缓存校验与重试 ----------

class TestVideoDownloadFlow:
    def _patch_core(self, downloader, audio_info, download_results):
        """统一 patch：audio_info 固定、URL 返回固定、download_audio 依次返回 download_results"""
        p1 = patch.object(downloader, "get_audio_info", return_value=audio_info)
        p2 = patch.object(downloader, "get_audio_url",
                          return_value=("http://audio.example/a.m4s", audio_info))
        p3 = patch.object(downloader, "download_audio", side_effect=download_results)
        return p1, p2, p3

    def test_verification_failure_retries_then_succeeds(self, video_dl, tmp_path):
        target = str(tmp_path / "a.m4s")
        p1, p2, p3 = self._patch_core(
            video_dl,
            {"id": 30216, "bandwidth": 66560, "codecs": "mp4a.40.2",
             "format": "dash", "timelength": 123456},
            [(True, target)] * MAX_ATTEMPTS,
        )
        with p1, p2, p3 as dl_mock, \
             patch("downloaders.bilibili_video.verify_audio_file",
                   side_effect=[(False, "时长不足"), (False, "时长不足"), (True, "ok")]), \
             patch("downloaders.bilibili_video.cache_manager") as cache_mock, \
             patch("os.remove"), patch("os.replace"):
            cache_mock.get_cached_file.return_value = None  # 无缓存，走下载
            ok, result = video_dl.download("BV1xx", "ck", save_dir=str(tmp_path))
        assert ok, result
        assert dl_mock.call_count == MAX_ATTEMPTS
        assert cache_mock.save_to_cache.called

    def test_all_attempts_fail_returns_incomplete_error(self, video_dl, tmp_path):
        target = str(tmp_path / "a.m4s")
        p1, p2, p3 = self._patch_core(
            video_dl,
            {"id": 30216, "bandwidth": 66560, "codecs": "mp4a.40.2",
             "format": "dash", "timelength": 123456},
            [(True, target)] * MAX_ATTEMPTS,
        )
        with p1, p2, p3 as dl_mock, \
             patch("downloaders.bilibili_video.verify_audio_file",
                   return_value=(False, "音频不完整: 实际时长 1.0s，期望 123.5s")), \
             patch("downloaders.bilibili_video.cache_manager") as cache_mock, \
             patch("os.remove"):
            cache_mock.get_cached_file.return_value = None
            ok, result = video_dl.download("BV1xx", "ck", save_dir=str(tmp_path))
        assert not ok
        assert "不完整" in result
        assert dl_mock.call_count == MAX_ATTEMPTS

    def test_download_audio_failure_retries(self, video_dl, tmp_path):
        results = [(False, "boom")] * (MAX_ATTEMPTS - 1) + [(True, str(tmp_path / "a.m4s"))]
        p1, p2, p3 = self._patch_core(
            video_dl,
            {"id": 30216, "bandwidth": 66560, "codecs": "mp4a.40.2",
             "format": "dash", "timelength": 123456},
            results,
        )
        with p1, p2, p3 as dl_mock, \
             patch("downloaders.bilibili_video.verify_audio_file",
                   return_value=(True, "ok")), \
             patch("downloaders.bilibili_video.cache_manager") as cache_mock, \
             patch("os.replace"):
            cache_mock.get_cached_file.return_value = None
            ok, result = video_dl.download("BV1xx", "ck", save_dir=str(tmp_path))
        assert ok, result
        assert dl_mock.call_count == MAX_ATTEMPTS
        assert cache_mock.save_to_cache.called

    def test_bad_cached_file_is_removed_and_redownloaded(self, video_dl, tmp_path):
        bad_cache = tmp_path / "cached.m4s"
        bad_cache.write_bytes(b"truncated")
        target = str(tmp_path / "a.m4s")
        p1, p2, p3 = self._patch_core(
            video_dl,
            {"id": 30216, "bandwidth": 66560, "codecs": "mp4a.40.2",
             "format": "dash", "timelength": 123456},
            [(True, target)],
        )
        with p1, p2, p3 as dl_mock, \
             patch("downloaders.bilibili_video.verify_audio_file",
                   side_effect=[(False, "时长不足"), (True, "ok")]), \
             patch("downloaders.bilibili_video.cache_manager") as cache_mock, \
             patch("os.replace"):
            cache_mock.get_cached_file.return_value = str(bad_cache)
            ok, result = video_dl.download("BV1xx", "ck", save_dir=str(tmp_path))
        assert ok, result
        assert not bad_cache.exists()  # 坏缓存被删除
        assert dl_mock.called

    def test_download_lands_file_via_atomic_rename(self, video_dl, tmp_path):
        # 并发安全契约：下载写唯一临时名，verify 通过后原子 rename 到正式名
        target = str(tmp_path / "a.m4s")
        p1, p2, p3 = self._patch_core(
            video_dl,
            {"id": 30216, "bandwidth": 66560, "codecs": "mp4a.40.2",
             "format": "dash", "timelength": 123456},
            [(True, target)],
        )
        with p1, p2, p3, \
             patch("downloaders.bilibili_video.verify_audio_file",
                   return_value=(True, "ok")), \
             patch("downloaders.bilibili_video.cache_manager") as cache_mock, \
             patch("os.replace") as replace_mock:
            cache_mock.get_cached_file.return_value = None
            ok, result = video_dl.download("BV1xx", "ck", save_dir=str(tmp_path))
        assert ok, result
        assert replace_mock.called

    def test_part_filename_unique_per_process(self, video_dl, tmp_path):
        # 跨进程并发契约：临时名含 pid，两个进程不会写同一个 .part 文件
        captured = []
        core = {"id": 30216, "bandwidth": 66560, "codecs": "mp4a.40.2",
                "format": "dash", "timelength": 123456}
        for pid in (111, 222):
            with patch.object(video_dl, "get_audio_info", return_value=core), \
                 patch.object(video_dl, "get_audio_url",
                              return_value=("http://audio.example/a.m4s", core)), \
                 patch.object(video_dl, "download_audio",
                              side_effect=lambda u, c, f, _pid=pid: captured.append((_pid, f)) or (False, "boom")), \
                 patch("downloaders.bilibili_video.cache_manager") as cache_mock, \
                 patch("downloaders.bilibili_video.os.getpid", return_value=pid):
                cache_mock.get_cached_file.return_value = None
                video_dl.download("BV1xx", "ck", save_dir=str(tmp_path))
        assert len(captured) == MAX_ATTEMPTS * 2
        assert all(str(pid) in name for pid, name in captured)
        assert len({name for _, name in captured}) == len(captured)  # 所有临时名互不相同

    def test_verify_failed_part_file_is_removed(self, video_dl, tmp_path):
        # verify 失败后临时 part 文件必须被清理
        target = str(tmp_path / "a.m4s")
        p1, p2, p3 = self._patch_core(
            video_dl,
            {"id": 30216, "bandwidth": 66560, "codecs": "mp4a.40.2",
             "format": "dash", "timelength": 123456},
            [(True, target)] * MAX_ATTEMPTS,
        )
        with p1, p2, p3, \
             patch("downloaders.bilibili_video.verify_audio_file",
                   return_value=(False, "音频不完整")), \
             patch("downloaders.bilibili_video.cache_manager") as cache_mock, \
             patch("downloaders.bilibili_video.os.remove") as rm_mock:
            cache_mock.get_cached_file.return_value = None
            ok, result = video_dl.download("BV1xx", "ck", save_dir=str(tmp_path))
        assert not ok
        assert rm_mock.call_count == MAX_ATTEMPTS  # 每次 attempt 的 part 文件都被删

    def test_url_failure_retries_then_reports(self, video_dl, tmp_path):
        # URL 三连失败（网络瞬断）也应走满重试并报错
        with patch.object(video_dl, "get_audio_info",
                          return_value={"id": 30216, "bandwidth": 66560,
                                        "codecs": "mp4a.40.2", "format": "dash",
                                        "timelength": 123456}), \
             patch.object(video_dl, "get_audio_url", return_value=None) as url_mock, \
             patch.object(video_dl, "download_audio") as dl_mock, \
             patch("downloaders.bilibili_video.cache_manager") as cache_mock:
            cache_mock.get_cached_file.return_value = None
            ok, result = video_dl.download("BV1xx", "ck", save_dir=str(tmp_path))
        assert not ok
        assert "无法获取音频URL" in result
        assert url_mock.call_count == MAX_ATTEMPTS
        assert not dl_mock.called

    def test_good_cached_file_short_circuits(self, video_dl, tmp_path):
        good_cache = tmp_path / "cached.m4s"
        good_cache.write_bytes(b"fine")
        p1, _, _ = self._patch_core(
            video_dl,
            {"id": 30216, "bandwidth": 66560, "codecs": "mp4a.40.2",
             "format": "dash", "timelength": 123456},
            [],
        )
        with p1, \
             patch.object(video_dl, "download_audio") as dl_mock, \
             patch("downloaders.bilibili_video.verify_audio_file",
                   return_value=(True, "ok")), \
             patch("downloaders.bilibili_video.cache_manager") as cache_mock:
            cache_mock.get_cached_file.return_value = str(good_cache)
            ok, result = video_dl.download("BV1xx", "ck", save_dir=str(tmp_path))
        assert ok
        assert result["file_path"] == str(good_cache)
        assert not dl_mock.called

    def test_cache_without_timelength_still_verified_for_decodability(self, video_dl, tmp_path):
        good_cache = tmp_path / "cached.m4s"
        good_cache.write_bytes(b"fine")
        p1, _, _ = self._patch_core(
            video_dl,
            {"id": 30216, "bandwidth": 66560, "codecs": "mp4a.40.2", "format": "dash"},
            [],
        )
        with p1, \
             patch("downloaders.bilibili_video.verify_audio_file",
                   return_value=(True, "ok")) as verify_mock, \
             patch("downloaders.bilibili_video.cache_manager") as cache_mock:
            cache_mock.get_cached_file.return_value = str(good_cache)
            ok, _ = video_dl.download("BV1xx", "ck", save_dir=str(tmp_path))
        assert ok
        # 无 timelength 时 expected_duration_ms 应为 None（仅可解码性校验）
        assert verify_mock.call_args.kwargs.get("expected_duration_ms") is None


class TestVideoTimelengthExtraction:
    def test_get_audio_url_extracts_timelength(self, video_dl):
        resp = MagicMock()
        resp.text = VIDEO_HTML
        resp.raise_for_status = lambda: None
        with patch("downloaders.bilibili_video.requests.get", return_value=resp):
            result = video_dl.get_audio_url("BV1xx", "ck")
        url, info = result
        assert info["timelength"] == 123456

    def test_get_audio_info_carries_timelength(self, video_dl):
        resp = MagicMock()
        resp.text = VIDEO_HTML
        resp.raise_for_status = lambda: None
        with patch("downloaders.bilibili_video.requests.get", return_value=resp):
            info = video_dl.get_audio_info("BV1xx", "ck")
        assert info["timelength"] == 123456



# ---------- episode 下载器 ----------

class TestEpisodeDownloadFlow:
    def _patch_core(self, downloader, audio_info, download_results):
        p1 = patch.object(downloader, "get_audio_info", return_value=audio_info)
        p2 = patch.object(downloader, "get_audio_url",
                          return_value=("http://audio.example/b.m4s", audio_info))
        p3 = patch.object(downloader, "download_audio", side_effect=download_results)
        return p1, p2, p3

    def test_verification_failure_retries_then_succeeds(self, episode_dl, tmp_path):
        target = str(tmp_path / "b.m4s")
        p1, p2, p3 = self._patch_core(
            episode_dl,
            {"id": 30280, "bandwidth": 66560, "codecs": "mp4a.40.2",
             "format": "dash", "timelength": 234567},
            [(True, target)] * MAX_ATTEMPTS,
        )
        with p1, p2, p3 as dl_mock, \
             patch("downloaders.bilibili_episode.verify_audio_file",
                   side_effect=[(False, "时长不足"), (False, "时长不足"), (True, "ok")]), \
             patch("downloaders.bilibili_episode.cache_manager") as cache_mock, \
             patch("os.replace"):
            cache_mock.get_cached_file.return_value = None  # 无缓存，走下载
            ok, result = episode_dl.download("2289525", "ck", save_dir=str(tmp_path))
        assert ok, result
        assert dl_mock.call_count == MAX_ATTEMPTS
        assert cache_mock.save_to_cache.called

    def test_bad_cached_file_is_removed_and_redownloaded(self, episode_dl, tmp_path):
        bad_cache = tmp_path / "cached.m4s"
        bad_cache.write_bytes(b"truncated")
        target = str(tmp_path / "b.m4s")
        p1, p2, p3 = self._patch_core(
            episode_dl,
            {"id": 30280, "bandwidth": 66560, "codecs": "mp4a.40.2",
             "format": "dash", "timelength": 234567},
            [(True, target)],
        )
        with p1, p2, p3 as dl_mock, \
             patch("downloaders.bilibili_episode.verify_audio_file",
                   side_effect=[(False, "时长不足"), (True, "ok")]), \
             patch("downloaders.bilibili_episode.cache_manager") as cache_mock, \
             patch("os.replace"):
            cache_mock.get_cached_file.return_value = str(bad_cache)
            ok, result = episode_dl.download("2289525", "ck", save_dir=str(tmp_path))
        assert ok, result
        assert not bad_cache.exists()
        assert dl_mock.called

    def test_size_mismatch_fails_and_removes_file(self, episode_dl, tmp_path):
        target = tmp_path / "b.m4s"
        chunks = [b"x" * 1000] * 5
        with patch("downloaders.bilibili_episode.requests.get",
                   return_value=make_response(chunks, content_length=10000)):
            ok, detail = episode_dl.download_audio("http://x", "ck", str(target))
        assert not ok
        assert "不完整" in detail
        assert not target.exists()

    def test_network_incomplete_does_not_hint_cookie(self, episode_dl, tmp_path):
        # 纯网络中断（下载不完整）不该被误归因为 cookie 失效
        target = str(tmp_path / "b.m4s")
        p1, p2, p3 = self._patch_core(
            episode_dl,
            {"id": 30280, "bandwidth": 66560, "codecs": "mp4a.40.2",
             "format": "dash", "timelength": 234567},
            [(False, "下载失败: 下载不完整: 已下载 5 字节，期望 100 字节")] * MAX_ATTEMPTS,
        )
        with p1, p2, p3, \
             patch("downloaders.bilibili_episode.cache_manager") as cache_mock:
            cache_mock.get_cached_file.return_value = None
            ok, result = episode_dl.download("2289525", "ck", save_dir=str(tmp_path))
        assert not ok
        assert "cookie" not in result.lower()

    def test_verify_failed_part_file_is_removed(self, episode_dl, tmp_path):
        # 与 video 侧对等：verify 失败后临时 part 文件必须被清理
        target = str(tmp_path / "b.m4s")
        p1, p2, p3 = self._patch_core(
            episode_dl,
            {"id": 30280, "bandwidth": 66560, "codecs": "mp4a.40.2",
             "format": "dash", "timelength": 234567},
            [(True, target)] * MAX_ATTEMPTS,
        )
        with p1, p2, p3, \
             patch("downloaders.bilibili_episode.verify_audio_file",
                   return_value=(False, "音频不完整")), \
             patch("downloaders.bilibili_episode.cache_manager") as cache_mock, \
             patch("downloaders.bilibili_episode.os.remove") as rm_mock:
            cache_mock.get_cached_file.return_value = None
            ok, result = episode_dl.download("2289525", "ck", save_dir=str(tmp_path))
        assert not ok
        assert rm_mock.call_count == MAX_ATTEMPTS

    def test_url_failure_retries_then_reports(self, episode_dl, tmp_path):
        # episode 侧同样：URL 三连失败走满重试并报错
        with patch.object(episode_dl, "get_audio_info",
                          return_value={"id": 30280, "bandwidth": 66560,
                                        "codecs": "mp4a.40.2", "format": "dash",
                                        "timelength": 234567}), \
             patch.object(episode_dl, "get_audio_url", return_value=None) as url_mock, \
             patch.object(episode_dl, "download_audio") as dl_mock, \
             patch("downloaders.bilibili_episode.cache_manager") as cache_mock:
            cache_mock.get_cached_file.return_value = None
            ok, result = episode_dl.download("2289525", "ck", save_dir=str(tmp_path))
        assert not ok
        assert "无法获取音频URL" in result
        assert url_mock.call_count == MAX_ATTEMPTS
        assert not dl_mock.called


class TestEpisodeTimelengthExtraction:
    def test_get_audio_url_extracts_timelength(self, episode_dl):
        resp = MagicMock()
        resp.text = EPISODE_HTML
        resp.raise_for_status = lambda: None
        with patch("downloaders.bilibili_episode.requests.get", return_value=resp):
            result = episode_dl.get_audio_url("2289525", "ck")
        url, info = result
        assert info["timelength"] == 234567
