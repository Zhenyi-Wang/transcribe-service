"""抖音下载器单测：slidesinfo 解析/图集防御/选档/ffmpeg 调用/缓存"""
import unittest
from unittest.mock import patch, MagicMock
from downloaders.douyin import DouyinDownloader, SLIDESINFO_URL


def _detail(images=None, bit_rate=None, duration=90000):
    play = {"uri": "v0d00fg10000c...", "url_list": ["https://v26.douyinvod.com/x/video/tos/..."]}
    return {
        "aweme_id": "7376234567890123456",
        "desc": "测试视频标题",
        "duration": duration,
        "author": {"nickname": "测试作者", "sec_uid": "MS4wLjABAAAAtest"},
        "video": {"play_addr": play, "bit_rate": bit_rate or [
            {"bit_rate": 2000000, "play_addr": play},
            {"bit_rate": 800000, "play_addr": play},
        ]},
        **({"images": images} if images is not None else {}),
    }


class TestFetchAwemeDetail(unittest.TestCase):
    @patch("downloaders.douyin.requests.get")
    def test_normal_video(self, mock_get):
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"aweme_details": [_detail()], "filter_list": []}
        mock_get.return_value = resp
        d = DouyinDownloader().fetch_aweme_detail("7376234567890123456")
        self.assertIsNotNone(d)
        self.assertEqual(d["desc"], "测试视频标题")
        # 请求参数：aweme_ids 数组、移动 UA、无 Referer
        args, kwargs = mock_get.call_args
        self.assertIn("aweme_ids=%5B%227376234567890123456%22%5D", args[0]
                      )  # requests 自动编码 [\"id\"]
        self.assertNotIn("Referer", kwargs["headers"])
        self.assertIn("iPhone", kwargs["headers"]["User-Agent"])

    @patch("downloaders.douyin.requests.get")
    def test_empty_then_retry_with_request_source(self, mock_get):
        empty = MagicMock(status_code=200)
        empty.json.return_value = {"aweme_details": [], "filter_list": []}
        ok = MagicMock(status_code=200)
        ok.json.return_value = {"aweme_details": [_detail()], "filter_list": []}
        mock_get.side_effect = [empty, ok]
        d = DouyinDownloader().fetch_aweme_detail("7376234567890123456")
        self.assertIsNotNone(d)
        self.assertEqual(mock_get.call_count, 2)
        self.assertIn("request_source=200", mock_get.call_args_list[1][0][0])

    @patch("downloaders.douyin.requests.get")
    def test_deleted_video_returns_none(self, mock_get):
        empty = MagicMock(status_code=200)
        empty.json.return_value = {"aweme_details": [], "filter_list": [{"reason": 8}]}
        mock_get.return_value = empty
        self.assertIsNone(DouyinDownloader().fetch_aweme_detail("123"))


class TestSelectPlayUrl(unittest.TestCase):
    def test_lowest_bitrate_selected(self):
        d = DouyinDownloader()
        play = d._select_play_url(_detail())
        self.assertEqual(play["url"], "https://v26.douyinvod.com/x/video/tos/...")

    def test_fallback_to_play_addr(self):
        detail = _detail(bit_rate=[])
        detail["video"]["bit_rate"] = []
        play = DouyinDownloader()._select_play_url(detail)
        self.assertIsNotNone(play["url"])


class TestDownloadDouyinAudio(unittest.TestCase):
    @patch("downloaders.douyin.cache_manager")
    @patch("downloaders.douyin.verify_audio_file", return_value=(True, "ok"))
    @patch("downloaders.douyin.subprocess.run")
    @patch("downloaders.douyin.DouyinDownloader.fetch_aweme_detail")
    def test_success_flow(self, mock_detail, mock_run, mock_verify, mock_cache):
        import tempfile, os
        mock_detail.return_value = _detail()
        mock_cache.get_cached_file.return_value = None

        def _save(url, file_path, *a, **kw):
            return file_path
        mock_cache.save_to_cache.side_effect = _save
        # ffmpeg 实际产出文件
        def _run(cmd, **kw):
            out = cmd[cmd.index("-y") + 1]
            with open(out, "wb") as f:
                f.write(b"fake-m4a")
            return MagicMock(returncode=0)
        mock_run.side_effect = _run

        with tempfile.TemporaryDirectory() as td:
            ok, result = DouyinDownloader().download_douyin_audio("7376234567890123456", save_dir=td)
        self.assertTrue(ok)
        self.assertTrue(result["file_path"].endswith(".m4a"))
        self.assertEqual(result["audio_id"], "dy")
        # ffmpeg 命令：-vn -c:a copy、UA、无 referer 头
        cmd = mock_run.call_args[0][0]
        self.assertIn("-vn", cmd)
        i = cmd.index("-c:a"); self.assertEqual(cmd[i + 1], "copy")
        i = cmd.index("-user_agent"); self.assertIn("iPhone", cmd[i + 1])
        self.assertNotIn("-referer", cmd)

    @patch("downloaders.douyin.cache_manager")
    @patch("downloaders.douyin.DouyinDownloader.fetch_aweme_detail")
    def test_gallery_rejected(self, mock_detail, mock_cache):
        mock_detail.return_value = _detail(images=[{"url_list": ["https://x/1.jpg"]}])
        ok, msg = DouyinDownloader().download_douyin_audio("7376234567890123456")
        self.assertFalse(ok)
        self.assertIn("图集", str(msg))

    @patch("downloaders.douyin.cache_manager")
    @patch("downloaders.douyin.DouyinDownloader.fetch_aweme_detail")
    def test_not_found(self, mock_detail, mock_cache):
        mock_detail.return_value = None
        ok, msg = DouyinDownloader().download_douyin_audio("123")
        self.assertFalse(ok)


    @patch("downloaders.douyin.cache_manager")
    @patch("downloaders.douyin.verify_audio_file", return_value=(True, "ok"))
    @patch("downloaders.douyin.subprocess.run")
    @patch("downloaders.douyin.DouyinDownloader.fetch_aweme_detail")
    def test_retry_reparses_play_url_and_removes_failed_partial_file(
            self, mock_detail, mock_run, mock_verify, mock_cache):
        import tempfile, os
        first_detail = _detail()
        second_detail = _detail()
        second_detail["video"]["play_addr"]["url_list"] = ["https://cdn/new-video"]
        for bitrate in second_detail["video"]["bit_rate"]:
            bitrate["play_addr"]["url_list"] = ["https://cdn/new-video"]
        mock_detail.side_effect = [first_detail, second_detail]
        mock_cache.get_cached_file.return_value = None
        mock_cache.save_to_cache.side_effect = lambda url, file_path, *a, **kw: file_path
        partial_paths = []

        def _run(cmd, **kw):
            out = cmd[cmd.index("-y") + 1]
            partial_paths.append(out)
            if mock_run.call_count == 1:
                with open(out, "wb") as f:
                    f.write(b"partial")
                return MagicMock(returncode=1, stderr=b"failed")
            with open(out, "wb") as f:
                f.write(b"fake-m4a")
            return MagicMock(returncode=0, stderr=b"")

        mock_run.side_effect = _run
        with tempfile.TemporaryDirectory() as td:
            ok, result = DouyinDownloader().download_douyin_audio("7376234567890123456", save_dir=td)
            self.assertTrue(ok)
            self.assertEqual(mock_detail.call_count, 2)
            self.assertEqual(result["audio_url"], "https://cdn/new-video")
            self.assertFalse(os.path.exists(partial_paths[0]))

    @patch("downloaders.douyin.cache_manager")
    @patch("downloaders.douyin.verify_audio_file", return_value=(True, "ok"))
    @patch("downloaders.douyin.subprocess.run")
    @patch("downloaders.douyin.DouyinDownloader.fetch_aweme_detail")
    def test_cached_file_returns_cached_url_without_ffmpeg(
            self, mock_detail, mock_run, mock_verify, mock_cache):
        import tempfile, os
        mock_detail.return_value = _detail()
        with tempfile.TemporaryDirectory() as td:
            cached_file = os.path.join(td, "cached.m4a")
            with open(cached_file, "wb") as f:
                f.write(b"cached-audio")
            mock_cache.get_cached_file.return_value = cached_file
            ok, result = DouyinDownloader().download_douyin_audio("7376234567890123456", save_dir=td)

        self.assertTrue(ok)
        self.assertEqual(result["audio_url"], "cached://7376234567890123456")
        mock_run.assert_not_called()

    @patch("downloaders.douyin.cache_manager")
    @patch("downloaders.douyin.verify_audio_file", return_value=(True, "ok"))
    @patch("downloaders.douyin.subprocess.run")
    @patch("downloaders.douyin.DouyinDownloader.fetch_aweme_detail")
    def test_timeout_removes_partial_file(self, mock_detail, mock_run, mock_verify, mock_cache):
        import tempfile, os, subprocess
        mock_detail.side_effect = [_detail(), _detail()]
        mock_cache.get_cached_file.return_value = None
        mock_cache.save_to_cache.side_effect = lambda url, file_path, *a, **kw: file_path
        partial_paths = []

        def _run(cmd, **kw):
            out = cmd[cmd.index("-y") + 1]
            partial_paths.append(out)
            if mock_run.call_count == 1:
                with open(out, "wb") as f:
                    f.write(b"partial")
                raise subprocess.TimeoutExpired(cmd, timeout=kw["timeout"])
            with open(out, "wb") as f:
                f.write(b"fake-m4a")
            return MagicMock(returncode=0, stderr=b"")

        mock_run.side_effect = _run
        with tempfile.TemporaryDirectory() as td:
            ok, _ = DouyinDownloader().download_douyin_audio("7376234567890123456", save_dir=td)
            self.assertTrue(ok)
            self.assertFalse(os.path.exists(partial_paths[0]))


if __name__ == "__main__":
    unittest.main()
