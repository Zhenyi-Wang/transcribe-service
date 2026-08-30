"""下载完整性校验测试

覆盖两层防护：
1. integrity 模块：ffprobe 末包时长校验（真实 ffmpeg 生成的音频文件）
2. 下载器：content-length 比对、验证失败重试、坏缓存拒绝（mock 网络与下载）
"""
import os
import subprocess
import sys
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downloaders.integrity import verify_audio_file, get_audio_end_time, FfprobeUnavailableError


# ---------- 真实音频 fixture ----------

def _make_m4s(path, seconds):
    """用 ffmpeg 生成指定时长的 fMP4(m4s) 音频，模拟 B 站 DASH 音频"""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         f"sine=frequency=440:duration={seconds}",
         "-c:a", "aac", "-movflags", "dash", "-f", "mp4", str(path), "-y"],
        check=True,
    )


@pytest.fixture(scope="module")
def full_audio(tmp_path_factory):
    """完整 10 秒音频"""
    path = tmp_path_factory.mktemp("audio") / "full.m4s"
    _make_m4s(path, 10)
    return str(path)


@pytest.fixture(scope="module")
def truncated_audio(tmp_path_factory, full_audio):
    """截断到 1/5 大小的音频（模拟下载中断）"""
    path = tmp_path_factory.mktemp("audio") / "truncated.m4s"
    size = os.path.getsize(full_audio) // 5
    with open(full_audio, "rb") as f:
        data = f.read(size)
    with open(path, "wb") as f:
        f.write(data)
    return str(path)


class TestFfprobeFailurePaths:
    def test_ffprobe_timeout_raises_unavailable(self, tmp_path):
        import subprocess
        target = tmp_path / "a.m4s"
        target.write_bytes(b"x")
        with patch("downloaders.integrity.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=120)):
            with pytest.raises(FfprobeUnavailableError):
                get_audio_end_time(str(target))

    def test_ffprobe_missing_raises_unavailable(self, tmp_path):
        target = tmp_path / "a.m4s"
        target.write_bytes(b"x")
        with patch("downloaders.integrity.subprocess.run",
                   side_effect=OSError("ffprobe not found")):
            with pytest.raises(FfprobeUnavailableError):
                get_audio_end_time(str(target))

    def test_ffprobe_unavailable_skips_verification(self, tmp_path):
        # 环境故障（ffprobe 不可用）不判文件失败——避免误删好缓存
        target = tmp_path / "a.m4s"
        target.write_bytes(b"x")
        with patch("downloaders.integrity.subprocess.run",
                   side_effect=OSError("ffprobe not found")):
            ok, detail = verify_audio_file(str(target), expected_duration_ms=10000)
        assert ok, detail
        assert "跳过" in detail

    def test_na_duration_time_does_not_drop_packet(self, tmp_path):
        # duration_time 为 N/A 时不应连同有效 pts 一起丢包
        target = tmp_path / "a.m4s"
        target.write_bytes(b"x")
        fake = MagicMock()
        fake.stdout = "0.0,0.021333\n10.0,N/A\n"
        with patch("downloaders.integrity.subprocess.run", return_value=fake):
            end = get_audio_end_time(str(target))
        assert end == 10.0


# ---------- get_audio_end_time ----------

class TestGetAudioEndTime:
    def test_full_audio_reports_real_duration(self, full_audio):
        end = get_audio_end_time(full_audio)
        assert end is not None
        assert 9.5 <= end <= 10.5

    def test_truncated_audio_reports_short_duration(self, truncated_audio, full_audio):
        assert get_audio_end_time(full_audio) > 8
        assert get_audio_end_time(truncated_audio) < 3

    def test_corrupt_file_returns_none(self, tmp_path):
        corrupt = tmp_path / "corrupt.m4s"
        corrupt.write_bytes(os.urandom(4096))
        assert get_audio_end_time(str(corrupt)) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert get_audio_end_time(str(tmp_path / "nope.m4s")) is None


# ---------- verify_audio_file ----------

class TestVerifyAudioFile:
    def test_full_audio_passes_with_expected_duration(self, full_audio):
        ok, detail = verify_audio_file(full_audio, expected_duration_ms=10000)
        assert ok, detail

    def test_truncated_audio_fails_with_expected_duration(self, truncated_audio):
        ok, detail = verify_audio_file(truncated_audio, expected_duration_ms=10000)
        assert not ok
        assert "时长" in detail or "duration" in detail.lower()

    def test_corrupt_file_fails(self, tmp_path):
        corrupt = tmp_path / "corrupt.m4s"
        corrupt.write_bytes(os.urandom(4096))
        ok, detail = verify_audio_file(str(corrupt), expected_duration_ms=10000)
        assert not ok

    def test_missing_file_fails(self, tmp_path):
        ok, detail = verify_audio_file(str(tmp_path / "nope.m4s"), expected_duration_ms=10000)
        assert not ok

    def test_without_expected_duration_only_checks_decodability(self, truncated_audio, full_audio):
        # 无期望时长时只能检出完全损坏，可解码的截断文件不判失败
        ok, _ = verify_audio_file(full_audio)
        assert ok, _
        ok, _ = verify_audio_file(truncated_audio)
        assert ok, _

    def test_tolerance_allows_small_drift(self, full_audio):
        # 实际 ~10.02s：期望 10.5s（差 <3s）应通过，期望 20s（差 >3s）应失败
        ok, detail = verify_audio_file(full_audio, expected_duration_ms=10500)
        assert ok, detail
        ok, detail = verify_audio_file(full_audio, expected_duration_ms=20000)
        assert not ok, detail
