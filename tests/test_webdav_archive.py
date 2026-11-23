"""webdav_archive.move_to_processed 的行为测试"""
import os

import pytest

from webdav_archive import move_to_processed


@pytest.fixture
def webdav(tmp_path):
    """构造 inbox/processed 骨架，返回 (webdav_base, 相对路径工具)"""
    base = tmp_path / "webdav"
    (base / "inbox").mkdir(parents=True)
    (base / "processed").mkdir()
    return str(base)


def write(base, rel, content="x"):
    p = os.path.join(base, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(content)
    return p


def test_normal_move_moves_file(webdav):
    write(webdav, "inbox/youtube/ch/a.mp3")
    assert move_to_processed(webdav, "inbox/youtube/ch/a.mp3") is True
    assert os.path.isfile(os.path.join(webdav, "processed/youtube/ch/a.mp3"))
    assert not os.path.exists(os.path.join(webdav, "inbox/youtube/ch/a.mp3"))


def test_idempotent_skip_when_source_gone(webdav):
    # 重置任务场景：文件已在 processed，inbox 无此文件 → 跳过不报错
    assert move_to_processed(webdav, "inbox/youtube/ch/gone.mp3") is False


def test_skip_when_path_already_processed(webdav):
    # 回退读取场景：relative_path 已被改写为 processed/...，
    # 归档必须跳过，不得把文件移进 processed/processed 嵌套目录
    write(webdav, "processed/youtube/ch/a.mp3")
    assert move_to_processed(webdav, "processed/youtube/ch/a.mp3") is False
    assert os.path.isfile(os.path.join(webdav, "processed/youtube/ch/a.mp3"))
    assert not os.path.exists(os.path.join(webdav, "processed/processed"))


def test_skip_processed_with_leading_slash(webdav):
    # 防御：绝对路径形式传入时同样按终态处理
    write(webdav, "processed/youtube/ch/a.mp3")
    assert move_to_processed(webdav, "/processed/youtube/ch/a.mp3") is False
    assert not os.path.exists(os.path.join(webdav, "processed/processed"))


def test_avoid_overwrite_renames_existing(webdav):
    write(webdav, "inbox/youtube/ch/a.mp3", "new")
    write(webdav, "processed/youtube/ch/a.mp3", "old")
    assert move_to_processed(webdav, "inbox/youtube/ch/a.mp3") is True
    assert open(os.path.join(webdav, "processed/youtube/ch/a.mp3")).read() == "new"
    assert open(os.path.join(webdav, "processed/youtube/ch/a.bak1.mp3")).read() == "old"


def test_companion_subtitles_moved(webdav):
    write(webdav, "inbox/youtube/ch/a.mp3")
    write(webdav, "inbox/youtube/ch/a.srt")
    write(webdav, "inbox/youtube/ch/a.vtt")
    assert move_to_processed(webdav, "inbox/youtube/ch/a.mp3") is True
    assert os.path.isfile(os.path.join(webdav, "processed/youtube/ch/a.srt"))
    assert os.path.isfile(os.path.join(webdav, "processed/youtube/ch/a.vtt"))
    assert not os.path.exists(os.path.join(webdav, "inbox/youtube/ch/a.srt"))


def test_metadata_moved_once_per_dir(webdav):
    write(webdav, "inbox/youtube/ch/a.mp3")
    write(webdav, "inbox/youtube/ch/b.mp3")
    write(webdav, "inbox/youtube/ch/metadata.json", "{}")
    assert move_to_processed(webdav, "inbox/youtube/ch/a.mp3") is True
    assert os.path.isfile(os.path.join(webdav, "processed/youtube/ch/metadata.json"))
    # 第二个文件移动时 metadata 源已不在，静默跳过
    assert move_to_processed(webdav, "inbox/youtube/ch/b.mp3") is True


def test_empty_source_dir_removed(webdav):
    write(webdav, "inbox/youtube/ch/a.mp3")
    assert move_to_processed(webdav, "inbox/youtube/ch/a.mp3") is True
    assert not os.path.exists(os.path.join(webdav, "inbox/youtube/ch"))


def test_non_empty_source_dir_kept(webdav):
    write(webdav, "inbox/youtube/ch/a.mp3")
    write(webdav, "inbox/youtube/ch/b.mp3")
    assert move_to_processed(webdav, "inbox/youtube/ch/a.mp3") is True
    assert os.path.isdir(os.path.join(webdav, "inbox/youtube/ch"))


def test_failure_is_silent(webdav, monkeypatch):
    write(webdav, "inbox/youtube/ch/a.mp3")
    import webdav_archive as wa

    def boom(*a, **k):
        raise RuntimeError("mount dead")

    monkeypatch.setattr(wa.shutil, "move", boom)
    assert move_to_processed(webdav, "inbox/youtube/ch/a.mp3") is False
    # 源文件未被误删
    assert os.path.isfile(os.path.join(webdav, "inbox/youtube/ch/a.mp3"))
