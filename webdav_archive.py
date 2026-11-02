"""WebDAV 归档：转录成功后把文件从 inbox 移到 processed。

2026-11-02 改造：接替原 mryk24 NoteFlow「创建任务即移动」的职责
（该实现依赖 mryk 侧挂载且失败静默，挂载故障曾致 14 天积压无人知晓）。

语义与原 NoteFlow moveToProcessed 对齐：
- 转录成功才算数；移动失败只记日志，不影响转录返回（调用方再兜一层）
- 幂等：源文件不在 inbox（重置任务经 processed 回退命中的场景）直接跳过
- 防覆盖：processed 已有同名文件时改 .bak.N 保留旧件
- 伴移：同名字幕（.srt/.vtt/.ass/.ssa/.lrc）与同目录 metadata.json
- 源目录移空后删除该目录（不递归）
"""
import logging
import os
import shutil

logger = logging.getLogger(__name__)

COMPANION_EXTS = ("srt", "vtt", "ass", "ssa", "lrc")


def _avoid_overwrite(target: str) -> None:
    """目标已存在时把旧文件改名为 .bak.N。"""
    if not os.path.exists(target):
        return
    root, ext = os.path.splitext(target)
    n = 1
    while os.path.exists(f"{root}.bak{n}{ext}"):
        n += 1
    backup = f"{root}.bak{n}{ext}"
    os.rename(target, backup)
    logger.info(f"processed 已有同名文件，旧件改名: {target} -> {backup}")


def _to_processed_path(webdav_base: str, relative_path: str) -> str:
    """inbox/xxx -> processed/xxx；非 inbox 前缀原样映射。"""
    rel = relative_path[len("inbox/"):] if relative_path.startswith("inbox/") else relative_path
    return os.path.join(webdav_base, "processed", rel)


def move_to_processed(webdav_base: str, relative_path: str) -> bool:
    """转录成功后把 inbox 文件归档到 processed。返回是否执行了移动。

    任何失败都只记日志并返回 False，不抛出（归档失败不应影响转录结果）。
    """
    try:
        src = os.path.join(webdav_base, relative_path)
        if not os.path.isfile(src):
            # 源不在 inbox：已被移走或经 processed 回退读取，幂等跳过
            logger.info(f"归档跳过（源不在 inbox）: {relative_path}")
            return False

        dst = _to_processed_path(webdav_base, relative_path)
        dst_dir = os.path.dirname(dst)
        os.makedirs(dst_dir, exist_ok=True)

        _avoid_overwrite(dst)
        shutil.move(src, dst)
        logger.info(f"归档完成: {src} -> {dst}")

        # 伴移同名字幕
        src_root = os.path.splitext(src)[0]
        dst_root = os.path.splitext(dst)[0]
        for ext in COMPANION_EXTS:
            companion = f"{src_root}.{ext}"
            if os.path.isfile(companion):
                companion_dst = f"{dst_root}.{ext}"
                _avoid_overwrite(companion_dst)
                shutil.move(companion, companion_dst)
                logger.info(f"伴移字幕: {companion} -> {companion_dst}")

        # 伴移 metadata.json（同目录级，各文件转录完只会成功伴移一次）
        meta_src = os.path.join(os.path.dirname(src), "metadata.json")
        if os.path.isfile(meta_src):
            meta_dst = os.path.join(dst_dir, "metadata.json")
            _avoid_overwrite(meta_dst)
            shutil.move(meta_src, meta_dst)
            logger.info(f"伴移 metadata: {meta_src} -> {meta_dst}")

        # 源目录移空则删除（不递归）
        src_dir = os.path.dirname(src)
        if not os.listdir(src_dir):
            os.rmdir(src_dir)
            logger.info(f"源目录已空，删除: {src_dir}")
        return True
    except Exception as e:
        logger.error(f"归档失败（不影响转录结果）: {relative_path}: {e}")
        return False
