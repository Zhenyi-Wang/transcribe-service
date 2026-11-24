"""B 站 Cookie 本地持有与自同步（2026-11-24）。

真值源是生产 NoteFlowConfig（mryk24），本服务只本地持有副本：
- 执行器从本地文件读 Cookie（bili_cookie.local，chmod 600，不入 git）；
  /jobs 载荷不再携带（jobs.py 的剥离逻辑保留，作为回显安全的最后防线）
- 失效自愈链：下载失败 → nav 探测区分登录类/风控类 → ssh ops 通道从生产同步
  → 仍失效则触发无人值守刷新脚本（claude -p /update-bilibili-cookie），
  刷新仍失败由脚本自身 tellme 兜底（scripts/refresh_bili_cookie.sh）
- 防抖：无人值守刷新 6 小时内最多触发一次；从生产同步幂等且廉价，不设防抖

安全边界：home 机本可 ssh mryk（密钥持有者即 root 等价），本地落一份 Cookie
不扩大攻击面；Cookie 值永不写日志。
"""
import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import requests

from logger_config import setup_logger

logger = setup_logger("bili_cookie")

_ROOT = Path(__file__).resolve().parent
COOKIE_PATH = _ROOT / "bili_cookie.local"
REFRESH_MARKER_PATH = _ROOT / "bili_refresh.last"
REFRESH_SCRIPT = _ROOT / "scripts" / "refresh_bili_cookie.sh"
REFRESH_LOG = _ROOT / "logs" / "bili_refresh.log"
REFRESH_DEBOUNCE_SECONDS = 6 * 3600

# ops 本机直连通道（10002 仅 VPS 本机可达，免鉴权）；见 mryk24 conventions.md
_OPS_SYNC_CMD = [
    "ssh", "mryk", "curl -s -m 15 http://127.0.0.1:10002/api/noteflow/config/get",
]
_NAV_URL = "https://api.bilibili.com/x/web-interface/nav"
_NAV_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/",
}


def load_cookie() -> str:
    try:
        return COOKIE_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def save_cookie(cookie: str) -> None:
    COOKIE_PATH.write_text(cookie.strip(), encoding="utf-8")
    os.chmod(COOKIE_PATH, 0o600)


def sync_from_production() -> bool:
    """从生产 NoteFlowConfig 拉取最新 Cookie 落盘。Cookie 值永不写日志。"""
    try:
        proc = subprocess.run(_OPS_SYNC_CMD, capture_output=True, text=True,
                              timeout=25)
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.error(f"从生产同步 Cookie 失败（ssh/网络）: {e}")
        return False
    if proc.returncode != 0:
        logger.error(f"从生产同步 Cookie 失败（ssh 退出码 {proc.returncode}）")
        return False
    try:
        payload = json.loads(proc.stdout)
        cookie = payload["data"]["bilibili"]["cookie"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"从生产同步 Cookie 失败（响应解析）: {e}")
        return False
    if not cookie:
        logger.error("从生产同步 Cookie 失败：生产配置中 cookie 为空")
        return False
    save_cookie(cookie)
    logger.info(f"已从生产同步 Cookie（长度 {len(cookie)}，"
                f"含 SESSDATA: {'SESSDATA' in cookie}）")
    return True


def probe_login(cookie: str) -> bool:
    """nav 接口探测登录态。True = 已登录；探测异常按 False 处理。"""
    if not cookie:
        return False
    try:
        resp = requests.get(_NAV_URL, timeout=10,
                            headers={**_NAV_HEADERS, "Cookie": cookie})
        data = resp.json()
    except (requests.RequestException, ValueError) as e:
        logger.warning(f"nav 登录态探测失败（按未登录处理）: {e}")
        return False
    return bool(data.get("code") == 0 and data.get("data", {}).get("isLogin"))


def trigger_unattended_refresh() -> bool:
    """触发无人值守刷新脚本，6 小时防抖。返回是否真的触发了。"""
    now = time.time()
    try:
        last = REFRESH_MARKER_PATH.read_text().strip()
        if last and now - float(last) < REFRESH_DEBOUNCE_SECONDS:
            logger.info("无人值守刷新处于防抖窗口内，跳过触发")
            return False
    except OSError:
        pass
    if not REFRESH_SCRIPT.exists():
        logger.error(f"刷新脚本不存在: {REFRESH_SCRIPT}")
        return False
    REFRESH_LOG.parent.mkdir(exist_ok=True)
    with open(REFRESH_LOG, "ab") as lf:
        subprocess.Popen(["bash", str(REFRESH_SCRIPT)], stdout=lf, stderr=lf,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    REFRESH_MARKER_PATH.write_text(str(now))
    logger.info("已触发无人值守 Cookie 刷新（后台执行，日志 logs/bili_refresh.log）")
    return True


def ensure_fresh_cookie(video_id: str = "") -> tuple[str, str]:
    """下载失败后的登录态自愈入口。返回 (cookie, 说明)。

    - 现有 Cookie 登录有效 → 原样返回（失败与登录态无关，疑似风控/其他）
    - 登录失效 → 从生产同步，有效则返回新 Cookie
    - 同步后仍失效 → 触发无人值守刷新（本次任务大概率仍失败，由脚本 tellme
      兜底；刷新成功后的任务用上新 Cookie）
    """
    current = load_cookie()
    if probe_login(current):
        return current, "cookie 有效"
    logger.info(f"登录态探测未通过，尝试从生产同步（{video_id}）")
    sync_from_production()
    fresh = load_cookie()
    if fresh and probe_login(fresh):
        return fresh, "已从生产同步新 cookie"
    trigger_unattended_refresh()
    return (fresh or current), "本地与生产 cookie 均失效，已触发无人值守刷新"


async def bili_cookie_probe_loop():
    """每日主动探测：登录失效当天自愈，不等任务撞墙。"""
    while True:
        await asyncio.sleep(24 * 3600)
        try:
            cookie = load_cookie()
            if not cookie:
                sync_from_production()
                continue
            if not probe_login(cookie):
                logger.info("每日探测发现 Cookie 失效，尝试从生产同步")
                sync_from_production()
                if not probe_login(load_cookie()):
                    trigger_unattended_refresh()
        except Exception as e:  # 后台任务绝不因探测异常退出
            logger.error(f"每日 Cookie 探测异常: {e}")
