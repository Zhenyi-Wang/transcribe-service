"""server.py /jobs 端点与拦截集成测试（Task 8）。

spec：docs/superpowers/specs/2026-11-10-transcribe-job-queue-design.md
§4（API 契约全部）/§3（鉴权与暂停拦截约束）/§6 执行模型末段（装配）。

覆盖（brief Step 1 清单 + Review Focus #2 Task 10 补项）：
1.  四源提交 → 终态全链路（桩执行体即时完成）
2.  result 与现有端点同构（webdav/upload 全等；bilibili/douyin 仅 + timing.download）
3.  GET/DELETE 未知 id 404（评审 R4A 勘误：422 仅限提交格式不合法）
4.  暂停提交 503（/jobs 与 /jobs/upload；GET /jobs 不拦）
5.  query 命中 + missing + 单批 >500 静默截断按 missing（评审 R5A）
6.  GET /jobs 终态 cap 100 + cookie 不回显
7.  提交 job_id 不合法 422 / Pydantic 字段错 422 / 未知 source 422
8.  DELETE 三态：queued 即时 cancelled / running → cancelling → cancelled /
    终态幂等返回当前态
9.  timing.download 注入（note_download_time → 终态映射）
10. upload 同 id 重放：新上传文件立即删除
11. 队列满 429：Retry-After 响应头与 body 字段双写（评审 R1A）
12. cleanup_stale_tmp_files(protected)：活跃路径与其 .job-* 硬链接别名跳过、
    未登记超龄 .job-* 照常清理（评审 R2A 修正措辞）
13. 取消后 webdav 不归档 + 缓存写入零发生 + result null（Review Focus #2）
14. webdav 失败不归档
15. 上游暂停集成：UpstreamPausedError 穿透真实执行体 → error+paused 三键、
    不归档 + 快照暂停期 worker 自冷（Task 10，spec §9）
16. fixwave F3：归档探针（job 取消标志置位 → 跳过归档、不进提交点）+
    归档提交点（归档开始后取消到达 → done 胜出取消，文件已归档、result 一致）
"""
import asyncio
import json
import os
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server
import webdav_archive
from backends.asr_engine_backend import UpstreamPausedError
from downloaders.base import DownloadCancelled
from jobs import Job, JobManager
from pause_manager import PauseManager
from transcribe import JobCancelledError


# ---------- 桩与公共装置 ----------

class StubConfig:
    def __init__(self, **values):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


STUB_RESULT = {
    "status": "success", "type": "srt", "version": "1.0", "lang": "zh",
    "body": [{"text": "你好", "start": 0.0, "end": 1.0}],
    "rtf": 0.5, "audio_duration": 8.0, "speakers": ["甲"],
    "timing": {"total": 1.23},
}


def _h():
    return {"Authorization": f"Bearer {server.config.api_token}"} \
        if server.config.api_token else {}


@pytest.fixture
def api(request, tmp_path, monkeypatch):
    """隔离的 /jobs API 环境：暂停态/预加载/缓存清理全部打桩，
    JobManager 换新实例（jobs.max_queue 可经 parametrize 注入）。"""
    max_queue = getattr(request, "param", 1000)
    monkeypatch.chdir(tmp_path)

    pm = PauseManager(state_file=tmp_path / "pause.json", notify_fn=lambda: None)
    monkeypatch.setattr(server, "pause_manager", pm)
    monkeypatch.setattr(server, "_pause_asr_engine", lambda hours: "mocked")
    monkeypatch.setattr(server, "_resume_asr_engine", lambda: "mocked")
    monkeypatch.setattr(server, "_release_gpu_for_pause", lambda gen: {"mocked": True})
    monkeypatch.setattr(server, "_query_asr_paused", lambda: False)
    monkeypatch.setattr(server.manager, "load_model_if_needed", lambda: None, raising=False)
    monkeypatch.setattr(server.cache_manager, "cleanup_expired_cache", lambda: None,
                        raising=False)

    # 转录桩：记录调用；mode=block 时阻塞到放行；放行后先探取消（与真实
    # process_transcription 的「转录返回后探针 → 缓存写入前」次序同构），
    # 命中抛 JobCancelledError，未命中先记缓存再返回结果。
    calls = []
    cache_writes = []
    state = {"mode": "instant", "gate": threading.Event()}

    cache_recorder = lambda *a, **kw: cache_writes.append((a, kw))

    async def fake_process(audio_file_path, original_filename=None, *args, **kw):
        calls.append({"path": audio_file_path, "name": original_filename,
                      "args": args, "kw": kw})
        if state["mode"] == "block":
            await asyncio.get_running_loop().run_in_executor(
                None, state["gate"].wait, 10)
        if state["mode"] == "paused":
            # asr-engine 返回 503 paused body → UpstreamPausedError（Task 10 上游暂停集成）
            raise UpstreamPausedError("asr engine paused",
                                      resume_at="2026-11-10T10:00:00",
                                      paused_at="2026-11-10T09:00:00")
        probe = kw.get("cancelled")
        if probe is not None and probe():
            raise JobCancelledError("cancelled before cache write")
        cache_recorder(audio_file_path)
        if state["mode"] == "error":
            return {"status": "error", "message": "stub error"}
        return dict(STUB_RESULT)

    monkeypatch.setattr(server.transcription_service, "process_transcription",
                        fake_process)
    monkeypatch.setattr(server.cache_manager, "save_transcript_to_cache",
                        cache_recorder, raising=False)

    # webdav 根指向测试目录（绝不能打到真实 /mnt/webdav）+ 预置全链路用文件
    real_get = server.config.get

    def fake_config_get(key, default=None):
        if key == "webdav.base_path":
            return str(tmp_path / "webdav")
        return real_get(key, default)

    monkeypatch.setattr(server.config, "get", fake_config_get)
    inbox_file = tmp_path / "webdav" / "inbox" / "x.mp3"
    inbox_file.parent.mkdir(parents=True)
    inbox_file.write_bytes(b"RIFF")

    # 下载桩：先报进度，再在取消门上等待（块边界语义），随后落一个真实小文件
    dl_files = []

    def fake_bili_download(video_id, cookie, save_dir="tmp", page=1,
                           progress_cb=None, cancel_flag=None):
        if progress_cb:
            progress_cb(50, 100)
        if cancel_flag is not None and cancel_flag.wait(timeout=5):
            raise DownloadCancelled(f"下载已取消: {video_id}")
        f = Path(save_dir) / f"{video_id}_p{page}_audio_1.m4a"
        f.write_bytes(b"RIFF")
        dl_files.append(str(f))
        return True, {"file_path": str(f), "audio_url": "https://example/a",
                      "audio_id": "42"}

    def fake_douyin_download(aweme_id, save_dir="tmp"):
        f = Path(save_dir) / f"{aweme_id}.mp3"
        f.write_bytes(b"RIFF")
        dl_files.append(str(f))
        return True, {"file_path": str(f), "audio_url": "https://example/d",
                      "audio_id": "7"}

    monkeypatch.setattr(server.downloader, "download_bilibili_audio",
                        fake_bili_download)
    monkeypatch.setattr(server.downloader_douyin, "download_douyin_audio",
                        fake_douyin_download)

    # diarize_only / merge 桩
    diar_calls = []

    async def fake_diarize_only(path, video_id=None, download_time=0.0,
                                wait_deadline=0.0, stage_cb=None):
        diar_calls.append({"fn": "only", "path": path, "video_id": video_id,
                           "wait_deadline": wait_deadline, "stage_cb": stage_cb})
        return {"status": "success", "video_id": video_id, "turns": [],
                "speakers": [], "timing": {"download": download_time, "total": 0.1}}

    async def fake_diarize_merge(body, path, video_id=None, download_time=0.0,
                                 wait_deadline=0.0, stage_cb=None):
        diar_calls.append({"fn": "merge", "path": path, "video_id": video_id,
                           "wait_deadline": wait_deadline, "stage_cb": stage_cb})
        return {"status": "success", "video_id": video_id, "body": body,
                "speakers": [], "annotated": True, "reason": None,
                "timing": {"download": download_time, "total": 0.1}}

    monkeypatch.setattr(server, "diarize_only", fake_diarize_only)
    monkeypatch.setattr(server, "diarize_merge_subtitle", fake_diarize_merge)

    # JobManager 隔离实例 + 六注入点装配
    jm = JobManager(StubConfig(**{"jobs.max_queue": max_queue,
                                  "jobs.retry_after_seconds": 45}))
    server._setup_job_manager(jm)
    monkeypatch.setattr(server, "job_manager", jm)

    with TestClient(server.app) as client:
        yield SimpleNS(client=client, jm=jm, calls=calls, state=state,
                       dl_files=dl_files, diar_calls=diar_calls,
                       cache_writes=cache_writes, tmp_path=tmp_path)
        state["gate"].set()   # 放行所有阻塞桩，保证 shutdown 干净收尾


class SimpleNS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ---------- 17. Cookie 本地持有（2026-11-24 /jobs 停传 Cookie）----------

def test_bilibili_loads_local_cookie(api, monkeypatch):
    """载荷不再携带 Cookie：执行器应从本地副本加载（回归 /jobs 丢 Cookie 缺陷）。"""
    captured = []

    def cap_download(video_id, cookie, save_dir="tmp", page=1,
                     progress_cb=None, cancel_flag=None):
        captured.append(cookie)
        f = Path(api.tmp_path) / f"{video_id}_p{page}_audio.m4a"
        f.write_bytes(b"RIFF")
        return True, {"file_path": str(f), "audio_url": "https://example/a",
                      "audio_id": "42"}

    monkeypatch.setattr(server.downloader, "download_bilibili_audio", cap_download)
    monkeypatch.setattr(server.bili_cookie_manager, "load_cookie",
                        lambda: "LOCAL_CK")
    r = submit_json(api.client, {"source": "bilibili", "bvid": "BVlocalck"})
    assert r.status_code == 202
    body = wait_terminal(api.client, r.json()["job_id"])
    assert body["status"] == "done"
    assert captured == ["LOCAL_CK"]


def test_bilibili_self_heal_retries_with_refreshed_cookie(api, monkeypatch):
    """下载失败 → 自愈产出新 Cookie → 自动重试一次并成功。"""
    calls_ = []
    state = {"n": 0}

    def flaky_download(video_id, cookie, save_dir="tmp", page=1,
                       progress_cb=None, cancel_flag=None):
        calls_.append(cookie)
        state["n"] += 1
        if state["n"] == 1:
            return False, "无法在页面中找到 __playinfo__ 数据"
        f = Path(api.tmp_path) / f"{video_id}_audio.m4a"
        f.write_bytes(b"RIFF")
        return True, {"file_path": str(f), "audio_url": "https://example/a",
                      "audio_id": "42"}

    monkeypatch.setattr(server.downloader, "download_bilibili_audio", flaky_download)
    monkeypatch.setattr(server.bili_cookie_manager, "load_cookie", lambda: "")
    monkeypatch.setattr(server.bili_cookie_manager, "ensure_fresh_cookie",
                        lambda *a, **k: ("FRESH_CK", "已从生产同步新 cookie"))
    r = submit_json(api.client, {"source": "bilibili", "bvid": "BVheal"})
    body = wait_terminal(api.client, r.json()["job_id"])
    assert body["status"] == "done"
    assert calls_ == ["", "FRESH_CK"]


def test_bilibili_self_heal_disabled_keeps_failure_semantics(api, monkeypatch):
    """自愈不可用（返回空 cookie）时保持原有失败语义：单次尝试、错误落终态。"""
    calls_ = []

    def always_fail(video_id, cookie, save_dir="tmp", page=1,
                    progress_cb=None, cancel_flag=None):
        calls_.append(cookie)
        return False, "no playinfo"

    monkeypatch.setattr(server.downloader, "download_bilibili_audio", always_fail)
    monkeypatch.setattr(server.bili_cookie_manager, "load_cookie", lambda: "")
    monkeypatch.setattr(server.bili_cookie_manager, "ensure_fresh_cookie",
                        lambda *a, **k: (None, ""))
    r = submit_json(api.client, {"source": "bilibili", "bvid": "BVdead"})
    body = wait_terminal(api.client, r.json()["job_id"])
    assert body["status"] == "error"
    assert len(calls_) == 1
    assert "音频下载失败" in (body.get("result") or {}).get("message", "")


def wait_terminal(client, job_id, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}", headers=_h()).json()
        if body.get("status") in ("done", "error", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} 未在 {timeout}s 内到达终态: "
                         f"{client.get(f'/jobs/{job_id}', headers=_h()).json()}")


def wait_running(client, job_id, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}", headers=_h()).json()
        if body.get("status") == "running":
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} 未进入 running")


def submit_json(client, payload):
    return client.post("/jobs", json=payload, headers=_h())


# ---------- 1+2+9. 四源全链路 / result 同构 / timing.download ----------

def test_four_sources_full_chain(api):
    c, t0 = api.client, time.monotonic()

    r = submit_json(c, {"source": "bilibili", "bvid": "BV1abc",
                        "page": 2, "context": None, "diarize": True,
                        "diarize_only": False, "body": None, "no_cache": False})
    assert r.status_code == 202
    bili = r.json()
    assert bili["status"] == "queued" and bili["job_id"]

    r = submit_json(c, {"source": "webdav", "path": "inbox/x.mp3"})
    assert r.status_code == 202
    wd = r.json()

    r = submit_json(c, {"source": "douyin", "aweme_id": "1234567890123456789"})
    assert r.status_code == 202
    dy = r.json()

    r = c.post("/jobs/upload", data={"diarize": "false", "context": ""},
               files={"file": ("a.wav", b"RIFF", "audio/wav")}, headers=_h())
    assert r.status_code == 202
    up = r.json()

    b_done = wait_terminal(c, bili["job_id"])
    w_done = wait_terminal(c, wd["job_id"])
    d_done = wait_terminal(c, dy["job_id"])
    u_done = wait_terminal(c, up["job_id"])

    # bilibili：STUB + timing.download 注入；转录/下载回调链已接线
    assert b_done["status"] == "done"
    assert b_done["result"]["timing"]["download"] >= 0
    assert isinstance(b_done["result"]["timing"]["download"], float)
    assert b_done["request"] == {"bvid": "BV1abc", "page": 2, "context": None,
                                 "diarize": True, "diarize_only": False,
                                 "body": None, "no_cache": False}
    assert "cookie" not in b_done["request"]
    call = api.calls[0]
    assert call["args"][:2] == ("https://example/a", "BV1abc_p2")   # audio_url, cache_bvid(page 编码)
    assert call["kw"]["diarize"] is True
    assert call["kw"]["progress_cb"] is not None
    assert call["kw"]["cancelled"] is not None
    assert call["kw"]["stage_cb"] is not None
    assert t0 < call["kw"]["diarize_wait_deadline"] < t0 + 700      # 分离忙退避期限 ~600s
    # backend name 回填（采样器归属依赖）
    assert api.jm._jobs[bili["job_id"]].backend_name == server.config.backend_name

    # webdav：result 与现有端点完全同构（无 download 注入）
    assert w_done["status"] == "done"
    assert w_done["result"] == STUB_RESULT

    # douyin：+ timing.download
    assert d_done["status"] == "done"
    assert d_done["result"]["timing"]["download"] >= 0

    # upload：result 同构（无 download 注入）；内部键路径与原始文件名
    assert u_done["status"] == "done"
    assert u_done["result"] == STUB_RESULT
    up_call = api.calls[-1]
    assert up_call["path"].startswith("tmp/jobfile_")
    assert up_call["path"].endswith(".wav")
    assert up_call["name"] == "a.wav"


def test_bilibili_diarize_only_phase_and_deadline(api):
    c = api.client
    r = submit_json(c, {"source": "bilibili", "bvid": "BVdio",
                        "diarize_only": True})
    assert r.status_code == 202
    done = wait_terminal(c, r.json()["job_id"])
    assert done["status"] == "done"
    assert done["phase"] == "formatting"      # diarize_only：downloading → formatting → done
    assert done["percent"] is None            # percent 恒 null（spec §4.2）
    d = api.diar_calls[0]
    assert d["fn"] == "only" and d["video_id"] == "BVdio"
    assert d["stage_cb"] is not None
    assert d["wait_deadline"] > time.monotonic()   # 600s 忙退避期限已注入


def test_bilibili_diarize_only_with_body_uses_merge(api):
    c = api.client
    body = [{"text": "句一"}]
    r = submit_json(c, {"source": "bilibili", "bvid": "BVmerge",
                        "diarize_only": True, "body": body})
    assert r.status_code == 202
    done = wait_terminal(c, r.json()["job_id"])
    assert done["status"] == "done"
    assert api.diar_calls[0]["fn"] == "merge"
    assert done["result"]["body"] == body


# ---------- 3+7. 404 与 422 ----------

def test_unknown_id_404(api):
    c = api.client
    assert c.get("/jobs/no-such-id", headers=_h()).status_code == 404
    assert c.delete("/jobs/no-such-id", headers=_h()).status_code == 404
    body = c.get("/jobs/no-such-id", headers=_h()).json()
    assert "job not found" in body["detail"]


def test_submit_validation_422(api):
    c = api.client
    # job_id 格式不合法 → JobManager ValueError → 422
    for bad in ("bad id!", "a" * 129, "with#hash", ".", ".."):
        r = submit_json(c, {"source": "bilibili", "bvid": "BV1",
                            "job_id": bad})
        assert r.status_code == 422, (bad, r.status_code)
    # Pydantic 字段错 → 422
    pass  # cookie 已不再随载荷传递（2026-11-24），缺 cookie 是合法提交
    assert submit_json(c, {"source": "douyin", "aweme_id": "abc"}).status_code == 422     # 格式错
    assert submit_json(c, {"source": "nope"}).status_code == 422                          # 未知 source
    assert submit_json(c, {"source": "webdav"}).status_code == 422                        # 缺 path
    # 合法 job_id 形状可提交（隔离字母数字）
    r = submit_json(c, {"source": "bilibili", "bvid": "BVok",
                        "job_id": "cuid123-1"})
    assert r.status_code == 202
    assert wait_terminal(c, r.json()["job_id"])["status"] == "done"


def test_upload_invalid_job_id_422_and_no_file_left(api, tmp_path):
    c = api.client
    r = c.post("/jobs/upload", data={"job_id": "bad id!"},
               files={"file": ("a.wav", b"RIFF", "audio/wav")}, headers=_h())
    assert r.status_code == 422
    # 被拒提交的文件就地删除，tmp 无残留
    assert not list(Path("tmp").glob("jobfile_*"))


# ---------- 4. 暂停提交 503 ----------

def test_paused_submit_503(api):
    c = api.client
    assert c.post("/pause", json={"hours": 1}, headers=_h()).status_code == 200
    # 两个提交端点均 503（POST 才拦）
    r = submit_json(c, {"source": "bilibili", "bvid": "BV1"})
    assert r.status_code == 503
    assert r.json()["paused"] is True and r.json()["resume_at"]
    r = c.post("/jobs/upload", files={"file": ("a.wav", b"RIFF", "audio/wav")},
               headers=_h())
    assert r.status_code == 503
    # GET /jobs 不在拦截集（仅 POST 提交端点）
    assert c.get("/jobs", headers=_h()).status_code == 200
    # 恢复后提交恢复
    assert c.post("/resume", headers=_h()).status_code == 200
    assert submit_json(c, {"source": "bilibili", "bvid": "BVresume",
                           }).status_code == 202


# ---------- 5. query 命中 / missing / >500 截断 ----------

def test_query_hit_missing_and_truncation(api):
    c = api.client
    ids = []
    for i in range(4):
        r = submit_json(c, {"source": "douyin", "aweme_id": f"123456789000000000{i}"})
        assert r.status_code == 202
        ids.append(r.json()["job_id"])
    wait_terminal(c, ids[0])
    for jid in ids[1:]:
        wait_terminal(c, jid)

    # 命中 + missing
    r = c.post("/jobs/query", json={"ids": ids + ["unk1", "unk2"]}, headers=_h())
    assert r.status_code == 200
    out = r.json()
    assert [j["job_id"] for j in out["jobs"]] == ids
    assert all("result" not in j for j in out["jobs"])   # 每项不含 result
    assert out["missing"] == ["unk1", "unk2"]

    # 单批 >500：静默截断，超出部分按 missing 返回（R5A）——503 个 id：
    # 前 500 全处理（4 真实 + 496 未知），后 3 个（2 未知 + 1 真实）按 missing
    long_ids = ids + [f"unk{i}" for i in range(498)] + [ids[0]]
    assert len(long_ids) == 503
    out = c.post("/jobs/query", json={"ids": long_ids}, headers=_h()).json()
    assert len(out["jobs"]) == 4                          # 截断窗口内的 4 个真实任务
    assert len(out["missing"]) == 499                     # 窗口内 496 未知 + 截断的 3 个
    assert long_ids[500] in out["missing"]                # 第 501 个（截断）→ missing
    assert long_ids[502] in out["missing"]                # 第 503 个（真实但被截断）→ missing


# ---------- 6. 列表 cap 与 cookie 不回显 ----------

def test_list_cap_and_cookie_not_echoed(api):
    c = api.client
    ids = []
    for i in range(105):
        r = submit_json(c, {"source": "douyin", "aweme_id": f"22345678900000000{i:02d}"})
        assert r.status_code == 202, r.text
        ids.append(r.json()["job_id"])

    deadline = time.time() + 60
    while time.time() < deadline:
        q = c.post("/jobs/query", json={"ids": ids}, headers=_h()).json()
        if all(j["status"] == "done" for j in q["jobs"]):
            break
        time.sleep(0.2)
    assert all(j["status"] == "done" for j in q["jobs"])

    items = c.get("/jobs", headers=_h()).json()
    assert len(items) == 100                              # 终态 cap 100
    assert all("cookie" not in j["request"] for j in items)
    single = c.get(f"/jobs/{ids[0]}", headers=_h()).json()
    assert "cookie" not in single["request"]
    # status 过滤 + query 项同样不回显
    done_items = c.get("/jobs", params={"status": "done"}, headers=_h()).json()
    assert len(done_items) == 100 and all(j["status"] == "done" for j in done_items)
    q = c.post("/jobs/query", json={"ids": ids[:5]}, headers=_h()).json()
    assert all("cookie" not in j["request"] for j in q["jobs"])


# ---------- 8. DELETE 三态 ----------

def test_delete_three_states(api):
    c, state = api.client, api.state
    state["mode"] = "block"
    r1 = submit_json(c, {"source": "bilibili", "bvid": "BVrun"})
    j1 = r1.json()["job_id"]
    wait_running(c, j1)
    r2 = submit_json(c, {"source": "bilibili", "bvid": "BVqueued", "cookie": "ck"})
    j2 = r2.json()["job_id"]

    # queued → 即时 cancelled
    r = c.delete(f"/jobs/{j2}", headers=_h())
    assert r.status_code == 200
    assert r.json() == {"job_id": j2, "status": "cancelled"}

    # running → cancelling（已接受语义：返回请求后即时状态）
    r = c.delete(f"/jobs/{j1}", headers=_h())
    assert r.status_code == 200
    assert r.json()["status"] == "cancelling"

    state["gate"].set()
    done1 = wait_terminal(c, j1)
    assert done1["status"] == "cancelled"

    # 终态幂等：cancelled / done 各自返回当前态
    assert c.delete(f"/jobs/{j1}", headers=_h()).json()["status"] == "cancelled"

    state["mode"] = "instant"
    r3 = submit_json(c, {"source": "bilibili", "bvid": "BVdone", "cookie": "ck"})
    j3 = r3.json()["job_id"]
    assert wait_terminal(c, j3)["status"] == "done"
    assert c.delete(f"/jobs/{j3}", headers=_h()).json() == {"job_id": j3, "status": "done"}


# ---------- 10. upload 重放删新文件 ----------

def test_upload_replay_deletes_new_file(api):
    c, state = api.client, api.state
    state["mode"] = "block"
    files = {"file": ("first.wav", b"RIFF", "audio/wav")}
    r1 = c.post("/jobs/upload", data={"job_id": "upload-1"}, files=files, headers=_h())
    assert r1.status_code == 202
    job_id = r1.json()["job_id"]
    assert len(list(Path("tmp").glob("jobfile_*"))) == 1

    # 同 id 重放：200 + dedup，新上传文件立即删除
    r2 = c.post("/jobs/upload", data={"job_id": "upload-1"},
                files={"file": ("second.wav", b"RIFF2", "audio/wav")}, headers=_h())
    assert r2.status_code == 200
    assert r2.json()["dedup"] is True
    assert r2.json()["job_id"] == job_id
    assert len(list(Path("tmp").glob("jobfile_*"))) == 1   # 新文件已删，原文件仍在

    state["gate"].set()
    assert wait_terminal(c, job_id)["status"] == "done"
    assert not list(Path("tmp").glob("jobfile_*"))          # 终态内部键删除


# ---------- 11. 队列满 429 头+body 双写 ----------

@pytest.mark.parametrize("api", [1], indirect=True)
def test_queue_full_429_header_and_body(api):
    c, state = api.client, api.state
    state["mode"] = "block"
    r1 = submit_json(c, {"source": "bilibili", "bvid": "BVq1", "cookie": "ck"})
    assert r1.status_code == 202
    wait_running(c, r1.json()["job_id"])
    r2 = submit_json(c, {"source": "bilibili", "bvid": "BVq2", "cookie": "ck"})
    assert r2.status_code == 429
    body = r2.json()
    assert body["detail"] == "queue full"
    assert body["retry_after"] == 45                        # fixture 注入的 config 值
    assert r2.headers["Retry-After"] == str(body["retry_after"])   # 头与 body 双写
    state["gate"].set()
    wait_terminal(c, r1.json()["job_id"])
    # 腾出后可再次提交
    assert submit_json(c, {"source": "bilibili", "bvid": "BVq3",
                           }).status_code == 202


# ---------- 12. cleanup protected 行为 ----------

def test_cleanup_protected_paths_and_aliases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    old = time.time() - 25 * 3600

    active = tmpdir / "jobfile_active.wav"
    active.write_bytes(b"x")
    os.utime(active, (old, old))
    alias = tmpdir / ".job-abc.wav"
    os.link(active, alias)                 # 真硬链接别名（同 inode，继承旧 mtime）
    stale_alias = tmpdir / ".job-dead.wav"
    stale_alias.write_bytes(b"x")
    os.utime(stale_alias, (old, old))
    fresh = tmpdir / "jobfile_fresh.wav"
    fresh.write_bytes(b"x")                # 未登记但未超龄
    unreg_old = tmpdir / "temp_other.wav"
    unreg_old.write_bytes(b"x")
    os.utime(unreg_old, (old, old))

    server.cleanup_stale_tmp_files(protected={str(active)})

    assert active.exists()                 # 活跃登记路径跳过
    assert alias.exists()                  # 其 .job-* 硬链接别名跳过（inode 豁免）
    assert fresh.exists()                  # 未超龄不动
    assert not stale_alias.exists()        # 未登记超龄 .job-* 照常清理（R2A）
    assert not unreg_old.exists()          # 未登记超龄普通文件照常清理


def test_cleanup_without_protected_keeps_legacy_behavior(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    old = time.time() - 25 * 3600
    f = tmpdir / "temp_old.wav"
    f.write_bytes(b"x")
    os.utime(f, (old, old))
    server.cleanup_stale_tmp_files()
    assert not f.exists()


# ---------- 13. 取消后 webdav 不归档 + 缓存零写入（Review Focus #2） ----------

def test_cancel_webdav_no_archive_no_cache(api, tmp_path, monkeypatch):
    c, state = api.client, api.state
    # webdav 根已由 fixture 隔离到 tmp_path/webdav；此处只加归档记录器
    audio = tmp_path / "webdav" / "inbox" / "song.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"RIFF")

    archive_calls = []
    monkeypatch.setattr(webdav_archive, "move_to_processed",
                        lambda base, rel: archive_calls.append((base, rel)))

    state["mode"] = "block"
    r = submit_json(c, {"source": "webdav", "path": "inbox/song.mp3"})
    jid = r.json()["job_id"]
    wait_running(c, jid)

    assert c.delete(f"/jobs/{jid}", headers=_h()).status_code == 200
    state["gate"].set()
    done = wait_terminal(c, jid)

    assert done["status"] == "cancelled"
    assert done["result"] is None                        # 终态无 result（部分结果不返回）
    assert archive_calls == []                           # 取消不归档
    assert api.cache_writes == []                        # 缓存写入零发生（探针先于缓存）


def test_webdav_error_no_archive(api, tmp_path, monkeypatch):
    c, state = api.client, api.state
    audio = tmp_path / "webdav" / "inbox" / "bad.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"RIFF")

    archive_calls = []
    monkeypatch.setattr(webdav_archive, "move_to_processed",
                        lambda base, rel: archive_calls.append((base, rel)))

    state["mode"] = "error"
    r = submit_json(c, {"source": "webdav", "path": "inbox/bad.mp3"})
    done = wait_terminal(c, r.json()["job_id"])
    assert done["status"] == "error"                     # 响应 dict status==error → job error
    assert done["result"]["message"] == "stub error"
    assert archive_calls == []                           # 失败不归档
    # 成功路径归档回归（同环境反向断言）
    state["mode"] = "instant"
    r = submit_json(c, {"source": "webdav", "path": "inbox/bad.mp3", "no_cache": True})
    done = wait_terminal(c, r.json()["job_id"])
    assert done["status"] == "done"
    assert archive_calls == [(str(tmp_path / "webdav"), "inbox/bad.mp3")]


# ---------- 13b. F3：归档探针（取消标志置位 → 跳过归档） ----------

def test_webdav_cancel_flag_skips_archive(tmp_path, monkeypatch):
    """fixwave F3 归档探针：process_transcription 成功返回但 job 取消标志已置位
    （取消发生在转录末段内部探针之后、归档开始之前）→ 跳过归档，源文件不动，
    不进入提交点。执行体直调（不经端点），隔离到 tmp_path/webdav。"""
    monkeypatch.chdir(tmp_path)
    real_get = server.config.get

    def fake_config_get(key, default=None):
        if key == "webdav.base_path":
            return str(tmp_path / "webdav")
        return real_get(key, default)

    monkeypatch.setattr(server.config, "get", fake_config_get)
    audio = tmp_path / "webdav" / "inbox" / "probe.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"RIFF")

    archive_calls = []
    monkeypatch.setattr(webdav_archive, "move_to_processed",
                        lambda base, rel: archive_calls.append((base, rel)))

    async def fake_process(*a, **kw):
        return dict(STUB_RESULT)          # 转录成功返回（标志在末段探针之后置位）

    monkeypatch.setattr(server.transcription_service, "process_transcription",
                        fake_process)

    job = Job(job_id="w", source="webdav", request={"path": "inbox/probe.mp3"})
    job.cancel_event.set()              # 归档探针读取的 job 级取消标志
    sinks = JobManager()._make_sinks(job)

    result = asyncio.run(server._exec_webdav(job, sinks, dict(job.request)))
    assert result["status"] == "success"
    assert archive_calls == [], "取消标志置位 → 归档被跳过"
    assert audio.exists(), "源文件未被动"
    assert job.commit_done is False, "未进入提交点"


# ---------- 13c. F3：归档提交点（done 胜出取消） ----------

def test_webdav_archive_commit_point(api, tmp_path, monkeypatch):
    """fixwave F3 提交点语义：归档一旦开始即为提交点——取消在归档期间到达时
    DELETE 返回 cancelling（已受理），但归档完成后 job 落 done（胜出取消），
    result 与已归档文件一致。"""
    c, state = api.client, api.state
    audio = tmp_path / "webdav" / "inbox" / "cm.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"RIFF")

    original_move = webdav_archive.move_to_processed
    archive_started = threading.Event()
    release_archive = threading.Event()

    def gated_move(base, rel):
        archive_started.set()
        release_archive.wait(timeout=10)
        return original_move(base, rel)

    monkeypatch.setattr(webdav_archive, "move_to_processed", gated_move)

    state["mode"] = "instant"
    r = submit_json(c, {"source": "webdav", "path": "inbox/cm.mp3"})
    jid = r.json()["job_id"]
    wait_running(c, jid)
    assert archive_started.wait(timeout=10), "归档未开始"

    # 归档进行中取消 → 已受理 cancelling（此时提交点已成立）
    assert c.delete(f"/jobs/{jid}", headers=_h()).json()["status"] == "cancelling"

    release_archive.set()
    body = wait_terminal(c, jid)
    assert body["status"] == "done", "提交点：done 胜出取消"
    assert body["result"]["status"] == "success"
    assert (tmp_path / "webdav" / "processed" / "cm.mp3").exists(), "文件已归档"
    assert not audio.exists()


# ---------- 14. webdav inbox → processed 回退 ----------

def test_webdav_inbox_processed_fallback(api, tmp_path, monkeypatch):
    c = api.client
    archive_calls = []
    monkeypatch.setattr(webdav_archive, "move_to_processed",
                        lambda base, rel: archive_calls.append((base, rel)))
    # inbox 不存在、processed 存在 → 回退
    processed = tmp_path / "webdav" / "processed" / "moved.mp3"
    processed.parent.mkdir(parents=True)
    processed.write_bytes(b"RIFF")

    r = submit_json(c, {"source": "webdav", "path": "inbox/moved.mp3"})
    done = wait_terminal(c, r.json()["job_id"])
    assert done["status"] == "done"
    assert archive_calls == [(str(tmp_path / "webdav"), "processed/moved.mp3")]

    # 完全不存在 → error dict（与现端点同构四键）
    r = submit_json(c, {"source": "webdav", "path": "inbox/ghost.mp3"})
    done = wait_terminal(c, r.json()["job_id"])
    assert done["status"] == "error"
    assert done["result"]["type"] == server.config.subtitle_config["type"]
    assert done["result"]["body"] == []


# ---------- 15. 上游暂停集成：error 元数据 + worker 自冷（spec §9） ----------

def test_upstream_paused_error_metadata_and_worker_selfcool(api, monkeypatch):
    """asr-engine 被独立暂停（本地未暂停）→ 执行链抛 UpstreamPausedError：
    (1) 暂停信号穿透真实 webdav 执行体（不得被 _exec_webdav 吞成 _job_error_dict）
        → job error + paused/resume_at/paused_at 三键，且不归档；
    (2) 快照暂停期间 worker 自冷不取新件，恢复后继续取件完成。"""
    c, state = api.client, api.state
    archive_calls = []
    monkeypatch.setattr(webdav_archive, "move_to_processed",
                        lambda base, rel: archive_calls.append((base, rel)))

    state["mode"] = "paused"
    r = submit_json(c, {"source": "webdav", "path": "inbox/x.mp3"})
    assert r.status_code == 202
    done = wait_terminal(c, r.json()["job_id"])
    assert done["status"] == "error"
    res = done["result"]
    assert res["status"] == "error"
    assert res["paused"] is True
    assert res["resume_at"] == "2026-11-10T10:00:00"
    assert res["paused_at"] == "2026-11-10T09:00:00"
    assert res["message"] == "asr engine paused", "暂停信号原样穿透（非泛化错误 dict）"
    assert "body" not in res, "不得呈 _job_error_dict 形状"
    assert archive_calls == [], "暂停中止不归档"

    # worker 自冷：快照暂停期间（刷新循环 60s 不会在测试窗口内覆写）不取新件
    monkeypatch.setitem(server._asr_paused_snapshot, "paused", True)
    state["mode"] = "instant"
    r2 = submit_json(c, {"source": "webdav", "path": "inbox/x.mp3", "no_cache": True})
    assert r2.status_code == 202
    jid2 = r2.json()["job_id"]
    time.sleep(1.2)                       # > WORKER_IDLE_WAIT，至少两轮取件门检查
    assert c.get(f"/jobs/{jid2}", headers=_h()).json()["status"] == "queued", \
        "上游独立暂停期间 worker 自冷不取件"

    monkeypatch.setitem(server._asr_paused_snapshot, "paused", False)
    done2 = wait_terminal(c, jid2)
    assert done2["status"] == "done", "恢复后 worker 继续取件"
