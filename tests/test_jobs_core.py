"""JobManager 核心测试：存储/提交语义/查询/取消（queued 分支）/清扫（Task 3）。

spec：docs/superpowers/specs/2026-11-10-transcribe-job-queue-design.md
§4.1（提交）/§4.2（查询）/§4.3（取消状态机，执行联动归 Task 7）/§6（清扫）/§7（配置）。

纯逻辑测试：直接构造 JobManager（不 start），终态/进度槽经内部字段直接置位
（worker 执行归 Task 7；`_executor_factory=None` 时任务保持 queued）。
"""
import asyncio
import json
from pathlib import Path

import pytest

import transcribe as T
from jobs import ACTIVE_STATUSES, Job, JobManager, QueueFullError

# GET /jobs/{id} §4.2 完整键集（Task 8 直接消费该形状）
GET_KEYS = {"job_id", "request", "status", "phase", "percent", "message",
            "diarize_stage", "queue_position", "submitted_at", "started_at",
            "finished_at", "result"}


class StubConfig:
    """config.get 点号访问桩（数值经 `jobs.*` 注入）。"""

    def __init__(self, **values):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


def make_mgr(**cfg):
    return JobManager(StubConfig(**cfg) if cfg else None)


def bili(**over):
    """bilibili 提交参数（含 cookie，用于回显安全断言）。"""
    p = {"bvid": "BV1xx", "page": 1, "cookie": "SECRET-TOKEN", "context": None,
         "diarize": False, "diarize_only": False, "body": None, "no_cache": False}
    p.update(over)
    return p


def webdav(path="inbox/a.mp3", **over):
    p = {"path": path, "context": None, "diarize": False, "no_cache": False}
    p.update(over)
    return p


def finish(mgr, job_id, status="done", result=None):
    """测试辅助：直接终态迁移（真实迁移由 Task 7 worker 驱动）。"""
    return mgr._mark_finished(mgr._jobs[job_id], status, result)


# ---------- 1. job_id 格式校验（spec §4.1：不透明字符串，422 路径） ----------

@pytest.mark.asyncio
async def test_job_id_validation():
    mgr = make_mgr()
    bad_ids = ["abc#def",          # fragment 截断字符
               "a/b",              # 路径分隔
               "..",               # URL 归一化会吞掉
               ".",
               "",
               "a" * 129,          # 超长
               "任务",              # 非字符集
               "a b",              # 空格
               123]                # 非字符串
    for bad in bad_ids:
        with pytest.raises(ValueError):
            await mgr.submit("webdav", webdav(), job_id=bad)
    assert mgr._jobs == {}, "不合法 job_id 不得产生任何记录"

    ok = await mgr.submit("webdav", webdav(), job_id="NF-cuid_1:2@3-4.5")
    assert ok["status"] == "queued"
    await mgr.submit("webdav", webdav("x.mp3"), job_id="a" * 128)  # 边界长度合法


@pytest.mark.asyncio
async def test_submit_without_job_id_generates_uuid():
    mgr = make_mgr()
    r = await mgr.submit("webdav", webdav())
    assert set(r) == {"job_id", "status"} and r["status"] == "queued"
    assert len(r["job_id"]) == 32  # uuid4().hex


# ---------- 2. 同 id 重放：活跃+终态均重放，绝不替换 ----------

@pytest.mark.asyncio
async def test_same_id_replay_active_and_terminal():
    mgr = make_mgr()
    first = await mgr.submit("bilibili", bili(), job_id="t1")
    assert "dedup" not in first
    original = mgr._jobs["t1"]

    # 参数不同也重放（job_id 是唯一幂等键）
    replay = await mgr.submit("bilibili", bili(bvid="OTHER"), job_id="t1")
    assert replay["dedup"] is True
    assert set(replay) == GET_KEYS | {"dedup"}
    assert replay["job_id"] == "t1" and replay["status"] == "queued"
    assert len(mgr._jobs) == 1 and mgr._jobs["t1"] is original, "重放绝不替换记录"

    # 终态重放：完整 GET 结构含 result
    result = {"status": "success", "body": [{"text": "hi"}]}
    finish(mgr, "t1", "done", result)
    replay2 = await mgr.submit("bilibili", bili(), job_id="t1")
    assert replay2["dedup"] is True
    assert replay2["status"] == "done" and replay2["result"] == result
    assert len(mgr._jobs) == 1


# ---------- 3. identity 去重（无 job_id 时，活跃口径含 cancelling） ----------

@pytest.mark.asyncio
async def test_identity_dedup_basic_and_clamp_normalization():
    mgr = make_mgr()
    r1 = await mgr.submit("bilibili", bili())
    r2 = await mgr.submit("bilibili", bili(cookie="DIFFERENT"))  # cookie 不入 identity
    assert r2["dedup"] is True and r2["job_id"] == r1["job_id"]
    assert set(r2) == GET_KEYS | {"dedup"}
    assert len(mgr._jobs) == 1

    # context 经 clamp_asr_context 归一：超长 context 与钳制后等价
    r3 = await mgr.submit("bilibili", bili(context="x" * 2000))
    assert "dedup" not in r3 and len(mgr._jobs) == 2
    clamped = T.clamp_asr_context("x" * 2000)
    assert len(clamped) < 2000
    r4 = await mgr.submit("bilibili", bili(context=clamped))
    assert r4["dedup"] is True and r4["job_id"] == r3["job_id"]


@pytest.mark.asyncio
async def test_identity_differs_on_identity_dimensions():
    mgr = make_mgr()
    base = await mgr.submit("bilibili", bili())
    for label, params in [
        ("diarize", bili(diarize=True)),
        ("diarize_only", bili(diarize_only=True)),
        ("no_cache", bili(no_cache=True)),
        ("body", bili(body={"x": 1})),
        ("page", bili(page=2)),
    ]:
        r = await mgr.submit("bilibili", params)
        assert "dedup" not in r, f"{label} 差异必须产生新任务"
    assert len(mgr._jobs) == 6

    # 各源 locator 独立
    w1 = await mgr.submit("webdav", webdav())
    w2 = await mgr.submit("webdav", webdav())
    assert w2["dedup"] is True and w2["job_id"] == w1["job_id"]
    w3 = await mgr.submit("webdav", webdav("inbox/b.mp3"))
    assert "dedup" not in w3
    d1 = await mgr.submit("douyin", {"aweme_id": "123", "context": None,
                                     "diarize": False, "no_cache": False})
    d2 = await mgr.submit("douyin", {"aweme_id": "123", "context": None,
                                     "diarize": False, "no_cache": False})
    assert d2["dedup"] is True and d2["job_id"] == d1["job_id"]


@pytest.mark.asyncio
async def test_identity_hit_requires_active_status():
    mgr = make_mgr()
    r1 = await mgr.submit("bilibili", bili())
    # cancelling 仍属活跃 → 命中
    mgr._jobs[r1["job_id"]].status = "cancelling"
    hit = await mgr.submit("bilibili", bili())
    assert hit["dedup"] is True and hit["job_id"] == r1["job_id"]
    # 终态不命中 → 新任务
    finish(mgr, r1["job_id"], "done", {})
    fresh = await mgr.submit("bilibili", bili())
    assert "dedup" not in fresh and fresh["job_id"] != r1["job_id"]
    assert len(mgr._jobs) == 2


@pytest.mark.asyncio
async def test_upload_has_no_identity_only_job_id_dedup(tmp_path):
    f1 = tmp_path / "jobfile_a.mp3"
    f1.write_bytes(b"x")
    f2 = tmp_path / "jobfile_b.mp3"
    f2.write_bytes(b"y")
    mgr = make_mgr()
    u1 = await mgr.submit("upload", {"diarize": False}, upload_tmp_path=str(f1))
    u2 = await mgr.submit("upload", {"diarize": False}, upload_tmp_path=str(f2))
    assert "dedup" not in u2 and u2["job_id"] != u1["job_id"]


# ---------- 4. 队列满 429（max_queue 含 running 与 cancelling） ----------

@pytest.mark.asyncio
async def test_queue_full_and_active_counting():
    mgr = make_mgr(**{"jobs.max_queue": 2})
    a = await mgr.submit("webdav", webdav("1"))
    b = await mgr.submit("webdav", webdav("2"))
    with pytest.raises(QueueFullError) as ei:
        await mgr.submit("webdav", webdav("3"))
    assert ei.value.detail == "queue full"
    assert ei.value.retry_after == 30  # 默认 Retry-After 30s

    # queued 取消释放名额
    await mgr.cancel(a["job_id"])
    c = await mgr.submit("webdav", webdav("3"))
    assert c["status"] == "queued"

    # running 与 cancelling 都计入活跃
    mgr._jobs[b["job_id"]].status = "running"
    await mgr.cancel(b["job_id"])          # → cancelling
    assert mgr._jobs[b["job_id"]].status == "cancelling"
    with pytest.raises(QueueFullError):
        await mgr.submit("webdav", webdav("4"))
    # 终态释放名额
    finish(mgr, b["job_id"], "cancelled")
    ok = await mgr.submit("webdav", webdav("4"))
    assert ok["status"] == "queued"


@pytest.mark.asyncio
async def test_retry_after_configurable():
    mgr = make_mgr(**{"jobs.max_queue": 0, "jobs.retry_after_seconds": 7})
    with pytest.raises(QueueFullError) as ei:
        await mgr.submit("webdav", webdav())
    assert ei.value.retry_after == 7


# ---------- 5. 取消：queued 即时 cancelled；running→cancelling（联动归 Task 7） ----------

@pytest.mark.asyncio
async def test_cancel_queued_dequeues_immediately():
    mgr = make_mgr()
    a = await mgr.submit("webdav", webdav("1"))
    b = await mgr.submit("webdav", webdav("2"))
    res = await mgr.cancel(a["job_id"])
    assert res == {"job_id": a["job_id"], "status": "cancelled"}
    job = mgr._jobs[a["job_id"]]
    assert job.status == "cancelled" and job.finished_at is not None
    assert a["job_id"] not in mgr._ready
    # 队列位置前移 + cancelled 不再有队列位置
    assert mgr.get(b["job_id"])["queue_position"] == 1
    assert mgr.get(a["job_id"])["queue_position"] is None
    # 终态幂等
    assert (await mgr.cancel(a["job_id"]))["status"] == "cancelled"
    # 未知 id → None（server 层转 404）
    assert await mgr.cancel("ghost") is None
    assert mgr.get("ghost") is None


@pytest.mark.asyncio
async def test_cancel_running_transition_and_exec_hook():
    mgr = make_mgr()
    r = await mgr.submit("webdav", webdav())
    mgr._jobs[r["job_id"]].status = "running"
    calls = []
    mgr._cancel_exec = calls.append  # Task 7 注入点
    res = await mgr.cancel(r["job_id"])
    assert res == {"job_id": r["job_id"], "status": "cancelling"}
    assert calls == [mgr._jobs[r["job_id"]]], "running 取消应触发执行联动钩子一次"
    # cancelling 重复取消：幂等，不再触发联动
    res2 = await mgr.cancel(r["job_id"])
    assert res2["status"] == "cancelling"
    assert calls == [mgr._jobs[r["job_id"]]]


# ---------- 6. GET 形状 / cookie 永不回显 / 进度槽拼装 ----------

@pytest.mark.asyncio
async def test_get_shape_queue_position_and_no_cookie_echo():
    mgr = make_mgr()
    r1 = await mgr.submit("bilibili", bili())
    r2 = await mgr.submit("bilibili", bili(bvid="BV2"))
    v = mgr.get(r1["job_id"])
    assert set(v) == GET_KEYS
    assert "cookie" not in v["request"]
    assert "SECRET-TOKEN" not in json.dumps(v)
    assert v["request"]["bvid"] == "BV1xx", "request 为提交参数摘要（cookie 剔除，其余保留）"
    assert v["status"] == "queued" and v["queue_position"] == 1
    assert v["phase"] is None and v["percent"] is None and v["result"] is None
    assert v["submitted_at"] and v["started_at"] is None and v["finished_at"] is None
    assert mgr.get(r2["job_id"])["queue_position"] == 2
    # list 不回显 cookie
    assert "SECRET-TOKEN" not in json.dumps(mgr.list_jobs())


@pytest.mark.asyncio
async def test_view_merges_progress_slots():
    """进度分槽快照：GET 拼装转录槽/分离槽（Task 7 采样器写槽）。"""
    mgr = make_mgr()
    r = await mgr.submit("webdav", webdav())
    job = mgr._jobs[r["job_id"]]
    job.status = "running"
    job.transcript_slot = {"phase": "transcribing", "percent": 40, "message": "3/8"}
    job.diarize_slot = {"diarize_stage": "embedding"}
    v = mgr.get(r["job_id"])
    assert v["phase"] == "transcribing" and v["percent"] == 40
    assert v["message"] == "3/8" and v["diarize_stage"] == "embedding"
    assert v["queue_position"] is None  # running 无队列位置
    # query 同样拼装且不含 result
    item = mgr.query([r["job_id"]])["jobs"][0]
    assert item["percent"] == 40 and "result" not in item


# ---------- 7. query 命中 + missing 形状 ----------

@pytest.mark.asyncio
async def test_query_hit_and_missing():
    mgr = make_mgr()
    a = await mgr.submit("webdav", webdav("1"))
    b = await mgr.submit("webdav", webdav("2"))
    out = mgr.query([a["job_id"], "ghost", b["job_id"]])
    assert set(out) == {"jobs", "missing"}
    assert out["missing"] == ["ghost"]
    assert [j["job_id"] for j in out["jobs"]] == [a["job_id"], b["job_id"]]
    assert all("result" not in j and set(j) == GET_KEYS - {"result"} for j in out["jobs"])
    # 空入参
    assert mgr.query([]) == {"jobs": [], "missing": []}


# ---------- 8. list_jobs：倒序 / cap 100 只作用终态 ----------

@pytest.mark.asyncio
async def test_list_jobs_reverse_and_terminal_cap():
    mgr = make_mgr()
    jids = []
    for i in range(105):
        r = await mgr.submit("webdav", webdav(f"p{i}"))
        jids.append(r["job_id"])
        finish(mgr, r["job_id"], "done", {"i": i})
    listed = mgr.list_jobs()
    assert len(listed) == 100, "cap 100 只作用于终态任务"
    assert listed[0]["job_id"] == jids[-1], "倒序：最新终态在前"
    active = await mgr.submit("webdav", webdav("live"))
    assert len(mgr.list_jobs()) == 101, "活跃任务不受 cap 影响"
    assert len(mgr.list_jobs(status="done")) == 100
    queued = mgr.list_jobs(status="queued")
    assert [j["job_id"] for j in queued] == [active["job_id"]]


# ---------- 9. 清扫：TTL 整条移除 / 字节预算逐最旧（单条超限保留） ----------

@pytest.mark.asyncio
async def test_ttl_sweep_removes_only_expired_terminal():
    mgr = make_mgr(**{"jobs.record_ttl_minutes": 0})
    a = await mgr.submit("webdav", webdav("1"))
    b = await mgr.submit("webdav", webdav("2"))
    c = await mgr.submit("webdav", webdav("3"))
    finish(mgr, a["job_id"], "done", {"ok": True})
    finish(mgr, c["job_id"], "cancelled")
    await mgr._sweep_once()
    assert a["job_id"] not in mgr._jobs, "TTL 到期终态整条移除"
    assert c["job_id"] not in mgr._jobs
    assert b["job_id"] in mgr._jobs, "活跃记录绝不被清扫"
    assert mgr.query([a["job_id"]])["missing"] == [a["job_id"]]

    # 默认 TTL 下刚完成的记录保留
    mgr2 = make_mgr()
    r = await mgr2.submit("webdav", webdav("x"))
    finish(mgr2, r["job_id"], "done", {})
    await mgr2._sweep_once()
    assert r["job_id"] in mgr2._jobs


@pytest.mark.asyncio
async def test_byte_budget_evicts_oldest_and_keeps_oversized_single():
    mgr = make_mgr(**{"jobs.max_result_bytes": 100})
    jids = []
    for i in range(3):
        r = await mgr.submit("webdav", webdav(str(i)))
        jids.append(r["job_id"])
        finish(mgr, r["job_id"], "done", {"pad": "x" * 80})  # ~91B/条
    await mgr._sweep_once()
    assert jids[0] not in mgr._jobs and jids[1] not in mgr._jobs
    assert jids[2] in mgr._jobs, "从最旧终态逐条驱逐至预算内"

    # 单条超限保留：只剩它自己时不再驱逐
    big = await mgr.submit("webdav", webdav("big"))
    finish(mgr, big["job_id"], "done", {"pad": "y" * 500})  # ~511B > 100
    await mgr._sweep_once()
    assert jids[2] not in mgr._jobs, "次旧记录被驱逐"
    assert big["job_id"] in mgr._jobs, "单条超限的记录保留"

    # 活跃记录不参与预算
    live = await mgr.submit("webdav", webdav("live"))
    await mgr._sweep_once()
    assert live["job_id"] in mgr._jobs


# ---------- 10. upload 登记：submit 登记 / 终态注销 / 重放删新文件 / 清扫维护 ----------

@pytest.mark.asyncio
async def test_upload_registry_lifecycle_and_replay_deletes_new_file(tmp_path):
    f1 = tmp_path / "jobfile_a.mp3"
    f1.write_bytes(b"x")
    f2 = tmp_path / "jobfile_b.mp3"
    f2.write_bytes(b"y")
    mgr = make_mgr()
    await mgr.submit("upload", {"diarize": False}, job_id="u1", upload_tmp_path=str(f1))
    assert str(f1) in mgr._active_uploads
    assert mgr._jobs["u1"].upload_path == str(f1)

    r2 = await mgr.submit("upload", {"diarize": False}, job_id="u1",
                          upload_tmp_path=str(f2))
    assert r2["dedup"] is True
    assert not f2.exists(), "同 id upload 重放：新上传文件立即删除"
    assert f1.exists(), "原上传文件保留"
    assert str(f2) not in mgr._active_uploads

    finish(mgr, "u1", "done", {})
    assert str(f1) not in mgr._active_uploads, "终态注销登记"


@pytest.mark.asyncio
async def test_sweep_prunes_vanished_upload_paths(tmp_path):
    alive = tmp_path / "alive.mp3"
    alive.write_bytes(b"x")
    ghost = tmp_path / "ghost.mp3"  # 不存在
    mgr = make_mgr()
    mgr.register_upload(str(alive))
    mgr.register_upload(str(ghost))
    await mgr._sweep_once()
    assert str(alive) in mgr._active_uploads
    assert str(ghost) not in mgr._active_uploads, "已消失路径由清扫摘除"
    mgr.unregister_upload(str(alive))
    assert str(alive) not in mgr._active_uploads


# ---------- 11. start/stop：factory 未注入不取件；清扫任务拉起与干净停止 ----------

@pytest.mark.asyncio
async def test_start_without_executor_factory_keeps_queued_and_stop_clean():
    mgr = make_mgr()
    r = await mgr.submit("webdav", webdav())
    await mgr.start()
    await asyncio.sleep(0.05)
    assert mgr._jobs[r["job_id"]].status == "queued", "_executor_factory=None 不取件"
    assert mgr._worker_task is None
    assert mgr._sweep_task is not None and not mgr._sweep_task.done()
    await mgr.start()  # 幂等
    assert mgr._sweep_task is not None
    await mgr.stop()
    assert mgr._sweep_task is None, "stop 后后台任务句柄清空（可再次 start 重拉）"
