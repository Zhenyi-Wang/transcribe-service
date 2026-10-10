"""工作进程必须真的退出，而不是仅取消等待；测试不加载 GPU 模型。"""
import asyncio
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _module():
    assert importlib.util.find_spec("diarization.worker") is not None, "分离尚未隔离到可终止进程"
    return importlib.import_module("diarization.worker")


def test_worker_has_killable_boundary():
    _module()


@pytest.fixture
def worker_module(monkeypatch):
    module = _module()
    popen = subprocess.Popen

    def spawn_stub(command, **kwargs):
        # 保留真正的 Pipe、进程组、终止与等待机制，只替换昂贵的推理体。
        command = [sys.executable, "-m", "tests.diarization_worker_stub", command[3]]
        return popen(command, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", spawn_stub)
    monkeypatch.setattr(module, "_manager_settings", lambda: {})
    return module


def _running(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0]
        return state != "Z"
    except FileNotFoundError:
        return False


async def _written(path):
    async def wait():
        while not path.exists():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), 5)


@pytest.mark.asyncio
async def test_success_reuses_worker_and_unload_reaps_it(tmp_path, worker_module):
    worker = worker_module.DiarizationWorker(memory_mb=256, max_jobs=10)
    try:
        turns = await worker.run(str(tmp_path / "ok"), timeout=5)
        pid = worker.pid
        assert [(t.start, t.end, t.speaker) for t in turns] == [(0.0, 4.0, 0), (4.0, 8.0, 1)]
        assert worker.is_loaded
        await worker.run(str(tmp_path / "ok"), timeout=5)
        assert worker.pid == pid
        assert worker.unload() is True
        assert not _running(pid)
        assert not worker.is_loaded
        assert worker.unload() is False
    finally:
        worker.unload()


@pytest.mark.asyncio
async def test_timeout_kills_stubborn_worker_and_decoder(tmp_path, worker_module):
    worker = worker_module.DiarizationWorker(memory_mb=256, max_jobs=10)
    path = tmp_path / "hang"
    task = asyncio.create_task(worker.run(str(path), timeout=1))
    try:
        await _written(path)
        pid, decoder_pid = map(int, path.read_text().split())
        with pytest.raises(asyncio.TimeoutError):
            await task
        assert not _running(pid)
        async def decoder_stopped():
            while _running(decoder_pid):
                await asyncio.sleep(0.01)
        await asyncio.wait_for(decoder_stopped(), 3)
        assert not worker.is_loaded
        assert await worker.run(str(tmp_path / "ok"), timeout=5)
    finally:
        task.cancel()
        worker.unload()


@pytest.mark.asyncio
async def test_cancel_reaps_process_before_propagating(tmp_path, worker_module):
    worker = worker_module.DiarizationWorker(memory_mb=256, max_jobs=10)
    path = tmp_path / "hang"
    task = asyncio.create_task(worker.run(str(path), timeout=10))
    try:
        await _written(path)
        pid = worker.pid
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not _running(pid)
        assert not worker.is_loaded
    finally:
        worker.unload()


@pytest.mark.asyncio
async def test_busy_request_is_rejected_before_spawning_or_decoding(tmp_path, worker_module):
    worker = worker_module.DiarizationWorker(memory_mb=256, max_jobs=10)
    path = tmp_path / "slow"
    first = asyncio.create_task(worker.run(str(path), timeout=5))
    try:
        await _written(path)
        pid = worker.pid
        with pytest.raises(worker_module.DiarizationBusyError):
            await worker.run(str(tmp_path / "never_decoded"), timeout=5)
        assert worker.pid == pid
        assert await first
    finally:
        first.cancel()
        worker.unload()


@pytest.mark.asyncio
async def test_memory_limit_reaps_worker_and_does_not_poison_next_job(tmp_path, worker_module):
    worker = worker_module.DiarizationWorker(memory_mb=64, max_jobs=10)
    path = tmp_path / "memory"
    try:
        with pytest.raises(worker_module.DiarizationMemoryError):
            await worker.run(str(path), timeout=5)
        assert not _running(int(path.read_text()))
        assert not worker.is_loaded
        assert await worker.run(str(tmp_path / "ok"), timeout=5)
    finally:
        worker.unload()


@pytest.mark.asyncio
async def test_crash_is_reported_and_next_job_starts_fresh(tmp_path, worker_module):
    worker = worker_module.DiarizationWorker(memory_mb=256, max_jobs=10)
    try:
        with pytest.raises(RuntimeError, match="7|退出|exited"):
            await worker.run(str(tmp_path / "crash"), timeout=5)
        assert not worker.is_loaded
        assert await worker.run(str(tmp_path / "ok"), timeout=5)
    finally:
        worker.unload()


@pytest.mark.asyncio
async def test_failed_job_recycles_pipeline_instead_of_reusing_partial_state(tmp_path, worker_module):
    worker = worker_module.DiarizationWorker(memory_mb=256, max_jobs=10)
    try:
        with pytest.raises(RuntimeError, match="bad audio"):
            await worker.run(str(tmp_path / "fail"), timeout=5)
        assert not worker.is_loaded
        assert await worker.run(str(tmp_path / "ok"), timeout=5)
    finally:
        worker.unload()


@pytest.mark.asyncio
async def test_job_limit_recycles_worker_after_success(tmp_path, worker_module):
    worker = worker_module.DiarizationWorker(memory_mb=256, max_jobs=1)
    try:
        assert await worker.run(str(tmp_path / "ok"), timeout=5)
        assert not worker.is_loaded
        assert worker.pid is None
    finally:
        worker.unload()


@pytest.mark.asyncio
async def test_pause_unload_generation_check_is_inside_job_lock(tmp_path, worker_module):
    worker = worker_module.DiarizationWorker(memory_mb=256, max_jobs=10)
    path = tmp_path / "slow"
    first = asyncio.create_task(worker.run(str(path), timeout=5))
    try:
        await _written(path)
        state = {"abort": False}
        unloaded = asyncio.create_task(asyncio.to_thread(worker.unload, lambda: state["abort"]))
        state["abort"] = True  # resume 发生在等锁期间
        await first
        assert await unloaded is False
        assert worker.is_loaded
    finally:
        first.cancel()
        worker.unload()


@pytest.mark.parametrize("kwargs", [
    {"memory_mb": 0}, {"memory_mb": -1}, {"memory_mb": float("nan")},
    {"max_jobs": 0}, {"max_jobs": -2},
])
def test_invalid_worker_limits_cannot_disable_protection(kwargs):
    with pytest.raises(ValueError):
        _module().DiarizationWorker(**kwargs)


@pytest.mark.asyncio
async def test_parent_sigkill_does_not_leave_a_running_worker(tmp_path):
    import signal
    marker = tmp_path / "worker-pid"
    parent = subprocess.Popen([sys.executable, "-m", "tests.diarization_parent_stub", str(marker)],
                              cwd=str(Path(__file__).resolve().parent.parent))
    worker_pid = None
    try:
        await _written(marker)
        worker_pid = int(marker.read_text())
        assert _running(worker_pid)
        parent.kill()  # 只终止本测试拥有的父进程，绕过优雅关闭来测试真实协议
        await asyncio.to_thread(parent.wait, timeout=3)
        async def wait_for_child():
            while _running(worker_pid):
                await asyncio.sleep(0.02)
        await asyncio.wait_for(wait_for_child(), 2)
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=3)
        if worker_pid and _running(worker_pid):
            os.killpg(worker_pid, signal.SIGKILL)


@pytest.mark.asyncio
async def test_unreaped_process_is_quarantined_not_reused_or_replaced(monkeypatch):
    module = _module()
    assert hasattr(module, "DiarizationStopError"), "未退出的进程需要明确隔离状态"

    class StuckProcess:
        pid = 987654321
        alive = True

        def poll(self):
            return None if self.alive else 0

        def wait(self, timeout):
            if self.alive:
                raise subprocess.TimeoutExpired("stuck", timeout)
            return 0

    class Connection:
        def send(self, job):
            raise AssertionError("不能向已终止但未回收的进程继续提交")

        def close(self):
            pass

    def forbidden_spawn(*args, **kwargs):
        raise AssertionError("旧进程未退出时不能再创建一个")

    monkeypatch.setattr(module.os, "killpg", lambda *args: None)
    monkeypatch.setattr(module.subprocess, "Popen", forbidden_spawn)
    worker = module.DiarizationWorker(settings={})
    process = StuckProcess()
    worker._process, worker._connection, worker._loaded = process, Connection(), True
    try:
        with pytest.raises(module.DiarizationStopError):
            worker.unload()
        assert not worker.is_loaded
        with pytest.raises(module.DiarizationStopError):
            await worker.run("unused", timeout=1)
        assert worker._process is process  # 仍拥有句柄；不能丢弃活进程再生成另一个
        process.alive = False
        assert worker.unload()
        assert worker.pid is None
    finally:
        process.alive = False
        worker.unload()

