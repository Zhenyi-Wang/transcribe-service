"""用于父服务异常退出测试：真实工作器，只替换推理依赖。"""
import asyncio
import subprocess
import sys

from diarization import worker as W

popen = subprocess.Popen


def spawn_stub(command, **kwargs):
    return popen([sys.executable, "-m", "tests.diarization_worker_stub", command[3], "production"], **kwargs)


W.subprocess.Popen = spawn_stub
worker = W.DiarizationWorker(memory_mb=256, max_jobs=8, settings={})
try:
    asyncio.run(worker.run(sys.argv[1], timeout=120))
finally:
    worker.unload()
