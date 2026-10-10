"""可回收的单进程分离工作器：父进程只传文件路径，不解码/排队持有 PCM。"""
import asyncio
import atexit
import logging
import math
import os
import signal
import subprocess
import sys
import threading
import time
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path

from config import config

logger = logging.getLogger("diarization")


class DiarizationBusyError(RuntimeError):
    pass


class DiarizationMemoryError(RuntimeError):
    pass


class DiarizationStopError(RuntimeError):
    """进程尚未退出（例如 FUSE 不可中断等待），保留句柄并禁止再提交。"""


def _positive_int(value, name):
    try:
        number = float(value)
        valid = math.isfinite(number) and number >= 1 and number.is_integer()
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} 必须是正整数")
    return int(number)


def _rss_mb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    return 0.0


def _manager_settings():
    return {
        "backend": config.diarization_backend,
        "cluster_threshold": config.diarization_cluster_threshold,
        "min_cluster_size": config.diarization_min_cluster_size,
        "num_speakers": config.diarization_num_speakers,
        "embedding_model": config.diarization_embedding_model,
        "hf_token": config.diarization_hf_token,
        "max_speakers": config.diarization_max_speakers,
        "max_num_embeddings": config.diarization_max_num_embeddings,
        "max_reconstruction_mb": config.diarization_max_reconstruction_mb,
        "reconstruction_batch_chunks": config.diarization_reconstruction_batch_chunks,
        "cudnn_conv_algo_search": config.diarization_cudnn_conv_algo_search,
    }


class DiarizationWorker:
    def __init__(self, memory_mb=None, max_jobs=None, settings=None):
        self.memory_mb = _positive_int(
            memory_mb if memory_mb is not None else config.get("diarization.worker_memory_mb", 4096),
            "diarization.worker_memory_mb")
        self.max_jobs = _positive_int(
            max_jobs if max_jobs is not None else config.get("diarization.worker_max_jobs", 8),
            "diarization.worker_max_jobs")
        self._settings = settings
        self._job_lock = threading.Lock()
        self._process = None
        self._connection = None
        self._loaded = False
        self._jobs = 0
        self._terminating = False

    @property
    def pid(self):
        return self._process.pid if self._process is not None else None

    @property
    def is_loaded(self):
        process = self._process
        return bool(self._loaded and process is not None and process.poll() is None)

    def _start(self):
        if self._process is not None:
            if not self._terminating and self._process.poll() is None:
                return
            self._stop()  # 未退出的旧进程不能复用，也不能丢掉句柄再启动一个
        parent, child = Pipe(duplex=True)
        try:
            # 不使用 fork 或 multiprocessing spawn：前者继承 CUDA，后者重导入 server.py。
            process = subprocess.Popen(
                [sys.executable, "-m", "diarization.worker", str(child.fileno()), str(os.getpid())],
                cwd=str(Path(__file__).resolve().parent.parent),
                pass_fds=(child.fileno(),), start_new_session=True,
            )
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        self._process, self._connection = process, parent
        self._loaded, self._jobs = False, 0
        logger.info("分离工作进程启动: pid=%s rss_limit=%sMiB", process.pid, self.memory_mb)

    def _stop(self):
        process = self._process
        if process is None:
            return False
        self._loaded = False
        self._terminating = True
        # start_new_session 保证该进程组只包含工作器及其 ffmpeg 子进程。
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        # 即使主进程已退出，也清理可能忽略 TERM 的解码子进程。
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            # D 状态不能强杀：仍持有进程/Pipe，之后只重试回收，不重复解码/启动。
            raise DiarizationStopError(f"分离进程 pid={process.pid} 尚未退出，已隔离，禁止再提交") from None
        if self._connection is not None:
            self._connection.close()
        self._connection, self._process = None, None
        self._jobs, self._terminating = 0, False
        logger.info("分离工作进程已回收: pid=%s", process.pid)
        return True

    def unload(self, should_abort=None):
        # 与原暂停协议一致：等当前任务结束后，再检查世代，避免 resume 后误卸载。
        with self._job_lock:
            if should_abort is not None and should_abort():
                return False
            return self._stop()

    async def run(self, audio_path, timeout):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("分离超时必须是有限正数")
        if not self._job_lock.acquire(blocking=False):
            raise DiarizationBusyError("分离工作进程繁忙，本次不解码、不排队")
        deadline = time.monotonic() + timeout
        try:
            self._start()
            # 配置只经私有 Pipe 传递，不进入命令行/日志（包含 HF 凭据）。
            settings = self._settings if self._settings is not None else _manager_settings()
            self._connection.send({"audio_path": audio_path, "settings": settings})
            while True:
                rss = _rss_mb(self.pid)
                if rss > self.memory_mb:
                    raise DiarizationMemoryError(
                        f"分离工作进程 RSS {rss:.0f}MiB 超过 {self.memory_mb}MiB，触发回收")
                if time.monotonic() >= deadline:
                    raise asyncio.TimeoutError()
                if self._connection.poll():
                    try:
                        message = self._connection.recv()
                    except EOFError:
                        code = self._process.wait(timeout=1)
                        raise RuntimeError(f"分离工作进程意外退出: {code}") from None
                    event = message.get("event")
                    if event == "result":
                        from diarization.manager import SpeakerTurn
                        self._loaded = True
                        self._jobs += 1
                        turns = [SpeakerTurn(*t) for t in message["turns"]]
                        logger.info("分离工作进程完成: pid=%s rss=%.0fMiB duration=%.1fs jobs=%s",
                                    self.pid, rss, message.get("audio_duration", 0), self._jobs)
                        if self._jobs >= self.max_jobs:
                            self._stop()
                        return turns
                    if event == "error":
                        raise RuntimeError(message["message"])
                    if event == "loaded":
                        self._loaded = True
                    if event == "decoded":
                        logger.info("分离音频已解码: pid=%s duration=%.1fs pcm=%.1fMiB rss=%.0fMiB",
                                    self.pid, message["audio_duration"], message["pcm_mb"], rss)
                elif self._process.poll() is not None:
                    raise RuntimeError(f"分离工作进程意外退出: {self._process.returncode}")
                await asyncio.sleep(0.05)
        except BaseException as e:
            # wait_for 超时和客户端取消也走这里；隔离状态不重复等待/丢弃活进程。
            if not isinstance(e, DiarizationStopError):
                self._stop()
            raise
        finally:
            self._job_lock.release()


_worker = None
_worker_lock = threading.Lock()


def get_worker():
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = DiarizationWorker()
        return _worker


def unload_global(should_abort=None):
    return _worker.unload(should_abort) if _worker is not None else False


def is_loaded():
    return _worker is not None and _worker.is_loaded


def _exit_cleanup():
    if _worker is not None:
        _worker._stop()


atexit.register(_exit_cleanup)


def _die_with_parent(parent_pid):
    """Linux 在父服务异常退出时杀掉本工作器，不等长推理结束再发现 Pipe 断开。"""
    if sys.platform != "linux":
        return
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if parent_pid <= 1 or os.getppid() != parent_pid:
        raise SystemExit(0)  # 父进程在设置信号前已退出，不启动模型/推理


def _main(fd, parent_pid=None):
    _die_with_parent(parent_pid if parent_pid is not None else os.getppid())
    # 子进程不导入 server/transcribe，不继承父进程的 CUDA/线程/PCM。
    from diarization.manager import DiarizationManager, _writable_samples
    from qwen_asr_gguf.inference.audio import load_audio
    from logger_config import setup_logger
    setup_logger("diarization")
    connection = Connection(fd)
    manager = None
    connection.send({"event": "ready"})
    try:
        while True:
            request = connection.recv()
            try:
                if manager is None:
                    manager = DiarizationManager(**request["settings"])
                # 在加载模型前替换只读缓冲，让原 bytes/array 及时释放，避免长期双份 PCM。
                samples = _writable_samples(load_audio(request["audio_path"]))
                duration = len(samples) / 16000
                connection.send({"event": "decoded", "audio_duration": duration,
                                 "pcm_mb": samples.nbytes / (1024 * 1024)})
                with manager._infer_lock:
                    manager._load()
                connection.send({"event": "loaded"})
                turns = manager.diarize(samples)
                connection.send({"event": "result", "audio_duration": duration,
                                 "turns": [[t.start, t.end, t.speaker] for t in turns]})
                del samples, turns
            except Exception as e:
                logger.exception("分离工作进程任务失败")
                connection.send({"event": "error", "message": f"{type(e).__name__}: {e}"})
                return  # 失败后不复用可能处于半初始化状态的 CUDA/ORT 会话
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


if __name__ == "__main__":
    _main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else None)
