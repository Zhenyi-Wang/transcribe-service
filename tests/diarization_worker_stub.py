"""工作进程生命周期测试用替身：真实子进程/Pipe，不加载模型。"""
import os
import signal
import subprocess
import sys
import time
from multiprocessing.connection import Connection
from pathlib import Path


def main():
    conn = Connection(int(sys.argv[1]))
    conn.send({"event": "ready"})
    while True:
        job = conn.recv()
        path = Path(job["audio_path"])
        if path.name == "hang":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            # 解码器一样会有子进程；超时必须连同它回收。
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
            path.write_text(f"{os.getpid()} {child.pid}")
            time.sleep(120)
        if path.name == "memory":
            path.write_text(str(os.getpid()))
            buffer = bytearray(128 * 1024 * 1024)
            time.sleep(120)
        if path.name == "fail":
            conn.send({"event": "error", "message": "bad audio"})
            continue
        if path.name == "crash":
            os._exit(7)
        if path.name == "slow":
            path.write_text(str(os.getpid()))
            time.sleep(0.2)
        conn.send({"event": "result", "turns": [[0.0, 4.0, 0], [4.0, 8.0, 1]],
                   "audio_duration": 8.0})


def production_main():
    """只替换模型/音频依赖，执行生产 _main 的 Pipe 与父进程退出协议。"""
    import threading
    import types
    from diarization.manager import _writable_samples

    class Samples:
        nbytes = 64000
        flags = types.SimpleNamespace(writeable=True)

        def __init__(self, path):
            self.path = Path(path)

        def __len__(self):
            return 16000

    class FakeManager:
        def __init__(self, **kwargs):
            self._infer_lock = threading.Lock()

        def _load(self):
            pass

        def diarize(self, samples):
            samples.path.write_text(str(os.getpid()))
            time.sleep(120)
            return []

    manager = types.ModuleType("diarization.manager")
    manager.DiarizationManager = FakeManager
    manager.SpeakerTurn = types.SimpleNamespace
    manager.get_manager = FakeManager
    manager._writable_samples = _writable_samples
    audio = types.ModuleType("qwen_asr_gguf.inference.audio")
    audio.load_audio = Samples
    sys.modules["diarization.manager"] = manager
    sys.modules["qwen_asr_gguf.inference.audio"] = audio
    from diarization.worker import _main
    _main(int(sys.argv[1]))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "production":
        production_main()
    else:
        main()
