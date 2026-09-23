#!/usr/bin/env python3
"""pyannote 3.1 GPU 基准：RTF / 显存 / 分离结果"""
import sys, time, resource, wave
import numpy as np
import torch

def load_wav(path):
    with wave.open(path) as w:
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        sr = w.getframerate()
    return samples, sr

def main():
    from pyannote.audio import Pipeline
    wav_path = sys.argv[1]
    num_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else None

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipeline.to(torch.device("cuda"))
    t_load = time.perf_counter() - t0

    samples, sr = load_wav(wav_path)
    duration = len(samples) / sr
    waveform = torch.from_numpy(samples).unsqueeze(0)  # (1, T)

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    t0 = time.perf_counter()
    diarization = pipeline({"waveform": waveform, "sample_rate": sr}, **kwargs)
    t_proc = time.perf_counter() - t0

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
    peak_resv = torch.cuda.max_memory_reserved() / 1024**2
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    print(f"audio={duration:.1f}s  load={t_load:.1f}s  process={t_proc:.1f}s  RTF={t_proc/duration:.3f}")
    print(f"VRAM peak: allocated={peak_alloc:.0f}MB  reserved={peak_resv:.0f}MB  host_rss={peak_rss:.0f}MB")

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append((turn.start, turn.end, speaker))
    labels = sorted({t[2] for t in turns})
    print(f"speakers={len(labels)}: {labels}")
    for s, e, spk in sorted(turns):
        print(f"  [{s:7.2f} - {e:7.2f}] {spk}")

if __name__ == "__main__":
    main()
