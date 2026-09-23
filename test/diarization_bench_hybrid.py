#!/usr/bin/env python3
"""pyannote GPU 分割/批量推理 + wespeaker CN-Celeb 中文 embedding（ONNX）混合管线"""
import sys, time, resource, wave
import numpy as np
import torch

def load_wav(path):
    with wave.open(path) as w:
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return samples, w.getframerate()

def main():
    from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
    wav_path = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7046
    onnx_path = "/home/zhenyi/models/diarization/wespeaker_cnceleb_resnet34_LM.onnx"

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pipeline = SpeakerDiarization(
        segmentation="pyannote/segmentation-3.0",
        embedding=onnx_path,               # 本地 ONNX → ONNXWeSpeakerPretrainedSpeakerEmbedding
        clustering="AgglomerativeClustering",
        segmentation_batch_size=32,
        embedding_batch_size=32,
    )
    pipeline.instantiate({
        "segmentation": {"min_duration_off": 0.0},
        "clustering": {"method": "centroid", "threshold": thr, "min_cluster_size": 12},
    })
    pipeline.to(torch.device("cuda"))
    t_load = time.perf_counter() - t0

    samples, sr = load_wav(wav_path)
    duration = len(samples) / sr
    waveform = torch.from_numpy(samples).unsqueeze(0)

    t0 = time.perf_counter()
    diar = pipeline({"waveform": waveform, "sample_rate": sr})
    t_proc = time.perf_counter() - t0

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"audio={duration:.1f}s  load={t_load:.1f}s  process={t_proc:.1f}s  RTF={t_proc/duration:.3f}")
    print(f"VRAM peak alloc={peak_alloc:.0f}MB  host_rss={peak_rss:.0f}MB  thr={thr}")

    from collections import defaultdict
    stat = defaultdict(float)
    ranges = defaultdict(list)
    for turn, _, spk in diar.itertracks(yield_label=True):
        stat[spk] += turn.end - turn.start
        ranges[spk].append((round(turn.start, 1), round(turn.end, 1)))
    print(f"speakers={len(stat)}")
    for spk, dur in sorted(stat.items(), key=lambda x: -x[1]):
        segs = sorted(ranges[spk])
        head = segs[:8] if len(segs) > 8 else segs
        print(f"  {spk}: {dur:.0f}s, {len(segs)} 段, 样例: {head}")

if __name__ == "__main__":
    main()
