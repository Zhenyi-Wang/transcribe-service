#!/usr/bin/env python3
"""pyannote 聚类超参扫描：threshold × min_cluster_size"""
import sys, time, wave
import numpy as np
import torch

def load_wav(path):
    with wave.open(path) as w:
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return samples, w.getframerate()

def main():
    from pyannote.audio import Pipeline
    wav_path = sys.argv[1]
    thresholds = [float(x) for x in sys.argv[2].split(",")]
    min_sizes = [int(x) for x in sys.argv[3].split(",")]

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipeline.to(torch.device("cuda"))

    samples, sr = load_wav(wav_path)
    waveform = torch.from_numpy(samples).unsqueeze(0)
    audio = {"waveform": waveform, "sample_rate": sr}

    print(f"{'thr':>5} {'minsz':>5} | {'spk':>3} {'turns':>5} {'proc':>5} | 各簇时长分布")
    for thr in thresholds:
        for ms in min_sizes:
            pipeline.instantiate({
                "segmentation": {"min_duration_off": 0.0},
                "clustering": {"method": "centroid", "threshold": thr, "min_cluster_size": ms},
            })
            t0 = time.perf_counter()
            diar = pipeline(audio)
            dt = time.perf_counter() - t0
            from collections import defaultdict
            stat = defaultdict(float)
            turns = 0
            ranges = defaultdict(list)
            for turn, _, spk in diar.itertracks(yield_label=True):
                stat[spk] += turn.end - turn.start
                turns += 1
                ranges[spk].append((round(turn.start), round(turn.end)))
            dist = ", ".join(f"{k[-2:]}:{v:.0f}s" for k, v in sorted(stat.items(), key=lambda x: -x[1]))
            print(f"{thr:>5} {ms:>5} | {len(stat):>3} {turns:>5} {dt:>4.1f}s | {dist}")
            # 输出小簇的时间范围（用于与文本对照）
            for k, v in sorted(stat.items(), key=lambda x: x[1])[:-1] if len(stat) > 1 else []:
                segs = sorted(ranges[k])[:6]
                print(f"        {k[-2:]} 段落样例: {segs}")

if __name__ == "__main__":
    main()
