#!/usr/bin/env python3
"""sherpa-onnx 说话人分离基准测试：RTF / 峰值内存 / 分离结果"""
import sys, time, wave, resource
import numpy as np
import sherpa_onnx

def load_wav(path):
    with wave.open(path) as w:
        assert w.getsampwidth() == 2, "expect 16bit wav"
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        sr = w.getframerate()
    return samples, sr

def main():
    seg_model = sys.argv[1]
    emb_model = sys.argv[2]
    wav_path = sys.argv[3]
    num_speakers = int(sys.argv[4]) if len(sys.argv) > 4 else -1
    threshold = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5

    t0 = time.perf_counter()
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=seg_model, window_shift_ratio=0.1,
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb_model),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers, threshold=threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    assert config.validate(), "config invalid"
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    t_load = time.perf_counter() - t0

    samples, sr = load_wav(wav_path)
    duration = len(samples) / sr

    t0 = time.perf_counter()
    result = sd.process(samples=samples)
    t_proc = time.perf_counter() - t0

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"audio={duration:.1f}s  load={t_load:.1f}s  process={t_proc:.1f}s  RTF={t_proc/duration:.3f}  peak_rss={peak_rss:.0f}MB")

    segs = result.sort_by_start_time()
    speakers = sorted({s.speaker for s in segs})
    print(f"speakers={len(speakers)}: {speakers}")
    for s in segs:
        print(f"  [{s.start:7.2f} - {s.end:7.2f}] spk{s.speaker}")

if __name__ == "__main__":
    main()
