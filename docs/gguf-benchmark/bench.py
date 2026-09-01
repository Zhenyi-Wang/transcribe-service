#!/usr/bin/env python3
# coding=utf-8
"""Qwen3-ASR GGUF 后端起基准：与生产配置一致（含 aligner），多轮 RTF 对比。

用法（funasr 环境 + 项目 PYTHONPATH，同 run.sh）：
    PYTHONPATH=~/ownprojects/transcribe-service python docs/gguf-benchmark/bench.py \
        --audio <mp3> --runs 5

输出：每轮 RTF 与平均值；JSON 结果 append 到 results_<lib_date>.json。
"""
import argparse
import json
import time
from pathlib import Path

from qwen_asr_gguf.inference.schema import ASREngineConfig, AlignerConfig
from qwen_asr_gguf.inference.asr import QwenASREngine
from qwen_asr_gguf.inference.audio import load_audio

DEFAULT_MODEL_DIR = str(Path.home() / "models/qwen3-asr-gguf")


def run_once(engine, audio, chunk_size=40.0, memory=2):
    t0 = time.time()
    res = engine.asr(
        audio=audio,
        context=None,
        language=None,
        chunk_size_sec=chunk_size,
        memory_chunks=memory,
        temperature=0.4,
        rollback_num=5,
    )
    wall = time.time() - t0
    return wall, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=str(Path.home() / "ownprojects/transcribe-service/test/test.mp3"))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--llm", default="qwen3_asr_llm.q4_k.gguf")
    ap.add_argument("--out", default=None, help="JSON 结果文件（默认 docs/gguf-benchmark/results_<libmtime>.json）")
    args = ap.parse_args()

    config = ASREngineConfig(
        model_dir=args.model_dir,
        llm_fn=args.llm,
        encoder_frontend_fn="qwen3_asr_encoder_frontend.int4.onnx",
        encoder_backend_fn="qwen3_asr_encoder_backend.int4.onnx",
        onnx_provider="CUDA",
        llm_use_gpu=True,
        enable_aligner=True,
        align_config=AlignerConfig(
            model_dir=args.model_dir,
            llm_fn="qwen3_aligner_llm.q4_k.gguf",
            encoder_frontend_fn="qwen3_aligner_encoder_frontend.int4.onnx",
            encoder_backend_fn="qwen3_aligner_encoder_backend.int4.onnx",
            onnx_provider="CUDA",
        ),
    )

    audio = load_audio(args.audio)
    duration = len(audio) / 16000
    print(f"audio: {args.audio}  duration={duration:.2f}s  runs={args.runs}  llm={args.llm}")

    engine = QwenASREngine(config)
    try:
        rows = []
        for i in range(args.runs):
            wall, res = run_once(engine, audio)
            st = res.performance
            rtf = wall / duration
            rows.append({
                "run": i + 1,
                "rtf": round(rtf, 4),
                "wall_s": round(wall, 2),
                "encode_s": round(st.get("encode_time", 0), 2),
                "prefill_s": round(st.get("prefill_time", 0), 2),
                "prefill_tokens": st.get("prefill_tokens", 0),
                "decode_s": round(st.get("decode_time", 0), 2),
                "decode_tokens": st.get("decode_tokens", 0),
                "align_s": round(st.get("align_time", 0), 2),
                "text_len": len(res.text),
            })
            print(f"  run{i+1}: RTF={rtf:.4f} wall={wall:.2f}s "
                  f"(enc={rows[-1]['encode_s']}s pref={rows[-1]['prefill_s']}s "
                  f"gen={rows[-1]['decode_s']}s align={rows[-1]['align_s']}s)")
            rtf_avg = sum(r["rtf"] for r in rows) / len(rows)
            print(f"  -> avg RTF={rtf_avg:.4f}")

        summary = {
            "audio": args.audio,
            "llm": args.llm,
            "runs": rows,
            "avg_rtf": round(sum(r["rtf"] for r in rows) / len(rows), 4),
            "lib_mtime": Path().cwd().joinpath("lib", "libllama.so").stat().st_mtime if False else None,
        }

        import os
        from datetime import datetime
        lib = os.path.expanduser("~/ownprojects/transcribe-service/lib/libllama.so")
        lib_m = datetime.fromtimestamp(os.path.getmtime(lib)).strftime("%Y%m%d_%H%M")
        summary["lib_mtime"] = lib_m
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_{lib_m}.json")
        with open(out, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(f"saved -> {out}  avg_rtf={summary['avg_rtf']}")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
