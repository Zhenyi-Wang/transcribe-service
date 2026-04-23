"""导出 0.6B ForcedAligner Decoder 为 F16 GGUF"""
import sys
import os
import json
import torch
from pathlib import Path

ALIGNER_MODEL_DIR = "/tmp/qwen3-aligner-0.6b"
EXPORT_DIR = "/tmp/aligner_decoder_hf"
MODEL_GGUF = os.path.expanduser("~/models/qwen3-asr-gguf/qwen3_aligner_llm.f16.gguf")

os.makedirs(EXPORT_DIR, exist_ok=True)

print("=" * 60)
print("Step 1: 提取 Aligner Decoder 权重")
print("=" * 60)

sys.path.insert(0, "/home/zhenyi/.local/lib/python3.10/site-packages")
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration

print(f"Loading model from: {ALIGNER_MODEL_DIR}")
model = Qwen3ASRForConditionalGeneration.from_pretrained(
    ALIGNER_MODEL_DIR, trust_remote_code=True, device_map="cpu", torch_dtype=torch.float32,
    local_files_only=True
)

text_config = model.config.thinker_config.text_config
llm_config_dict = text_config.to_dict()
llm_config_dict["architectures"] = ["Qwen3VLForConditionalGeneration"]
llm_config_dict["model_type"] = "qwen3_vl"

with open(os.path.join(EXPORT_DIR, "config.json"), "w", encoding="utf-8") as f:
    json.dump(llm_config_dict, f, indent=2, ensure_ascii=False)
print("Saved config.json")

from safetensors.torch import save_file

as_state_dict = model.state_dict()
new_state_dict = {}
for key in as_state_dict:
    if key.startswith("thinker.model."):
        new_key = key.replace("thinker.model.", "model.")
        new_state_dict[new_key] = as_state_dict[key]
    elif key.startswith("thinker.lm_head."):
        new_key = key.replace("thinker.lm_head.", "lm_head.")
        new_state_dict[new_key] = as_state_dict[key].clone()

print(f"Extracted {len(new_state_dict)} tensors")
save_file(new_state_dict, os.path.join(EXPORT_DIR, "model.safetensors"))
print("Saved model.safetensors")

from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(ALIGNER_MODEL_DIR, trust_remote_code=True)
    tokenizer.save_pretrained(EXPORT_DIR)
    print("Saved tokenizer")
except Exception as e:
    print(f"Warning: Failed to save tokenizer: {e}")

del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\nStep 1 complete!\n")

print("=" * 60)
print("Step 2: 转换为 F16 GGUF")
print("=" * 60)

CONVERT_LIB_DIR = "/tmp/Qwen3-ASR-GGUF/qwen_asr_gguf/export"
sys.path.insert(0, CONVERT_LIB_DIR)

import convert_hf_to_gguf
from convert_hf_to_gguf import ModelBase, TextModel

def patched_load_hparams(dir_model, is_mistral_format):
    with open(os.path.join(dir_model, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    if "llm_config" in config:
        config["text_config"] = config["llm_config"]
    if "thinker_config" in config:
        config["text_config"] = config["thinker_config"]["text_config"]
    return config

def patched_get_vocab_base_pre(self, tokenizer):
    return "qwen2"

ModelBase.load_hparams = staticmethod(patched_load_hparams)
TextModel.get_vocab_base_pre = patched_get_vocab_base_pre

sys.argv = [
    "convert_hf_to_gguf.py",
    EXPORT_DIR,
    "--outfile", MODEL_GGUF,
    "--outtype", "f16",
    "--verbose"
]

print(f"Converting to F16 GGUF...")
try:
    convert_hf_to_gguf.main()
    print(f"\nF16 GGUF saved: {MODEL_GGUF}")
except Exception as e:
    print(f"Conversion failed: {e}")
    import traceback
    traceback.print_exc()
