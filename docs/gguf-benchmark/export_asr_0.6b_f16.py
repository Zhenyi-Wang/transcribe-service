"""导出 0.6B ASR Decoder 为 F16 GGUF"""
import sys, os, json, torch
sys.path.insert(0, "/home/zhenyi/.local/lib/python3.10/site-packages")

MODEL_DIR = "/tmp/qwen3-asr-0.6b"
EXPORT_DIR = "/tmp/asr_06b_decoder_hf"
MODEL_GGUF = os.path.expanduser("~/models/qwen3-asr-gguf/qwen3_asr_0.6b_llm.f16.gguf")

os.makedirs(EXPORT_DIR, exist_ok=True)

print("Loading 0.6B ASR model...")
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
model = Qwen3ASRForConditionalGeneration.from_pretrained(
    MODEL_DIR, trust_remote_code=True, device_map="cpu", torch_dtype=torch.float32, local_files_only=True
)

text_config = model.config.thinker_config.text_config
d = text_config.to_dict()
d["architectures"] = ["Qwen3VLForConditionalGeneration"]
d["model_type"] = "qwen3_vl"
with open(os.path.join(EXPORT_DIR, "config.json"), "w") as f:
    json.dump(d, f, indent=2, ensure_ascii=False)

from safetensors.torch import save_file
sd = model.state_dict()
new_sd = {}
for k in sd:
    if k.startswith("thinker.model."):
        new_sd[k.replace("thinker.model.", "model.")] = sd[k]
    elif k.startswith("thinker.lm_head."):
        new_sd[k.replace("thinker.lm_head.", "lm_head.")] = sd[k].clone()
print(f"Extracted {len(new_sd)} tensors")
save_file(new_sd, os.path.join(EXPORT_DIR, "model.safetensors"))

from transformers import AutoTokenizer
try:
    AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True).save_pretrained(EXPORT_DIR)
except: pass
del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None
print("Step 1 done\n")

# Convert to GGUF
sys.path.insert(0, "/tmp/Qwen3-ASR-GGUF/qwen_asr_gguf/export")
import convert_hf_to_gguf
from convert_hf_to_gguf import ModelBase, TextModel

def patched_load_hparams(dir_model, is_mistral_format):
    with open(os.path.join(dir_model, "config.json")) as f:
        config = json.load(f)
    if "thinker_config" in config:
        config["text_config"] = config["thinker_config"]["text_config"]
    return config

ModelBase.load_hparams = staticmethod(patched_load_hparams)
TextModel.get_vocab_base_pre = lambda self, tk: "qwen2"

sys.argv = ["convert_hf_to_gguf.py", EXPORT_DIR, "--outfile", MODEL_GGUF, "--outtype", "f16", "--verbose"]
try:
    convert_hf_to_gguf.main()
    print(f"\nSaved: {MODEL_GGUF}")
except Exception as e:
    print(f"Failed: {e}")
    import traceback; traceback.print_exc()
