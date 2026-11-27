import json
from pathlib import Path
import sys
from time import perf_counter

import torch

assert torch.cuda.device_count() == 1
capability = torch.cuda.get_device_capability()
assert capability == (12, 0), capability
assert torch.cuda.is_bf16_supported()
weights = Path(__file__).resolve().parent / "model"
free_bytes, total_bytes = torch.cuda.mem_get_info()
assert free_bytes >= 28 * 2**30, free_bytes
torch.cuda.set_per_process_memory_fraction(24 * 2**30 / total_bytes, 0)
left = torch.ones((64, 64), dtype=torch.bfloat16, device="cuda")
assert (left @ left)[0, 0].item() == 64
del left
torch.cuda.synchronize()
print(json.dumps({
    "python": sys.version.split()[0], "torch": torch.__version__,
    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
    "capability": capability, "arch_list": torch.cuda.get_arch_list(),
    "bf16_matmul": "pass", "free_before_mib": free_bytes // 2**20,
}), flush=True)
if sys.argv[1:] == ["--kernel-only"]:
    sys.exit(0)

import accelerate
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
start = perf_counter()
tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    weights, local_files_only=True, use_safetensors=True,
    dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager",
).eval()
torch.cuda.synchronize()
load_seconds = perf_counter() - start
assert next(model.parameters()).dtype == torch.bfloat16
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "请用一句话说明为什么天空是蓝色的。"}],
    tokenize=True, add_generation_prompt=True, enable_thinking=False,
    return_dict=True, return_tensors="pt",
).to("cuda")
torch.cuda.synchronize()
start = perf_counter()
with torch.inference_mode():
    result = model.generate(**inputs, max_new_tokens=8, do_sample=False)
torch.cuda.synchronize()
generation_seconds = perf_counter() - start
new_tokens = result[0, inputs.input_ids.shape[1]:]
assert new_tokens.numel() > 0
print(json.dumps({
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__, "model_dtype": str(model.dtype),
    "batch_size": 1, "load_seconds": round(load_seconds, 3),
    "generation_seconds": round(generation_seconds, 3),
    "generated_tokens": new_tokens.numel(),
    "tokens_per_second": round(new_tokens.numel() / generation_seconds, 2),
    "peak_allocated_mib": torch.cuda.max_memory_allocated() // 2**20,
    "peak_reserved_mib": torch.cuda.max_memory_reserved() // 2**20,
    "generated_text": tokenizer.decode(new_tokens, skip_special_tokens=True),
}, ensure_ascii=False), flush=True)
