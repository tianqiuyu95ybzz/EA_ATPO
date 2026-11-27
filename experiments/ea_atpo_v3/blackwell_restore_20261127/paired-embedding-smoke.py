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
from experiments.ea_atpo_v2.training.extract_state_embeddings import PublicState, extract_batches

initial = [{"role": "user", "content": "我的植物叶片最近发黄，可能是什么原因？"}]
after_ask = initial + [
    {"role": "assistant", "content": "最近浇水频率和盆底排水情况如何？"},
    {"role": "user", "content": "每天浇水，托盘经常有积水。"},
]
states = [initial, after_ask]
samples = []
with torch.inference_mode():
    for messages in states:
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors="pt",
        ).to("cuda")
        torch.cuda.synchronize()
        start = perf_counter()
        result = model.generate(**tokens, max_new_tokens=8, do_sample=False)
        torch.cuda.synchronize()
        new_tokens = result[0, tokens.input_ids.shape[1]:]
        assert new_tokens.numel() > 0
        samples.append({
            "tokens": new_tokens.numel(),
            "seconds": round(perf_counter() - start, 3),
            "text": tokenizer.decode(new_tokens, skip_special_tokens=True),
        })
    public_states = [
        PublicState("synthetic-1", name, tuple(messages))
        for name, messages in zip(("initial", "after_ask"), states)
    ]
    start = perf_counter()
    vectors, prompt_hashes = extract_batches(
        model.model, tokenizer, public_states, device="cuda",
        max_length=256, batch_size=1, normalize_l2=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    torch.cuda.synchronize()
    embedding_seconds = perf_counter() - start
assert len(vectors) == 2
assert all(len(vector) == model.config.hidden_size for vector in vectors)
assert prompt_hashes[0] != prompt_hashes[1]
with (weights.parent / "synthetic-embeddings.json").open("w", encoding="utf-8") as output:
    json.dump({"vectors": vectors, "prompt_hashes": prompt_hashes}, output)
print(json.dumps({
    "batch_size": 1, "load_seconds": round(load_seconds, 3),
    "paired_generations": samples, "embedding_count": len(vectors),
    "embedding_dimensions": len(vectors[0]),
    "embedding_seconds": round(embedding_seconds, 3),
    "peak_allocated_mib": torch.cuda.max_memory_allocated() // 2**20,
    "peak_reserved_mib": torch.cuda.max_memory_reserved() // 2**20,
}, ensure_ascii=False), flush=True)
