# Environment Audit — 2026-11-19

- Qwen3-32B BF16 revision: `9216db5781bf21249d130ec9da846c4624c16137`
- Model download complete: **False** (0 / None shards; 0 complete shard bytes; 8 partial files, 8,807,862,064 partial bytes; 8,809,565,036 bytes in the model directory)
- NVIDIA runtime visible through `nvidia-smi`: **True**; PyTorch build: `cu128`; BF16 capability: **not yet verified** (will be checked by the CUDA smoke).
- Workspace free disk: **42.23 TiB**
- GPUs failing the reserved SFT gate (>=150,000 MiB free, <=20% utilization): GPU 0: 110835 MiB free, 0% util, 1 compute processes, GPU 1: 110835 MiB free, 61% util, 1 compute processes, GPU 2: 44071 MiB free, 0% util, 3 compute processes, GPU 3: 101237 MiB free, 1% util, 1 compute processes, GPU 4: 96725 MiB free, 98% util, 4 compute processes, GPU 5: 118047 MiB free, 82% util, 3 compute processes, GPU 6: 118729 MiB free, 74% util, 2 compute processes, GPU 7: 106069 MiB free, 76% util, 2 compute processes
- All download, HF, datasets, XDG, TRITON and temporary caches are configured under `experiments/expert_sft`.

## Package Versions

```json
{
  "torch": "2.7.1+cu128",
  "transformers": "4.54.1",
  "peft": "0.21.2",
  "accelerate": "1.15.0",
  "datasets": "4.0.0",
  "pandas": "2.3.3",
  "pyarrow": "19.0.1",
  "huggingface_hub": "0.36.2",
  "trl": null,
  "bitsandbytes": null
}
```

This audit records availability only. It does not start or stop any model service or GPU process.
