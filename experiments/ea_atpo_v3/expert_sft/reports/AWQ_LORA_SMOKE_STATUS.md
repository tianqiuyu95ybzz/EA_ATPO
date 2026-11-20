# AWQ LoRA smoke status

**Blocked before model load by GPU resource policy.** At the latest comprehensive check, every allowed GPU (0, 2, 3, 4, 5, 6) had an existing compute process. GPU 5 had approximately 179,609 MiB free but had a foreign process, so it also failed the zero-other-process gate. GPUs 1 and 7 are prohibited. No process was interrupted.

A deliberate gate test on occupied GPU 0 stopped before any AWQ weight was loaded. Its preserved `outputs/awq_smoke_gatecheck_20261120/smoke_report.json` records `stage=resource_gate` and zero optimizer steps. That older diagnostic labels the run `FAILED`, meaning the resource gate failed; it is **not** evidence that AWQ or PEFT is incompatible. There has been no model forward, backward, optimizer update, checkpoint round-trip or inference in this V3 smoke.

`scripts/verify_awq_lora.py` is prepared to use only synthetic non-medical text, the pinned Qwen3-32B-AWQ source lock and GPTQModel `torch_awq`, a new independent adapter and output directory. It enforces the no-sharing gate immediately before model load and the 32-GiB post-load reserve, checks finite nonzero LoRA gradients and changed weights for 10 actual optimizer steps, saves and reloads the adapter, checks tensor identity and attempts a real post-reload generation. Since no safe GPU was available, those model-level paths are **not yet validated**, and this script must not be treated as proof of training support.

Once an eligible card is free, run a fresh attempt using the existing isolated AWQ Python. Replace `GPU` with an allowed physical GPU and use a new output path:

```bash
CUDA_VISIBLE_DEVICES=GPU /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/experiments/expert_sft/multiturn_dialogue/.venv_awq/bin/python \
  experiments/ea_atpo_v3/expert_sft/scripts/verify_awq_lora.py --gpu GPU \
  --model-dir /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ \
  --output-dir experiments/ea_atpo_v3/expert_sft/outputs/awq_smoke_NEW
```

The script's source and small-file syntax checks are not a substitute for that real run. Formal A and B training is blocked until the smoke, completed human review and applicable resource gates all pass. No BF16, other model or alternative training algorithm is selected as a fallback.
