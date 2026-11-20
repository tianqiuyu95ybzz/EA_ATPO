# AWQ LoRA smoke status

**The required model-level smoke passed** after GPU 5 became available and passed both pre-load no-sharing checks. GPUs 1 and 7 remain prohibited. This is a compatibility smoke with synthetic non-medical text, not Expert A or B training and not evidence of medical quality.

A deliberate earlier gate test on occupied GPU 0 stopped before any AWQ weight was loaded. Its preserved `outputs/awq_smoke_gatecheck_20261120/smoke_report.json` records `stage=resource_gate` and zero optimizer steps. That diagnostic labels the resource gate `FAILED`; it is not an AWQ/PEFT incompatibility result.

The successful attempt used Qwen3-32B-AWQ revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`, GPTQModel 7.12.0 `AwqTorchLinear`, PEFT 0.21.3 and PyTorch 2.15.0+cu132. All source-lock model and tokenizer hashes matched. Ten actual AdamW steps each produced finite loss, nonzero LoRA gradients and a changed LoRA parameter. The rank-16 adapter was saved, loaded onto a newly loaded AWQ base, matched the saved tensors exactly, and produced a nonempty synthetic inference output after reload. Its SHA-256 is `0962465510ba0b3027c8be66de77f2e931e4f2853a19d5680a7f827b80ca8368`. The checkpoint remains in ignored PFS output and is not an A/B adapter.

The attempt took 160.351 seconds on NVIDIA B20Z GPU 5. PyTorch peak allocated/reserved memory was 21,455.2/21,766 MiB; the observed device usage after the steps was 22,646 MiB. No foreign GPU process appeared at the checks. Per-step losses and gradient norms, model hashes, package versions and seed are in `data/awq_smoke_manifest.json`.

Transformers emitted a `qzeros_format` missing-buffer warning for seven projection types across the model. Inspection of the installed GPTQModel code shows that the selected `AwqTorchLinear` forward directly dequantizes the stored `qweight`, `qzeros` and `scales`; the warning concerns an inherited format marker. This inspection and the successful smoke do not establish numerical parity with vLLM or clinical quality, so the warning remains a recorded limitation rather than an assumed accuracy guarantee.

Once an eligible card is free, run a fresh attempt using the existing isolated AWQ Python. Replace `GPU` with an allowed physical GPU and use a new output path:

```bash
CUDA_VISIBLE_DEVICES=GPU /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/experiments/expert_sft/multiturn_dialogue/.venv_awq/bin/python \
  experiments/ea_atpo_v3/expert_sft/scripts/verify_awq_lora.py --gpu GPU \
  --model-dir /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ \
  --output-dir experiments/ea_atpo_v3/expert_sft/outputs/awq_smoke_NEW
```

Formal A and B training remains blocked by the incomplete human review and must pass a fresh resource gate at launch. No BF16, other model or alternative training algorithm was selected as a fallback.
