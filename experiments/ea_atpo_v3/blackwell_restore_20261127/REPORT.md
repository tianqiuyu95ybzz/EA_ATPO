# V3 Blackwell restore and BF16 smoke

The restore used the V3 migration commit `f3fb006ebcfc464911d43520def073cbb95a0eda`. The checkout and fetched upstream matched. The checksum sidecar and all 35 migration payload entries passed on the server. Five CPU Oracle unit tests passed.

## Runtime and model

- NVIDIA RTX PRO 6000 Blackwell Server Edition, GPU 0, compute capability 12.0, NVIDIA driver 580.95.05.
- Python 3.11.15, PyTorch 2.11.0+cu130 with an sm_120 build, CUDA runtime 13.0, Transformers 5.6.0 and Accelerate 1.13.0. Native BF16 matrix multiplication passed.
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218) at revision `b968826d9c46dd6066d109eabc6255188de91218`. The direct Hub endpoint timed out from the host. Files were retrieved via a mirror and verified against the existing audit digests. All five safetensors shards and five model metadata files passed SHA-256. This run used raw pretrained weights.

## Results

| Check | Batch | Result | PyTorch peak allocated |
| --- | --- | --- | --- |
| One synthetic prompt | 1 | Load 5.477 s, 8 tokens in 1.732 s (4.62 tokens/s) | 15,636 MiB |
| One scripted pair | 1 | Load 4.953 s, two 8-token generations in 0.855 s and 0.330 s | 15,644 MiB for paired generation and embedding together |
| Frozen public-state embedding extraction | 1 | Two finite 4,096-dimensional vectors in 0.244 s | Included in the paired run |

The scripts used seed 0, BF16, eager attention, the snapshot chat template, thinking disabled and greedy decoding. Both runs finished with exit status 0. The pair used a synthetic non-medical dialogue with scripted feedback and the existing V2 embedding extractor. These are engineering checks. No clinical scoring or Router fitting was performed. Transformers warned that sampling flags inherited from the snapshot may be ignored during greedy decoding.

Both GPUs had active workloads. GPU 0 showed 95–99% device utilization during the initial continuous check and about 79,643 MiB free. The smoke process peaked at 16,310 MiB in periodic NVIDIA process monitoring. The pre-existing compute process and its memory snapshot matched before and after both runs. The other workload's application throughput was unavailable, so the sample cannot establish its performance impact or sustained batch capacity.

## Reproduction

Start from the restored migration checkout and run the commands from its root. Copy the two Python files in this directory to an isolated run directory and place the verified model files in its `model/` subdirectory. Set `RUN_DIR` to that absolute run directory and activate the environment described above.

```bash
sha256sum -c MIGRATION_SHA256SUMS.sha256
sha256sum -c MIGRATION_SHA256SUMS
awk '$2 ~ /\/Qwen3-8B\// {sub(/^.*\//,"",$2); print $1 "  " $2}' \
  experiments/ea_atpo_v3/atpo_checkpoint_audit_20261121/MODEL_SHA256SUMS > "$RUN_DIR/base.sha256"
(cd "$RUN_DIR/model" && sha256sum -c ../base.sha256)
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s experiments/ea_atpo_v3/oracle_expert/tests -v
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
timeout 180 python -u "$RUN_DIR/smoke-bf16.py"
PYTHONPATH="$PWD" timeout 120 python -u "$RUN_DIR/paired-embedding-smoke.py"
```

On the server, raw monitoring logs, output vectors and verified model files are kept in an access-controlled run directory. The generation log SHA-256 is `5597c8228eb4adae0f6c91738d814d383754415586ca5c9edf7d0e31be10d3c5`. The synthetic embedding JSON SHA-256 is `d480a145f85f6e81546dfb0e20c58cadfd07652bd96b5069911bacf571879ff7`. The previous PFS mount was unavailable. Formal runs still require a usable case cohort and validated feedback, reference answers and scoring. A formal SFT→ATPO 8B Base also requires verified checkpoint lineage.
