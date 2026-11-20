# V3 execution and admission decision

Status at this handoff: **formal training and evaluation are blocked**, not completed. The 291 Base responses and a separate real 10-step AWQ LoRA compatibility smoke are complete. All 291 Gold reviews remain `PENDING`. An additional synthetic GPU integration check of the new A/B loss code was interrupted by SIGTERM; its source is unknown and it has no completed result. No PPO/ATPO RL or Router training was started.

| Requested conclusion | Current result |
| --- | --- |
| 1. Expert A trained? | No. No A adapter, optimizer, epoch checkpoint or training metric exists. |
| 2. Expert B trained? | No. Its independent 19,341,527,816-byte AWQ initialization copy is present and hash-verified; no B adapter, optimizer, checkpoint or training metric exists. |
| 3. Expert C verified/trained? | No. `zky001/med_qwen` is blocked: empty local candidate directory, inaccessible file tree, no immutable revision or identified license, and no real inference. |
| 4–7. Highest correction rate, lowest degradation, largest Net Gain, closest to Oracle? | Not determinable: no reviewed Gold, trained Expert outputs or V3 paired evaluation exists. |
| 8. Reliable expert complementarity? | Not established. Historical comparisons are not V3 results. |
| 9. Router admission? | Not met. The numerical admission criteria remain unfrozen and no validated Expert result exists. |
| 10. GitHub? | The isolated branch contains the reproducible code, configs, privacy-safe manifests, audit and status reports. Final local/remote commit equality is checked separately after the last push. |
| 11. Missing/uncommitted artifacts? | Formal A/B/C adapters, optimizer checkpoints, reviewed Gold, paired dataset and evaluation results do not exist. Private data, the B weight copy and smoke adapter are intentionally Git-ignored PFS artifacts; see `data/pfs_artifact_inventory_v2.json`. |

The source audit independently verified 205/42/44 states and 70/15/15 case-isolated Train/Dev/Internal Test splits. They originate from simulated patient dialogues, including 100 zero-reply states; the 14,256 original ATPO rows are single-turn. The V2 50-case/146-state/438-trajectory Pilot overlaps Train exactly and was not added as extra training data. The 400-case frozen holdout was not opened.

The new verified Base responses match the 291 frozen requests and include raw-output SHA-256 values. The private human review queue contains all of them. Its priority list flags 169 disagreements and 122 agreements with **unreviewed historical** labels; neither count is Base accuracy or a clinical judgment. A real reviewer must assess all 291, record risk/evidence and mark ambiguous items appropriately. No physician certification is claimed.

The separate Qwen3-32B-AWQ `torch_awq`/PEFT smoke performed 10 actual optimizer steps, each with nonzero gradients and changed LoRA weights, then saved/reloaded its adapter and performed post-reload inference. The load emitted a `qzeros_format` warning; source inspection suggests an inherited format marker, but cross-backend numerical parity remains unverified. The later loss-integration interruption is distinct from the successful 10-step smoke and must be resolved before the current trainer is used.

The A/B training, prediction, Dev-calibration and paired-evaluation code has been prepared and tested on synthetic examples; it has not been validated end to end on a trained checkpoint or reviewed medical data. `TRAINING_PROTOCOL.md` and `AWQ_LORA_SMOKE_STATUS.md` give recovery commands. A prior Base startup attempt used the wrong attention backend and attempted a FlashInfer JIT build in a shared PFS cache; only this run's own processes were stopped, the cache was left intact, and the successful retry isolated caches and used TRITON_ATTN. No other user's process was stopped by this run.
