# V3 data audit

The auditable source is the existing **model-simulated** 100-case EA-ATPO Pilot, not actual patient interviews. The audited SFT snapshots contain 291 distinct states; 191 have at least one simulated patient reply and 100 are initial states with no patient reply. The medical answer labels are historical source labels, not yet independently approved Gold.

| Split | Cases | States | States with 0 / 1 / 2 patient replies |
| --- | ---: | ---: | ---: |
| Train | 70 | 205 | 70 / 70 / 65 |
| Dev (`validation`) | 15 | 42 | 15 / 14 / 13 |
| Internal Test | 15 | 44 | 15 / 15 / 14 |

The three files' SHA-256 hashes match the existing dataset manifest. The audit independently checked case membership against the frozen split, unique state IDs, public dialogue/trace reconstruction, dialogue prompt hashes, public question identity against the official source and historical answer-label consistency. Case overlap across splits is zero. These structural and label-provenance checks do not establish that any answer is medically correct.

The original ATPO source has 14,256 rows, all with the two-role `system,user` prompt structure. It is not a 14,256-example multi-turn patient corpus. Three V2 public states were previously flagged for possible question/option anomalies; their clinical ambiguity must be investigated in review and they are not silently excluded.

The V2 Pilot contains 50 unique cases, 146 unique states and 438 action trajectories. All 50 cases belong to Train, none to Dev or Internal Test, and all 146 states already exist in the 205 Train states. No V2 trajectory is added as another SFT example. The V2 `ANSWER` output may be considered only as a candidate paired Base response after the full model/tokenizer/prompt/decoding and raw-output checks; no acceptance is implied by matching case IDs.

The audit did not open the 400-case frozen holdout. The earlier manifest records a pre-existing holdout selection, but this run makes no new independent claim about its content. Privacy-preserving per-sample provenance, split hashes and input-file hashes are in `data/`; patient text and answer labels are deliberately excluded from Git.

Reproduce into a new output directory, never over the frozen artifacts:

```bash
python3 experiments/ea_atpo_v3/expert_sft/scripts/audit_data.py \
  --source-root /pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1 \
  --output-dir experiments/ea_atpo_v3/expert_sft/data/audit_recheck_NEW
```
