# EA-ATPO V1 Attempt09 results

## Outcome

The 10-step small-scale PPO run completed with exit code 0 on GPUs 4 and 6. All 10 optimizer steps produced finite Actor/Critic losses and gradients; no OOM occurred. The runner saved checkpoints at every step, with the latest at `checkpoints/global_step_10/`. This verifies that the repaired Expert protocol and failure isolation can complete a small run. It does not establish training stability or policy improvement: the Critic explained variance stayed negative throughout, and there was no matched Original ATPO baseline in this run.

## Run and data

- Actor/Critic: Qwen3-8B with FSDP across GPUs 4 and 6; Expert and Patient shared the configured SGLang service through role-specific prompts.
- Data: 20 official training cases sampled with seed 1, four natural on-policy trajectories per training case batch; 20 held-out validation cases. The 20 unique hashed train case IDs and 20 validation IDs had zero overlap.
- PPO: 10 optimizer steps, train batch size 2, `M=N=4`, at most 3 Actor actions, at most 1 ASK and 1 CONSULT, response cap 512 tokens.
- Reward: correctness reward 1.0; ASK/CONSULT/DEFER costs 0.05/0.20/0.60. Costs are included once in tree edge rewards; the episode score is reporting only.
- No forced-action outcomes were inserted into PPO.

## Effective trajectories and action coverage

| Measure | Result |
|---|---:|
| Training rollout slots | 80 |
| PPO-eligible trajectories | 79/80 (98.75%) |
| Protocol-invalid Actor trajectories excluded | 1 |
| ASK budget violations retained as policy outcomes | 2 |
| Correct training episodes | 36/80 (45.0%) |
| Natural decision events: ANSWER / ASK / CONSULT / DEFER | 73 / 76 / 66 / 4 |
| Initial decisions: ANSWER / ASK / CONSULT / DEFER | 6 / 74 / 0 / 0 |

DEFER was naturally sampled four times, once in step 1 and three times in step 3. It was not sampled in steps 4–10. Thus every action received at least some natural trajectory coverage, but DEFER coverage is sparse and the initial policy strongly favors ASK. This is not evidence that the policy has learned a useful DEFER strategy.

The only excluded trajectory was an invalid Actor action in step 9. Two ASK budget violations remained eligible and were treated as policy constraint outcomes, not protocol failures. Each step had 8 eligible trajectories except step 9, which had 7.

## Expert protocol telemetry

Across training and validation, 74 raw Expert completions were preserved in `traces/all_rollouts.jsonl` with completion tokens, latency, exact SGLang finish reason, and parser error. All 74 finish reasons were `stop`; there were no retries, token-cap completions, or parse failures. In training, this covers 66 CONSULT and 4 DEFER requests; validation added four CONSULT requests. The protocol-level explicit-answer leakage detector found 0/66 CONSULT leaks, and the strict DEFER parser found 0/4 format failures. This measures the implemented syntactic checks, not an independent semantic review.

Trace privacy checks found no gold/ground-truth/atomic-fact/hidden-fact fields and no raw patient response or public case text. Case and state identifiers are hashes. A limited email/phone/SSN-pattern scan found no matches in the raw Expert completions; this is a pattern screen, not a general-purpose de-identification guarantee. Correctness booleans and terminal reward remain as post-episode evaluation outcomes; the gold answer itself is not stored.

## PPO and memory diagnostics

| Metric across 10 steps | Range |
|---|---:|
| Actor PG loss | -0.705 to 0.373 |
| Critic value loss | 0.0297 to 8.097 |
| Actor grad norm before clipping | 4.34 to 117.16 |
| Critic grad norm before clipping | 12.58 to 1064.72 |
| Critic Return mean | -0.355 to 0.865 |
| Critic Return observed range | -0.80 to 1.00 |
| Advantage mean | -0.274 to 0.648 |
| Advantage observed range | -2.279 to 2.991 |
| Critic explained variance | -203.59 to -0.56 (negative at all 10 steps) |
| Peak allocated / reserved memory | 76.30 / 89.14 GiB |

Gradient norms in the log are pre-clipping; the configured gradient clip is 1.0. Losses, Returns, Advantages, and gradients remained finite, but the negative explained variance shows that the Critic did not fit Returns reliably in this small run. `custom/effective_rate` is a legacy compatibility metric fixed at zero; the 79/80 eligibility rate above is computed from trace-level training eligibility instead.

The legacy `static/q_value_variance_mean/std` diagnostics are `NaN` because that diagnostic receives an empty list; they are not PPO losses, rewards, or Critic targets.

At the resource gate, GPUs 4 and 6 each had about 110.9 GiB and 110.7 GiB free, above the runner's 105,000 MiB threshold. Peak reported allocated/reserved memory remained about 76.30/89.14 GiB. No other process was stopped.

## Held-out validation

The final checkpoint was evaluated once on 20 held-out cases: 4/20 correct (20%), 20/20 answers parseable, mean net reward 0.15, mean 1,162.4 total tokens, and mean measured serial latency 1.69 seconds per case. There were four Expert requests on validation; action-event counts were ANSWER 20, ASK 4, CONSULT 4, DEFER 0. These are small-sample smoke metrics, not a comparative result. No same-budget Original ATPO or no-cost four-action baseline was run here.

## Artifacts and reproduction

- Per-step metrics and action coverage: `ppo_step_metrics.csv`, `action_coverage.csv`.
- Validation summary: `validation_summary.json`.
- Redacted train/validation traces with raw Expert completions: `traces/train.jsonl`, `traces/validation.jsonl`; combined source: `traces/all_rollouts.jsonl`.
- Run configuration, data hashes, privacy audit, and artifact hashes: `run_manifest.json`.
- Full runtime log remains on the experiment filesystem at `logs/ea_atpo_v1_smoke_20261115T101426Z.log`; the GitHub artifact is path-scrubbed at `logs/ea_atpo_v1_smoke_20261115T101426Z_redacted.log`.
- Checkpoints: `checkpoints/global_step_1/` through `checkpoints/global_step_10/` on the experiment filesystem. Checkpoints and converted Parquet files are not included in the Git push because of their size and/or data content.
- Re-run command: `./run_ea_atpo_v1_stability.sh` (requires the configured Qwen3-8B cache and the GPU memory gate).

No large-scale training, new network module, or new loss was introduced.
