# EA-ATPO V2 Phase 1

Method: **EA-ATPO V2: Counterfactual Value-Guided Sequential Learning to Defer with Active Information Acquisition**.

This isolated directory starts the V2 three-action workflow. It preserves all V1 code and results. Phase 1 collects paired counterfactuals only; it does not train an Actor, Q network, Critic, SFT Expert, PPO, or RL policy.

## Audited components and reuse

| Component | Existing location | V2 use |
|---|---|---|
| Frozen public dialogue states | `experiments/real_case_pilot_20261103/outputs/full/pilot_public_states.jsonl` | Reused as starting states; every case is filtered through the frozen case split before selection. The state-construction trace, labels, and old action results are not passed to models. |
| Case split | `experiments/ea_atpo_value_router_20261105/artifacts/case_split.json` | Reused by `case_id`: 70 train, 15 validation, 15 internal-test cases. Phase 1 samples train cases only. |
| Original single-turn source | `experiments/expert_sft/data/source/dataset/merged_train_dataset.jsonl` | Supplies the indexed source prompt, hidden atomic facts for the Patient role, and the answer key for post-rollout scoring. It remains described as single-turn source data. |
| Four-action protocol and collector | `ea_atpo_stage/recipe/ea_atpo/protocol.py`, `collect_counterfactuals.py` | Their public-state boundary, role separation, parser patterns, and OpenAI-compatible request conventions were audited. V2 uses new three-action protocol code because the V1 runner includes CONSULT and has different continuation semantics. The user-owned V1 collector diff is untouched. |
| Original ATPO tree search | `ATPO-official/codes/recipe/atpo/tree_search_manager.py`, `user_assistant_agent_loop.py` | Audited and preserved. Not invoked in this first data pilot; later tree search can consume the learned Q/VOI estimates without changing upstream files. |
| V1 Actor/Critic training | `ea_atpo_stage/recipe/ea_atpo/ea_agent_loop.py` and original ATPO worker interfaces | Not used. V2 Phase 1 is inference-only and collects action returns. |
| Patient simulator | V1 role prompt pattern, now isolated in `src/v2_protocol.py` | The Qwen3-8B service receives hidden atomic facts only in the Patient prompt and a separate patient question. It is a simulated patient, not a real patient. |
| Base and Expert | Existing Phase 2 service configuration | Base is pinned Qwen3-8B; DEFER Expert is pinned Qwen3-32B-AWQ. CONSULT is disabled in this V2 protocol. |

## Three-action protocol

- **ANSWER:** Qwen3-8B produces a final option and ends the branch.
- **ASK:** Qwen3-8B produces one patient question. The Patient simulator sees the source atomic facts and that question. The patient reply is appended as public dialogue, then the same Base service is forced to produce the terminal answer. This frozen continuation is `ASK_ONCE_THEN_FORCE_BASE_ANSWER_V1`; it permits no second ASK or DEFER within the branch. Earlier ASK turns in the starting state count toward the episode maximum of three.
- **DEFER:** Qwen3-32B-AWQ independently produces a terminal answer using only the current public state. The Expert endpoint is not called for ANSWER or ASK branches.

The V1 CONSULT path is not part of V2. Gold is accessed only after all three branch calls for a paired state have finished. Hidden atomic facts are passed only to the Patient role. Each action's accuracy reward and ASK/DEFER cost are stored separately; the net return is `accuracy - 0.05 * ask_count - 0.60 * defer_count`. Token counts and latency remain separate from this utility. One rollout per state/action gives `return_std: null`; a missing/invalid action receives `valid_mask: false`, never an imputed value.

Operational ASK checks distinguish a non-empty response, exact-text novelty against the public state, and an explicit UNKNOWN response. These do not claim semantic information gain.

## Run commands

All run artifacts and raw trajectories stay under the ignored, project-local `outputs/` directory. No model service startup is permitted by this config.

```bash
python3 experiments/ea_atpo_v2/scripts/run_counterfactual_pilot.py --preflight --cases 10
# After both pinned services pass preflight, exercise all three real roles on one state first.
python3 experiments/ea_atpo_v2/scripts/run_counterfactual_pilot.py --execute --smoke
# Run the frozen 10-case / 29-state pilot only after the role smoke passes.
python3 experiments/ea_atpo_v2/scripts/run_counterfactual_pilot.py --execute --cases 10
python3 experiments/ea_atpo_v2/scripts/build_q_dataset.py \
  --trajectories experiments/ea_atpo_v2/outputs/<run>/pilot_trajectories.jsonl \
  --output experiments/ea_atpo_v2/outputs/<run>/q_regression_dataset.jsonl
python3 -m unittest discover -s experiments/ea_atpo_v2/tests -v
```

`--execute` first requires both configured `/v1/models` endpoints to respond and match their pinned aliases. Loopback requests explicitly bypass shell HTTP proxy variables, so a local proxy cannot impersonate a model-service response. It refuses to start model calls if that check fails. `--smoke` exercises Base ANSWER, Expert DEFER, and Base ASK → Patient reply → Base answer without scoring Gold; it must pass before the paired pilot. The runner never starts or stops a service. Select 10–20 training cases only; the 400-case Frozen Holdout is not opened.

## Q-regression interface

`src/q_dataset.py` builds one row per `(case_id, state_id)`:

```text
Q_dataset(state) -> {
  Q_ASK, Q_ANSWER, Q_DEFER,
  valid_mask,
  return_std,
  valid_sample_count,
  attempted_sample_count,
  metadata
}
```

The Q targets are empirical means of valid, paired returns under one continuation policy. They already include the registered action costs; downstream training must not subtract the same costs again. If an action has no valid rollout, its target stays null and its mask is false. Gold, expert output, future replies, and hidden facts are absent from the feature payload.

## Phase 1 status

See `reports/PHASE1_COUNTERFACTUAL_EXECUTION.md` for the audit and exact preflight result. A successful static check is not a model rollout. The first-stage execution pauses when the configured model services are unavailable; no V2 performance claim is produced from V1 trajectories.
