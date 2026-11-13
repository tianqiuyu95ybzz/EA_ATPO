# EA-ATPO V1 Implementation Record

## Current status

The method implementation is gated on a passing original ATPO training smoke. The official source is a read-only checkout at `ATPO-official/` (commit `25af83688c06917f94245a4db06be7d0924c090c`). The SM100 RMSNorm and rotary kernels pass isolated checks with official `sgl-kernel==0.2.9+cu128`. Attempt 18 confirmed that two-GPU FSDP plus SGLang TP on GPUs 4 and 6 clears the prior weight-sync OOM after lowering SGLang's memory fraction to 0.35 and enabling multi-stage wake-up. It then stopped before generation because the preprocessed Parquet lacked the `api_key` field expected by upstream AgentLoop. The smoke runner now detects and repairs this schema issue using a non-secret placeholder; the one-turn baseline does not call ATPO's external Patient API. Attempt 19 will retry the same model sharding and two-step PPO smoke. GPUs 1 and 7 remain excluded. See `EA_ATPO_SMOKE_TEST.md` and `experiments/ea_atpo_v1_smoke/run_manifest.json`.

## V1 design to implement after the baseline passes

### Policy interface

The existing Actor emits one action tag and its content in a single assistant response:

- `ANSWER`: final answer; terminate.
- `ASK`: patient question; append the patient reply and continue with the same Actor.
- `CONSULT`: specific expert question; append expert advice and continue with the same Actor.
- `DEFER`: request an independent expert final answer; terminate.

Tags will be ordinary text in the sampled response, so no new vocabulary rows or special-token embeddings are needed. The generated response mask already covers the tag and action content. Patient replies and expert messages remain non-Actor text with response-mask value zero. Tests will verify those masks from actual token IDs before any multi-step training run.

### Reward and value targets

For each action edge, record a single reward:

`r_t = correctness_reward_t - ask_cost * I(ASK) - consult_cost * I(CONSULT) - defer_cost * I(DEFER)`

The correctness term is nonzero only at a terminal answer. Costs are charged once at the action edge. The existing tree backup receives this net reward; tree returns train the Critic, and the corresponding action-level advantages train the Actor. Since agent-loop outputs already carry advantages and returns, the trainer must use them directly and skip a second GAE pass or cost subtraction.

The existing clipped PPO and Critic MSE losses remain in place. No extra information-gain reward, clinical-harm reward, new Critic head, or Router loss is introduced in V1.

### Information boundaries

- The patient simulator receives the case's hidden atomic facts and the ASK question. Its reply alone is appended to the public dialogue.
- The expert receives only the public dialogue and the Actor's specific CONSULT question. It receives neither hidden facts nor gold.
- DEFER uses the same public-information boundary, with the expert responsible for the terminal answer.
- Gold is available only to post-rollout reward calculation and final evaluation.
- After ASK or CONSULT, the same Actor chooses the next action. DEFER terminates the episode.

### Planned isolation from the baseline

EA code will live in a new `codes/recipe/ea_atpo/` package and be selected through the existing `agent_loop_config_path`. The initial implementation should avoid changes to `codes/recipe/atpo/`, `codes/verl/`, and the original PPO loss. Planned modules are the EA agent loop/tree runner, action parser and validators, role-isolated simulator prompts, cost/reward configuration, and smoke assertions for action-token masks and parameter deltas.

## Required smoke evidence before small training

The EA smoke must show all four actions executing; ASK/CONSULT continuations returning to the same Actor; DEFER terminating; no gold/hidden-fact path into Actor or Expert prompts; costs charged exactly once; action tags included in PPO masks; and nonzero parameter deltas for both Actor and Critic. Failure traces terminate safely and are saved with the run manifest.
