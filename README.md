# EA-ATPO research project

This repository contains research additions and experiment plans for extending
ATPO multi-turn medical dialogue with sequential patient and expert
collaboration. It does not contain the official ATPO checkout, private data,
model weights, checkpoints, or previous server logs.

## Start here

- [Project layout and upstream dependency](PROJECT_LAYOUT.md)
- [Experiment and implementation handoff](EA_ATPO_HANDOFF_2026-10-29.md)
- [Takeover review, literature update, and new method candidate](EA_ATPO_RESEARCH_REVIEW_AND_NEW_METHOD_2026-11-02.md)
- [Roadmap and innovation plan](EA_ATPO_ROADMAP_2026-11-02.md)
- [Real-data experiment preregistration and annotation protocol](EA_ATPO_EXPERIMENT_PREREGISTRATION_2026-11-03.md)
- [Real-case feasibility pilot: preflight, runner, and data recovery](experiments/real_case_pilot_20261103/README.md)
- [Reliability-conditioned router synthetic pilot](experiments/reliability_conditioned_router_20261102/README.md)

The synthetic pilot is only a routing-logic feasibility check. It does not
replace the required paired rollout and held-out evaluation with the project's
medical benchmark and model checkpoints.
