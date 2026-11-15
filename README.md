# EA-ATPO research project

This repository contains research additions and experiment plans for extending
ATPO multi-turn medical dialogue with sequential patient and expert
collaboration. It does not contain the official ATPO checkout, private data,
model weights, checkpoints, or previous server logs.

## Start here

- [Documentation index and current sources of truth](DOCUMENTATION_INDEX.md)
- [Project layout and upstream dependency](PROJECT_LAYOUT.md)
- [Research review and literature positioning](EA_ATPO_RESEARCH_REVIEW_AND_NEW_METHOD_2026-11-02.md)
- [Current roadmap and innovation status](EA_ATPO_ROADMAP_2026-11-02.md)
- [Real-data experiment preregistration and annotation protocol](EA_ATPO_EXPERIMENT_PREREGISTRATION_2026-11-03.md)
- [EA-ATPO V1 implementation record](EA_ATPO_V1_IMPLEMENTATION.md)
- [EA-ATPO V1 latest results and stability report](EA_ATPO_V1_SMALL_SCALE_RESULTS.md)
- [Published experiment results index](PUBLISHED_RESULTS_INDEX.md)
- [Real-case feasibility pilot: preflight, runner, and data recovery](experiments/real_case_pilot_20261103/README.md)
- [Reliability-conditioned router synthetic pilot](experiments/reliability_conditioned_router_20261102/README.md)

The synthetic pilot is only a routing-logic feasibility check. It does not
replace the required paired rollout and held-out evaluation with the project's
medical benchmark and model checkpoints.
