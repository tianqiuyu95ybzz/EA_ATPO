# EA-ATPO Stage 2 feasibility study

This package keeps the reproduced official ATPO implementation unchanged and
defines the leakage-safe four-action interface needed before full EA-ATPO training.

## Semantics

- `ANSWER`: the base agent gives the terminal diagnosis.
- `ASK`: the base agent asks one question; only the patient simulator sees hidden
  `atomic_facts`.
- `CONSULT`: an expert reads the public dialogue plus a focused consultation
  question and returns short advice. It cannot output a final option.
- `DEFER`: an expert reviews the public dialogue and makes the terminal decision.

The expert never receives `atomic_facts`, `ground_truth`, `answer_idx`, or the gold
answer explanation. Runtime validators fail closed if privileged content crosses
this boundary.

## Value definition

The collected target is pure clinical outcome value:

`Q_out(s,a) = E[R_clinical + gamma V(s') | s,a]`

Costs are applied once, only by the router:

`U(s,a) = Q_out(s,a) - lambda_cost C(a) - lambda_harm H(s,a)`

This separation prevents the earlier double-counting issue. The feasibility study
must report the optimal-action distribution and value-gap distribution across
500--1000 information states before a router is trained.

## Required checks before GPU rollout

Run from the `codes-repro` directory:

```bash
python -m unittest recipe.ea_atpo.test_protocol -v
```

After starting the three configured endpoints, run a 12-state pilot:

```bash
python -m recipe.ea_atpo.collect_counterfactuals \
  --data data/train.parquet \
  --output ../outputs/ea_atpo/feasibility_pilot.jsonl \
  --num-states 12 --rollouts-per-action 1
```

The JSONL contains every action rollout and its trace. The adjacent summary JSON
contains the optimal-action distribution and utility gaps.
