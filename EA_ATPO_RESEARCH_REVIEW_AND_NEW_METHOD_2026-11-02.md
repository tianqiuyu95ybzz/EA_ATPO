# EA-ATPO takeover review and new research direction

**Review date:** 2026-11-02  
**Scope:** public repository, local innovation note, current literature, and an
executable synthetic routing pilot. This is a research handoff, not a clinical
validation report.

## 1. Can this project be taken over?

Yes, for the research code, experiments, and paper development. The public
EA-ATPO repository has a useful implementation of the four-action protocol,
counterfactual rollout collector, leakage guards, utility code, patches, and
dated experiment handoffs. The [official ATPO upstream](https://github.com/Quark-Medical/ATPO)
separately publishes its RL code and adapted benchmark JSONL data, but neither
is checked out here. Its README says the SFT training code is not included and
points users to LLaMA-Factory or VeRL; it also says its service URL was anonymized. Qwen checkpoints,
EA-ATPO's previous rollout files, and old server outputs are absent here.

The current container has Python 3.12 but no `torch`, `transformers`, `pandas`,
`pyarrow`, or `datasets` installation. In the 2026-11-03 resource snapshot,
GPUs 1 and 3 were free while the others were in use; no project model weights,
Parquet data, or configured inference services were present. I did not start a
GPU job. After implementing and checking the CONSULT collector, protocol and
collector tests pass **17/17** locally.

This means the project can move forward on protocol, research design, and
CPU-feasible experiments now. A Qwen3-8B ATPO run or a real counterfactual
medical evaluation needs the upstream checkout, the approved dataset/model
assets, reachable inference endpoints, the missing runtime dependencies, and
GPU allocation for the selected models.

The upstream training path is therefore recoverable, but not turnkey: it needs
an external SFT setup, model weights, service deployment/endpoint configuration,
and VeRL dependencies in addition to an available GPU.

## 2. What the original idea establishes—and what it does not

The local `tqy/创新点.txt` proposes a useful clinical interaction setting:
ANSWER, ASK the patient, request a non-terminal expert hint, or transfer the
case for a terminal expert answer. That framing should remain the project
motivation and task definition.

Recent work means that the broad claims are not enough as standalone
contributions:

- ATPO already trains multi-turn medical information seeking with uncertainty-
  guided tree expansion. [ATPO, ICLR 2026](https://openreview.net/forum?id=2bv3B8B9bl)
- ProMed explicitly rewards clinically useful information gain for medical
  questions. [ProMed, ACL 2026](https://aclanthology.org/2026.acl-long.1500/)
- MedCIA combines structured clinical intake, a profile-anchored patient
  simulator, and ATPO for information acquisition. [MedCIA, Findings of EMNLP
  2026](https://aclanthology.org/2026.findings-emnlp.2387/)
- Typed evidence states, unresolved checks, and support reliability are already
  explicit in MIND. [MIND, EMNLP 2026](https://aclanthology.org/2026.emnlp-main.1417/)
- Sequential deferral and guided human deferral also predate this project.
  [Sequential Learning-to-Defer](https://arxiv.org/abs/2109.06312),
  [Guided Deferral Systems](https://openreview.net/forum?id=4c5rg9y4me)
- General L2D work now studies varying expert reliability and choosing
  expert-conditioned advice. [Varying Experts, AISTATS 2026](https://proceedings.mlr.press/v300/montreuil26b.html),
  [Learning-to-Defer with Advice, 2026 preprint](https://arxiv.org/abs/2603.14324)

Therefore, “add expert actions,” “reward new facts,” and “use information gain”
should not be presented as new by themselves. The more defensible opening is
the interaction between sequential medical dialogue, the different semantics
of CONSULT versus DEFER, and the reliability of each interaction channel.
This literature pass is focused rather than exhaustive; it does not certify
global novelty.

## 3. New candidate contribution

### Reliability-calibrated sequential collaboration

Learn whether a particular interaction is likely to help *before* spending the
patient's time or expert budget. Keep three quantities distinct:

1. **Patient response availability:** probability that a proposed ASK receives
   interpretable evidence, estimated from the public state and question.
2. **Hint utility:** probability/expected value that a CONSULT hint is
   evidence-grounded and improves the base agent's continuation.
3. **Takeover accuracy:** probability that DEFER's independent terminal answer
   is correct.

The hint and takeover heads must be calibrated separately: a model can be a
useful consultant without being a better autonomous diagnostician, or vice
versa. A patient refusal is stored as `UNKNOWN`; it is neither positive nor
negative clinical evidence. Its availability signal can still update the
expected value of asking again.

At each decision, estimate action outcome and resource cost separately, then
select under the remaining patient and expert budgets. A risk-sensitive version
uses a validation-calibrated lower bound for each action's outcome estimate:

```text
U(s, a) = LCBα[Q_out(s, a)] - λ_patient C_patient(s, a)
                           - λ_expert C_expert(s, a)
```

The policy may choose any legal action at every state; the action space is not
forced to escalate one level at a time. Reliability estimates at inference must
use only the public dialogue and calibration learned from training cases.
Hidden `atomic_facts`, gold answers, and future simulator responses remain
unavailable to the router.

This is a **candidate research contribution**, not a novelty claim. It is close
enough to recent varying-expert and expert-advice work that the paper must make
the sequential patient channel, non-terminal hint versus terminal takeover,
and held-out channel-shift evaluation precise.

## 4. Executed pilot

I added a reproducible CPU-only simulation at
[`experiments/reliability_conditioned_router_20261102/`](experiments/reliability_conditioned_router_20261102/).
It uses 30 seeds, 20 simulated outcomes per action/context in training, and
5,000 held-out synthetic states per seed. The test mixture shifts toward harder
cases, lower patient answerability, and lower expert reliability. Every router
is allocated exactly 0.20 expert-work units per test case; metrics report
expected accuracy under the generator's known outcome model.

| Router features | Expected accuracy | Expert units / case |
|---|---:|---:|
| Always ANSWER | 56.92% | 0.00 |
| No reliability, pooled action values | 60.51% | 0.20 |
| Difficulty only | 61.74% | 0.20 |
| Patient reliability + difficulty | 60.97% | 0.20 |
| Expert reliability + difficulty | **63.11%** | 0.20 |
| Both reliability signals + difficulty | 62.33% | 0.20 |
| Oracle synthetic values | 65.76% | 0.20 |

Paired across seeds, adding the expert-reliability feature over pooled action
values improved expected accuracy by **2.60 percentage points** (95% CI
**[1.83, 3.37]**); over difficulty-only values the difference was **1.37 pp**
(95% CI **[0.67, 2.08]**). Combining both reliability features did **not** beat
expert-reliability-only (**−0.78 pp**, 95% CI **[−1.49, −0.07]**).

Interpretation: the pilot supports testing case-conditional expert quality in
the actual router. It does not support assuming that patient-answerability and
expert-quality features add independently. This experiment assumes the
reliability bins are known perfectly and uses hand-written outcome equations;
it is not evidence of real medical accuracy or of an improvement to Qwen/ATPO.
It generated no medical cases, consultation questions, or expert text: CONSULT
was represented only by a Bernoulli outcome probability in the toy simulator.

There is no ready-made human CONSULT corpus in the current EA-ATPO checkout.
The official ATPO data are adapted public medical multiple-choice cases; the
handoff records historical splits of 14,256 training and 1,268 test items,
which must be reverified when data are restored. On 2026-11-03, I updated the
local collector so the base agent generates a focused CONSULT question, passes
it to the expert, and can continue with ASK/CONSULT/ANSWER/DEFER (one CONSULT
per episode by default). Initial state generation, public dialogue, and JSONL
traces now omit private `<think>` text. The mocked collector and protocol suite
passes **17/17** tests. No live endpoint-backed rollout has run: there is no
project Parquet/model checkpoint in the workspace, the configured API ports are
not serving, and `aiohttp`, `pandas`, and `pyarrow` are absent. GPUs 1 and 3
were free in the latest snapshot, but no project model is available to load.
The generated consultation language and data quality remain unverified.
Construction details are in
[the roadmap](EA_ATPO_ROADMAP_2026-11-02.md).

The same repository's prior handoff reports a high refusal rate in forced ASK
branches. That motivates patient-channel analysis, but it does not make the
synthetic patient feature result real-world evidence.

## 5. Next real-data experiment

After models and data are restored:

1. Freeze the chosen ATPO base agent and use a patient-case train/validation/
   test split. Do not expose hidden facts or labels to the gate.
2. On training states, collect paired ANSWER/ASK/CONSULT/DEFER rollouts. Label
   patient responses as answerable, unanswerable, or ambiguous; separately score
   CONSULT hint usefulness and DEFER final correctness.
3. Fit public-state-only estimates for answerability, hint usefulness, and
   takeover accuracy. Cross-fit them by case; calibrate action-value intervals
   and cost weights on validation only.
4. Compare the existing outcome-value gate with ablations that add each
   reliability signal. Match expert-work units and patient burden. Include
   low-response and low-expert-quality stress slices, calibration error,
   unknown-as-negative rate, invalid advice, and paired case-bootstrap CIs.
5. Keep the signal only if it improves the preregistered held-out
   same-budget endpoint. If the full two-channel model fails, retain only the
   supported individual signal or report the negative result.

Do not begin expensive joint ATPO training until this gate passes the existing
independent-test and expert-protocol checks.
