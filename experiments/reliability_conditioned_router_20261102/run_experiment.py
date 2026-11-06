#!/usr/bin/env python3
"""Synthetic feasibility study for reliability-conditioned EA-ATPO routing.

This is an executable decision-policy simulation, not a medical benchmark.  It
uses only the Python standard library so it can run without model weights, GPU
access, or private project data.
"""

from __future__ import annotations

import json
import math
import platform
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ACTIONS = ("ANSWER", "ASK", "CONSULT", "DEFER")
FEATURES = {
    "no_reliability": (),
    "difficulty_only": (0,),
    "patient_reliability": (0, 1),
    "expert_reliability": (0, 2),
    "both_reliabilities": (0, 1, 2),
}
PATIENT_COST = {"ANSWER": 0.0, "ASK": 0.04, "CONSULT": 0.0, "DEFER": 0.0}
EXPERT_COST = {"ANSWER": 0, "ASK": 0, "CONSULT": 1, "DEFER": 2}
PATIENT_COST_WEIGHT = 0.10
EXPERT_BUDGET_PER_CASE = 0.20
TRAIN_ROLLOUTS_PER_CONTEXT_ACTION = 20
TEST_CASES_PER_SEED = 5_000
SEEDS = tuple(range(30))

# Context is (difficulty, patient answerability, expert reliability). These are
# pre-action calibration signals in the toy simulator, not hidden patient facts.
CONTEXTS = tuple(
    (difficulty, answerability, expert_reliability)
    for difficulty in (0, 1)
    for answerability in (0.20, 0.85)
    for expert_reliability in (0.40, 0.90)
)


def true_outcome_probabilities(context: tuple[int, float, float]) -> dict[str, float]:
    """Synthetic probability of a correct terminal diagnosis after each action."""
    difficulty, answerability, expert_reliability = context
    answer = 0.90 - 0.44 * difficulty
    ask = min(0.98, answer + answerability * 0.18 * (1 - 0.25 * difficulty))
    helpful_consult = min(0.98, answer + 0.30)
    misleading_consult = max(0.02, answer - 0.30)
    consult = (
        expert_reliability * helpful_consult
        + (1 - expert_reliability) * misleading_consult
    )
    defer = min(0.90, 0.45 + 0.32 * expert_reliability - 0.04 * difficulty)
    return {
        "ANSWER": answer,
        "ASK": ask,
        "CONSULT": consult,
        "DEFER": defer,
    }


def _context_key(context: tuple[int, float, float], feature_indices: tuple[int, ...]):
    return tuple(context[index] for index in feature_indices)


def collect_training_rollouts(seed: int):
    rng = random.Random(seed)
    samples: dict[tuple[tuple[int, float, float], str], list[int]] = defaultdict(list)
    for context in CONTEXTS:
        probabilities = true_outcome_probabilities(context)
        for action in ACTIONS:
            samples[(context, action)] = [
                int(rng.random() < probabilities[action])
                for _ in range(TRAIN_ROLLOUTS_PER_CONTEXT_ACTION)
            ]
    return samples


def fit_value_table(samples, feature_indices: tuple[int, ...]):
    totals: dict[tuple[object, str], list[int]] = defaultdict(lambda: [0, 0])
    for (context, action), outcomes in samples.items():
        key = _context_key(context, feature_indices)
        totals[(key, action)][0] += sum(outcomes)
        totals[(key, action)][1] += len(outcomes)
    return {
        key: {
            action: totals[(key, action)][0] / totals[(key, action)][1]
            for action in ACTIONS
        }
        for key, _action in totals
    }


def make_test_contexts(seed: int, count: int):
    """Sample a shifted case mix: more difficult cases and weaker channels."""
    rng = random.Random(seed)
    result = []
    for _ in range(count):
        difficulty = rng.choices((0, 1), weights=(0.25, 0.75))[0]
        answerability = rng.choices((0.20, 0.85), weights=(0.75, 0.25))[0]
        expert_reliability = rng.choices((0.40, 0.90), weights=(0.70, 0.30))[0]
        result.append((difficulty, answerability, expert_reliability))
    return result


def _select_with_budget(value_table, feature_indices, contexts, budget_units, tie_seed):
    local_actions: list[str] = []
    expert_upgrades = []
    tie_rng = random.Random(tie_seed)

    for index, context in enumerate(contexts):
        key = _context_key(context, feature_indices)
        values = value_table[key]
        utility = {
            action: values[action] - PATIENT_COST_WEIGHT * PATIENT_COST[action]
            for action in ACTIONS
        }
        local = max(("ANSWER", "ASK"), key=lambda action: (utility[action], -EXPERT_COST[action]))
        expert = max(("CONSULT", "DEFER"), key=lambda action: utility[action])
        local_actions.append(local)
        gain = utility[expert] - utility[local]
        burden = EXPERT_COST[expert]
        expert_upgrades.append((gain / burden, tie_rng.random(), index, expert, burden))

    # Allocate a fixed expert budget to the cases with the largest predicted
    # utility gain per expert-work unit. A policy with a poor ranking may spend
    # budget on negative-gain cases; that is useful for a matched-budget test.
    # Deferred cases cost two units.
    expert_upgrades.sort(reverse=True)
    chosen_expert_actions: dict[int, str] = {}
    spent = 0
    for _priority, _tie, index, action, burden in expert_upgrades:
        if spent + burden <= budget_units:
            chosen_expert_actions[index] = action
            spent += burden

    actions = [chosen_expert_actions.get(i, action) for i, action in enumerate(local_actions)]
    return actions, spent


def _expected_accuracy(contexts: Iterable[tuple[int, float, float]], actions: list[str]) -> float:
    total = sum(
        true_outcome_probabilities(context)[action]
        for context, action in zip(contexts, actions)
    )
    return total / len(actions)


def _policy_metrics(contexts, actions, spent):
    count = len(actions)
    action_counts = Counter(actions)
    return {
        "expected_accuracy": _expected_accuracy(contexts, actions),
        "expert_units_per_case": spent / count,
        "patient_ask_rate": action_counts["ASK"] / count,
        "action_rates": {action: action_counts[action] / count for action in ACTIONS},
        "low_answerability_accuracy": _expected_accuracy(
            [c for c, a in zip(contexts, actions) if c[1] == 0.20],
            [a for c, a in zip(contexts, actions) if c[1] == 0.20],
        ),
        "low_expert_reliability_accuracy": _expected_accuracy(
            [c for c, a in zip(contexts, actions) if c[2] == 0.40],
            [a for c, a in zip(contexts, actions) if c[2] == 0.40],
        ),
    }


def _mean_ci(values: list[float]):
    mean = statistics.mean(values)
    if len(values) < 2:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    # t_(0.975, 29) for the fixed 30-seed paired pilot.
    half_width = 2.045 * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "ci95_low": mean - half_width, "ci95_high": mean + half_width}


def run():
    per_seed = []
    policy_names = (*FEATURES.keys(), "oracle", "always_answer")
    for seed in SEEDS:
        samples = collect_training_rollouts(seed)
        contexts = make_test_contexts(seed + 100_000, TEST_CASES_PER_SEED)
        budget = int(EXPERT_BUDGET_PER_CASE * len(contexts))
        metrics = {}

        for name, indices in FEATURES.items():
            table = fit_value_table(samples, indices)
            actions, spent = _select_with_budget(table, indices, contexts, budget, seed + 200_000)
            metrics[name] = _policy_metrics(contexts, actions, spent)

        oracle_table = {
            context: true_outcome_probabilities(context)
            for context in CONTEXTS
        }
        oracle_actions, oracle_spent = _select_with_budget(
            oracle_table, (0, 1, 2), contexts, budget, seed + 200_000
        )
        metrics["oracle"] = _policy_metrics(contexts, oracle_actions, oracle_spent)
        answer_actions = ["ANSWER"] * len(contexts)
        metrics["always_answer"] = _policy_metrics(contexts, answer_actions, 0)
        per_seed.append({"seed": seed, "metrics": metrics})

    summary = {}
    for name in policy_names:
        summary[name] = {}
        for metric in (
            "expected_accuracy",
            "expert_units_per_case",
            "patient_ask_rate",
            "low_answerability_accuracy",
            "low_expert_reliability_accuracy",
        ):
            summary[name][metric] = _mean_ci(
                [entry["metrics"][name][metric] for entry in per_seed]
            )
        summary[name]["action_rates"] = {
            action: statistics.mean(
                entry["metrics"][name]["action_rates"][action] for entry in per_seed
            )
            for action in ACTIONS
        }

    paired_differences = {}
    for comparator in (
        "always_answer",
        "no_reliability",
        "difficulty_only",
        "patient_reliability",
        "expert_reliability",
    ):
        deltas = [
            entry["metrics"]["both_reliabilities"]["expected_accuracy"]
            - entry["metrics"][comparator]["expected_accuracy"]
            for entry in per_seed
        ]
        paired_differences[f"both_reliabilities_minus_{comparator}"] = _mean_ci(deltas)
    for comparator in ("no_reliability", "difficulty_only"):
        deltas = [
            entry["metrics"]["expert_reliability"]["expected_accuracy"]
            - entry["metrics"][comparator]["expected_accuracy"]
            for entry in per_seed
        ]
        paired_differences[f"expert_reliability_minus_{comparator}"] = _mean_ci(deltas)

    return {
        "experiment": "synthetic_reliability_conditioned_router",
        "claim_scope": "synthetic decision-policy feasibility only; not medical or LLM performance",
        "config": {
            "seeds": list(SEEDS),
            "train_rollouts_per_context_action": TRAIN_ROLLOUTS_PER_CONTEXT_ACTION,
            "test_cases_per_seed": TEST_CASES_PER_SEED,
            "expert_budget_units_per_case": EXPERT_BUDGET_PER_CASE,
            "expert_cost_units": EXPERT_COST,
            "patient_ask_cost": PATIENT_COST["ASK"],
            "patient_cost_weight": PATIENT_COST_WEIGHT,
            "train_mix": "balanced over difficulty, patient answerability, and expert reliability bins",
            "test_mix": "75% difficult; 75% low patient-answerability; 70% low expert reliability",
            "budget_protocol": "each policy is allocated the same 0.20 expert-work units per test case",
        },
        "summary": summary,
        "paired_differences": paired_differences,
        "per_seed": per_seed,
        "runtime": {"python": platform.python_version(), "dependencies": "Python standard library only"},
    }


if __name__ == "__main__":
    output = run()
    result_path = Path(__file__).with_name("results.json")
    result_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": output["summary"], "paired_differences": output["paired_differences"]}, indent=2))
    print(f"Wrote {result_path}")
