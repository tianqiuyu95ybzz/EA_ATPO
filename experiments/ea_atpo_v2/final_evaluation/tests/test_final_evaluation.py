from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.ea_atpo_v2.final_evaluation.case_manifest import HistoryIndex, build_case_bundle
from experiments.ea_atpo_v2.final_evaluation.evaluate import evaluate
from experiments.ea_atpo_v2.final_evaluation.metrics import (
    oracle_upper_bound_components,
    paired_bootstrap,
    policy_case_components,
    risk_coverage_rows,
    validate_outcomes,
    validate_predictions,
)


def make_outcomes(case_id: str, base: bool, expert: bool, ask: bool) -> dict:
    return {
        "case_id": case_id,
        "base_correct": base,
        "expert_correct": expert,
        "action_outcomes": {
            "ASK": {"final_correct": ask, "valid": True, "ask_count": 1, "defer_count": 0},
            "ANSWER": {"final_correct": base, "valid": True, "ask_count": 0, "defer_count": 0},
            "DEFER": {"final_correct": expert, "valid": True, "ask_count": 0, "defer_count": 1},
        },
    }


def make_predictions(case_ids: list[str]) -> list[dict]:
    scores = {
        "q_guided_distillation": [
            {"ASK": 0.8, "ANSWER": 0.1, "DEFER": 0.2},
            {"ASK": 0.2, "ANSWER": 0.1, "DEFER": 0.8},
            {"ASK": 0.1, "ANSWER": 0.7, "DEFER": 0.2},
            {"ASK": 0.1, "ANSWER": 0.8, "DEFER": 0.2},
        ],
        "relative_advantage": [
            {"ASK": 0.4, "ANSWER": 0.0, "DEFER": -0.2},
            {"ASK": -0.1, "ANSWER": 0.0, "DEFER": 0.3},
            {"ASK": -0.1, "ANSWER": 0.0, "DEFER": -0.2},
            {"ASK": 0.2, "ANSWER": 0.0, "DEFER": -0.1},
        ],
        "learned_router": [
            {"ASK": 0.6, "ANSWER": 0.1, "DEFER": 0.2},
            {"ASK": 0.1, "ANSWER": 0.1, "DEFER": 0.9},
            {"ASK": 0.1, "ANSWER": 0.8, "DEFER": 0.2},
            {"ASK": 0.1, "ANSWER": 0.7, "DEFER": 0.3},
        ],
    }
    actions = {
        "q_guided_distillation": ["ASK", "DEFER", "ANSWER", "ANSWER"],
        "relative_advantage": ["ASK", "DEFER", "ANSWER", "ASK"],
        "learned_router": ["ASK", "DEFER", "ANSWER", "ANSWER"],
    }
    return [
        {"case_id": case_id, "policy": policy, "selected_action": actions[policy][i],
         "action_scores": scores[policy][i], "model_revision": "synthetic-only"}
        for policy in actions for i, case_id in enumerate(case_ids)
    ]


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.case_ids = ["synthetic-1", "synthetic-2", "synthetic-3", "synthetic-4"]
        rows = [
            make_outcomes(self.case_ids[0], False, True, True),
            make_outcomes(self.case_ids[1], True, False, False),
            make_outcomes(self.case_ids[2], False, False, True),
            make_outcomes(self.case_ids[3], True, True, True),
        ]
        predictions = make_predictions(self.case_ids)
        self.outcomes = validate_outcomes(rows, self.case_ids)
        self.predictions = validate_predictions(predictions, self.case_ids)

    def test_case_level_metrics_and_cost_are_recomputed_once(self):
        components = policy_case_components(self.case_ids, self.outcomes, self.predictions)
        point, _ = paired_bootstrap(components, replicates=250, seed=11)
        self.assertEqual(point["always_answer"]["accuracy"]["estimate"], 0.5)
        self.assertEqual(point["always_answer"]["utility"]["estimate"], 0.5)
        self.assertAlmostEqual(point["always_ask"]["mean_cost"]["estimate"], 0.01)
        self.assertAlmostEqual(point["always_ask"]["utility"]["estimate"], 0.74)
        self.assertAlmostEqual(point["always_defer"]["mean_cost"]["estimate"], 0.05)

    def test_expert_correction_and_correct_answer_degradation(self):
        components = policy_case_components(self.case_ids, self.outcomes, self.predictions)
        point, _ = paired_bootstrap(components, replicates=250, seed=12)
        self.assertAlmostEqual(point["always_defer"]["expert_correction_rate"]["estimate"], 0.5)
        self.assertAlmostEqual(point["always_defer"]["correct_answer_degradation_rate"]["estimate"], 0.5)
        self.assertIsNone(point["always_answer"]["expert_correction_rate"]["estimate"])

    def test_oracle_is_separate_and_is_an_upper_bound(self):
        oracle = oracle_upper_bound_components(self.case_ids, self.outcomes)
        learned = policy_case_components(self.case_ids, self.outcomes, self.predictions)
        oracle_utility = sum(row["utility"] for row in oracle) / len(oracle)
        self.assertGreaterEqual(oracle_utility, max(sum(row["utility"] for row in rows) / len(rows)
                                                    for rows in learned.values()))
        with self.assertRaisesRegex(ValueError, "learned policies"):
            validate_predictions([{"case_id": self.case_ids[0], "policy": "oracle_upper_bound",
                                   "selected_action": "ANSWER", "defer_score": 0.0,
                                   "non_defer_action": "ANSWER"}], self.case_ids)

    def test_prediction_payload_rejects_gold_or_outcome_labels(self):
        rows = make_predictions(self.case_ids)
        rows[0]["gold"] = "A"
        with self.assertRaisesRegex(ValueError, "Gold field"):
            validate_predictions(rows, self.case_ids)

    def test_missing_counterfactual_action_fails_closed(self):
        rows = [make_outcomes(case_id, False, False, False) for case_id in self.case_ids]
        del rows[0]["action_outcomes"]["DEFER"]
        with self.assertRaisesRegex(ValueError, "complete ASK/ANSWER/DEFER"):
            validate_outcomes(rows, self.case_ids)

    def test_risk_coverage_uses_prediction_scores_not_gold(self):
        rows = risk_coverage_rows(self.case_ids, self.outcomes, self.predictions)
        baseline_answer = next(row for row in rows if row["policy"] == "always_answer")
        baseline_defer = next(row for row in rows if row["policy"] == "always_defer")
        self.assertEqual(baseline_answer["coverage"], 1.0)
        self.assertIsNone(baseline_defer["risk"])
        qguided = [row for row in rows if row["policy"] == "q_guided_distillation"]
        self.assertEqual(sum(row["curve_kind"] == "observed_policy" for row in qguided), 1)
        self.assertGreaterEqual(len(qguided), 5)

    def test_paired_bootstrap_is_reproducible(self):
        components = policy_case_components(self.case_ids, self.outcomes, self.predictions)
        first = paired_bootstrap(components, replicates=400, seed=991)
        second = paired_bootstrap(components, replicates=400, seed=991)
        self.assertEqual(first, second)
        _, paired = first
        self.assertIn("ci95_low", paired["q_guided_distillation"]["utility"])

    def test_end_to_end_cli_interface_writes_new_outputs_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "case_manifest.json"
            outcomes = root / "outcomes.jsonl"
            predictions = root / "predictions.jsonl"
            manifest.write_text(json.dumps({"manifest_id": "synthetic", "case_ids": self.case_ids,
                                            "cases": [{"case_id": cid} for cid in self.case_ids],
                                            "case_id_list_sha256": "synthetic"}), encoding="utf-8")
            outcome_rows = [self.outcomes[cid] for cid in self.case_ids]
            outcomes.write_text("".join(json.dumps(row) + "\n" for row in outcome_rows), encoding="utf-8")
            prediction_rows = make_predictions(self.case_ids)
            predictions.write_text("".join(json.dumps(row) + "\n" for row in prediction_rows), encoding="utf-8")
            output_dir = root / "new-evaluation"
            summary = evaluate(manifest, outcomes, predictions, output_dir, replicates=100, bootstrap_seed=4)
            self.assertEqual(summary["case_count"], 4)
            self.assertTrue((output_dir / "metrics.csv").is_file())
            self.assertTrue((output_dir / "paired_deltas_vs_always_answer.csv").is_file())
            self.assertTrue((output_dir / "risk_coverage.csv").is_file())
            with self.assertRaises(FileExistsError):
                evaluate(manifest, outcomes, predictions, output_dir, replicates=100, bootstrap_seed=4)

    def test_case_manifest_selection_is_deterministic_and_label_blind(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "synthetic_source.jsonl"
            records = [
                {"index": i, "prompt": [{"role": "user", "content": f"synthetic question {i}"}],
                 "ground_truth": "not-read-by-selector"}
                for i in range(4)
            ]
            records[3]["prompt"] = records[2]["prompt"]
            source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            history = HistoryIndex(case_ids={"row-0:item-0"})
            first, audit = build_case_bundle(source, history, sample_size=2, selection_seed="fixture")
            second, _ = build_case_bundle(source, history, sample_size=2, selection_seed="fixture")
        self.assertEqual(first["cases"], second["cases"])
        self.assertEqual(audit["test_overlap_with_historical_case_ids"], 0)
        serialized = json.dumps(first)
        self.assertNotIn("synthetic question", serialized)
        self.assertNotIn("ground_truth", serialized)


if __name__ == "__main__":
    unittest.main()
