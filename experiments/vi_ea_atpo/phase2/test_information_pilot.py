"""CPU-only tests for the VI-EA-ATPO pilot adapters and cost accounting."""
from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "ea_atpo_stage"))

import run_information_pilot as pilot  # noqa: E402
from recipe.ea_atpo.collect_counterfactuals import ACTION_REGEX, APICall  # noqa: E402
from recipe.ea_atpo.protocol import (  # noqa: E402
    Action,
    PublicState,
    build_expert_messages,
    parse_action,
    validate_expert_view,
)


class PilotAccountingTests(unittest.TestCase):
    def test_normalized_utility_counts_each_action_once(self) -> None:
        value = pilot.utility(True, 1, 1, 1, 0.05, 0.20, 0.60)
        self.assertAlmostEqual(value, 0.15)

    def test_costs_do_not_change_task_correctness(self) -> None:
        self.assertEqual(pilot.utility(False, 1, 0, 0, 0.05, 0.20, 0.60), -0.05)
        self.assertEqual(pilot.utility(False, 0, 1, 0, 0.05, 0.20, 0.60), -0.20)

    def test_gpu_gate_requires_both_memory_and_utilization_headroom(self) -> None:
        self.assertTrue(pilot.safe_gpu_available([{"memory_free_mib": 30000, "utilization_pct": 10}]))
        self.assertFalse(pilot.safe_gpu_available([{"memory_free_mib": 30000, "utilization_pct": 51}]))
        self.assertFalse(pilot.safe_gpu_available([{"memory_free_mib": 24000, "utilization_pct": 10}]))

    def test_pre_action_features_do_not_contain_outcomes_or_hidden_facts(self) -> None:
        state = PublicState("case-hash", ({"role": "user", "content": "Public symptom question?"},))
        features = pilot.public_features(state)
        self.assertEqual(features["uncertainty_status"], "unavailable_from_current_completion_api")
        self.assertNotIn("gold", features)
        self.assertNotIn("expert_correct", features)
        self.assertNotIn("atomic_facts", features)

    def test_summary_compares_each_branch_to_answer_defer_and_frozen_static(self) -> None:
        rows = []
        actions = {
            "DIRECT_ANSWER": ({"ask": 0, "consult": 0, "defer": 0}, True),
            "DIRECT_DEFER": ({"ask": 0, "consult": 0, "defer": 1}, False),
            "ASK_TASK_ANSWER": ({"ask": 1, "consult": 0, "defer": 0}, True),
            "ASK_DECISION": ({"ask": 1, "consult": 0, "defer": 1}, True),
            "CONSULT_TASK_ANSWER": ({"ask": 0, "consult": 1, "defer": 0}, True),
            "CONSULT_DECISION": ({"ask": 0, "consult": 1, "defer": 0}, True),
        }
        for case_id, split, answer_ok, defer_ok in (
            ("a", "train", True, False), ("b", "train", False, True),
            ("c", "validation", True, False), ("d", "validation", False, True),
        ):
            for intervention, (counts, default_correct) in actions.items():
                correct = answer_ok if intervention == "DIRECT_ANSWER" else defer_ok if intervention == "DIRECT_DEFER" else default_correct
                if intervention == "ASK_DECISION":
                    correct = answer_ok
                rows.append({"case_id": case_id, "split": split, "intervention": intervention,
                             "action_counts": counts, "terminal_correct": correct})
        costs = {"ask": 0.05, "consult": 0.20, "defer_grid": [0.0, 0.6]}
        outcome, analysis = pilot.summarize(rows, costs)
        self.assertTrue(any(row["strategy"] == "CONSULT_TASK_ANSWER" and row["cost_defer"] == 0.6
                            for row in outcome))
        consult = next(row for row in outcome if row["split"] == "validation"
                       and row["strategy"] == "CONSULT_TASK_ANSWER" and row["cost_defer"] == 0.6)
        self.assertEqual(consult["expert_calls"], 2)
        self.assertEqual(consult["consult_rate"], 1.0)
        self.assertTrue(any(row["analysis"] == "utility_delta_vs_frozen_static_action" and row.get("group") == "helpful_defer"
                            for row in analysis))

    def test_protocol_keeps_consult_and_defer_distinct(self) -> None:
        state = PublicState("case-hash", ({"role": "user", "content": "Public case with options A-D."},))
        consult = build_expert_messages(state, Action.CONSULT, "differential diagnosis")
        defer = build_expert_messages(state, Action.DEFER)
        self.assertIn("Expert Advice:", consult[0]["content"])
        self.assertIn("Expert Final Answer:", defer[0]["content"])
        self.assertEqual(parse_action("Final Answer: B")[0], Action.ANSWER)
        self.assertEqual(parse_action("Defer to Expert")[0], Action.DEFER)

    def test_hidden_patient_fact_is_rejected_from_expert_view(self) -> None:
        hidden = "A highly distinctive hidden fact string"
        state = PublicState("case-hash", ({"role": "user", "content": hidden},))
        messages = build_expert_messages(state, Action.CONSULT, "next useful step")
        with self.assertRaises(ValueError):
            validate_expert_view(messages, atomic_facts=[hidden])


class FakeChatAPI:
    def __init__(self, role: str):
        self.role = role
        self.messages_seen = []

    async def generate(self, messages, *, temperature, max_tokens=768, retries=3, structured_outputs=None):
        self.messages_seen.append(messages)
        if self.role == "base":
            regex = (structured_outputs or {}).get("regex")
            if regex == ACTION_REGEX[Action.ASK]:
                content = "Question: Is the symptom present today?"
            elif regex == ACTION_REGEX[Action.CONSULT]:
                content = "Consult Expert: What differential best fits the visible symptoms?"
            elif regex == pilot.TERMINAL_DECISION_REGEX:
                content = "Defer to Expert"
            else:
                content = "Final Answer: B"
        elif self.role == "patient":
            content = "The symptom is present today."
        elif "Expert Advice:" in str(messages[0].get("content", "")):
            content = "Expert Advice: Compare the visible symptoms against the differential."
        else:
            content = "Expert Final Answer: A"
        return APICall(content=content, prompt_tokens=20, completion_tokens=3, latency_s=0.01)


class PilotAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_consult_branch_uses_expert_advice_then_distinct_defer(self) -> None:
        hidden = "HIDDEN_PATIENT_FACT_SHOULD_NOT_LEAK_12345"
        secret_gold_text = "SECRET_GOLD_DIAGNOSIS_NOT_FOR_PROMPTS"
        state = PublicState("case-hash", ({"role": "user", "content": "Public case and options A-D."},))
        row = {
            "case_id": "case-hash", "split": "train", "state": state,
            "atomic_facts": [hidden],
            "ground_truth": {"answer": "A", "answer_info": secret_gold_text},
            "sample_seed": 20261118,
        }
        base, patient, expert = FakeChatAPI("base"), FakeChatAPI("patient"), FakeChatAPI("expert")
        task, decision = await pilot.evaluate_information_branch(Action.CONSULT, row, base, patient, expert)
        self.assertFalse(task["terminal_correct"])
        self.assertTrue(decision["terminal_correct"])
        self.assertEqual(task["action_counts"], {"ask": 0, "consult": 1, "defer": 0})
        self.assertEqual(decision["action_counts"], {"ask": 0, "consult": 1, "defer": 1})
        self.assertAlmostEqual(pilot.utility(True, 0, 1, 1, 0.05, 0.20, 0.60), 0.20)
        expert_messages = json.dumps(expert.messages_seen, ensure_ascii=False)
        base_messages = json.dumps(base.messages_seen, ensure_ascii=False)
        self.assertNotIn(hidden, expert_messages)
        self.assertNotIn(secret_gold_text, expert_messages)
        self.assertNotIn(secret_gold_text, base_messages)
        self.assertIn("Expert Advice:", decision["acquired_information"])
        self.assertEqual(decision["continuation_policy_version"], "terminal-answer-or-defer-v1")

    async def test_ask_gives_hidden_facts_only_to_patient(self) -> None:
        hidden = "HIDDEN_PATIENT_FACT_SHOULD_NOT_LEAK_12345"
        secret_gold_text = "SECRET_GOLD_DIAGNOSIS_NOT_FOR_PROMPTS"
        state = PublicState("case-hash", ({"role": "user", "content": "Public case and options A-D."},))
        row = {"case_id": "case-hash", "split": "train", "state": state,
               "atomic_facts": [hidden],
               "ground_truth": {"answer": "A", "answer_info": secret_gold_text}}
        base, patient, expert = FakeChatAPI("base"), FakeChatAPI("patient"), FakeChatAPI("expert")
        task, decision = await pilot.evaluate_information_branch(Action.ASK, row, base, patient, expert)
        self.assertIn(hidden, json.dumps(patient.messages_seen, ensure_ascii=False))
        self.assertNotIn(hidden, json.dumps(base.messages_seen, ensure_ascii=False))
        self.assertNotIn(secret_gold_text, json.dumps(patient.messages_seen, ensure_ascii=False))
        self.assertIn("present today", task["acquired_information"])
        self.assertEqual(task["action_counts"]["ask"], 1)
        self.assertEqual(decision["action_counts"]["ask"], 1)


if __name__ == "__main__":
    unittest.main()
