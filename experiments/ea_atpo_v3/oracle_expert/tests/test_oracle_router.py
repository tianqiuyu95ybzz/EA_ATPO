import unittest

from experiments.ea_atpo_v3.oracle_expert.scripts.evaluate_oracle_router import (
    ASK_COST,
    case_metric_vectors,
    oracle_conditional_result,
    row_result,
    selected_oracle_result,
)


def state(case_id, base, ask, defer, router_action):
    return {
        "case_id": case_id,
        "correctness": {"ASK": ask, "ANSWER": base, "DEFER": defer},
        "action_counts": {"ASK": 1, "ANSWER": 0, "DEFER": 1},
        "router": {"17": {"action": router_action, "scores": [0.0, 0.0, 0.0]}},
    }


class OracleSemanticsTests(unittest.TestCase):
    def test_oracle_conditional_preserves_correct_base_and_repairs_errors(self):
        self.assertEqual(oracle_conditional_result(1), (1, 0))
        self.assertEqual(oracle_conditional_result(0), (1, 1))

    def test_learned_defer_is_cancelled_for_correct_base(self):
        self.assertEqual(selected_oracle_result("DEFER", 1, 0), (1, 0, "ANSWER"))
        self.assertEqual(selected_oracle_result("DEFER", 0, 0), (1, 1, "DEFER"))

    def test_learned_ask_keeps_actual_ask_outcome(self):
        self.assertEqual(selected_oracle_result("ASK", 0, 1), (1, 0, "ASK"))
        self.assertEqual(selected_oracle_result("ASK", 1, 0), (0, 0, "ASK"))

    def test_defer_cost_is_charged_once(self):
        row = state("case-a", 0, 1, 1, "DEFER")
        result = row_result(row, "learned_actual", 17, 0.2)
        self.assertAlmostEqual(result["utility"], 0.8)
        row_oracle = state("case-a", 1, 0, 0, "DEFER")
        oracle = row_result(row_oracle, "learned_oracle", 17, 0.2)
        self.assertEqual(oracle["final_correct"], 1)
        self.assertEqual(oracle["defer_count"], 0)
        self.assertAlmostEqual(oracle["utility"], 1.0)

    def test_case_macro_keeps_all_states_in_case_cluster(self):
        rows = [
            state("case-a", 1, 1, 1, "ANSWER"),
            state("case-a", 1, 1, 1, "ANSWER"),
            state("case-b", 0, 0, 0, "ANSWER"),
        ]
        result = case_metric_vectors(rows, "always_answer", 0.2)
        # Case A contributes 1.0, case B contributes 0.0, then cases are equal-weighted.
        self.assertAlmostEqual(float(result["accuracy"].mean()), 0.5)
        self.assertNotAlmostEqual(float(result["accuracy"].mean()), 2.0 / 3.0)
        self.assertAlmostEqual(ASK_COST, 0.01)


if __name__ == "__main__":
    unittest.main()
