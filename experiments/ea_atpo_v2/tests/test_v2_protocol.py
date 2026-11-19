import sys
import unittest
from pathlib import Path

V2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_DIR / "src"))
sys.path.insert(0, str(V2_DIR / "scripts"))

from q_dataset import build_q_dataset, q_dataset_for_state  # noqa: E402
from run_counterfactual_pilot import direct_loopback_opener  # noqa: E402
from v2_protocol import (  # noqa: E402
    Action,
    ASK_QUESTION_REGEX,
    FINAL_ANSWER_REGEX,
    PublicState,
    action_net_return,
    assert_no_privileged_payload,
    build_base_messages,
    build_defer_messages,
    build_patient_messages,
    extend_after_ask,
    parse_answer,
    parse_question,
    valid_ask_transition,
)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.state = PublicState(
            case_id="fixture-case",
            state_id="fixture-state",
            state_index=0,
            initial_messages=(
                {"role": "system", "content": "V1 may use CONSULT or DEFER actions."},
                {"role": "user", "content": "A generic question with options A-D."},
            ),
        )

    def test_three_actions_and_ask_limit(self):
        self.assertEqual([item.value for item in self.state.legal_actions()], ["ASK", "ANSWER", "DEFER"])
        exhausted = PublicState("c", "s", 0, self.state.initial_messages, asked_count=3, max_asks=3)
        self.assertEqual([item.value for item in exhausted.legal_actions()], ["ANSWER", "DEFER"])

    def test_base_prompt_replaces_legacy_four_action_system_prompt(self):
        prompt = build_base_messages(self.state, Action.ASK)
        self.assertNotIn("CONSULT", prompt[0]["content"])
        self.assertNotIn("DEFER", prompt[0]["content"])
        self.assertEqual(prompt[1], self.state.messages[1])

    def test_ask_continuation_is_terminal_answer_only(self):
        with self.assertRaises(ValueError):
            build_base_messages(self.state, Action.ASK, stage="after_ask")
        updated = extend_after_ask(self.state, "When did the symptom start?", "It began yesterday.")
        self.assertEqual(updated.asked_count, 1)
        self.assertEqual(updated.dialogue[-1]["content"], "It began yesterday.")
        self.assertIn("Do not ask", build_base_messages(updated, Action.ANSWER, stage="after_ask")[0]["content"])
        self.assertTrue(valid_ask_transition(self.state, updated, "When did the symptom start?", "It began yesterday."))
        self.assertFalse(valid_ask_transition(self.state, self.state, "When did the symptom start?", "It began yesterday."))

    def test_action_output_parsers(self):
        self.assertEqual(parse_question("Question: When did this begin?"), "When did this begin?")
        self.assertEqual(parse_answer("Final Answer: c"), "C")
        self.assertEqual(parse_answer("Final Answer: D"), "D")
        with self.assertRaises(ValueError):
            parse_answer("Final Answer: A plus explanation")
        with self.assertRaises(ValueError):
            parse_answer("Expert Final Answer: D")
        with self.assertRaises(ValueError):
            parse_answer("Final Answer: None")

    def test_forced_single_choice_prompts_and_generation_contracts(self):
        base_prompt = build_base_messages(self.state, Action.ANSWER)[0]["content"]
        self.assertIn("choose exactly one", base_prompt)
        self.assertIn("Final Answer: <A-D>", base_prompt)
        self.assertIn("A-D", FINAL_ANSWER_REGEX)
        expert_prompt = build_defer_messages(self.state)[0]["content"]
        self.assertIn("Final Answer: <A-D>", expert_prompt)
        self.assertNotIn("Expert Final Answer", expert_prompt)
        ask_prompt = build_base_messages(self.state, Action.ASK)[0]["content"]
        self.assertIn("Ground the question in the visible symptoms or history", ask_prompt)
        self.assertIn("Question:", ASK_QUESTION_REGEX)

    def test_base_and_expert_prompts_use_public_state_only(self):
        hidden = "A deliberately hidden fixture fact longer than twelve chars"
        for messages in (build_base_messages(self.state, Action.ANSWER), build_defer_messages(self.state)):
            assert_no_privileged_payload(messages, public_state=self.state, unrevealed_facts=[hidden])
            self.assertNotIn("V1 may use CONSULT", " ".join(message["content"] for message in messages))
        contaminated = [{"role": "user", "content": hidden}]
        with self.assertRaises(ValueError):
            assert_no_privileged_payload(contaminated, public_state=self.state, unrevealed_facts=[hidden])

    def test_only_patient_role_receives_hidden_facts(self):
        hidden = "A deliberately hidden fixture fact longer than twelve chars"
        patient_messages = build_patient_messages([hidden], "What happened?")
        self.assertIn(hidden, " ".join(message["content"] for message in patient_messages))
        base_messages = build_base_messages(self.state, Action.ASK)
        self.assertNotIn(hidden, " ".join(message["content"] for message in base_messages))

    def test_reward_cost_is_applied_once(self):
        result = action_net_return(1.0, ask_count=1, defer_count=0, ask_cost=0.05, defer_cost=0.60)
        self.assertAlmostEqual(result["return"], 0.95)
        deferred = action_net_return(1.0, ask_count=0, defer_count=1, ask_cost=0.05, defer_cost=0.60)
        self.assertAlmostEqual(deferred["return"], 0.40)

    def test_loopback_endpoints_bypass_http_proxy(self):
        opener = direct_loopback_opener()
        handlers = [handler for handler in opener.handlers if handler.__class__.__name__ == "ProxyHandler"]
        self.assertTrue(not handlers or all(handler.proxies == {} for handler in handlers))


class QDatasetTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "case_id": "train-case",
            "state_id": "state-1",
            "state_index": 0,
            "public_state": [{"role": "user", "content": "public fixture only"}],
            "public_state_digest": "abc",
        }

    def rollout(self, action, value, seed, valid=True):
        return {
            **self.state,
            "action": action,
            "valid": valid,
            "return": value if valid else None,
            "return_mean": value if valid else None,
            "return_std": None,
            "sample_count": 1,
            "valid_sample_count": int(valid),
            "rollout_seed": seed,
            "continuation_policy_id": "fixed-v1",
        }

    def test_valid_mask_does_not_impute_missing_actions(self):
        rows = [self.rollout("ANSWER", 1.0, 1), self.rollout("ASK", None, 2, valid=False)]
        item = q_dataset_for_state(self.state, rows)
        self.assertEqual(item["Q_ANSWER"], 1.0)
        self.assertIsNone(item["Q_DEFER"])
        self.assertFalse(item["valid_mask"]["DEFER"])
        self.assertFalse(item["valid_mask"]["ASK"])
        self.assertIsNone(item["return_std"]["Q_ANSWER"])

    def test_replicates_are_aggregated_and_case_split_enforced(self):
        rows = [self.rollout("ASK", 0.2, 1), self.rollout("ASK", 0.4, 2)]
        dataset = build_q_dataset(rows, {"train-case"})
        self.assertEqual(len(dataset), 1)
        self.assertAlmostEqual(dataset[0]["Q_ASK"], 0.3)
        self.assertEqual(dataset[0]["valid_sample_count"]["ASK"], 2)
        with self.assertRaises(ValueError):
            build_q_dataset(rows, {"some-other-case"})

    def test_mixed_continuation_policies_are_rejected(self):
        first = self.rollout("ASK", 0.2, 1)
        second = self.rollout("ASK", 0.4, 2)
        second["continuation_policy_id"] = "different-policy"
        with self.assertRaises(ValueError):
            q_dataset_for_state(self.state, [first, second])


if __name__ == "__main__":
    unittest.main()
