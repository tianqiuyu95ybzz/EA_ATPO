import unittest

from recipe.ea_atpo.protocol import (
    Action,
    PublicState,
    build_expert_messages,
    parse_action,
    unrevealed_atomic_facts,
    validate_expert_response,
    validate_expert_view,
)
from recipe.ea_atpo.utility import UtilityConfig, compute_utilities, select_action


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.state = PublicState(
            sample_id="7",
            initial_messages=({"role": "user", "content": "Visible case. Options: A, B, C, D"},),
            dialogue=({"role": "assistant", "content": "Question: Any fever?"}, {"role": "user", "content": "No fever."}),
        )

    def test_expert_view_contains_only_public_state(self):
        messages = build_expert_messages(self.state, Action.CONSULT, "What should be distinguished next?")
        validate_expert_view(messages, atomic_facts=["Hidden laboratory result is positive."], ground_truth={"answer": "B", "answer_info": "Gold explanation"})

    def test_explicit_privileged_key_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_expert_view([{"role": "user", "content": "atomic_facts: secret"}])

    def test_hidden_fact_value_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_expert_view([{"role": "user", "content": "Hidden laboratory result is positive."}], atomic_facts=["Hidden laboratory result is positive."])

    def test_already_public_atomic_fact_is_not_hidden(self):
        state = PublicState("x", ({"role": "user", "content": "The patient has fever."},))
        self.assertEqual(unrevealed_atomic_facts(state, ["The patient has fever.", "A hidden rash is present."]), ["A hidden rash is present."])

    def test_public_option_text_is_not_mistaken_for_gold_leakage(self):
        messages = [{"role": "user", "content": "Options: {'A': 'Give aspirin', 'B': 'Obtain an ECG'}"}]
        validate_expert_view(messages, ground_truth={"answer": "B", "answer_info": "Obtain an ECG"})

    def test_consult_accepts_post_thinking_protocol(self):
        value = validate_expert_response(Action.CONSULT, "<think>private</think>\nExpert Advice: obtain an ECG")
        self.assertEqual(value, "Expert Advice: obtain an ECG")

    def test_consult_cannot_answer(self):
        self.assertEqual(validate_expert_response(Action.CONSULT, "Expert Advice: compare time course and exposures"), "Expert Advice: compare time course and exposures")
        with self.assertRaises(ValueError):
            validate_expert_response(Action.CONSULT, "Expert Advice: option B is correct")

    def test_defer_is_a_terminal_expert_answer(self):
        self.assertEqual(validate_expert_response(Action.DEFER, "Expert Final Answer: C"), "Expert Final Answer: C")

    def test_action_parser(self):
        self.assertEqual(parse_action("</think>\nQuestion: Any fever?")[0], Action.ASK)
        self.assertEqual(parse_action("Final Answer: D"), (Action.ANSWER, "D"))


class UtilityTest(unittest.TestCase):
    def test_cost_changes_selected_action_without_double_counting(self):
        q = {Action.ANSWER: .60, Action.ASK: .70, Action.CONSULT: .80, Action.DEFER: .95}
        chosen, utilities = select_action(q, UtilityConfig())
        self.assertEqual(chosen, Action.ASK)
        self.assertAlmostEqual(utilities[Action.DEFER], .35)

    def test_requires_all_counterfactuals(self):
        with self.assertRaises(ValueError):
            compute_utilities({Action.ANSWER: 1.0}, UtilityConfig())


if __name__ == "__main__":
    unittest.main()
