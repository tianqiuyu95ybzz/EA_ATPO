import asyncio
import unittest

from recipe.ea_atpo.collect_counterfactuals import APICall, _continue_base, generate_states, rollout_action
from recipe.ea_atpo.protocol import Action, PublicState


class FakeAPI:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.generation_kwargs = []

    async def generate(self, messages, **kwargs):
        self.requests.append([dict(message) for message in messages])
        self.generation_kwargs.append(dict(kwargs))
        if not self.responses:
            raise AssertionError("unexpected model call")
        return APICall(self.responses.pop(0), 10, 5, 0.01)


class ConsultCollectionTest(unittest.TestCase):
    def setUp(self):
        self.state = PublicState(
            "case-1",
            ({"role": "user", "content": "Public case. Options: A, B, C, D"},),
        )
        self.facts = ["A hidden patient fact is present."]
        self.gold = {"answer": "A", "answer_info": "Private explanation"}

    def test_consult_question_is_agent_generated_and_think_is_not_exposed(self):
        focus = "Which finding best distinguishes the two leading options?"
        base = FakeAPI(
            f"<think>private chain of thought</think>Consult Expert: {focus}",
            "<think>another private thought</think>Final Answer: A",
        )
        patient = FakeAPI()
        expert = FakeAPI(
            "<think>private expert chain</think>Expert Advice: Compare the time course and obtain a targeted test."
        )

        record = asyncio.run(
            rollout_action(
                Action.CONSULT,
                self.state,
                self.facts,
                self.gold,
                base,
                patient,
                expert,
                temperature=0.2,
                max_turns=3,
            )
        )

        expert_prompt = "\n".join(message["content"] for message in expert.requests[0])
        self.assertIn(focus, expert_prompt)
        self.assertNotIn("private chain of thought", expert_prompt)
        self.assertNotIn(self.facts[0], expert_prompt)
        self.assertNotIn("Private explanation", expert_prompt)
        self.assertTrue(record["valid"])
        self.assertEqual(record["answer"], "A")
        self.assertEqual(base.generation_kwargs[0]["temperature"], 0.2)
        public_text = "\n".join(message["content"] for message in record["public_messages"])
        self.assertIn(f"Consult Expert: {focus}", public_text)
        self.assertNotIn("private chain of thought", public_text)
        serialized_record = str(record)
        self.assertNotIn("private chain of thought", serialized_record)
        self.assertNotIn("private expert chain", serialized_record)

    def test_sequential_consult_then_defer_is_terminal(self):
        base = FakeAPI(
            "Consult Expert: What evidence would change the differential?",
            "Defer to Expert",
        )
        patient = FakeAPI()
        expert = FakeAPI(
            "Expert Advice: Reassess the timeline and one discriminating symptom.",
            "Expert Final Answer: B",
        )

        answer, trace, final_state = asyncio.run(
            _continue_base(
                base,
                patient,
                expert,
                self.state,
                self.facts,
                temperature=0.0,
                max_turns=3,
            )
        )

        self.assertEqual(answer, "B")
        self.assertEqual(
            sum(item.get("actor") == "base" and item.get("action") == "CONSULT" for item in trace),
            1,
        )
        self.assertEqual(
            sum(item.get("actor") == "base" and item.get("action") == "DEFER" for item in trace),
            1,
        )
        self.assertEqual(len(expert.requests), 2)
        final_expert_prompt = "\n".join(message["content"] for message in expert.requests[1])
        self.assertIn("Expert Advice: Reassess", final_expert_prompt)
        self.assertNotIn("Defer to Expert", final_expert_prompt)
        self.assertNotIn(self.facts[0], final_expert_prompt)
        self.assertIn("Expert Final Answer: B", "\n".join(m["content"] for m in final_state.messages))

    def test_base_can_consult_again_after_consult(self):
        base = FakeAPI(
            "Consult Expert: What should be checked first?",
            "Consult Expert: Which finding distinguishes the leading choices?",
            "Final Answer: B",
        )
        patient = FakeAPI()
        expert = FakeAPI(
            "Expert Advice: Clarify the onset and progression.",
            "Expert Advice: Compare the time course against the listed choices.",
        )

        record = asyncio.run(
            rollout_action(
                Action.CONSULT,
                self.state,
                self.facts,
                self.gold,
                base,
                patient,
                expert,
                temperature=0.0,
                max_turns=4,
            )
        )

        self.assertTrue(record["valid"])
        self.assertEqual(record["answer"], "B")
        self.assertEqual(record["patient_turns"], 0)
        self.assertEqual(sum(x.get("actor") == "expert" for x in record["trace"]), 2)
        self.assertEqual(record["total_prompt_tokens"], 50)

    def test_base_can_ask_after_consult(self):
        base = FakeAPI(
            "Consult Expert: What history is most informative?",
            "Question: When did the symptom begin?",
            "Final Answer: A",
        )
        patient = FakeAPI("It started yesterday.")
        expert = FakeAPI("Expert Advice: Establish the symptom timeline.")

        record = asyncio.run(
            rollout_action(
                Action.CONSULT,
                self.state,
                self.facts,
                self.gold,
                base,
                patient,
                expert,
                temperature=0.0,
                max_turns=4,
            )
        )

        self.assertTrue(record["valid"])
        self.assertEqual(record["answer"], "A")
        self.assertEqual(record["patient_turns"], 1)
        self.assertEqual(record["patient_completion_tokens"], 5)

    def test_defer_counterfactual_calls_only_terminal_expert(self):
        base = FakeAPI()
        patient = FakeAPI()
        expert = FakeAPI("Expert Final Answer: C")

        record = asyncio.run(
            rollout_action(
                Action.DEFER,
                self.state,
                self.facts,
                {"answer": "C", "answer_info": "Private explanation"},
                base,
                patient,
                expert,
                temperature=0.0,
                max_turns=3,
            )
        )

        self.assertTrue(record["valid"])
        self.assertEqual(record["answer"], "C")
        self.assertEqual(len(expert.requests), 1)
        self.assertEqual(base.requests, [])
        prompt = "\n".join(message["content"] for message in expert.requests[0])
        self.assertIn("Make the terminal decision now", prompt)
        self.assertNotIn("Private explanation", prompt)

    def test_patient_unknown_is_kept_as_unknown_in_public_history(self):
        base = FakeAPI(
            "Question: Does the patient report symptom X?",
            "Final Answer: D",
        )
        patient = FakeAPI("The patient cannot answer this question.")
        expert = FakeAPI()

        record = asyncio.run(
            rollout_action(
                Action.ASK,
                self.state,
                self.facts,
                {"answer": "D"},
                base,
                patient,
                expert,
                temperature=0.0,
                max_turns=3,
            )
        )

        self.assertTrue(record["valid"])
        self.assertIn(
            "The patient cannot answer this question.",
            "\n".join(message["content"] for message in record["public_messages"]),
        )

    def test_state_generation_removes_private_think_before_expert_can_see_state(self):
        base = FakeAPI(
            "<think>private differential</think>Question: When did symptom X begin?",
            "<think>not used</think>Final Answer: A",
        )
        patient = FakeAPI("<think>patient private reasoning</think>It began yesterday.")
        row = {
            "extra_info": {
                "prompt": [{"role": "user", "content": "Public case. Options: A, B, C, D"}],
                "atomic_facts": self.facts,
                "index": "case-1",
            },
            "reward_model": {"ground_truth": self.gold},
        }

        states = asyncio.run(
            generate_states([row], base, patient, limit=3, temperature=0.0, max_turns=3)
        )

        state_text = "\n".join(m["content"] for m in states[1][0].messages)
        self.assertIn("Question: When did symptom X begin?", state_text)
        self.assertIn("It began yesterday.", state_text)
        self.assertNotIn("private differential", state_text)
        self.assertNotIn("patient private reasoning", state_text)

    def test_turn_limit_without_terminal_answer_has_failure_reason(self):
        base = FakeAPI("Question: Any fever?", "Question: Any rash?")
        patient = FakeAPI("No fever.", "No rash.")
        expert = FakeAPI()

        record = asyncio.run(
            rollout_action(
                Action.ASK,
                self.state,
                self.facts,
                self.gold,
                base,
                patient,
                expert,
                temperature=0.0,
                max_turns=2,
            )
        )

        self.assertFalse(record["valid"])
        self.assertEqual(record["error"], "no_terminal_answer")


if __name__ == "__main__":
    unittest.main(verbosity=2)
