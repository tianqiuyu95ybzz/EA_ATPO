"""E0: verify whether ATPO's calculated question reward enters tree values."""

import asyncio
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from recipe.atpo import user_assistant_agent_loop as agent_loop
from recipe.atpo.tree_search_manager import TreeNode


COEFFICIENTS = {
    "format_score": 0.1,
    "verify_score": 0.4,
    "effective_score": 0.5,
    "correctness_score": 3.0,
    "exceed_max_turn_penalty": -1.0,
}


class RewardPathAudit(unittest.TestCase):
    def setUp(self):
        self.initial = HumanMessage(content="Public case and options")
        self.runner = agent_loop.TreeSearchRunner(
            initial_messages=[self.initial],
            params={
                "call_critic_enabled": True,
                "gamma": 1.0,
                "only_use_critic_value": True,
                "M_trajectories": 2,
                "N_candidates": 4,
            },
            model=None,
            config={"configurable": {"ground_truth": {"answer": "A"}}},
            reward_coefficients=COEFFICIENTS,
            statistic_metrics={},
        )
        self.runner.root.critic_state_value = 0.2

    def question_node(self, verifier, patient_answer):
        assistant = AIMessage(content="<think>Reasoning</think>Question: What is the symptom?")
        patient = HumanMessage(content=patient_answer)
        node = TreeNode(
            messages=[self.initial, assistant, patient],
            parent=self.runner.root,
            assistant_message=assistant,
            verifier_response=verifier,
            is_terminal=False,
        )
        self.runner.root.children = [node]
        return node

    def evaluate(self, node):
        async def fake_critic(*_args, **_kwargs):
            return 0.4

        with patch.object(agent_loop, "call_critic_model", fake_critic):
            asyncio.run(self.runner._evaluate_nodes([node]))

    def test_new_answer_reward_is_computed_but_not_used_in_tree(self):
        node = self.question_node("<Normal>", "The patient has chest pain.")
        info = self.runner._calculate_turn_reward(node)
        self.assertAlmostEqual(info["process_reward"], 1.0)
        self.assertTrue(info["details"]["is_effective"])
        self.evaluate(node)
        self.assertEqual(node.action_reward, 0.0)
        self.assertAlmostEqual(node.q_value, 0.4)
        node.explore_num = 1
        node.mdp_state_value = 0.4
        self.runner._perform_tree_backup({self.runner.root, node})
        self.assertAlmostEqual(self.runner.root.mdp_state_value, 0.0 + 0.4)
        self.runner._calculate_advantage_and_returns_for_all_nodes(
            {self.runner.root, node}
        )
        self.assertAlmostEqual(node.advantage, 0.2)
        self.assertAlmostEqual(node.returns, 0.4)

    def test_repeated_question_process_score_is_also_excluded(self):
        node = self.question_node("<Repeated>", "The patient has chest pain.")
        info = self.runner._calculate_turn_reward(node)
        self.assertAlmostEqual(info["process_reward"], 0.1)
        self.assertTrue(info["details"]["is_repeated"])
        self.evaluate(node)
        self.assertEqual(node.action_reward, 0.0)
        self.assertAlmostEqual(node.q_value, 0.4)

    def test_process_reward_switch_changes_tree_q_and_advantage(self):
        self.runner.params["enable_process_reward"] = True
        node = self.question_node("<Normal>", "The patient has chest pain.")
        self.evaluate(node)
        self.assertAlmostEqual(node.action_reward, 1.0)
        self.assertAlmostEqual(node.q_value, 1.4)
        node.explore_num = 1
        node.mdp_state_value = 0.4
        self.runner._perform_tree_backup({self.runner.root, node})
        self.assertAlmostEqual(self.runner.root.mdp_state_value, 1.4)
        self.runner._calculate_advantage_and_returns_for_all_nodes(
            {self.runner.root, node}
        )
        self.assertAlmostEqual(node.advantage, 1.2)

    def test_unanswerable_normal_question_still_gets_positive_process_score(self):
        self.runner.params["enable_process_reward"] = True
        node = self.question_node(
            "<Normal>", "The patient cannot answer this question."
        )
        info = self.runner._calculate_turn_reward(node)
        self.assertFalse(info["details"]["is_effective"])
        self.assertTrue(info["details"]["is_reject_human"])
        self.assertAlmostEqual(info["process_reward"], 0.5)
        self.evaluate(node)
        self.assertAlmostEqual(node.action_reward, 0.5)

    def test_terminal_correct_answer_enters_tree(self):
        assistant = AIMessage(content="<think>Reasoning</think>Final Answer: A")
        node = TreeNode(
            messages=[self.initial, assistant],
            parent=self.runner.root,
            assistant_message=assistant,
            is_terminal=True,
        )
        info = self.runner._calculate_turn_reward(node)
        self.assertAlmostEqual(info["outcome_reward"], 3.0)
        self.evaluate(node)
        self.assertAlmostEqual(node.action_reward, 3.0)
        self.assertAlmostEqual(node.q_value, 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
