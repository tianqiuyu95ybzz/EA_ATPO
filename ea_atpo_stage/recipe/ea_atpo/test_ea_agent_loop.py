"""Deterministic protocol smoke for the EA tree loop (not a model-quality test)."""

import asyncio
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import convert_to_messages
from transformers import AutoTokenizer

import recipe.atpo.user_assistant_agent_loop as original_loop
import recipe.ea_atpo.ea_agent_loop as ea_loop
from recipe.ea_atpo.protocol import PublicState
from recipe.atpo.chat_model import convert_to_agent_output


class _DeterministicClient:
    """Stand-in for generation only; it does not supply experimental results."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prompts = []

    async def generate(self, *, request_id, prompt_ids, sampling_params):
        prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        self.prompts.append(prompt)
        if "Patient atomic facts:" in prompt:
            text = "The patient reports fever."
        elif "You are a senior medical consultant" in prompt:
            text = "Expert Advice: Compare the symptom timeline with the listed options."
        elif "You are the senior clinician taking over this case" in prompt:
            text = "Expert Final Answer: B"
        elif "Patient response:" in prompt or "Expert advice response:" in prompt:
            text = "Final Answer: B"
        else:
            text = [
                "Final Answer: A",
                "Question: When did the symptom start?",
                "Consult Expert: Which finding distinguishes the leading options?",
                "Defer to Expert",
            ][int(request_id.rsplit("-", 1)[-1])]
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return SimpleNamespace(token_ids=token_ids, log_probs=[-0.02] * len(token_ids))


async def _fixed_critic_value(messages, is_terminal, config):
    return 0.0 if is_terminal else 0.2


class EAFourActionLoopSmokeTest(unittest.TestCase):
    def test_four_actions_mask_cost_continue_and_isolate_roles(self):
        model_path = os.environ.get(
            "EA_ATPO_MODEL_PATH",
            "/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B",
        )
        if not (Path(model_path) / "tokenizer_config.json").is_file():
            self.skipTest(f"Qwen3 tokenizer unavailable at {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        client = _DeterministicClient(tokenizer)
        model = ea_loop.EAChatModel.model_construct(
            model_name="qwen3",
            client=client,
            tokenizer=tokenizer,
            max_tokens=512,
            temperature=1.0,
            top_p=0.8,
            repetition_penalty=1.0,
        )
        messages = convert_to_messages(
            [
                {"role": "system", "content": "Choose one EA-ATPO action."},
                {"role": "user", "content": "Public case. Options: A, B, C, D."},
            ]
        )
        public_state = PublicState(
            sample_id="protocol-unit-case",
            initial_messages=({"role": "user", "content": "Public case. Options: A, B, C, D."},),
            remaining_interaction_budget=3,
        )
        config = {
            "configurable": {
                "model": model,
                "server_manager": client,
                "tokenizer": tokenizer,
                "critic_worker": object(),
                "sampling_params": {"temperature": 1.0, "top_p": 0.8, "logprobs": True},
                "max_assistant_turns": 3,
                "max_actor_actions": 3,
                "max_asks": 1,
                "max_consults": 1,
                "atomic_facts": ["A 40-year-old patient reports fever."],
                "ground_truth": {"answer": "B", "answer_info": "private gold explanation"},
                "correctness_reward": 1.0,
            }
        }
        params = {
            "M_trajectories": 4,
            "N_candidates": 4,
            "variance_threshold": 1.2,
            "pruning_enabled": False,
            "call_critic_enabled": True,
            "only_use_critic_value": True,
            "gamma": 1.0,
            "lam": 1.0,
        }
        costs = {
            "ask_cost": 0.05,
            "consult_cost": 0.20,
            "defer_cost": 0.60,
            "lambda_ask": 1.0,
            "lambda_consult": 1.0,
            "lambda_defer": 1.0,
        }

        async def run():
            runner = ea_loop.EAATPOSearchRunner(
                messages, params, model, config, {}, {}, public_state, costs
            )
            return runner, await runner.run()

        with patch.object(ea_loop, "call_critic_model", _fixed_critic_value), patch.object(
            original_loop, "call_critic_model", _fixed_critic_value
        ):
            runner, leaves = asyncio.run(run())

        traces = [runner.episode_trace(leaf) for leaf in leaves]
        first_actions = {trace["action_trace"][0]["action"] for trace in traces}
        self.assertEqual(first_actions, {"ANSWER", "ASK", "CONSULT", "DEFER"})
        self.assertTrue(any(event["action"] == "ASK" and not event["terminated"] for t in traces for event in t["action_trace"]))
        self.assertTrue(any(event["action"] == "CONSULT" and not event["terminated"] for t in traces for event in t["action_trace"]))
        self.assertTrue(any(event["termination_reason"] == "expert_deferred_answer" for t in traces for event in t["action_trace"]))

        expected_cost = {"ANSWER": 0.0, "ASK": 0.05, "CONSULT": 0.20, "DEFER": 0.60}
        expected_answer = {"ANSWER": "A", "ASK": "B", "CONSULT": "B", "DEFER": "B"}
        for leaf, trace in zip(leaves, traces):
            first_action = trace["action_trace"][0]["action"]
            self.assertAlmostEqual(trace["evaluation"]["resource_cost_total"], expected_cost[first_action])
            self.assertEqual(trace["final_answer"], expected_answer[first_action])
            for event in trace["action_trace"]:
                state_text = repr(event["public_state_before"]) + repr(event["public_state_after"])
                self.assertNotIn("private gold explanation", state_text)
                if event["action"] in {"ANSWER", "ASK", "CONSULT", "DEFER"}:
                    self.assertGreater(event["actor_completion_tokens"], 0)

            output = convert_to_agent_output(leaf, 512)
            actor_tokens = sum(event["actor_completion_tokens"] for event in trace["action_trace"])
            self.assertEqual(sum(output.response_mask), actor_tokens)
            if first_action in {"ASK", "CONSULT"}:
                self.assertTrue(any(mask == 0 for mask in output.response_mask))
            if first_action == "DEFER":
                ea_loop.EAATPOAgentLoop._append_terminal_observation(
                    SimpleNamespace(tokenizer=tokenizer), output, leaf, 512
                )
                self.assertEqual(sum(output.response_mask), actor_tokens)
                self.assertTrue(any(mask == 0 for mask in output.response_mask))

        expert_prompts = [p for p in client.prompts if "You are a senior medical consultant" in p or "You are the senior clinician taking over" in p]
        self.assertEqual(len(expert_prompts), 2)
        for prompt in expert_prompts:
            self.assertNotIn("A 40-year-old patient reports fever.", prompt)
            self.assertNotIn("private gold explanation", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
