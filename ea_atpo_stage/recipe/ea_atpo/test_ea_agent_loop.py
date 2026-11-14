"""Deterministic protocol smoke for the EA tree loop (not a model-quality test)."""

import asyncio
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import convert_to_messages
import numpy as np
from transformers import AutoTokenizer
import torch
from verl import DataProto

import recipe.atpo.user_assistant_agent_loop as original_loop
import recipe.ea_atpo.ea_agent_loop as ea_loop
from recipe.ea_atpo.ea_reward_manager import EARewardManager
from recipe.ea_atpo.protocol import PublicState
from recipe.atpo.chat_model import convert_to_agent_output
from recipe.atpo.mt_metric_utils import compute_accuracy_metrics


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


class _MetricTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        answers = {1: "Final Answer: B", 2: "Expert Final Answer: A"}
        return answers[token_ids[0]]


class _FourStateForcedClient:
    """Test-only action driver; its outputs never enter the PPO trainer."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prompts = []
        self.cursors = {}
        self.actor_scripts = {
            "FORCED_ANSWER": {0: ["Final Answer: B"], 1: ["Final Answer: A"]},
            "FORCED_ASK": {
                0: ["Question: Does the patient report fever?", "Final Answer: B"],
                1: ["Final Answer: A"],
            },
            "FORCED_CONSULT": {
                0: ["Consult Expert: Which finding favors option B?", "Final Answer: B"],
                1: ["Final Answer: A"],
            },
            "FORCED_DEFER": {0: ["Defer to Expert"], 1: ["Final Answer: A"]},
        }

    async def generate(self, *, request_id, prompt_ids, sampling_params):
        prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        self.prompts.append(prompt)
        if "Patient atomic facts:" in prompt:
            text = "The patient reports fever."
        elif "You are a senior medical consultant" in prompt:
            text = "Expert Advice: Compare the timeline with the listed options."
        elif "You are the senior clinician taking over this case" in prompt:
            text = "Expert Final Answer: B"
        else:
            case = next((name for name in self.actor_scripts if name in prompt), None)
            if case is None:
                raise AssertionError("Actor prompt did not identify one of the four forced test states")
            candidate = int(request_id.rsplit("-", 1)[-1])
            key = (case, candidate)
            cursor = self.cursors.get(key, 0)
            script = self.actor_scripts[case][candidate]
            if cursor >= len(script):
                raise AssertionError(f"Unexpected extra Actor call for {case}, candidate {candidate}")
            text = script[cursor]
            self.cursors[key] = cursor + 1
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return SimpleNamespace(token_ids=token_ids, log_probs=[-0.02] * len(token_ids))


async def _fixed_critic_value(messages, is_terminal, config):
    return 0.0 if is_terminal else 0.2


class EAFourActionLoopSmokeTest(unittest.TestCase):
    def test_legal_action_context_and_failure_training_eligibility(self):
        initial = ea_loop._install_action_protocol(
            [{"role": "user", "content": "Public case."}],
            budget=3,
            max_asks=1,
            max_consults=1,
        )
        self.assertIn("Legal actions: ANSWER, ASK, CONSULT, DEFER", initial[0]["content"])

        after_ask = ea_loop._decision_observation(
            "Patient", "Cannot answer.", actor_actions_used=1, remaining=2,
            asked=1, max_asks=1, consulted=0, max_consults=1,
        )
        self.assertIn("Legal actions: ANSWER, CONSULT, DEFER", after_ask)
        self.assertNotIn("Legal actions: ANSWER, ASK,", after_ask)

        last_decision = ea_loop._decision_observation(
            "Expert advice", "Consider the options.", actor_actions_used=2, remaining=1,
            asked=1, max_asks=1, consulted=1, max_consults=1,
        )
        self.assertIn("Legal actions: ANSWER, DEFER", last_decision)

        for reason in ("invalid_actor_action: unsupported response",):
            self.assertEqual(ea_loop._failure_category(reason), "protocol_error")
        for reason in (
            "actor_generation_failed: timeout",
            "patient_call_failed: timeout",
            "consult_call_failed: timeout",
            "defer_call_failed: timeout",
        ):
            self.assertEqual(ea_loop._failure_category(reason), "environment_error")
        for reason in (
            "ask_budget_exhausted",
            "consult_budget_exhausted",
            "no_remaining_continuation_budget",
        ):
            self.assertEqual(ea_loop._failure_category(reason), "policy_constraint_violation")

    def test_four_independent_forced_states_cover_protocol_and_token_masks(self):
        """Force each action in a distinct state without creating PPO updates."""
        model_path = os.environ.get(
            "EA_ATPO_MODEL_PATH",
            "/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B",
        )
        if not (Path(model_path) / "tokenizer_config.json").is_file():
            self.skipTest(f"Qwen3 tokenizer unavailable at {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        client = _FourStateForcedClient(tokenizer)
        model = ea_loop.EAChatModel.model_construct(
            model_name="qwen3",
            client=client,
            tokenizer=tokenizer,
            max_tokens=512,
            temperature=1.0,
            top_p=0.8,
            repetition_penalty=1.0,
        )
        costs = {
            "ask_cost": 0.05,
            "consult_cost": 0.20,
            "defer_cost": 0.60,
            "lambda_ask": 1.0,
            "lambda_consult": 1.0,
            "lambda_defer": 1.0,
        }
        cases = [
            ("FORCED_ANSWER", ea_loop.Action.ANSWER, 0.0, ["ANSWER"], "base_answer", ""),
            ("FORCED_ASK", ea_loop.Action.ASK, 0.05, ["ASK", "ANSWER"], "base_answer", "Does the patient report fever?"),
            ("FORCED_CONSULT", ea_loop.Action.CONSULT, 0.20, ["CONSULT", "ANSWER"], "base_answer", "Which finding favors option B?"),
            ("FORCED_DEFER", ea_loop.Action.DEFER, 0.60, ["DEFER"], "expert_deferred_answer", ""),
        ]

        async def run_cases():
            results = []
            for case_name, target_action, expected_cost, expected_actions, terminal_reason, _ in cases:
                prompt = [
                    {"role": "system", "content": "Use the EA-ATPO action protocol."},
                    {"role": "user", "content": f"Public vignette {case_name}. Options A, B, C, D."},
                ]
                actor_messages = ea_loop._install_action_protocol(
                    prompt, budget=3, max_asks=1, max_consults=1
                )
                messages = convert_to_messages(actor_messages)
                public_state = PublicState(
                    sample_id=case_name,
                    initial_messages=({"role": "user", "content": prompt[1]["content"]},),
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
                        "atomic_facts": [f"private hidden fact for {case_name}: fever is reported"],
                        "ground_truth": {"answer": "B", "answer_info": "private gold explanation"},
                        "correctness_reward": 1.0,
                    }
                }
                params = {
                    "M_trajectories": 2,
                    "N_candidates": 2,
                    "variance_threshold": 1.2,
                    "pruning_enabled": False,
                    "call_critic_enabled": True,
                    "only_use_critic_value": True,
                    "gamma": 1.0,
                    "lam": 1.0,
                }
                runner = ea_loop.EAATPOSearchRunner(
                    messages,
                    params,
                    model,
                    config,
                    {},
                    {},
                    public_state,
                    costs,
                )
                leaves = await runner.run()
                traces = [(leaf, runner.episode_trace(leaf)) for leaf in leaves]
                target_text = {
                    ea_loop.Action.ANSWER: "Final Answer: B",
                    ea_loop.Action.ASK: "Question:",
                    ea_loop.Action.CONSULT: "Consult Expert:",
                    ea_loop.Action.DEFER: "Defer to Expert",
                }[target_action]
                target_leaf, trace = next(
                    (leaf, trace)
                    for leaf, trace in traces
                    if trace["action_trace"][0]["action"] == target_action.value
                    and trace["action_trace"][0]["actor_text"].startswith(target_text)
                )
                self.assertTrue(trace["evaluation"]["training_eligible"])
                output = convert_to_agent_output(target_leaf, 512)
                ea_loop.EAATPOAgentLoop._append_terminal_observation(
                    SimpleNamespace(tokenizer=tokenizer), output, target_leaf, 512
                )
                results.append((case_name, target_action, expected_cost, expected_actions, terminal_reason, trace, output, runner))
            return results

        with patch.object(ea_loop, "call_critic_model", _fixed_critic_value), patch.object(
            original_loop, "call_critic_model", _fixed_critic_value
        ):
            results = asyncio.run(run_cases())

        self.assertEqual([result[1] for result in results], [case[1] for case in cases])
        for case_name, action, expected_cost, expected_actions, terminal_reason, trace, output, runner in results:
            events = trace["action_trace"]
            self.assertEqual([event["action"] for event in events], expected_actions, case_name)
            self.assertEqual(events[-1]["termination_reason"], terminal_reason, case_name)
            self.assertTrue(events[-1]["terminated"], case_name)
            self.assertTrue(trace["evaluation"]["correct"], case_name)
            self.assertAlmostEqual(trace["evaluation"]["resource_cost_total"], expected_cost, msg=case_name)
            self.assertAlmostEqual(trace["evaluation"]["net_episode_reward"], 1.0 - expected_cost, msg=case_name)
            self.assertAlmostEqual(
                sum(float(event["action_cost"]) for event in events),
                -expected_cost,
                msg=f"cost must be charged exactly once: {case_name}",
            )
            self.assertTrue(all(event["return"] is not None and event["advantage"] is not None for event in events))
            expected_first_advantage = {
                ea_loop.Action.ANSWER: 0.8,
                ea_loop.Action.ASK: -0.05,
                ea_loop.Action.CONSULT: -0.20,
                ea_loop.Action.DEFER: 0.20,
            }[action]
            self.assertAlmostEqual(float(events[0]["advantage"]), expected_first_advantage, places=5, msg=case_name)
            self.assertAlmostEqual(
                float(runner.root.mdp_state_value), (1.0 - expected_cost) / 2.0, places=5, msg=case_name
            )

            actor_token_count = sum(int(event["actor_completion_tokens"]) for event in events)
            self.assertEqual(sum(output.response_mask), actor_token_count, case_name)
            actor_text = tokenizer.decode(
                [token for token, mask in zip(output.response_ids, output.response_mask) if mask == 1],
                skip_special_tokens=False,
            )
            observation_text = tokenizer.decode(
                [token for token, mask in zip(output.response_ids, output.response_mask) if mask == 0],
                skip_special_tokens=False,
            )
            actor_spans = []
            index = 0
            while index < len(output.response_mask):
                if output.response_mask[index] == 0:
                    self.assertEqual(float(output.advantages[index]), 0.0, case_name)
                    index += 1
                    continue
                start = index
                while index < len(output.response_mask) and output.response_mask[index] == 1:
                    index += 1
                actor_spans.append((start, index))
            self.assertEqual(
                [end - start for start, end in actor_spans],
                [int(event["actor_completion_tokens"]) for event in events],
                case_name,
            )
            for (start, end), event in zip(actor_spans, events):
                self.assertTrue(
                    all(abs(float(value) - float(event["advantage"])) < 1e-6 for value in output.advantages[start:end]),
                    case_name,
                )
                value_mask = output.value_response_mask[start:end]
                self.assertEqual(sum(value_mask), min(5, end - start), case_name)
                self.assertTrue(all(value == 1 for value in value_mask[:5]), case_name)
                self.assertTrue(all(value == 0 for value in value_mask[5:]), case_name)
            self.assertIn(
                {
                    ea_loop.Action.ANSWER: "Final Answer:",
                    ea_loop.Action.ASK: "Question:",
                    ea_loop.Action.CONSULT: "Consult Expert:",
                    ea_loop.Action.DEFER: "Defer to Expert",
                }[action],
                actor_text,
                case_name,
            )
            self.assertNotIn("Expert Final Answer: B", actor_text, case_name)
            if action is ea_loop.Action.ASK:
                self.assertIn("Does the patient report fever?", actor_text)
                self.assertIn("The patient reports fever.", observation_text)
            elif action is ea_loop.Action.CONSULT:
                self.assertIn("Which finding favors option B?", actor_text)
                self.assertIn("Expert Advice: Compare the timeline", observation_text)
            elif action is ea_loop.Action.DEFER:
                self.assertIn("Expert Final Answer: B", observation_text)
            self.assertEqual(
                len([1 for event in events if event["action"] == "ASK"]),
                int(action is ea_loop.Action.ASK),
                case_name,
            )

        expert_prompts = [
            prompt for prompt in client.prompts
            if "You are a senior medical consultant" in prompt
            or "You are the senior clinician taking over this case" in prompt
        ]
        self.assertEqual(len(expert_prompts), 2)
        for prompt in expert_prompts:
            self.assertNotIn("private hidden fact", prompt)
            self.assertNotIn("private gold explanation", prompt)

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

    def test_reward_manager_only_accounts_for_precomputed_ea_score(self):
        data = DataProto.from_dict(
            tensors={
                "rm_scores": torch.tensor([[0.0, 0.95], [0.0, -0.60]]),
                "responses": torch.tensor([[1, 0], [2, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]]),
            },
            non_tensors={
                "reward_model": np.array(
                    [
                        {"ground_truth": {"answer": "B"}},
                        {"ground_truth": {"answer": "C"}},
                    ],
                    dtype=object,
                )
            },
        )
        manager = EARewardManager(tokenizer=_MetricTokenizer(), num_examine=0)
        result = manager(data, return_dict=True)
        for observed, expected in zip(
            result["reward_tensor"].sum(dim=-1).tolist(), [0.95, -0.60]
        ):
            self.assertAlmostEqual(observed, expected)
        for observed, expected in zip(
            result["reward_extra_info"]["ea_net_episode_reward"], [0.95, -0.60]
        ):
            self.assertAlmostEqual(observed, expected)
        data.non_tensor_batch.update(
            {key: np.asarray(value) for key, value in result["reward_extra_info"].items()}
        )
        metrics = compute_accuracy_metrics(data)
        self.assertEqual(data.non_tensor_batch["is_valid"].tolist(), [1, 1])
        self.assertEqual(data.non_tensor_batch["is_correct"].tolist(), [1, 0])
        self.assertEqual(metrics["custom/num_valid"], 2)
        self.assertEqual(metrics["custom/num_correct"], 1)
        self.assertAlmostEqual(metrics["custom/incorrect_format_rate"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
