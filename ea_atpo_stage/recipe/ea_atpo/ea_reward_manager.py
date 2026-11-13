"""Reward-accounting adapter for EA-ATPO precomputed tree targets.

The agent loop has already computed the episode's net EA reward from the exact
tree-edge rewards. This adapter exposes that scalar to ATPO's ordinary reward
logging path. It does not compute another reward or alter Actor/Critic targets.
"""
from __future__ import annotations

from typing import Any

import torch

from verl import DataProto


class EARewardManager:
    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        reward_coefficients=None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" not in data.batch:
            raise RuntimeError(
                "EARewardManager requires AgentLoopOutput.reward_score from the EA tree trace"
            )
        reward_tensor = data.batch["rm_scores"].to(dtype=torch.float32)
        episode_scores = reward_tensor.sum(dim=-1).detach().cpu().tolist()
        if return_dict:
            # ATPO's trainer still logs its legacy multi-turn interaction
            # metrics after the PPO update. EA-ATPO does not use that protocol,
            # so derive answer validity/correctness from the already completed
            # episode trace and mark the unrelated legacy counters as zero.
            # This is post-rollout evaluation metadata; it does not affect the
            # precomputed EA rewards, returns, advantages, or PPO loss.
            statistics = data.non_tensor_batch.get("statistics")
            if statistics is None:
                raise RuntimeError(
                    "EARewardManager requires episode statistics to populate ATPO metrics"
                )

            episodes = []
            for item in statistics:
                if not isinstance(item, dict) or not isinstance(item.get("ea_episode"), dict):
                    raise RuntimeError(
                        "EARewardManager expected statistics['ea_episode'] from EAATPOAgentLoop"
                    )
                episodes.append(item["ea_episode"])

            valid_answers = {"A", "B", "C", "D"}
            final_answers = [episode.get("final_answer") for episode in episodes]
            evaluations = [episode.get("evaluation", {}) for episode in episodes]
            traces = [episode.get("action_trace", []) for episode in episodes]
            invalid_action_rates = [
                (sum(not bool(event.get("action_valid", False)) for event in trace) / len(trace))
                if trace else 0.0
                for trace in traces
            ]
            reward_extra_info = {
                "ea_net_episode_reward": episode_scores,
                "is_valid": [int(answer in valid_answers) for answer in final_answers],
                "is_correct": [int(bool(evaluation.get("correct", False))) for evaluation in evaluations],
                "incorrect_format_rate": invalid_action_rates,
                "multiple_rate": [0.0] * len(episodes),
                "repeated_rate": [0.0] * len(episodes),
                "verify_timeout_rate": [0.0] * len(episodes),
                "effective_rate": [0.0] * len(episodes),
                "human_reject_rate": [0.0] * len(episodes),
                "human_timeout_rate": [0.0] * len(episodes),
                "assistant_error_rate": invalid_action_rates,
            }
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
