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
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {"ea_net_episode_reward": episode_scores},
            }
        return reward_tensor
