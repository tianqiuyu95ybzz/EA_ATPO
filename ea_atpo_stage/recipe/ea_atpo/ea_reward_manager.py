"""Reward-accounting adapter for EA-ATPO precomputed tree targets.

The agent loop has already computed the episode's net EA reward from the exact
tree-edge rewards. This adapter exposes that scalar to ATPO's ordinary reward
logging path. It does not compute another reward or alter Actor/Critic targets.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
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
            # ATPO's legacy logger needs per-sample validity/correctness. Derive
            # those only after the model has finished, using the concatenated
            # completion and the gold field kept outside the model prompt.
            response_ids = data.batch["responses"]
            response_length = response_ids.shape[-1]
            if "attention_mask" in data.batch:
                response_attention = data.batch["attention_mask"][:, -response_length:].bool()
            else:
                response_attention = data.batch["response_mask"].bool()

            reward_models = data.non_tensor_batch.get("reward_model")
            if reward_models is None:
                raise RuntimeError("EARewardManager requires reward_model.ground_truth for post-hoc metrics")
            if isinstance(reward_models, Mapping):
                ground_truths = reward_models.get("ground_truth")
            else:
                ground_truths = [
                    item.get("ground_truth") if isinstance(item, Mapping) else None
                    for item in reward_models
                ]
            if ground_truths is None or len(ground_truths) != len(response_ids):
                raise RuntimeError("EARewardManager could not align ground truth with generated responses")

            valid_answers = []
            correct_answers = []
            for ground_truth, token_row, mask_row in zip(
                ground_truths, response_ids, response_attention, strict=True
            ):
                token_ids = token_row[mask_row].detach().cpu().tolist()
                completion = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                parsed_answers = re.findall(
                    r"(?:Final Answer|Expert Final Answer):\s*([A-D])\b",
                    completion,
                    flags=re.IGNORECASE,
                )
                answer = parsed_answers[-1].upper() if parsed_answers else ""
                if isinstance(ground_truth, Mapping):
                    ground_truth = ground_truth.get("answer", "")
                gold = str(ground_truth).strip().upper()
                valid_answers.append(int(bool(answer)))
                correct_answers.append(int(bool(answer) and answer == gold))

            # These original human-interaction counters do not apply to EA's
            # single-case protocol, so keep them explicitly zero.
            zeros = [0.0] * len(response_ids)
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    "ea_net_episode_reward": episode_scores,
                    "is_valid": valid_answers,
                    "is_correct": correct_answers,
                    "incorrect_format_rate": zeros,
                    "multiple_rate": zeros,
                    "repeated_rate": zeros,
                    "verify_timeout_rate": zeros,
                    "effective_rate": zeros,
                    "human_reject_rate": zeros,
                    "human_timeout_rate": zeros,
                    "assistant_error_rate": zeros,
                },
            }
        return reward_tensor
