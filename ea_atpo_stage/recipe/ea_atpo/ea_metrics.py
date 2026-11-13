"""Post-rollout EA metrics that bridge ATPO's legacy logging interface."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


_LEGACY_METRIC_KEYS = (
    "is_valid",
    "is_correct",
    "incorrect_format_rate",
    "multiple_rate",
    "repeated_rate",
    "verify_timeout_rate",
    "effective_rate",
    "human_reject_rate",
    "human_timeout_rate",
    "assistant_error_rate",
)


def attach_ea_metric_fields(data: Any) -> Any:
    """Copy per-episode evaluation values before ATPO discards statistics.

    ATPO's AgentLoopWorker consumes and removes ``statistics`` immediately
    after postprocessing. Its later reward/metric code expects several fields
    from the original interaction protocol. We preserve EA correctness and
    validity as ordinary batch metadata at this boundary. The remaining legacy
    counters are inapplicable to EA-ATPO and are explicitly zero.
    """
    statistics = data.non_tensor_batch.get("statistics")
    if statistics is None:
        raise RuntimeError("EA metric bridge requires AgentLoopOutput.statistics")

    episodes = []
    for item in statistics:
        if not isinstance(item, Mapping) or not isinstance(item.get("ea_episode"), Mapping):
            raise RuntimeError("EA metric bridge expected statistics['ea_episode']")
        episodes.append(item["ea_episode"])

    valid_answers = {"A", "B", "C", "D"}
    final_answers = [episode.get("final_answer") for episode in episodes]
    evaluations = [episode.get("evaluation", {}) for episode in episodes]
    traces = [episode.get("action_trace", []) for episode in episodes]
    invalid_action_rates = [
        sum(not bool(event.get("action_valid", False)) for event in trace) / len(trace)
        if trace
        else 0.0
        for trace in traces
    ]
    batch_fields = {
        "is_valid": np.asarray([int(answer in valid_answers) for answer in final_answers], dtype=np.int64),
        "is_correct": np.asarray(
            [int(bool(evaluation.get("correct", False))) for evaluation in evaluations], dtype=np.int64
        ),
        "incorrect_format_rate": np.asarray(invalid_action_rates, dtype=np.float32),
        "multiple_rate": np.zeros(len(episodes), dtype=np.float32),
        "repeated_rate": np.zeros(len(episodes), dtype=np.float32),
        "verify_timeout_rate": np.zeros(len(episodes), dtype=np.float32),
        "effective_rate": np.zeros(len(episodes), dtype=np.float32),
        "human_reject_rate": np.zeros(len(episodes), dtype=np.float32),
        "human_timeout_rate": np.zeros(len(episodes), dtype=np.float32),
        "assistant_error_rate": np.asarray(invalid_action_rates, dtype=np.float32),
    }
    data.non_tensor_batch.update(batch_fields)
    return data


def patch_agent_loop_worker(agent_loop_worker: Any) -> None:
    """Install the metric bridge on the ATPO Ray actor class at runtime."""
    metadata = getattr(agent_loop_worker, "__ray_metadata__", None)
    worker_class = getattr(metadata, "modified_class", None)
    if worker_class is None:
        raise RuntimeError("Could not locate ATPO AgentLoopWorker's Ray class")
    if getattr(worker_class, "_ea_metric_bridge_installed", False):
        return

    original_postprocess = worker_class._postprocess

    def postprocess_with_ea_metrics(self, inputs):
        data, repeat_time_list = original_postprocess(self, inputs)
        attach_ea_metric_fields(data)
        return data, repeat_time_list

    worker_class._postprocess = postprocess_with_ea_metrics
    worker_class._ea_metric_bridge_installed = True
