#!/usr/bin/env python3
"""Recompute Attempt09 Critic/action diagnostics from its saved logs and traces."""
from __future__ import annotations

import collections
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability"
OUT = RUN / "paired_evaluation"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse_steps(path: Path):
    result = {}
    for raw in path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw)
        if " - static/q_value_variance_mean:" not in line:
            continue
        row = {}
        for segment in line.split(" - "):
            if ":" not in segment:
                continue
            key, value = segment.split(":", 1)
            key = key.rsplit(")", 1)[-1].strip()
            try:
                row[key] = float(value.strip())
            except ValueError:
                continue
        step = int(row.get("training/global_step", 0))
        if step:
            result[step] = row
    return result


def mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(values) / len(values) if values else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ppo_rows = {int(r["step"]): r for r in csv.DictReader((RUN / "ppo_step_metrics.csv").open())}
    steps = parse_steps(RUN / "logs/ea_atpo_v1_smoke_20261115T101426Z.log")
    critic_rows = []
    for step in sorted(steps):
        row = steps[step]
        ret_var = row.get("critic/return_var")
        residual_var = row.get("critic/return_diff_var")
        ev = row.get("critic/vf_explained_var")
        recomputed = 1.0 - residual_var / (ret_var + 1e-5) if ret_var is not None and residual_var is not None else None
        critic_rows.append({
            "step": step,
            "ppo_eligible_trajectories": int(float(ppo_rows[step]["ppo_eligible_trajectories"])),
            "vf_loss": row.get("critic/vf_loss"),
            "vf_explained_variance": ev,
            "vf_explained_variance_recomputed": recomputed,
            "return_variance": ret_var,
            "residual_variance": residual_var,
            "return_mean": row.get("critic/returns/mean"),
            "return_min": row.get("critic/returns/min"),
            "return_max": row.get("critic/returns/max"),
            "value_mean": row.get("critic/values/mean"),
            "value_min": row.get("critic/values/min"),
            "value_max": row.get("critic/values/max"),
            "advantage_mean": row.get("critic/advantages/mean"),
            "advantage_min": row.get("critic/advantages/min"),
            "advantage_max": row.get("critic/advantages/max"),
            "actor_grad_norm_preclip": row.get("actor/grad_norm"),
            "critic_grad_norm_preclip": row.get("critic/grad_norm"),
        })
    with (OUT / "critic_step_diagnostics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(critic_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(critic_rows)

    episodes = [json.loads(x) for x in (RUN / "traces/train.jsonl").read_text().splitlines() if x.strip()]
    action_rows = []
    all_events = collections.defaultdict(list)
    order_counts = collections.Counter()
    per_step_counts = collections.defaultdict(collections.Counter)
    predecessor_counts = collections.Counter()
    initial_counts = collections.Counter()
    reward_check_failures = []
    eligibility = collections.Counter()
    for i, episode in enumerate(episodes):
        step = i // 8 + 1
        eligible = bool(episode.get("evaluation", {}).get("training_eligible"))
        eligibility["eligible" if eligible else "ineligible"] += 1
        if episode.get("action_trace"):
            initial_counts[episode["action_trace"][0].get("action") or "INVALID"] += 1
        expected = float(episode.get("evaluation", {}).get("terminal_correctness_reward", 0.0)) - float(episode.get("evaluation", {}).get("resource_cost_total", 0.0))
        actual = float(episode.get("evaluation", {}).get("net_episode_reward", 0.0))
        summed_edges = sum(float(event.get("net_action_reward", 0.0) or 0.0) for event in episode.get("action_trace", []))
        if abs(expected - actual) > 1e-6 or abs(actual - summed_edges) > 1e-6:
            reward_check_failures.append({"row": i, "expected": expected, "actual": actual, "summed_edges": summed_edges})
        for j, event in enumerate(episode.get("action_trace", [])):
            action = event.get("action") or "INVALID"
            prior_action = episode["action_trace"][j - 1].get("action") if j > 0 else "START"
            pos = j + 1
            order_counts[(pos, action)] += 1
            predecessor_counts[(prior_action, action)] += 1
            per_step_counts[step][action] += 1
            all_events[action].append({
                "valid": bool(event.get("action_valid", False)),
                "return": event.get("return"),
                "advantage": event.get("advantage"),
                "action_cost": event.get("action_cost"),
                "net_action_reward": event.get("net_action_reward"),
                "episode_correct": bool(episode.get("evaluation", {}).get("correct")),
                "episode_reward": actual,
                "joint_response_probability": event.get("joint_response_probability"),
                "joint_response_log_probability": event.get("joint_response_log_probability"),
            })

    for action, events in sorted(all_events.items()):
        action_rows.append({
            "action": action,
            "events": len(events),
            "valid_events": sum(x["valid"] for x in events),
            "valid_rate": mean([int(x["valid"]) for x in events]),
            "mean_state_return": mean([x["return"] for x in events]),
            "mean_advantage": mean([x["advantage"] for x in events]),
            "mean_action_cost": mean([x["action_cost"] for x in events]),
            "mean_edge_reward": mean([x["net_action_reward"] for x in events]),
            "mean_final_episode_reward_given_action": mean([x["episode_reward"] for x in events]),
            "final_episode_accuracy_given_action": mean([int(x["episode_correct"]) for x in events]),
            "mean_selected_full_response_probability": mean([x["joint_response_probability"] for x in events]),
            "mean_selected_full_response_log_probability": mean([x["joint_response_log_probability"] for x in events]),
        })
    with (OUT / "action_distribution.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(action_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(action_rows)

    action_position = [
        {"actor_decision_position": pos, "action": action, "count": count,
         "decision_states_at_position": len(episodes) if pos == 1 else sum(len(ep.get("action_trace", [])) >= pos for ep in episodes),
         "predecessor_action_counts_json": json.dumps({prev: n for (prev, curr), n in sorted(predecessor_counts.items()) if curr == action}, sort_keys=True)}
        for (pos, action), count in sorted(order_counts.items())
    ]
    with (OUT / "action_position_counts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(action_position[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(action_position)

    # Sequence and action-rate summaries intentionally contain no case text.
    defer_summary = []
    for i, episode in enumerate(episodes):
        for pos, event in enumerate(episode.get("action_trace", [])):
            if event.get("action") == "DEFER":
                defer_summary.append({
                    "training_step": i // 8 + 1,
                    "action_position": pos + 1,
                    "previous_actions": [x.get("action") for x in episode["action_trace"][:pos]],
                    "case_id_sha256": episode.get("case_id_sha256"),
                    "correct": bool(episode.get("evaluation", {}).get("correct")),
                    "net_episode_reward": episode.get("evaluation", {}).get("net_episode_reward"),
                    "action_cost": event.get("action_cost"),
                    "net_action_reward": event.get("net_action_reward"),
                    "return": event.get("return"),
                    "advantage": event.get("advantage"),
                    "full_response_probability": event.get("joint_response_probability"),
                })

    report = {
        "training_episode_count": len(episodes),
        "ppo_eligible_episode_count": eligibility["eligible"],
        "ineligible_episode_count": eligibility["ineligible"],
        "episode_accuracy": sum(bool(x.get("evaluation", {}).get("correct")) for x in episodes) / max(len(episodes), 1),
        "initial_action_counts": dict(initial_counts),
        "action_counts_by_step": {str(k): dict(v) for k, v in sorted(per_step_counts.items())},
        "action_position_counts": {f"{pos}_{a}": n for (pos, a), n in sorted(order_counts.items())},
        "predecessor_action_counts": {f"{previous}_to_{action}": n for (previous, action), n in sorted(predecessor_counts.items())},
        "defer_events": defer_summary,
        "reward_cost_identity_failures": reward_check_failures,
        "critic": {
            "steps": len(critic_rows),
            "all_explained_variance_negative": all(float(x["vf_explained_variance"]) < 0 for x in critic_rows),
            "mean_explained_variance": mean([x["vf_explained_variance"] for x in critic_rows]),
            "final_explained_variance": critic_rows[-1]["vf_explained_variance"],
            "min_explained_variance": min(x["vf_explained_variance"] for x in critic_rows),
            "max_explained_variance": max(x["vf_explained_variance"] for x in critic_rows),
            "mean_vf_loss": mean([x["vf_loss"] for x in critic_rows]),
            "min_return_variance": min(x["return_variance"] for x in critic_rows),
            "max_residual_variance": max(x["residual_variance"] for x in critic_rows),
            "explained_variance_formula_max_abs_error": max(abs(x["vf_explained_variance"] - x["vf_explained_variance_recomputed"]) for x in critic_rows),
        },
        "privacy": "Hashed case IDs and numerical/action metadata only; no public text, hidden facts, gold, patient response, consult question, or expert advice.",
    }
    (OUT / "training_diagnostic_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
