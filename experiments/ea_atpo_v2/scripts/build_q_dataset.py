#!/usr/bin/env python3
"""Convert V2 rollout JSONL to paired Q-regression rows (no label imputation)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
V2_DIR = HERE.parents[1]
PROJECT_DIR = V2_DIR.parents[1]
sys.path.insert(0, str(V2_DIR / "src"))
from q_dataset import build_q_dataset  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=V2_DIR / "configs" / "phase1_counterfactual.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    split_path = PROJECT_DIR / config["data"]["case_split"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = set(map(str, split["train"]))
    rollouts = read_jsonl(args.trajectories)
    rows = build_q_dataset(rollouts, train_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    valid_targets = sum(sum(bool(value) for value in row["valid_mask"].values()) for row in rows)
    print(json.dumps({
        "q_dataset_states": len(rows),
        "valid_action_targets": valid_targets,
        "target_actions": ["ASK", "ANSWER", "DEFER"],
        "costs_already_included": True,
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
