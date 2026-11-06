#!/usr/bin/env python3
"""Exercise all three LLM roles on one authentic ATPO training case."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from recipe.ea_atpo.collect_counterfactuals import ChatAPI, rollout_action
from recipe.ea_atpo.protocol import Action, PublicState


def as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] if value is not None else []


async def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frame = pd.read_parquet(args.data)
    if frame.empty:
        raise ValueError("training Parquet contains no cases")
    row = frame.iloc[0].to_dict()
    extra = row["extra_info"]
    if hasattr(extra, "tolist"):
        extra = extra.tolist()
    if not isinstance(extra, dict):
        raise ValueError("adapted training row lacks mapping extra_info")
    facts = as_list(extra.get("atomic_facts"))
    facts = [str(value) for value in facts if str(value).strip()]
    if not facts:
        raise ValueError("the authentic training row contains no atomic_facts")
    prompt = as_list(extra.get("prompt", row.get("prompt")))
    state = PublicState(
        f"service-smoke-row-0:item-{extra.get('index', row.get('index', 0))}",
        tuple({"role": str(m["role"]), "content": str(m["content"])} for m in prompt),
    )
    ground_truth = row.get("reward_model", {}).get("ground_truth", extra.get("ground_truth"))

    clients = {}
    for name in ("base", "patient", "expert"):
        settings = config["endpoints"][name]
        clients[name] = ChatAPI(
            settings["base_url"], settings["model"],
            timeout_s=float(config.get("timeout_s", 180)),
            enable_thinking=settings.get("enable_thinking"),
        )
    try:
        actions = (Action.ANSWER, Action.ASK, Action.CONSULT, Action.DEFER)
        outcomes = await asyncio.gather(*(
            rollout_action(
                action, state, facts, ground_truth,
                clients["base"], clients["patient"], clients["expert"],
                temperature=0.0,
                max_turns=int(config.get("max_assistant_turns", 10)),
                max_consults=int(config.get("max_consults_per_rollout", 2)),
            ) for action in actions
        ))
    finally:
        await asyncio.gather(*(client.close() for client in clients.values()), return_exceptions=True)

    records = []
    for action, outcome in zip(actions, outcomes):
        trace = outcome.get("trace", [])
        consult_questions = [event.get("consult_question") for event in trace if event.get("consult_question")]
        expert_advice = [event.get("response") for event in trace if event.get("actor") == "expert" and event.get("action") == "CONSULT"]
        patient_responses = [event.get("response") for event in trace if event.get("actor") == "patient"]
        records.append({
            "record_type": "role_smoke_trajectory",
            "case_id": state.sample_id,
            "state_id": state.state_id,
            "public_state": state.messages,
            "action": action.value,
            "consult_question": consult_questions,
            "expert_advice": expert_advice,
            "patient_response": patient_responses,
            "final_answer": outcome.get("answer"),
            "gold": outcome.get("gold"),
            "correctness": outcome.get("outcome"),
            "valid": outcome.get("valid"),
            "failure_reason": outcome.get("error"),
            "action_cost": {
                "expert_tokens": outcome.get("expert_prompt_tokens", 0) + outcome.get("expert_completion_tokens", 0),
                "total_tokens": outcome.get("total_prompt_tokens", 0) + outcome.get("total_completion_tokens", 0),
                "latency_seconds": outcome.get("total_latency_s", 0.0),
                "patient_turns": outcome.get("patient_turns", 0),
            },
            "trace": trace,
        })

    required_checks = {
        "base_answer": next(r for r in records if r["action"] == "ANSWER")["valid"],
        "patient_response": bool(next(r for r in records if r["action"] == "ASK")["patient_response"]),
        "consult_question": bool(next(r for r in records if r["action"] == "CONSULT")["consult_question"]),
        "expert_advice": bool(next(r for r in records if r["action"] == "CONSULT")["expert_advice"]),
        "defer_final": next(r for r in records if r["action"] == "DEFER")["valid"],
        "all_actions_valid": all(r["valid"] for r in records),
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "passed" if all(required_checks.values()) else "failed",
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "source_row_index": 0,
        "state_id": state.state_id,
        "model": config.get("model", {}),
        "checks": required_checks,
        "trajectory_count": len(records),
        "real_doctor_participated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("pilot_config.yaml"))
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "assets/train.pilot.parquet")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs/service_role_smoke.jsonl")
    args = parser.parse_args()
    manifest = asyncio.run(run(args))
    if manifest["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
