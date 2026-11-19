#!/usr/bin/env python3
"""Run a real, bounded Patient Unknown-review smoke on previously frozen ASK questions."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
V2_DIR = HERE.parents[1]
PROJECT_DIR = V2_DIR.parents[1]
sys.path.insert(0, str(V2_DIR / "src"))
sys.path.insert(0, str(HERE.parent))
from run_counterfactual_pilot import (  # noqa: E402
    APIError, chat_completion, case_id_for_source, hidden_facts_for_row,
    read_jsonl, read_source_rows,
)
from v2_protocol import build_patient_messages, build_patient_review_messages, patient_reply_is_unknown  # noqa: E402


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--state-ids-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected_ids = json.loads(args.state_ids_file.read_text(encoding="utf-8"))
    if not isinstance(selected_ids, list) or not 3 <= len(selected_ids) <= 5 or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("state id list must contain 3-5 unique state IDs")
    split = json.loads((PROJECT_DIR / config["data"]["case_split"]).read_text(encoding="utf-8"))
    train_ids = set(map(str, split["train"]))
    trajectories = read_jsonl(args.trajectories)
    asks = {str(row["state_id"]): row for row in trajectories if row.get("action") == "ASK"}
    if not set(map(str, selected_ids)).issubset(asks):
        raise ValueError("requested state IDs are missing from the frozen ASK trajectory file")
    selected = [asks[state_id] for state_id in selected_ids]
    source_indices = {int(row["source_row_index"]) for row in selected}
    source_rows = read_source_rows(PROJECT_DIR / config["data"]["source_jsonl"], source_indices)
    records = []
    counts = Counter()
    for ask in selected:
        source_index = int(ask["source_row_index"])
        source = source_rows[source_index]
        case_id = str(ask["case_id"])
        if case_id not in train_ids or case_id_for_source(source_index, source) != case_id:
            raise ValueError("smoke includes a case outside the frozen train split or source mapping mismatch")
        facts = hidden_facts_for_row(source)
        question = str(ask["ask_question"])
        seed = int(ask["rollout_seed"])
        first_error = None
        review_error = None
        review_error_elapsed = 0.0
        try:
            first = chat_completion(
                "patient", config["roles"]["patient"], build_patient_messages(facts, question),
                temperature=float(config["inference"]["patient_temperature"]),
                top_p=float(config["inference"]["top_p"]),
                max_tokens=int(config["inference"]["patient_max_tokens"]), seed=seed,
                timeout_seconds=int(config["inference"]["timeout_seconds"]),
                retries=int(config["inference"]["api_retries"]),
            )
            first_reply = first["content"].strip()
            first_unknown = patient_reply_is_unknown(first_reply)
            retry = None
            if first_unknown:
                try:
                    retry = chat_completion(
                        "patient", config["roles"]["patient"],
                        build_patient_review_messages(facts, question, first_reply),
                        temperature=float(config["inference"]["patient_temperature"]),
                        top_p=float(config["inference"]["top_p"]),
                        max_tokens=int(config["inference"]["patient_max_tokens"]), seed=seed,
                        timeout_seconds=int(config["inference"]["timeout_seconds"]),
                        retries=int(config["inference"]["api_retries"]),
                    )
                except APIError as exc:
                    review_error = f"{exc.role}_api_failure"
                    review_error_elapsed = exc.elapsed_seconds
            review_reply = retry["content"].strip() if retry else None
            review_accepted = bool(review_reply and not patient_reply_is_unknown(review_reply))
            final_reply = review_reply if review_accepted else first_reply
            counts["first_unknown"] += int(first_unknown)
            counts["review_accepted"] += int(review_accepted)
            counts["final_unknown"] += int(patient_reply_is_unknown(final_reply))
            prompt_tokens = int(first["prompt_tokens"]) + (int(retry["prompt_tokens"]) if retry else 0)
            completion_tokens = int(first["completion_tokens"]) + (int(retry["completion_tokens"]) if retry else 0)
            latency = (float(first["latency_seconds"])
                       + (float(retry["latency_seconds"]) if retry else review_error_elapsed))
            records.append({
                "case_id": case_id, "state_id": ask["state_id"],
                "public_state_digest": ask["public_state_digest"], "question": question,
                "rollout_seed": seed, "patient_model": config["roles"]["patient"],
                "first_reply": first_reply, "first_finish_reason": first["finish_reason"],
                "first_unknown": first_unknown,
                "review_reply": review_reply,
                "review_finish_reason": retry["finish_reason"] if retry else None,
                "review_accepted": review_accepted, "review_error": review_error,
                "selected_final_reply": final_reply,
                "final_unknown": patient_reply_is_unknown(final_reply),
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens, "latency_seconds": latency,
                "patient_fact_text_serialized": False, "gold_serialized": False,
            })
        except APIError as exc:
            first_error = f"{exc.role}_api_failure"
            records.append({"case_id": case_id, "state_id": ask["state_id"],
                            "public_state_digest": ask["public_state_digest"], "question": question,
                            "rollout_seed": seed, "failure_reason": first_error,
                            "patient_fact_text_serialized": False, "gold_serialized": False})
            counts["failed"] += 1
    summary = {
        "experiment": "real Patient Unknown-review targeted smoke",
        "selected_states": len(selected_ids), "successful_states": len(records) - counts["failed"],
        "counts": dict(counts),
        "review_prompt_version": config["roles"]["patient"].get("role_prompt_version"),
        "gold_used_as_input": False, "gold_serialized": False,
        "hidden_fact_text_serialized": False,
        "all_cases_train_only": True,
        "output_path": str(args.output.resolve()),
        "models": {"patient": config["roles"]["patient"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = {"summary": summary, "states": records}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
