#!/usr/bin/env python3
"""Freeze a new 400-case holdout from unseen ATPO-train rows, without Gold."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/expert_sft"
DATA = EXP / "data/reconstructed_compat/train.pilot.parquet"
ATTEMPT04 = ROOT / "experiments/real_case_pilot_20261103/audit/outputs/qwen32_router_generalization_20261118_attempt04"
PHASE2 = ROOT / "experiments/vi_ea_atpo/phase2"
OUT = EXP / "data/fresh_test_selection.json"
N = 400
SEED = 20261120
BASE_SYSTEM = (
    "You are answering a medical multiple-choice benchmark question. Use only the case and options "
    "in the user message. Select the single best option. Do not ask follow-up questions. "
    "Output exactly: Final Answer: <one uppercase option letter A-D>. Do not include an explanation."
)

import sys
sys.path.insert(0, str(ROOT / "ea_atpo_stage"))
from recipe.ea_atpo.protocol import Action, parse_action, visible_response


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def native(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def row_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    raw = native(row.get("prompt", (row.get("extra_info") or {}).get("prompt")))
    if not isinstance(raw, list):
        raise ValueError("Source case has no public message list")
    result = [{"role": "system", "content": BASE_SYSTEM}]
    for item in raw:
        role = str(item.get("role", ""))
        if role == "system":
            continue
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unexpected public role: {role}")
        result.append({"role": role, "content": str(item.get("content", ""))})
    if not any(m["role"] == "user" for m in result):
        raise ValueError("Source prompt lacks a public user case")
    return result


def identities(row: dict[str, Any]) -> tuple[str, str]:
    raw = native(row.get("prompt", (row.get("extra_info") or {}).get("prompt")))
    visible = []
    for item in raw:
        role = str(item.get("role", "user"))
        content = str(item.get("content", ""))
        if role == "system":
            continue
        if role == "assistant":
            try:
                action, payload = parse_action(content)
                if action is Action.ANSWER:
                    content = f"Final Answer: {payload.strip().upper()}"
                elif action is Action.ASK:
                    content = f"Question: {payload.strip()}"
                elif action is Action.CONSULT:
                    content = f"Consult Expert: {payload.strip()}"
                else:
                    content = "Defer to Expert"
            except ValueError:
                content = visible_response(content)
        visible.append({"role": "assistant" if role == "assistant" else "user", "content": content})
    case_payload = json.dumps(visible, ensure_ascii=False, sort_keys=True)
    messages = row_messages(row)
    prompt_payload = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return sha_text(case_payload), sha_text(prompt_payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=N)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Fresh test selection is already frozen: {args.output}")
    if sha_file(DATA) != "c586d8653460e8bef102f7c6231a1356dbfa88b5b35c3538219724efd4c1a5c5":
        raise RuntimeError("Source train Parquet is not the pinned row-identical historical dataset")

    cache = json.loads((ATTEMPT04 / "history_exclusion_cache.json").read_text(encoding="utf-8"))
    historical_hashes = set(cache.get("hashes", []))
    historical_indices = set(cache.get("train_indices", [])) | set(cache.get("test_indices", [])) | set(cache.get("ambiguous_indices", []))
    selection04 = json.loads((ATTEMPT04 / "case_selection.json").read_text(encoding="utf-8"))
    for split in ("train", "validation", "test"):
        for spec in selection04["rows"][split]:
            historical_indices.add(int(spec["source_row_index"]))
            historical_hashes.add(str(spec["case_hash"]))
            historical_hashes.add(str(spec["prompt_hash"]))
    # Also exclude the frozen Phase-2 ASK/CONSULT cohort, which was selected after Attempt04.
    phase2_csv = PHASE2 / "pilot_cases.csv"
    phase2_excluded = set()
    if phase2_csv.is_file():
        with phase2_csv.open(encoding="utf-8", newline="") as f:
            for record in csv.DictReader(f):
                if record.get("source_row_index", "").isdigit():
                    phase2_excluded.add(int(record["source_row_index"]))
                for key in ("case_hash", "prompt_hash", "public_state_sha256"):
                    if record.get(key):
                        historical_hashes.add(str(record[key]))
                # Phase-2 case_id values are already stable case hashes; exclude
                # them even if a CSV export omitted the corresponding hash column.
                candidate_id = str(record.get("case_id", ""))
                if re.fullmatch(r"[0-9a-fA-F]{64}", candidate_id):
                    historical_hashes.add(candidate_id.lower())
    historical_indices |= phase2_excluded

    # Selection needs only public prompts. Do not load Gold, explanations, or
    # hidden atomic facts from any row while freezing the fresh cohort.
    frame = pd.read_parquet(DATA, columns=["prompt"])
    sft_selection = json.loads((ATTEMPT04 / "case_selection.json").read_text(encoding="utf-8"))
    sft_indices = {int(x["source_row_index"]) for split in ("train", "validation") for x in sft_selection["rows"][split]}
    excluded_indices = historical_indices | sft_indices
    candidate_indices = list(range(len(frame)))
    random.Random(args.seed).shuffle(candidate_indices)
    selected = []
    selected_hashes: set[str] = set()
    skipped_index = skipped_hash = skipped_duplicate = 0
    for index in candidate_indices:
        if index in excluded_indices:
            skipped_index += 1
            continue
        row = frame.iloc[index].to_dict()
        case_hash, prompt_hash = identities(row)
        if case_hash in historical_hashes or prompt_hash in historical_hashes:
            skipped_hash += 1
            continue
        if case_hash in selected_hashes or prompt_hash in selected_hashes:
            skipped_duplicate += 1
            continue
        selected.append({
            "case_id": f"fresh-train-holdout-row-{index}",
            "source_row_index": int(index),
            "case_hash": case_hash,
            "prompt_hash": prompt_hash,
            "split": "test",
        })
        selected_hashes.update((case_hash, prompt_hash))
        if len(selected) == args.count:
            break
    if len(selected) != args.count:
        raise RuntimeError(f"Only found {len(selected)} eligible cases, expected {args.count}")
    for split in ("train", "validation"):
        sft_hashes = {x[field] for x in sft_selection["rows"][split] for field in ("case_hash", "prompt_hash")}
        if any(x[field] in sft_hashes for x in selected for field in ("case_hash", "prompt_hash")):
            raise RuntimeError(f"Fresh test overlaps frozen SFT {split} cases")
    manifest = {
        "status": "frozen_before_sft_checkpoint_selection",
        "seed": args.seed,
        "count": len(selected),
        "source_split": "ATPO train",
        "source_data_sha256": sha_file(DATA),
        "sampling_rule": "uniform seeded shuffle of source row indices after excluding historical identities, all Attempt04 train/validation/test cases, frozen Phase-2 pilot cases, and SFT train/validation cases",
        "historical_case_identities_excluded": True,
        "phase2_frozen_pilot_excluded": True,
        "sft_train_validation_excluded": True,
        "official_test_split_read": False,
        "gold_or_atomic_facts_read_for_selection": False,
        "gold_or_atomic_facts_stored": False,
        "excluded_index_count": len(excluded_indices),
        "excluded_hash_count": len(historical_hashes),
        "selection_diagnostics": {"skipped_index": skipped_index, "skipped_hash": skipped_hash, "skipped_duplicate": skipped_duplicate},
        "rows": selected,
        "case_ids_sha256": sha_text("\n".join(x["case_id"] for x in selected)),
        "case_hashes_sha256": sha_text("\n".join(x["case_hash"] for x in selected)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "count": len(selected), "seed": args.seed, "selection_sha256": sha_file(args.output), "excluded_index_count": len(excluded_indices), "excluded_hash_count": len(historical_hashes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
