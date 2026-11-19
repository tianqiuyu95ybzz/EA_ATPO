#!/usr/bin/env python3
"""Build answer-only SFT examples from the frozen 1k/300 ATPO train/val cases.

Only selected train and validation rows are read. The historical 400-case
holdout is never opened. Atomic facts and answer explanations are excluded from
the model input; no Gold reasoning is generated.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/expert_sft"
CONFIG = EXPERIMENT / "configs/domain_sft.json"
DEFAULT_SELECTION = ROOT / "experiments/real_case_pilot_20261103/audit/outputs/qwen32_router_generalization_20261118_attempt04/case_selection.json"
BASE_SYSTEM = (
    "You are answering a medical multiple-choice benchmark question. Use only the case and options "
    "in the user message. Select the single best option. Do not ask follow-up questions. "
    "Output exactly: Final Answer: <one uppercase option letter A-D>. Do not include an explanation."
)
SFT_SYSTEM = (
    "You are a medical multiple-choice expert. Use only the visible case and answer options. "
    "Do not assume hidden patient facts or access an answer key. Output exactly one line: "
    "Expert Final Answer: <one uppercase option letter A-D>. Do not include an explanation."
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def native(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def mapping(value: Any) -> dict[str, Any]:
    value = native(value)
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            result = json.loads(value)
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def messages_for_row(row: dict[str, Any], system_prompt: str) -> list[dict[str, str]]:
    raw = native(row.get("prompt", mapping(row.get("extra_info")).get("prompt")))
    if not isinstance(raw, list):
        raise ValueError("ATPO row prompt is not a message list")
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("ATPO prompt contains a non-message value")
        role = str(item.get("role", ""))
        if role == "system":
            continue
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unexpected visible prompt role: {role!r}")
        messages.append({"role": role, "content": str(item.get("content", ""))})
    if not any(m["role"] == "user" for m in messages):
        raise ValueError("ATPO prompt has no user case content")
    return messages


def extract_gold(row: dict[str, Any]) -> str:
    raw = row.get("reward_model", {}).get("ground_truth")
    if raw is None:
        raw = mapping(row.get("extra_info")).get("ground_truth", row.get("ground_truth"))
    raw = native(raw)
    if isinstance(raw, dict):
        raw = raw.get("answer", raw.get("label", raw.get("correct_option")))
    match = re.match(r"^\s*([A-D])(?:\b|[.)/:])", str(raw or ""), re.I)
    if not match:
        raise ValueError("Gold answer is missing or is not a single A-D choice")
    return match.group(1).upper()


def load_public_state_hasher():
    path = ROOT / "experiments/real_case_pilot_20261103/audit/qwen32_defer_light_helpers.py"
    spec = importlib.util.spec_from_file_location("sft_case_identity_helper", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load frozen case identity helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.public_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--parquet", type=Path, default=EXPERIMENT / "data/reconstructed_compat/train.pilot.parquet")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT / "data")
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    expected_sha = config["data"]["expected_source_sha256"]
    actual_sha = sha256_file(args.parquet)
    if actual_sha != expected_sha:
        raise RuntimeError(f"Frozen ATPO source data hash mismatch: expected {expected_sha}, got {actual_sha}")
    frame = pd.read_parquet(args.parquet)
    hasher = load_public_state_hasher()
    split_specs = selection.get("rows", {})
    # Consume only train/validation entries; test labels and source rows are never opened.
    selected = {name: split_specs.get(name, []) for name in ("train", "validation")}
    if {name: len(rows) for name, rows in selected.items()} != {"train": 1000, "validation": 300}:
        raise RuntimeError("Frozen case manifest must contain exactly 1000 train and 300 validation cases")

    all_ids: set[str] = set()
    all_case_hashes: set[str] = set()
    all_prompt_hashes: set[str] = set()
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    label_counts: Counter[str] = Counter()
    for split, specs in selected.items():
        for spec in specs:
            idx = int(spec["source_row_index"])
            if idx < 0 or idx >= len(frame):
                raise IndexError(f"Frozen source row index {idx} out of bounds")
            row = frame.iloc[idx].to_dict()
            case_id = str(spec["case_id"])
            _, case_hash, _, _ = hasher(row, idx)
            base_messages = messages_for_row(row, BASE_SYSTEM)
            prompt_hash = sha256_text(json.dumps(base_messages, ensure_ascii=False, sort_keys=True))
            if case_id != f"train-row-{idx}":
                raise ValueError(f"Frozen case_id/source index mismatch for row {idx}")
            if case_hash != str(spec["case_hash"]) or prompt_hash != str(spec["prompt_hash"]):
                raise ValueError(f"Public case/prompt hash mismatch for {case_id}; refusing data build")
            if case_id in all_ids or case_hash in all_case_hashes or prompt_hash in all_prompt_hashes:
                raise ValueError(f"Duplicate or cross-split leakage detected for {case_id}")
            answer = extract_gold(row)
            label_counts[answer] += 1
            messages = messages_for_row(row, SFT_SYSTEM)
            messages.append({"role": "assistant", "content": f"Expert Final Answer: {answer}"})
            rows_by_split[split].append({
                "case_id": case_id,
                "source_row_index": idx,
                "split": split,
                "case_hash": case_hash,
                "prompt_hash": prompt_hash,
                "messages": messages,
            })
            all_ids.add(case_id)
            all_case_hashes.add(case_hash)
            all_prompt_hashes.add(prompt_hash)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split, filename in (("train", "train_sft.jsonl"), ("validation", "validation_sft.jsonl")):
        path = args.output_dir / filename
        with path.open("w", encoding="utf-8") as f:
            for row in rows_by_split[split]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        outputs[split] = {"path": str(path.relative_to(ROOT)), "count": len(rows_by_split[split]), "sha256": sha256_file(path)}

    data_manifest = {
        "status": "built",
        "source_repository": "https://github.com/Quark-Medical/ATPO",
        "source_commit": config["data"]["atpo_commit"],
        "source_parquet_sha256": actual_sha,
        "selection_manifest_sha256": sha256_file(args.selection),
        "selection_seed": selection.get("seed"),
        "counts": {name: len(rows) for name, rows in rows_by_split.items()},
        "output_files": outputs,
        "gold_answer_format_counts": dict(sorted(label_counts.items())),
        "case_id_overlap": 0,
        "case_hash_overlap": 0,
        "prompt_hash_overlap": 0,
        "hidden_facts_in_model_messages": False,
        "gold_reasoning_synthesized": False,
        "historical_test_rows_used": 0,
        "gold_reliability_scope": "Official ATPO labels were checked for presence, A-D format, frozen row/hash identity, and duplicate/cross-split leakage. No clinical re-adjudication was performed.",
    }
    manifest_path = args.output_dir / "sft_data_manifest.json"
    manifest_path.write_text(json.dumps(data_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = f"""# SFT Data Audit

- Source: official ATPO train data at pinned commit `{config['data']['atpo_commit']}`.
- Frozen cases: 1,000 train and 300 validation. No historical holdout source rows or labels were read for dataset construction; only train/validation entries were consumed.
- Rebuilt Parquet SHA-256 matched the frozen source hash: `{actual_sha}`.
- Case ID, public-state hash, and prompt hash matched the prior selection manifest for every selected row.
- Missing or malformed A-D labels: 0; the builder stops on any invalid label.
- Duplicate IDs/hashes, prompt duplicates, and train/validation overlap: 0.
- Hidden `atomic_facts`, answer explanations, and Gold reasoning are excluded from model input. Targets contain only the official answer letter in `Expert Final Answer: X` format.
- Gold reliability is limited to source/schema/integrity checks; labels were not re-adjudicated by clinicians.

Outputs: `train_sft.jsonl`, `validation_sft.jsonl`, and `sft_data_manifest.json`.
"""
    (args.output_dir / "SFT_DATA_AUDIT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"counts": data_manifest["counts"], "train_sha256": outputs["train"]["sha256"], "validation_sha256": outputs["validation"]["sha256"], "source_sha256": actual_sha}, ensure_ascii=False))


if __name__ == "__main__":
    main()
