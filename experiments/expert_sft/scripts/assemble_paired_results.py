#!/usr/bin/env python3
"""Pair Base and Expert predictions by immutable case identity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def unique_index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in result:
            raise ValueError(f"{name}: missing or duplicate case_id {case_id!r}")
        result[case_id] = row
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--expert", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--split", choices=("train", "validation", "test"), required=True)
    args = ap.parse_args()
    base_rows, expert_rows = read_jsonl(args.base), read_jsonl(args.expert)
    base, expert = unique_index(base_rows, "Base"), unique_index(expert_rows, "Expert")
    if set(base) != set(expert):
        missing_base = sorted(set(expert) - set(base))[:5]
        missing_expert = sorted(set(base) - set(expert))[:5]
        raise ValueError(f"Base/Expert case sets differ; missing Base={missing_base}, missing Expert={missing_expert}")
    paired = []
    for case_id in sorted(base):
        b, e = base[case_id], expert[case_id]
        for key in ("case_hash", "prompt_hash", "gold"):
            if str(b.get(key)) != str(e.get(key)):
                raise ValueError(f"{case_id}: Base/Expert {key} mismatch")
        for row, name in ((b, "Base"), (e, "Expert")):
            if not isinstance(row.get("correct"), bool):
                raise ValueError(f"{case_id}: {name} correctness missing")
        features = dict(b.get("public_features") or {})
        # Backward-compatible with Base evaluator outputs written flat.
        for key in (
            "log_public_chars", "log_public_words", "option_count", "mean_option_chars",
            "question_marks", "digit_count", "clinical_lexicon_count",
            "base_entropy_normalized", "base_probability_margin", "base_confidence",
        ):
            if key in b:
                features[key] = b[key]
        paired.append({
            "case_id": case_id,
            "split": args.split,
            "case_hash": b["case_hash"],
            "prompt_hash": b["prompt_hash"],
            "base_predicted": b.get("predicted"),
            "expert_predicted": e.get("predicted"),
            "base_valid": bool(b.get("valid")),
            "expert_valid": bool(e.get("valid")),
            "base_correct": bool(b["correct"]),
            "expert_correct": bool(e["correct"]),
            "gold": b["gold"],
            "relative_gain": int(bool(e["correct"])) - int(bool(b["correct"])),
            "helpful": bool(e["correct"]) and not bool(b["correct"]),
            "harmful": bool(b["correct"]) and not bool(e["correct"]),
            "base_option_probabilities": b.get("option_probabilities"),
            "base_entropy_normalized": b.get("base_entropy_normalized"),
            "base_probability_margin": b.get("base_probability_margin"),
            "base_confidence": b.get("base_confidence"),
            "base_prompt_tokens": b.get("prompt_tokens"),
            "base_completion_tokens": b.get("completion_tokens"),
            "base_latency_s": b.get("latency_s"),
            "expert_prompt_tokens": e.get("prompt_tokens"),
            "expert_completion_tokens": e.get("completion_tokens"),
            "expert_latency_s": e.get("latency_s"),
            "public_features": features,
            "base_raw_output_sha256": b.get("raw_output_sha256"),
            "expert_raw_output_sha256": e.get("raw_output_sha256"),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite paired results: {args.output}")
    with args.output.open("w", encoding="utf-8") as f:
        for row in paired:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"status": "paired", "split": args.split, "n": len(paired), "output": str(args.output), "sha256": digest, "test_labels_are_posthoc_only": args.split == "test"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
