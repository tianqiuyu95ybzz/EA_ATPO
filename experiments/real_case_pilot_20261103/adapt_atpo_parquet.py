#!/usr/bin/env python3
"""Validate and normalize official ATPO Parquet rows without altering source data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


def as_native(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def as_mapping(value: Any) -> dict[str, Any]:
    value = as_native(value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def is_present(value: Any) -> bool:
    value = as_native(value)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def prompt_text(value: Any) -> str:
    value = as_native(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = []
        for message in value:
            message = as_mapping(message)
            if message:
                parts.append(str(message.get("content", "")))
            else:
                parts.append(str(message))
        return "\n".join(parts)
    return str(value or "")


def answer_letter(value: Any) -> str | None:
    value = as_native(value)
    if isinstance(value, Mapping):
        value = value.get("answer", value.get("label", value.get("correct_option")))
    match = re.match(r"^\s*([A-D])(?:\b|[.)/:])", str(value or ""), re.IGNORECASE)
    return match.group(1).upper() if match else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapt_split(input_path: Path, output_path: Path, split: str) -> dict[str, Any]:
    frame = pd.read_parquet(input_path)
    input_columns = [str(column) for column in frame.columns]
    missing_prompt = missing_gold = missing_facts = invalid_gold = 0
    option_counts: Counter[int] = Counter()
    extra_info_keys: Counter[str] = Counter()
    fact_types: Counter[str] = Counter()
    output_rows = []

    for row_index, raw in enumerate(frame.to_dict(orient="records")):
        row = dict(raw)
        extra = as_mapping(row.get("extra_info"))
        reward = as_mapping(row.get("reward_model"))
        prompt = row.get("prompt", extra.get("prompt"))
        gold = reward.get("ground_truth", extra.get("ground_truth", row.get("ground_truth")))
        facts = extra.get("atomic_facts", row.get("atomic_facts"))
        if not is_present(prompt):
            missing_prompt += 1
        if not is_present(gold):
            missing_gold += 1
        if answer_letter(gold) is None:
            invalid_gold += 1
        if not is_present(facts):
            missing_facts += 1
        else:
            fact_types[type(facts).__name__] += 1
        text = prompt_text(prompt)
        # Upstream ATPO stores the four answer options inside a Python-style
        # dictionary (``Options: {'A': ..., 'B': ...}``) on one line. Also
        # accept conventional one-option-per-line prompts such as ``A. ...``.
        option_labels = set(
            re.findall(
                r"(?m)(?:^|[\s,{])['\"]?([A-D])['\"]?(?:['\"]\s*:\s*|\s*[.)：:]\s*)",
                text,
            )
        )
        option_counts[len(option_labels)] += 1
        extra_info_keys.update(str(key) for key in extra)

        # The adapter writes a separate pilot-ready copy; source Parquet/JSONL
        # files and values are retained. No missing clinical facts are created.
        row["prompt"] = as_native(prompt)
        row["reward_model"] = {**reward, "ground_truth": as_native(gold)}
        row["extra_info"] = {**extra, "prompt": as_native(prompt)}
        if is_present(facts):
            row["extra_info"]["atomic_facts"] = as_native(facts)
        output_rows.append(row)

    if split == "train" and (missing_prompt or missing_gold or missing_facts or invalid_gold):
        # Do not silently sample around absent patient facts or labels.
        raise ValueError(
            f"train data failed required-field audit: missing_prompt={missing_prompt}, "
            f"missing_gold={missing_gold}, missing_atomic_facts={missing_facts}, invalid_gold={invalid_gold}"
        )
    if split == "test" and (missing_prompt or missing_gold or invalid_gold):
        raise ValueError(
            f"test data failed required-field audit: missing_prompt={missing_prompt}, "
            f"missing_gold={missing_gold}, invalid_gold={invalid_gold}"
        )

    adapted = pd.DataFrame(output_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adapted.to_parquet(output_path, index=False)
    return {
        "split": split,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "rows": len(frame),
        "input_columns": input_columns,
        "output_columns": [str(column) for column in adapted.columns],
        "missing_prompt": missing_prompt,
        "missing_ground_truth": missing_gold,
        "invalid_ground_truth_format": invalid_gold,
        "missing_atomic_facts": missing_facts,
        "atomic_facts_types": dict(fact_types),
        "prompt_option_label_count_distribution": dict(option_counts),
        "extra_info_keys": dict(extra_info_keys),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = {
        "adapter": "adapt_atpo_parquet.py",
        "train": adapt_split(args.train, args.output_dir / "train.pilot.parquet", "train"),
        "test": adapt_split(args.test, args.output_dir / "test.pilot.parquet", "test"),
    }
    manifest_path = args.output_dir / "adapter_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
