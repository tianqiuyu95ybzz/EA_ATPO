#!/usr/bin/env python3
"""Merge Q, policy, and external baseline scores by exact state identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import read_jsonl


IDENTITY = {"case_id", "state_id"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite merged predictions: {args.output}")
    indexed = []
    for path in args.inputs:
        rows = {}
        for row in read_jsonl(path):
            key = (str(row.get("case_id", "")), str(row.get("state_id", "")))
            if not all(key) or key in rows:
                raise ValueError(f"{path}: missing or duplicate state key {key}")
            rows[key] = row
        indexed.append((path, rows))
    reference_keys = set(indexed[0][1])
    for path, rows in indexed[1:]:
        if set(rows) != reference_keys:
            raise ValueError(f"state coverage mismatch in {path}: missing={len(reference_keys-set(rows))}, extra={len(set(rows)-reference_keys)}")
    merged = []
    for key in sorted(reference_keys):
        target = {"case_id": key[0], "state_id": key[1]}
        for path, rows in indexed:
            for name, value in rows[key].items():
                if name in IDENTITY:
                    continue
                if name in target and target[name] != value:
                    raise ValueError(f"conflicting {name} for state {key} while merging {path}")
                target[name] = value
        merged.append(target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged), encoding="utf-8")
    print(json.dumps({"status": "completed", "sources": [str(path.resolve()) for path in args.inputs],
                      "states": len(merged), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
