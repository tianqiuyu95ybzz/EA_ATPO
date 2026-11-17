#!/usr/bin/env python3
"""Fail closed before the 200-case run if the 20-case protocol smoke is bad."""
from __future__ import annotations
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent / "outputs/qwen32_defer_validation_20261119_attempt01/smoke_20"

def read(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

base = read(RUN / "base_answer_trajectories.jsonl")
expert = read(RUN / "qwen32_defer_trajectories.jsonl")
paired = read(RUN / "paired_protocol_smoke.jsonl")
summary = {
    "n_base": len(base), "n_expert": len(expert), "n_paired": len(paired),
    "base_valid": sum(bool(r.get("valid")) for r in base),
    "expert_valid": sum(bool(r.get("valid")) for r in expert),
    "base_uncertainty_valid": sum(bool(r.get("uncertainty_valid")) for r in base),
    "raw_base_saved": sum(isinstance(r.get("raw_response"), str) for r in base),
    "raw_expert_saved": sum(isinstance(r.get("raw_response"), str) for r in expert),
    "base_finish_reasons": sorted({str(r.get("finish_reason")) for r in base}),
    "expert_finish_reasons": sorted({str(r.get("finish_reason")) for r in expert}),
}
summary["pass"] = (
    summary["n_base"] == summary["n_expert"] == summary["n_paired"] == 20
    and summary["base_valid"] == 20 and summary["expert_valid"] == 20
    and summary["base_uncertainty_valid"] >= 16
    and summary["raw_base_saved"] == summary["raw_expert_saved"] == 20
)
(RUN / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
if not summary["pass"]:
    raise SystemExit("Protocol smoke did not pass; stopped before the 200-case experiment")
