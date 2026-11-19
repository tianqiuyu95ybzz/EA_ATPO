#!/usr/bin/env python3
"""Compare historical and current expert protocols without serializing answer keys."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
V2_DIR = HERE.parents[1]
PROJECT_DIR = V2_DIR.parents[1]
sys.path.insert(0, str(V2_DIR / "src"))
from v2_protocol import Action, PublicState, build_base_messages, build_defer_messages  # noqa: E402


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def score_agreement(rows: list[dict[str, Any]], cohort: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["state_id"])][str(row["action"])] = row
    if any(not {"ANSWER", "DEFER"}.issubset(actions) for actions in grouped.values()):
        raise ValueError(f"missing ANSWER/DEFER rows in {cohort}")
    outcomes = Counter()
    output_rows = []
    for state_id, actions in sorted(grouped.items()):
        base, expert = actions["ANSWER"], actions["DEFER"]
        same = base.get("final_answer") == expert.get("final_answer")
        bc = base.get("final_accuracy")
        ec = expert.get("final_accuracy")
        if bc == 1 and ec == 1:
            label = "both_correct"
        elif bc == 1 and ec == 0:
            label = "expert_regression"
        elif bc == 0 and ec == 1:
            label = "expert_correction"
        elif bc == 0 and ec == 0:
            label = "both_wrong"
        else:
            label = "unscored_or_invalid"
        outcomes[label] += 1
        output_rows.append({
            "cohort": cohort, "case_id": base["case_id"], "state_id": state_id,
            "base_prediction": base.get("final_answer"), "expert_prediction": expert.get("final_answer"),
            "exact_prediction_match": same, "base_correct": bc, "expert_correct": ec,
            "outcome": label,
        })
    n = len(grouped)
    summary = {
        "cohort": cohort, "states": n,
        "exact_prediction_matches": sum(bool(row["exact_prediction_match"]) for row in output_rows),
        "exact_prediction_match_rate": sum(bool(row["exact_prediction_match"]) for row in output_rows) / n if n else None,
        "prediction_disagreements": sum(not bool(row["exact_prediction_match"]) for row in output_rows),
        "both_correct": outcomes["both_correct"], "both_wrong": outcomes["both_wrong"],
        "expert_corrections": outcomes["expert_correction"], "expert_regressions": outcomes["expert_regression"],
        "unscored_or_invalid": outcomes["unscored_or_invalid"],
    }
    return summary, output_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-trajectories", type=Path, required=True)
    parser.add_argument("--new-first10-trajectories", type=Path, required=True)
    parser.add_argument("--new-50-trajectories", type=Path, required=True)
    parser.add_argument("--historical-run-manifest", type=Path, required=True)
    parser.add_argument("--historical-base-service-manifest", type=Path, required=True)
    parser.add_argument("--historical-expert-service-manifest", type=Path, required=True)
    parser.add_argument("--current-model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    historical = json.loads(args.historical_run_manifest.read_text(encoding="utf-8"))
    hist_base_service = json.loads(args.historical_base_service_manifest.read_text(encoding="utf-8"))
    hist_expert_service = json.loads(args.historical_expert_service_manifest.read_text(encoding="utf-8"))
    direct = load_module(PROJECT_DIR / "experiments/real_case_pilot_20261103/audit/run_direct_reliability_eval.py", "historical_direct_prompt")
    old_protocol = load_module(PROJECT_DIR / "ea_atpo_stage/recipe/ea_atpo/protocol.py", "historical_expert_protocol")
    model_root = args.current_model_root
    tokenizer_config = model_root / "Qwen3-8B/tokenizer_config.json"
    tokenizer_data = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    current_public = read_jsonl(args.new_first10_trajectories)[0]["public_state"]
    current_state = PublicState("hash-probe", "hash-probe", 0, tuple(current_public))
    old_state = old_protocol.PublicState("hash-probe", tuple(current_public))
    current_base_system = build_base_messages(current_state, Action.ANSWER)[0]["content"]
    current_expert_system = build_defer_messages(current_state)[0]["content"]
    historical_expert_system = old_protocol.build_expert_messages(old_state, old_protocol.Action.DEFER)[0]["content"]
    template = tokenizer_data.get("chat_template", "")
    current_base_config = json.loads((model_root / "Qwen3-8B/config.json").read_text(encoding="utf-8"))
    current_expert_config = json.loads((model_root / "Qwen3-32B-AWQ/config.json").read_text(encoding="utf-8"))

    old_rows = read_jsonl(args.old_trajectories)
    new_first10 = read_jsonl(args.new_first10_trajectories)
    new_50 = read_jsonl(args.new_50_trajectories)
    summaries = []
    prediction_rows = []
    for rows, name in ((old_rows, "previous_v2_10_case_run"),
                       (new_first10, "patient_review_v2_rerun_first10"),
                       (new_50, "patient_review_v2_50_case")):
        summary, records = score_agreement(rows, name)
        summaries.append(summary)
        prediction_rows.extend(records)

    historical_models = historical["models"]
    protocol = {
        "model_revisions_match": (
            historical_models["base"]["revision"] == "b968826d9c46dd6066d109eabc6255188de91218"
            and historical_models["expert"]["revision"] == "0499c3ac83fdef8810b907a23894ba91e95eddd8"
            and hist_base_service.get("revision") == "b968826d9c46dd6066d109eabc6255188de91218"
            and hist_expert_service.get("revision") == "0499c3ac83fdef8810b907a23894ba91e95eddd8"
        ),
        "historical_vllm_versions": [hist_base_service.get("framework_version"), hist_expert_service.get("framework_version")],
        "current_vllm_version": "0.33.0",
        "historical_and_current_tokenizer_template_sha256_current_pinned_assets": sha_text(template),
        "tokenizer_config_sha256_current_pinned_assets": hashlib.sha256(tokenizer_config.read_bytes()).hexdigest(),
        "historical_chat_template_hash_recorded_in_run_manifest": False,
        "thinking_disabled_both": not historical["generation"]["enable_thinking"],
        "temperature_top_p_match": historical["generation"]["temperature"] == 0.0 and historical["generation"]["top_p"] == 1.0,
        "historical_max_tokens": {"base": historical["generation"]["base_max_tokens"], "expert": historical["generation"]["expert_max_tokens"]},
        "v2_max_tokens": {"base": 24, "expert": 32},
        "seed_parameter_historical": "not sent by historical infer()",
        "seed_parameter_v2": "per-state-action rollout_seed is sent",
        "historical_structured_output_constraint": False,
        "v2_structured_output_regex": {"answer": "^Final Answer: [A-D]$", "ask": "^Question: [^\\r\\n]{8,320}$"},
        "historical_base_system_prompt_sha256": sha_text(direct.SYSTEM),
        "v2_base_answer_system_prompt_sha256": sha_text(current_base_system),
        "historical_defer_system_prompt_sha256": sha_text(old_protocol.DEFER_SYSTEM_PROMPT),
        "v2_defer_system_prompt_sha256": sha_text(current_expert_system),
        "historical_defer_output_prefix": "Expert Final Answer:",
        "v2_defer_output_prefix": "Final Answer:",
        "historical_base_parser": "one Final Answer A-D occurrence, or a bare letter",
        "historical_expert_parser": "strict Expert Final Answer A-D protocol validator",
        "v2_parser": "exact Final Answer: A-D; generation constrained with regex",
        "answer_scoring_concept": "parsed option letter equals answer key; parse failures do not count correct",
        "historical_cohort": "200 independent test cases from frozen historical DEFER evaluation",
        "v2_cohort": "50 train cases expanded to 146 public states; not the same cases or protocol",
        "historical_result_from_locked_report": {"base_accuracy": 0.46, "expert_accuracy": 0.555},
        "direct_performance_comparison_valid": False,
        "v2_current_base_config_quantization": current_base_config.get("quantization_config"),
        "v2_current_expert_quantization": current_expert_config.get("quantization_config"),
        "prediction_agreement": summaries,
        "interpretation": "Same pinned model revisions and core decoding temperature/top_p, but prompts, max token caps, seed handling, output prefixes, parser, constrained decoding, and cohort differ. Historical accuracy is not a controlled baseline for these V2 runs.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "expert_prediction_agreement.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("cohort", "case_id", "state_id", "base_prediction", "expert_prediction", "exact_prediction_match", "base_correct", "expert_correct", "outcome")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prediction_rows)
    (args.output_dir / "expert_protocol_compatibility.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prediction_agreement": summaries, "historical_v2_protocol_comparable": False,
                      "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
