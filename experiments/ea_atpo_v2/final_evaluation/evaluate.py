#!/usr/bin/env python3
"""Evaluate frozen V2 policies on the frozen case-ID test manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from . import ACTION_ORDER, ALL_POLICIES, DEFAULT_COSTS, LEARNED_POLICIES
from .metrics import (
    paired_bootstrap,
    paired_delta_rows,
    policy_case_components,
    risk_coverage_rows,
    summary_rows,
    validate_costs,
    validate_outcomes,
    validate_predictions,
    oracle_upper_bound_components,
    write_csv,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected an object")
            rows.append(value)
    return rows


def case_ids_from_manifest(manifest: dict[str, Any]) -> list[str]:
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("case manifest must contain a cases list")
    ids = [str(case.get("case_id", "")) for case in cases if isinstance(case, dict)]
    if len(ids) != len(cases) or not ids or any(not case_id for case_id in ids):
        raise ValueError("case manifest has malformed case records")
    if len(set(ids)) != len(ids):
        raise ValueError("case manifest has duplicate case_id values")
    return ids


def evaluate(case_manifest_path: Path, outcomes_path: Path, predictions_path: Path, output_dir: Path,
             *, costs: dict[str, float] | None = None, replicates: int = 10000,
             bootstrap_seed: int = 20261120) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output_dir}")
    registered_costs = validate_costs(costs)
    manifest = read_json(case_manifest_path)
    case_ids = case_ids_from_manifest(manifest)
    outcome_map = validate_outcomes(read_jsonl(outcomes_path), case_ids)
    prediction_map = validate_predictions(read_jsonl(predictions_path), case_ids)

    # Prediction rows are the only inputs to deployable selection. Gold-derived
    # action correctness is joined afterward, strictly for post-hoc metrics.
    deployable = policy_case_components(case_ids, outcome_map, prediction_map, registered_costs)
    metric_ci, paired_ci = paired_bootstrap(deployable, replicates=replicates, seed=bootstrap_seed)

    # This Oracle path consumes post-hoc correctness labels and produces only
    # an aggregate upper-bound row. It never enters the prediction map above.
    deployable["oracle_upper_bound"] = oracle_upper_bound_components(case_ids, outcome_map, registered_costs)
    oracle_metric_ci, oracle_paired_ci = paired_bootstrap(deployable, replicates=replicates, seed=bootstrap_seed)
    metric_ci["oracle_upper_bound"] = oracle_metric_ci["oracle_upper_bound"]
    paired_ci["oracle_upper_bound"] = oracle_paired_ci["oracle_upper_bound"]

    risk_rows = risk_coverage_rows(case_ids, outcome_map, prediction_map, registered_costs)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "metrics.csv", summary_rows(metric_ci, case_count=len(case_ids)))
    write_csv(output_dir / "paired_deltas_vs_always_answer.csv", paired_delta_rows(paired_ci, case_count=len(case_ids)))
    write_csv(output_dir / "risk_coverage.csv", risk_rows)
    result_manifest = {
        "schema_version": 1,
        "evaluation_id": manifest.get("manifest_id"),
        "case_manifest_sha256": sha256_file(case_manifest_path),
        "outcomes_sha256": sha256_file(outcomes_path),
        "predictions_sha256": sha256_file(predictions_path),
        "case_count": len(case_ids),
        "case_id_list_sha256": manifest.get("case_id_list_sha256"),
        "action_order": list(ACTION_ORDER),
        "evaluated_policies": list(ALL_POLICIES),
        "deployable_policy_predictions": list(LEARNED_POLICIES),
        "costs": registered_costs,
        "utility_recomputed_from_raw_correctness_and_call_counts": True,
        "oracle_is_posthoc_gold_upper_bound_only": True,
        "oracle_available_to_prediction_or_training_path": False,
        "bootstrap": {"unit": "case_id", "replicates": replicates, "seed": bootstrap_seed,
                      "confidence": 0.95, "paired_reference": "always_answer"},
        "risk_coverage": {"coverage": "non-DEFER fraction", "risk": "error rate among non-DEFER cases",
                          "risk_score_source": "prediction-only defer_score"},
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(result_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True,
                        help="post-hoc case/action correctness and raw call counts; keep local/private")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="prediction-only rows for Q-guided, Relative Advantage, Learned Router")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="must be a new directory; previous results are never overwritten")
    parser.add_argument("--ask-cost", type=float, default=DEFAULT_COSTS["ASK"])
    parser.add_argument("--defer-cost", type=float, default=DEFAULT_COSTS["DEFER"])
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261120)
    args = parser.parse_args()
    costs = {"ASK": args.ask_cost, "ANSWER": 0.0, "DEFER": args.defer_cost}
    result = evaluate(args.case_manifest, args.outcomes, args.predictions, args.output_dir,
                      costs=costs, replicates=args.bootstrap_replicates,
                      bootstrap_seed=args.bootstrap_seed)
    print(json.dumps({"status": "completed", "case_count": result["case_count"],
                      "output_dir": str(args.output_dir), "oracle_posthoc_only": True}, indent=2))


if __name__ == "__main__":
    main()
