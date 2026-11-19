#!/usr/bin/env python3
"""Fit a two-layer masked Q regressor from case-disjoint V2 Q-dataset rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .calibration import ActionValueCalibrator
from .data import load_q_dataset, q_sampling_summary, require_minimum_q_samples, require_single_continuation_policy, validate_official_train_cases, validate_q_audit_report
from .losses import masked_regression_loss
from .models import ConstantMeanQRegressor, QRegressor, RidgeQRegressor
from .trainer_utils import atomic_torch_save, optimize, prepare_feature_splits, seed_everything, serialize_json, split_train_validation


V2_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = V2_DIR.parents[1]
DEFAULT_CASE_SPLIT = PROJECT_ROOT / "experiments/ea_atpo_value_router_20261105/artifacts/case_split.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-dataset", type=Path, required=True, help="Q JSONL from the official train cases only")
    parser.add_argument("--real-data-confirmed", action="store_true", help="confirm leakage-checked real V2 counterfactual labels")
    parser.add_argument("--smoke-only", action="store_true", help="run an engineering smoke; output is not a generalization result")
    parser.add_argument("--case-split", type=Path, default=DEFAULT_CASE_SPLIT)
    parser.add_argument("--audit-report", type=Path, help="passed read-only report for this exact Q dataset; required for formal fitting")
    parser.add_argument("--public-features", type=Path, help="versioned frozen-Base embedding cache; required for formal fitting")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("constant_mean", "ridge", "mlp"), default="ridge")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--max-features", type=int, default=3000)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--minimum-k", type=int, default=3, help="formal minimum valid rollout count per observed state/action; smoke ignores this gate")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--loss", choices=("huber", "mse"), default="huber")
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    if not args.real_data_confirmed:
        raise ValueError("formal Q fitting requires --real-data-confirmed after auditing raw V2 trajectories")
    if not args.smoke_only and args.public_features is None:
        raise ValueError("formal Q fitting requires a versioned frozen-Base public embedding cache; TF-IDF is limited to smoke/ablation")
    if not args.smoke_only and args.audit_report is None:
        raise ValueError("formal Q fitting requires --audit-report for the exact Q dataset")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    seed_everything(args.seed)
    rows = load_q_dataset(args.q_dataset)
    if any(row.raw_metadata.get("synthetic_data", False) for row in rows):
        raise ValueError("synthetic Q rows are test fixtures only and cannot be used by the training CLI")
    source_case_count = len({row.case_id for row in rows})
    if args.smoke_only and source_case_count > 10:
        raise ValueError("--smoke-only accepts at most the ten-case engineering Pilot")
    split_audit = validate_official_train_cases(rows, args.case_split)
    audit_summary = validate_q_audit_report(args.audit_report, args.q_dataset) if args.audit_report else None
    continuation_policy_id = require_single_continuation_policy(rows)
    sampling_summary = q_sampling_summary(rows)
    if not args.smoke_only:
        official_train = set(map(str, json.loads(args.case_split.read_text(encoding="utf-8"))["train"]))
        observed_train = {row.case_id for row in rows}
        if observed_train != official_train:
            raise ValueError(
                "formal Q fitting requires the complete official train case pool; "
                f"observed={len(observed_train)}, expected={len(official_train)}, missing={len(official_train-observed_train)}"
            )
        require_minimum_q_samples(rows, args.minimum_k)
    train_rows, val_rows, train_ids, val_ids = split_train_validation(
        rows, validation_fraction=args.validation_fraction, seed=args.seed
    )
    if not all(row.costs_included_in_targets for row in rows):
        raise ValueError("V2 Q regression expects net returns with action costs already included")
    x_train, x_val, encoder, scaler, feature_source, feature_metadata = prepare_feature_splits(
        train_rows, val_rows, args.public_features, args.max_features
    )
    model_type = {"constant_mean": "q_constant_mean", "ridge": "q_ridge", "mlp": "q_regressor"}[args.model]
    if args.model == "constant_mean":
        model = ConstantMeanQRegressor(x_train.shape[1])
        model.fit_closed_form(train_rows)
        validation_loss_kind = "mse"
        fit = {"best_epoch": 0, "history": []}
    elif args.model == "ridge":
        model = RidgeQRegressor(x_train.shape[1])
        model.fit_closed_form(x_train, train_rows, alpha=args.ridge_alpha)
        validation_loss_kind = "mse"
        fit = {"best_epoch": 0, "history": []}
    else:
        model = QRegressor(x_train.shape[1], args.hidden_dim)
        validation_loss_kind = args.loss

    def loss_fn(prediction, target, mask, sample_count):
        return masked_regression_loss(prediction, target, mask, loss=args.loss, sample_count=sample_count)

    if args.model == "mlp":
        fit = optimize(model, x_train, train_rows, x_val, val_rows,
                       epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                       learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                       loss_fn=loss_fn, seed=args.seed)
    with torch.no_grad():
        val_prediction = model(torch.from_numpy(x_val)).cpu().numpy().astype(np.float64)
    val_target = np.stack([row.q_target for row in val_rows])
    val_mask = np.stack([row.valid_mask for row in val_rows])
    val_count = np.stack([row.sample_count for row in val_rows])
    val_loss = masked_regression_loss(
        torch.from_numpy(val_prediction.astype(np.float32)),
        torch.from_numpy(val_target.astype(np.float32)),
        torch.from_numpy(val_mask),
        loss=validation_loss_kind,
        sample_count=torch.from_numpy(val_count.astype(np.float32)),
    )
    fit["best_validation_loss"] = float(val_loss.item())
    reference_cost = np.stack([row.action_cost for row in val_rows])
    calibrator = ActionValueCalibrator().fit(val_prediction, val_target, val_mask, reference_cost)
    output = args.output_dir
    checkpoint_path = output / "q_regressor.pt"
    payload = {
        "format": "ea_atpo_v2_training_checkpoint_v1",
        "model_type": model_type,
        "action_order": ["ASK", "ANSWER", "DEFER"],
        "input_dim": int(x_train.shape[1]),
        "hidden_dim": int(args.hidden_dim) if args.model == "mlp" else None,
        "ridge_alpha": float(args.ridge_alpha) if args.model == "ridge" else None,
        "model_state": model.state_dict(),
        "feature_source": feature_source,
        "external_feature_metadata": feature_metadata,
        "public_encoder": encoder.to_json() if encoder else None,
        "feature_standardizer": scaler.to_json(),
        "action_value_calibrator": calibrator.to_json(),
        "reference_action_costs": {"ASK": 0.05, "ANSWER": 0.0, "DEFER": 0.60},
        "costs_included_in_q_target": True,
        "validation_loss_kind": validation_loss_kind,
        "training": {
            "best_epoch": fit["best_epoch"], "best_validation_loss": fit["best_validation_loss"],
            "seed": args.seed, "epochs_requested": args.epochs,
            "train_case_ids": sorted(train_ids), "validation_case_ids": sorted(val_ids),
            "source_q_dataset": str(args.q_dataset.resolve()),
            "source_q_row_count": len(rows),
            "continuation_policy_id": continuation_policy_id,
            "q_sampling_summary": sampling_summary,
            "minimum_k_required": None if args.smoke_only else args.minimum_k,
            "source_case_count": source_case_count,
            "train_rows": len(train_rows), "validation_rows": len(val_rows),
            "model": args.model,
            "ridge_alpha": float(args.ridge_alpha) if args.model == "ridge" else None,
            "training_mode": "smoke_only" if args.smoke_only else "formal",
            "not_generalization_evidence": bool(args.smoke_only),
            "official_case_split_audit": split_audit,
            "q_data_audit": audit_summary,
        },
    }
    atomic_torch_save(payload, checkpoint_path)
    serialize_json(output / "training_manifest.json", {
        "status": "completed",
        "task": f"Q {args.model} pipeline smoke" if args.smoke_only else f"Q {args.model} fit",
        "training_mode": "smoke_only" if args.smoke_only else "formal",
        "not_generalization_evidence": bool(args.smoke_only),
        "synthetic_data": False,
        "checkpoint": str(checkpoint_path.resolve()),
        "feature_source": feature_source,
        "external_feature_metadata": feature_metadata,
        "model": args.model,
        "ridge_alpha": float(args.ridge_alpha) if args.model == "ridge" else None,
        "continuation_policy_id": continuation_policy_id,
        "q_sampling_summary": sampling_summary,
        "minimum_k_required": None if args.smoke_only else args.minimum_k,
        "q_data_audit": audit_summary,
        "action_order": payload["action_order"],
        "train_case_ids": sorted(train_ids), "validation_case_ids": sorted(val_ids),
        "best_epoch": fit["best_epoch"], "best_validation_loss": fit["best_validation_loss"],
        "validation_value_calibration_fitted": True,
        "real_data_confirmed": True,
        "calibration_uses_validation_cases": True,
        "validation_metrics_are_not_independent_test_results": True,
    })
    print(json.dumps({"status": "completed", "checkpoint": str(checkpoint_path.resolve()),
                      "best_epoch": fit["best_epoch"], "best_validation_loss": fit["best_validation_loss"],
                      "train_cases": len(train_ids), "validation_cases": len(val_ids),
                      "model": args.model, "feature_source": feature_source}, indent=2))


if __name__ == "__main__":
    main()
