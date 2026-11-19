#!/usr/bin/env python3
"""Train a lightweight actor head by masked Q-guided policy distillation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_q_dataset, q_sampling_summary, require_minimum_q_samples, require_single_continuation_policy, validate_official_train_cases, validate_q_audit_report
from .losses import masked_distillation_loss
from .models import PolicyHead
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
    parser.add_argument("--public-features", type=Path, help="same versioned frozen-Base embedding cache used for Q fitting; required for formal fitting")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--max-features", type=int, default=3000)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--minimum-k", type=int, default=3, help="formal minimum valid rollout count per observed state/action; smoke ignores this gate")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    if not args.real_data_confirmed:
        raise ValueError("policy distillation requires --real-data-confirmed after auditing Q targets")
    if not args.smoke_only and args.public_features is None:
        raise ValueError("formal policy distillation requires the same versioned frozen-Base public embedding cache as Q fitting")
    if not args.smoke_only and args.audit_report is None:
        raise ValueError("formal policy distillation requires --audit-report for the exact Q dataset")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
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
                "formal policy distillation requires the complete official train case pool; "
                f"observed={len(observed_train)}, expected={len(official_train)}, missing={len(official_train-observed_train)}"
            )
        require_minimum_q_samples(rows, args.minimum_k)
    train_rows, val_rows, train_ids, val_ids = split_train_validation(
        rows, validation_fraction=args.validation_fraction, seed=args.seed
    )
    if not all(row.costs_included_in_targets for row in rows):
        raise ValueError("Q distillation expects the registered action costs to already be included in Q_target")
    x_train, x_val, encoder, scaler, feature_source, feature_metadata = prepare_feature_splits(
        train_rows, val_rows, args.public_features, args.max_features
    )
    model = PolicyHead(x_train.shape[1], args.hidden_dim)

    def loss_fn(logits, q_target, valid_mask, _sample_count):
        loss, _eligible = masked_distillation_loss(
            logits, q_target, valid_mask, temperature=args.temperature, allow_empty=True
        )
        return loss

    fit = optimize(model, x_train, train_rows, x_val, val_rows,
                   epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                   learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                   loss_fn=loss_fn, seed=args.seed)
    checkpoint_path = args.output_dir / "policy_head.pt"
    payload = {
        "format": "ea_atpo_v2_training_checkpoint_v1",
        "model_type": "policy_head",
        "action_order": ["ASK", "ANSWER", "DEFER"],
        "input_dim": int(x_train.shape[1]), "hidden_dim": int(args.hidden_dim),
        "model_state": model.state_dict(),
        "feature_source": feature_source,
        "external_feature_metadata": feature_metadata,
        "public_encoder": encoder.to_json() if encoder else None,
        "feature_standardizer": scaler.to_json(),
        "distillation_temperature": float(args.temperature),
        "q_targets_include_action_costs": True,
        "reference_action_costs": {"ASK": 0.05, "ANSWER": 0.0, "DEFER": 0.60},
        "training": {
            "best_epoch": fit["best_epoch"], "best_validation_loss": fit["best_validation_loss"],
            "seed": args.seed, "source_q_dataset": str(args.q_dataset.resolve()),
            "continuation_policy_id": continuation_policy_id,
            "q_sampling_summary": sampling_summary,
            "minimum_k_required": None if args.smoke_only else args.minimum_k,
            "train_case_ids": sorted(train_ids), "validation_case_ids": sorted(val_ids),
            "train_rows": len(train_rows), "validation_rows": len(val_rows),
            "training_mode": "smoke_only" if args.smoke_only else "formal",
            "not_generalization_evidence": bool(args.smoke_only),
            "official_case_split_audit": split_audit,
            "q_data_audit": audit_summary,
            "states_with_fewer_than_two_valid_actions_are_skipped": True,
        },
    }
    atomic_torch_save(payload, checkpoint_path)
    serialize_json(args.output_dir / "training_manifest.json", {
        "status": "completed", "task": "Q-guided policy distillation smoke" if args.smoke_only else "Q-guided policy distillation",
        "training_mode": "smoke_only" if args.smoke_only else "formal",
        "not_generalization_evidence": bool(args.smoke_only),
        "synthetic_data": False, "real_data_confirmed": True, "checkpoint": str(checkpoint_path.resolve()),
        "temperature": args.temperature, "best_epoch": fit["best_epoch"],
        "best_validation_loss": fit["best_validation_loss"], "feature_source": feature_source,
        "external_feature_metadata": feature_metadata,
        "continuation_policy_id": continuation_policy_id,
        "q_sampling_summary": sampling_summary,
        "minimum_k_required": None if args.smoke_only else args.minimum_k,
        "q_data_audit": audit_summary,
        "train_case_ids": sorted(train_ids), "validation_case_ids": sorted(val_ids),
        "validation_metrics_are_not_independent_test_results": True,
    })
    print(json.dumps({"status": "completed", "checkpoint": str(checkpoint_path.resolve()),
                      "temperature": args.temperature, "best_epoch": fit["best_epoch"],
                      "best_validation_loss": fit["best_validation_loss"],
                      "train_cases": len(train_ids), "validation_cases": len(val_ids)}, indent=2))


if __name__ == "__main__":
    main()
