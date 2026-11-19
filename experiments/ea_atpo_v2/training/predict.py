#!/usr/bin/env python3
"""Create Q-value or distilled-policy scores for public-state JSONL rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

from . import ACTION_NAMES
from .calibration import ActionValueCalibrator
from .data import load_public_embedding_cache, q_example_from_dict, read_jsonl
from .features import ExistingPublicStateEncoder, Standardizer, feature_matrix
from .models import ConstantMeanQRegressor, PolicyHead, QRegressor, RidgeQRegressor
from .trainer_utils import load_checkpoint


def load_unique_states(path: Path):
    by_key = {}
    for row in read_jsonl(path):
        example = q_example_from_dict(row)
        key = (example.case_id, example.state_id)
        fingerprint = hashlib.sha256(json.dumps(list(example.public_state), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if key in by_key and by_key[key][1] != fingerprint:
            raise ValueError(f"public-state contents differ for repeated state {key}")
        by_key.setdefault(key, (example, fingerprint, row))
    if not by_key:
        raise ValueError("no public states found")
    return list(by_key.values())


def legal_actions(public_state, max_asks: int = 3):
    asked = sum(
        message["role"] == "assistant" and re.match(r"\s*question\s*:", message["content"], re.IGNORECASE) is not None
        for message in public_state
    )
    return [action for action in ACTION_NAMES if action != "ASK" or asked < max_asks]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True, help="Q JSONL or public rollout JSONL")
    parser.add_argument("--public-features", type=Path, help="required for checkpoints trained with external embeddings")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite predictions: {args.output}")
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint.get("action_order") != list(ACTION_NAMES):
        raise ValueError(f"checkpoint action order differs from V2 protocol: {checkpoint.get('action_order')}")
    items = load_unique_states(args.states)
    examples = [entry[0] for entry in items]
    if checkpoint["public_encoder"] is None:
        if args.public_features is None:
            raise ValueError("this checkpoint needs --public-features from public_state_before_action_only")
        cache = load_public_embedding_cache(args.public_features, expected_examples=examples)
        expected_feature_metadata = checkpoint.get("external_feature_metadata")
        if not expected_feature_metadata:
            raise ValueError("checkpoint lacks the pinned external embedding configuration")
        if cache.metadata["embedding_config_hash"] != expected_feature_metadata["embedding_config_hash"]:
            raise ValueError("prediction embedding cache uses a different model/configuration than training")
        raw_features, _ = feature_matrix(examples, external_vectors=cache.vectors)
        feature_source = "external_public_state_embedding"
    else:
        encoder = ExistingPublicStateEncoder.from_json(checkpoint["public_encoder"])
        raw_features, _ = feature_matrix(examples, encoder=encoder)
        feature_source = checkpoint["feature_source"]
    scaler = Standardizer.from_json(checkpoint["feature_standardizer"])
    features = torch.from_numpy(scaler.transform(raw_features))
    model_type = checkpoint["model_type"]
    if model_type == "q_regressor":
        model = QRegressor(checkpoint["input_dim"], checkpoint["hidden_dim"])
    elif model_type == "q_constant_mean":
        model = ConstantMeanQRegressor(checkpoint["input_dim"])
    elif model_type == "q_ridge":
        model = RidgeQRegressor(checkpoint["input_dim"])
    elif model_type == "policy_head":
        model = PolicyHead(checkpoint["input_dim"], checkpoint["hidden_dim"])
    else:
        raise ValueError(f"unsupported checkpoint model_type: {model_type}")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        scores = model(features).cpu().numpy().astype(np.float64)

    q_model_types = {"q_regressor", "q_constant_mean", "q_ridge"}
    calibrator = ActionValueCalibrator.from_json(checkpoint["action_value_calibrator"]) if model_type in q_model_types else None
    reference_costs = np.asarray([checkpoint["reference_action_costs"][a] for a in ACTION_NAMES], dtype=np.float64)
    gross_values = calibrator.calibrate_gross(scores, reference_costs) if calibrator else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for i, (example, _fingerprint, raw) in enumerate(items):
            result = {
                "case_id": example.case_id,
                "state_id": example.state_id,
                "legal_actions": legal_actions(list(example.public_state)),
                "feature_source": feature_source,
                "training_continuation_policy_id": checkpoint.get("training", {}).get("continuation_policy_id"),
                "training_model_identity": (((checkpoint.get("training", {}).get("q_data_audit") or {}).get("real_run_provenance") or {}).get("models")),
            }
            if model_type in q_model_types:
                result["q_values_reference_net"] = {a: float(gross_values[i, j] - reference_costs[j]) for j, a in enumerate(ACTION_NAMES)}
                result["q_gross_values"] = {a: float(gross_values[i, j]) for j, a in enumerate(ACTION_NAMES)}
                result["q_reference_costs"] = {a: float(reference_costs[j]) for j, a in enumerate(ACTION_NAMES)}
                result["q_calibrated"] = True
            else:
                result["policy_logits"] = {a: float(scores[i, j]) for j, a in enumerate(ACTION_NAMES)}
                result["policy_temperature"] = float(checkpoint["distillation_temperature"])
                result["policy_reference_costs"] = dict(checkpoint["reference_action_costs"])
            for key in ("uncertainty_score", "sft_action"):
                if key in raw:
                    result[key] = raw[key]
            if raw.get("synthetic_data", False):
                result["synthetic_data"] = True
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "completed", "model_type": model_type, "states": len(items),
                      "feature_source": feature_source, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
