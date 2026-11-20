#!/usr/bin/env python3
"""Train and evaluate lightweight V2 value routers on a fixed case split.

This is a Train-only development study. It never reads official Validation,
Internal Test, or Frozen Holdout outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments.ea_atpo_v2.training.data import load_q_dataset, load_public_embedding_cache
from experiments.ea_atpo_v2.training.features import ExistingPublicStateEncoder, Standardizer
from experiments.ea_atpo_v2.training.models import ConstantMeanQRegressor

ACTION_ORDER = ("ASK", "ANSWER", "DEFER")
SEEDS = (17, 42, 2026)


def json_dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def load_split(path: Path, rows):
    split = json.loads(path.read_text(encoding="utf-8"))
    if split.get("unit") != "case_id" or split.get("train_case_count", 35) != 35:
        raise ValueError("expected the frozen case-level 35/15 split")
    train_ids, dev_ids = set(map(str, split["train"])), set(map(str, split["dev"]))
    if len(train_ids) != 35 or len(dev_ids) != 15 or train_ids & dev_ids:
        raise ValueError("invalid case-level split");
    observed = {r.case_id for r in rows}
    if observed != train_ids | dev_ids:
        raise ValueError("split case IDs do not match the cost-adjusted Q dataset")
    return train_ids, dev_ids, split


def make_features(rows, train_ids, dev_ids, feature_kind: str, embedding_path: Path | None, max_features: int):
    train = [r for r in rows if r.case_id in train_ids]
    dev = [r for r in rows if r.case_id in dev_ids]
    if feature_kind == "text":
        encoder = ExistingPublicStateEncoder(max_features).fit(train)
        raw_train, raw_dev = encoder.transform(train), encoder.transform(dev)
        feature_manifest = {"feature_kind": "audited_v1_tfidf_public_state", "vocabulary_fit_cases": 35,
                            "features_fit_on_train_only": True, "encoder": encoder.to_json()}
    elif feature_kind == "embedding":
        if embedding_path is None:
            raise ValueError("embedding feature group needs a validated frozen public-state embedding cache")
        cache = load_public_embedding_cache(embedding_path, expected_examples=rows)
        raw_train = np.stack([cache.vectors[(r.case_id, r.state_id)] for r in train]).astype(np.float32)
        raw_dev = np.stack([cache.vectors[(r.case_id, r.state_id)] for r in dev]).astype(np.float32)
        feature_manifest = {"feature_kind": "frozen_qwen3_8b_public_state_mean_embedding",
                            "model_id": cache.metadata["model_id"], "model_revision": cache.metadata["model_revision"],
                            "cache_sha256": cache.sha256, "feature_boundary": "public_state_before_action_only"}
        encoder = None
    else:
        raise ValueError(feature_kind)
    scaler = Standardizer().fit(raw_train)
    x_train, x_dev = scaler.transform(raw_train), scaler.transform(raw_dev)
    feature_manifest["standardizer_fit_cases"] = 35
    return train, dev, x_train, x_dev, encoder, scaler, feature_manifest


def arrays(rows):
    q = np.stack([r.q_target for r in rows]).astype(np.float32)
    mask = np.stack([r.valid_mask for r in rows]).astype(bool)
    count = np.stack([r.sample_count for r in rows]).astype(np.float32)
    accuracy = np.stack([r.q_target * 0.0 for r in rows]).astype(np.float32)
    # The cost-adjusted dataset stores raw correctness beside new targets in its paired source table.
    # Correctness is attached below from the verified raw rollout lookup.
    return q, mask, count, accuracy


def target_arrays(q, mask, mode):
    if mode == "absolute":
        return q.copy(), mask.copy()
    if mode == "relative":
        # Relative advantages are defined against ANSWER, requiring both labels.
        common = mask[:, [0, 1, 2]].copy()
        ask_valid = common[:, 0] & common[:, 1]
        defer_valid = common[:, 2] & common[:, 1]
        target = np.column_stack([q[:, 0] - q[:, 1], q[:, 2] - q[:, 1]])
        relative_mask = np.column_stack([ask_valid, defer_valid])
        return target.astype(np.float32), relative_mask
    if mode == "distill":
        return q.copy(), mask.copy()
    raise ValueError(mode)


def fit_ridge(x_train, target, mask, alpha):
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    outputs = target.shape[1]
    weights = np.zeros((x_train.shape[1], outputs), dtype=np.float64)
    bias = np.zeros(outputs, dtype=np.float64)
    x = np.asarray(x_train, dtype=np.float64)
    for j in range(outputs):
        use = mask[:, j]
        if not use.any():
            raise ValueError(f"no valid training targets for output {j}")
        xv, yv = x[use], target[use, j]
        xm, ym = xv.mean(0), float(yv.mean())
        xc, yc = xv - xm, yv - ym
        dual = np.linalg.solve(xc @ xc.T + alpha * np.eye(len(xc)), yc)
        weights[:, j] = xc.T @ dual
        bias[j] = ym - xm @ weights[:, j]
    return weights.astype(np.float32), bias.astype(np.float32)


class TinyHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)


def torch_loss(pred, target, mask, objective, temperature):
    if objective in ("absolute", "relative"):
        if pred.shape != target.shape or pred.shape != mask.shape or not bool(mask.any()):
            raise ValueError("invalid masked regression batch")
        err = F.smooth_l1_loss(pred, target, reduction="none")
        return err[mask].mean()
    if objective == "distill":
        if pred.shape != target.shape or pred.shape != mask.shape:
            raise ValueError("invalid masked distillation batch")
        eligible = mask.sum(1) >= 2
        if not bool(eligible.any()):
            raise ValueError("no state has at least two valid actions for policy distillation")
        logits, q, valid = pred[eligible], target[eligible], mask[eligible]
        losses = []
        for row_logits, row_q, row_valid in zip(logits, q, valid):
            log_pi = F.log_softmax(row_logits[row_valid], dim=0)
            target_pi = F.softmax(row_q[row_valid] / temperature, dim=0)
            losses.append(-(target_pi * log_pi).sum())
        return torch.stack(losses).mean()
    raise ValueError(objective)


def fit_mlp(x_train, target, mask, x_dev, y_dev, mask_dev, *, seed, objective,
            hidden_dim, epochs, patience, batch_size, learning_rate, weight_decay, temperature):
    seed_all(seed)
    model = TinyHead(x_train.shape[1], target.shape[1], hidden_dim)
    tx, ty, tm = (torch.as_tensor(x_train, dtype=torch.float32), torch.as_tensor(target, dtype=torch.float32),
                  torch.as_tensor(mask, dtype=torch.bool))
    vx, vy, vm = (torch.as_tensor(x_dev, dtype=torch.float32), torch.as_tensor(y_dev, dtype=torch.float32),
                  torch.as_tensor(mask_dev, dtype=torch.bool))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); order = rng.permutation(len(tx)); batch_losses = []
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], dtype=torch.long)
            pred = model(tx[idx]); loss = torch_loss(pred, ty[idx], tm[idx], objective, temperature)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            batch_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val = float(torch_loss(model(vx), vy, vm, objective, temperature))
        train_loss = float(np.mean(batch_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val})
        if val < best_loss - 1e-8:
            best_loss, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("MLP early stopping did not save a checkpoint")
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        train_pred = model(torch.as_tensor(x_train, dtype=torch.float32)).numpy()
        dev_pred = model(torch.as_tensor(x_dev, dtype=torch.float32)).numpy()
    return model, train_pred, dev_pred, {"history": history, "best_epoch": best_epoch,
                                         "best_validation_loss": best_loss,
                                         "best_train_loss": history[best_epoch - 1]["train_loss"]}


def safe_auc(labels, scores):
    y, s = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)
    p, n = int(y.sum()), int((~y).sum())
    if not p or not n:
        return None
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s), dtype=float); i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and s[order[j]] == s[order[i]]: j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2; i = j
    return float((ranks[y].sum() - p * (p + 1) / 2) / (p * n))


def policy_metrics(action_indices, rows, correctness_lookup):
    case_metrics = defaultdict(lambda: {"accuracy": [], "utility": [], "ask": [], "answer": [], "defer": [],
                                        "corr": 0.0, "spoil": 0.0, "basewrong_defer": 0.0,
                                        "basecorrect_defer": 0.0, "states": 0})
    for idx, row in zip(action_indices, rows):
        action = ACTION_ORDER[int(idx)]
        correct = correctness_lookup[(row.case_id, row.state_id)]
        selected_correct = float(correct[action])
        m = case_metrics[row.case_id]
        m["accuracy"].append(selected_correct)
        m["utility"].append(float(row.q_target[int(idx)]))
        m["ask"].append(float(action == "ASK")); m["answer"].append(float(action == "ANSWER")); m["defer"].append(float(action == "DEFER"))
        m["states"] += 1
        if action == "DEFER":
            base = float(correct["ANSWER"]); expert = float(correct["DEFER"])
            m["corr"] += float(base == 0 and expert == 1)
            m["spoil"] += float(base == 1 and expert == 0)
            m["basewrong_defer"] += float(base == 0)
            m["basecorrect_defer"] += float(base == 1)
    case_rows = {}
    for case, m in case_metrics.items():
        case_rows[case] = {name: (float(np.mean(m[name])) if name in ("accuracy", "utility", "ask", "answer", "defer") else float(m[name])) for name in m if name != "states"}
    # Equal case weighting is used throughout, matching case-cluster bootstrap.
    keys = ["accuracy", "utility", "ask", "answer", "defer"]
    result = {name: float(np.mean([case_rows[c][name] for c in case_rows])) for name in keys}
    # Also save pooled-state metrics/rates explicitly. The equal-case action
    # rates above are useful companion metrics but are not state fractions.
    chosen_actions = [ACTION_ORDER[int(idx)] for idx in action_indices]
    result.update({
        "accuracy_state_weighted": float(np.mean([
            correctness_lookup[(row.case_id, row.state_id)][action]
            for row, action in zip(rows, chosen_actions)
        ])),
        "utility_state_weighted": float(np.mean([
            row.q_target[int(idx)] for idx, row in zip(action_indices, rows)
        ])),
        "ask_rate_state_weighted": float(np.mean(action_indices == 0)),
        "answer_rate_state_weighted": float(np.mean(action_indices == 1)),
        "defer_rate_state_weighted": float(np.mean(action_indices == 2)),
    })
    selected_defer = int(np.sum(action_indices == 2))
    corr = sum(case_rows[c]["corr"] for c in case_rows); spoil = sum(case_rows[c]["spoil"] for c in case_rows)
    base_wrong = sum(case_rows[c]["basewrong_defer"] for c in case_rows)
    base_correct = sum(case_rows[c]["basecorrect_defer"] for c in case_rows)
    result.update({"defer_calls_state_count": selected_defer, "corrections_count": int(corr), "spoilage_count": int(spoil),
                   "correction_rate_given_base_wrong": corr / base_wrong if base_wrong else None,
                   "spoilage_rate_given_base_correct": spoil / base_correct if base_correct else None,
                   "case_metrics": case_rows})
    return result


def cluster_bootstrap_delta(policy_cases_by_seed, baseline_cases_by_seed, *, seed=20261120, replicates=5000):
    # A shared fixed baseline (e.g., Always ANSWER) is one row, while a learned
    # policy has one row per optimization seed. Repeat the shared baseline so
    # zip() does not silently reduce the interval to the first policy seed.
    if len(baseline_cases_by_seed) == 1 and len(policy_cases_by_seed) > 1:
        baseline_cases_by_seed = baseline_cases_by_seed * len(policy_cases_by_seed)
    if len(policy_cases_by_seed) != len(baseline_cases_by_seed):
        raise ValueError("policy and baseline seed counts must match or baseline must contain one shared row")
    case_ids = sorted(set.intersection(*(set(x) for x in policy_cases_by_seed), *(set(x) for x in baseline_cases_by_seed)))
    rng = np.random.default_rng(seed)
    samples = {"accuracy": [], "utility": []}
    for _ in range(replicates):
        drawn = rng.choice(case_ids, size=len(case_ids), replace=True)
        for metric in samples:
            seed_deltas = []
            for pol, base in zip(policy_cases_by_seed, baseline_cases_by_seed):
                seed_deltas.append(np.mean([pol[c][metric] - base[c][metric] for c in drawn]))
            samples[metric].append(float(np.mean(seed_deltas)))
    return {m: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for m, v in samples.items()}


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-dataset", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--max-features", type=int, default=3000)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    rows = load_q_dataset(args.q_dataset)
    train_ids, dev_ids, split = load_split(args.split, rows)
    raw_trajectories = [json.loads(s) for s in args.trajectories.read_text(encoding="utf-8").splitlines() if s.strip()]
    correct = {}; action_seen = defaultdict(set)
    for tr in raw_trajectories:
        key = (str(tr["case_id"]), str(tr["state_id"])); action = str(tr["action"])
        if action not in ACTION_ORDER or tr.get("valid") is not True:
            raise ValueError("all action outcome labels must come from valid raw real rollouts")
        if action in action_seen[key]: raise ValueError(f"duplicate action outcome for {key}/{action}")
        correct.setdefault(key, {})[action] = float(tr["final_accuracy"])
        action_seen[key].add(action)
    if set(correct) != {(r.case_id, r.state_id) for r in rows} or any(set(v) != set(ACTION_ORDER) for v in correct.values()):
        raise ValueError("raw rollout correctness does not align exactly with Q states/actions")
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    correctness_lookup = correct
    q_all = np.stack([r.q_target for r in rows]).astype(np.float32)
    mask_all = np.stack([r.valid_mask for r in rows]).astype(bool)
    if not mask_all.all():
        # This cohort is fully paired; generic fitting remains masked and fails closed for unsupported labels.
        if not np.all(mask_all.sum(1) >= 2): raise ValueError("every train/dev state needs at least two valid action targets")
    train_indices = [i for i,r in enumerate(rows) if r.case_id in train_ids]
    dev_indices = [i for i,r in enumerate(rows) if r.case_id in dev_ids]
    train_rows = [rows[i] for i in train_indices]; dev_rows = [rows[i] for i in dev_indices]
    q_train, q_dev = q_all[train_indices], q_all[dev_indices]
    mask_train, mask_dev = mask_all[train_indices], mask_all[dev_indices]
    feature_groups = ["text"] + (["embedding"] if args.embedding_cache is not None else [])
    results, prediction_rows, q_error_rows = [], [], []
    all_policy_case_metrics = defaultdict(list)
    for feature_kind in feature_groups:
        train_rows, dev_rows, x_train, x_dev, encoder, scaler, feat_meta = make_features(
            rows, train_ids, dev_ids, feature_kind, args.embedding_cache, args.max_features)
        y_train_abs, m_train_abs = target_arrays(q_train, mask_train, "absolute")
        y_dev_abs, m_dev_abs = target_arrays(q_dev, mask_dev, "absolute")
        y_train_rel, m_train_rel = target_arrays(q_train, mask_train, "relative")
        y_dev_rel, m_dev_rel = target_arrays(q_dev, mask_dev, "relative")

        # Frozen train-only constant Q baseline.
        const = ConstantMeanQRegressor(x_train.shape[1]); const.fit_closed_form(train_rows)
        const_pred_train = const(torch.from_numpy(x_train)).detach().numpy()
        const_pred_dev = const(torch.from_numpy(x_dev)).detach().numpy()
        static_l2d_action = 2 if float(np.mean(y_train_abs[:, 2] - y_train_abs[:, 1])) > 0 else 1

        methods = []
        for seed in SEEDS:
            # Ridge absolute Q regression.
            w, b = fit_ridge(x_train, y_train_abs, m_train_abs, args.ridge_alpha)
            methods.append(("ridge_absolute", seed, (x_train @ w + b, x_dev @ w + b), {"weight": w, "bias": b}, {"best_epoch": 0}))
            # Ridge relative advantages: ASK-ANSWER and DEFER-ANSWER.
            wr, br = fit_ridge(x_train, y_train_rel, m_train_rel, args.ridge_alpha)
            methods.append(("ridge_relative", seed,
                            (x_train @ wr + br, x_dev @ wr + br), {"weight": wr, "bias": br}, {"best_epoch": 0}))
            # Three small learned heads; each seed is initialized and trained independently.
            for method_name, objective, train_target, train_mask, dev_target, dev_mask in (
                ("mlp_absolute", "absolute", y_train_abs, m_train_abs, y_dev_abs, m_dev_abs),
                ("mlp_relative", "relative", y_train_rel, m_train_rel, y_dev_rel, m_dev_rel),
                ("q_guided_distillation", "distill", y_train_abs, m_train_abs, y_dev_abs, m_dev_abs),
            ):
                model, pred_train, pred_dev, fit = fit_mlp(
                    x_train, train_target, train_mask, x_dev, dev_target, dev_mask,
                    seed=seed, objective=objective, hidden_dim=args.hidden_dim, epochs=args.epochs,
                    patience=args.patience, batch_size=args.batch_size, learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay, temperature=args.temperature)
                state = model.state_dict()
                methods.append((method_name, seed, (pred_train, pred_dev), state, fit))

        # Save three-seed checkpoints, predictions, train/dev errors and policy-level outcomes.
        for method_name, seed, (pred_train, pred_dev), state, fit in methods:
            model_dir = args.output_dir / "checkpoints" / feature_kind / method_name
            model_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = model_dir / f"seed_{seed}.pt"
            payload = {"format": "ea_atpo_v2_router_training_v1", "model": method_name,
                       "feature_kind": feature_kind, "action_order": list(ACTION_ORDER), "seed": seed,
                       "state": state, "feature_metadata": feat_meta,
                       "encoder": encoder.to_json() if encoder else None,
                       "standardizer": scaler.to_json(),
                       "temperature": args.temperature if method_name == "q_guided_distillation" else None,
                       "reference_costs": {"ANSWER": 0.0, "ASK": 0.01, "DEFER": 0.05},
                       "costs_already_in_q_targets": True, "train_case_ids": sorted(train_ids),
                       "dev_case_ids": sorted(dev_ids), "training_metadata": fit}
            torch.save(payload, checkpoint)
            if method_name == "ridge_absolute":
                predq_train, predq_dev = pred_train, pred_dev
                actions_train, actions_dev = predq_train.argmax(1), predq_dev.argmax(1)
                target_dev, target_train = y_dev_abs, y_train_abs
                qmae_train = float(np.mean(np.abs(pred_train - y_train_abs)))
                qmae_dev = float(np.mean(np.abs(pred_dev - y_dev_abs)))
                rel_pred_train = np.column_stack([pred_train[:,0]-pred_train[:,1], pred_train[:,2]-pred_train[:,1]])
                rel_pred_dev = np.column_stack([pred_dev[:,0]-pred_dev[:,1], pred_dev[:,2]-pred_dev[:,1]])
                relmae_train = float(np.mean(np.abs(rel_pred_train-y_train_rel)))
                relmae_dev = float(np.mean(np.abs(rel_pred_dev-y_dev_rel)))
                q_auc_ask = safe_auc((y_dev_abs[:,0] - y_dev_abs[:,1]) > 0, pred_dev[:,0] - pred_dev[:,1])
                q_auc_defer = safe_auc((y_dev_abs[:,2] - y_dev_abs[:,1]) > 0, pred_dev[:,2] - pred_dev[:,1])
            elif method_name == "mlp_absolute":
                predq_train, predq_dev = pred_train, pred_dev
                actions_train, actions_dev = pred_train.argmax(1), pred_dev.argmax(1)
                qmae_train = float(np.mean(np.abs(pred_train - y_train_abs)))
                qmae_dev = float(np.mean(np.abs(pred_dev - y_dev_abs)))
                rel_pred_train = np.column_stack([pred_train[:,0]-pred_train[:,1], pred_train[:,2]-pred_train[:,1]])
                rel_pred_dev = np.column_stack([pred_dev[:,0]-pred_dev[:,1], pred_dev[:,2]-pred_dev[:,1]])
                relmae_train = float(np.mean(np.abs(rel_pred_train-y_train_rel)))
                relmae_dev = float(np.mean(np.abs(rel_pred_dev-y_dev_rel)))
                q_auc_ask = safe_auc((y_dev_abs[:,0] - y_dev_abs[:,1]) > 0, pred_dev[:,0] - pred_dev[:,1])
                q_auc_defer = safe_auc((y_dev_abs[:,2] - y_dev_abs[:,1]) > 0, pred_dev[:,2] - pred_dev[:,1])
            elif method_name == "ridge_relative":
                predq_train = np.column_stack([pred_train[:,0], np.zeros(len(pred_train)), pred_train[:,1]])
                predq_dev = np.column_stack([pred_dev[:,0], np.zeros(len(pred_dev)), pred_dev[:,1]])
                actions_train, actions_dev = predq_train.argmax(1), predq_dev.argmax(1)
                relmae_train = float(np.mean(np.abs(pred_train - y_train_rel)))
                relmae_dev = float(np.mean(np.abs(pred_dev - y_dev_rel)))
                qmae_train = qmae_dev = None
                q_auc_ask = safe_auc(y_dev_rel[:,0] > 0, pred_dev[:,0])
                q_auc_defer = safe_auc(y_dev_rel[:,1] > 0, pred_dev[:,1])
            elif method_name == "mlp_relative":
                predq_train = np.column_stack([pred_train[:,0], np.zeros(len(pred_train)), pred_train[:,1]])
                predq_dev = np.column_stack([pred_dev[:,0], np.zeros(len(pred_dev)), pred_dev[:,1]])
                actions_train, actions_dev = predq_train.argmax(1), predq_dev.argmax(1)
                relmae_train = float(np.mean(np.abs(pred_train - y_train_rel)))
                relmae_dev = float(np.mean(np.abs(pred_dev - y_dev_rel)))
                qmae_train = qmae_dev = None
                q_auc_ask = safe_auc(y_dev_rel[:,0] > 0, pred_dev[:,0])
                q_auc_defer = safe_auc(y_dev_rel[:,1] > 0, pred_dev[:,1])
            else:  # distillation head predicts action logits, not calibrated Q values.
                predq_train = pred_train * args.temperature
                predq_dev = pred_dev * args.temperature
                actions_train, actions_dev = pred_train.argmax(1), pred_dev.argmax(1)
                qmae_train = qmae_dev = relmae_train = relmae_dev = None
                q_auc_ask = q_auc_defer = None

            train_policy = policy_metrics(actions_train, train_rows, correctness_lookup)
            dev_policy = policy_metrics(actions_dev, dev_rows, correctness_lookup)
            # Save per-seed per-state predictions without including clinical text.
            for row, action, pred in zip(dev_rows, actions_dev, predq_dev):
                prediction_rows.append({"feature": feature_kind, "model": method_name, "seed": seed,
                                        "case_id": row.case_id, "state_id": row.state_id,
                                        "predicted_action": ACTION_ORDER[int(action)],
                                        "predicted_scores": [float(x) for x in pred],
                                        "q_target": [float(x) if math.isfinite(float(x)) else None for x in row.q_target],
                                        "valid_mask": [bool(x) for x in row.valid_mask]})
            for row_index, (row, pred) in enumerate(zip(dev_rows, predq_dev)):
                actual = correctness_lookup[(row.case_id, row.state_id)]
                q_error_rows.append({"feature": feature_kind, "model": method_name, "seed": seed,
                                     "case_id": row.case_id, "state_id": row.state_id,
                                     "q_target": [float(x) for x in row.q_target],
                                     "predicted_scores": [float(x) for x in pred],
                                     "prediction_semantics": "q_values" if method_name in ("ridge_absolute","mlp_absolute") else ("relative_advantages_with_answer_anchor_0" if method_name in ("ridge_relative","mlp_relative") else "distillation_logits_scaled_by_temperature"),
                                     "relative_target": [float(y_dev_rel[row_index,0]), float(y_dev_rel[row_index,1])],
                                     "correctness": actual})
            results.append({"feature": feature_kind, "model": method_name, "seed": seed,
                            "checkpoint": str(checkpoint.resolve()), "train_rows": len(train_rows), "dev_rows": len(dev_rows),
                            "train_loss_best_epoch": fit.get("best_train_loss"),
                            "validation_loss": fit.get("best_validation_loss"), "best_epoch": fit.get("best_epoch"),
                            "q_mae_train": qmae_train, "q_mae_dev": qmae_dev,
                            "relative_advantage_mae_train": relmae_train, "relative_advantage_mae_dev": relmae_dev,
                            "advantage_auc_ask_dev": q_auc_ask, "advantage_auc_defer_dev": q_auc_defer,
                            "train_policy": {k:v for k,v in train_policy.items() if k != "case_metrics"},
                            "dev_policy": {k:v for k,v in dev_policy.items() if k != "case_metrics"},
                            "dev_case_metrics": dev_policy["case_metrics"]})
            all_policy_case_metrics[(feature_kind, method_name)].append(dev_policy["case_metrics"])

        # Constant mean is an explicit score-based baseline, repeated over the 3 registered seeds.
        for seed in SEEDS:
            pred_train, pred_dev = const_pred_train, const_pred_dev
            actions_train, actions_dev = pred_train.argmax(1), pred_dev.argmax(1)
            checkpoint = args.output_dir / "checkpoints" / feature_kind / "constant_mean" / f"seed_{seed}.json"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            json_dump(checkpoint, {"action_order": list(ACTION_ORDER), "q_means": pred_train[0].tolist(),
                                   "seed": seed, "feature_kind": feature_kind, "train_cases": sorted(train_ids),
                                   "costs": {"ASK":0.01,"ANSWER":0.0,"DEFER":0.05}})
            train_policy, dev_policy = policy_metrics(actions_train, train_rows, correctness_lookup), policy_metrics(actions_dev, dev_rows, correctness_lookup)
            results.append({"feature": feature_kind, "model": "constant_mean", "seed": seed,
                            "checkpoint": str(checkpoint.resolve()), "train_rows": len(train_rows), "dev_rows": len(dev_rows),
                            "q_mae_train": float(np.mean(np.abs(pred_train-y_train_abs))),
                            "q_mae_dev": float(np.mean(np.abs(pred_dev-y_dev_abs))),
                            "relative_advantage_mae_train": float(np.mean(np.abs(np.column_stack([pred_train[:,0]-pred_train[:,1],pred_train[:,2]-pred_train[:,1]])-y_train_rel))),
                            "relative_advantage_mae_dev": float(np.mean(np.abs(np.column_stack([pred_dev[:,0]-pred_dev[:,1],pred_dev[:,2]-pred_dev[:,1]])-y_dev_rel))),
                            "advantage_auc_ask_dev": safe_auc(y_dev_rel[:,0]>0, pred_dev[:,0]-pred_dev[:,1]),
                            "advantage_auc_defer_dev": safe_auc(y_dev_rel[:,1]>0, pred_dev[:,2]-pred_dev[:,1]),
                            "train_policy": {k:v for k,v in train_policy.items() if k != "case_metrics"},
                            "dev_policy": {k:v for k,v in dev_policy.items() if k != "case_metrics"},
                            "dev_case_metrics": dev_policy["case_metrics"]})
            all_policy_case_metrics[(feature_kind,"constant_mean")].append(dev_policy["case_metrics"])

        # Non-learned policy baselines and offline oracle on the same paired states.
        train_mean_adv = float(np.mean(y_train_abs[:,2]-y_train_abs[:,1]))
        fixed_action = "DEFER" if train_mean_adv > 0 else "ANSWER"
        for policy, chosen in (("always_answer", 1), ("always_ask", 0), ("always_defer", 2),
                               ("static_l2d_train_mean", 2 if fixed_action=="DEFER" else 1),
                               ("oracle_offline_upper_bound", None)):
            if policy == "oracle_offline_upper_bound":
                actions = np.argmax(q_dev, axis=1)
            else:
                actions = np.full(len(dev_rows), chosen, dtype=int)
            m = policy_metrics(actions, dev_rows, correctness_lookup)
            results.append({"feature": feature_kind, "model": policy, "seed": None,
                            "checkpoint": None, "train_rows": len(train_rows), "dev_rows": len(dev_rows),
                            "q_mae_train": None, "q_mae_dev": None,
                            "relative_advantage_mae_train": None, "relative_advantage_mae_dev": None,
                            "advantage_auc_ask_dev": None, "advantage_auc_defer_dev": None,
                            "training_train_mean_defer_advantage": train_mean_adv,
                            "fixed_static_l2d_action": fixed_action if policy=="static_l2d_train_mean" else None,
                            "train_policy": None, "dev_policy": {k:v for k,v in m.items() if k!="case_metrics"},
                            "dev_case_metrics": m["case_metrics"]})
            all_policy_case_metrics[(feature_kind,policy)] = [m["case_metrics"]]

    # Case-clustered summary: all states of a case move together in each bootstrap draw.
    summaries = []
    by_feature_model = defaultdict(list)
    for item in results:
        by_feature_model[(item["feature"], item["model"])].append(item)
    for (feature, model_name), items in sorted(by_feature_model.items()):
        metrics = [item["dev_policy"] for item in items if item.get("dev_policy")]
        if not metrics:
            continue
        summary = {"feature": feature, "model": model_name, "seed_count": len(metrics)}
        for key in ("accuracy", "utility", "ask", "answer", "defer"):
            values = [float(m[key]) for m in metrics]
            summary[key + "_mean"] = float(np.mean(values))
            summary[key + "_seed_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        for key in ("accuracy_state_weighted", "utility_state_weighted", "ask_rate_state_weighted",
                    "answer_rate_state_weighted", "defer_rate_state_weighted"):
            values = [float(m[key]) for m in metrics]
            summary[key + "_mean"] = float(np.mean(values))
            summary[key + "_seed_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        for key in ("corrections_count", "spoilage_count", "defer_calls_state_count",
                    "correction_rate_given_base_wrong", "spoilage_rate_given_base_correct"):
            vals = [m.get(key) for m in metrics if m.get(key) is not None]
            summary[key] = float(np.mean(vals)) if vals else None
        for key in ("q_mae_train", "q_mae_dev", "relative_advantage_mae_train", "relative_advantage_mae_dev",
                    "advantage_auc_ask_dev", "advantage_auc_defer_dev"):
            vals = [float(item[key]) for item in items if isinstance(item.get(key), (int, float)) and math.isfinite(float(item[key]))]
            summary[key + "_mean"] = float(np.mean(vals)) if vals else None
            summary[key + "_seed_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else (0.0 if vals else None)
        baseline_items = by_feature_model[(feature, "always_answer")]
        pol_cases = [item["dev_case_metrics"] for item in items]
        base_cases = [item["dev_case_metrics"] for item in baseline_items]
        ci = cluster_bootstrap_delta(pol_cases, base_cases, seed=20261120, replicates=5000)
        summary["paired_vs_always_answer_delta_accuracy_ci95"] = ci["accuracy"]
        summary["paired_vs_always_answer_delta_utility_ci95"] = ci["utility"]
        summaries.append(summary)

    pair_counts = Counter()
    case_pair_deltas = defaultdict(list)
    dev_pair_count = 0
    for row in [r for r in rows if r.case_id in dev_ids]:
        outcomes = correctness_lookup[(row.case_id, row.state_id)]
        base_ok, expert_ok = int(outcomes["ANSWER"]), int(outcomes["DEFER"])
        pair_counts[(base_ok, expert_ok)] += 1
        case_pair_deltas[row.case_id].append(expert_ok - base_ok)
        dev_pair_count += 1
    case_ids_dev = sorted(case_pair_deltas)
    pair_deltas = {c: float(np.mean(case_pair_deltas[c])) for c in case_ids_dev}
    rng_pair = np.random.default_rng(20261120)
    boot_pair = []
    for _ in range(5000):
        sampled = rng_pair.choice(case_ids_dev, size=len(case_ids_dev), replace=True)
        boot_pair.append(float(np.mean([pair_deltas[c] for c in sampled])))
    dev_states = [r for r in rows if r.case_id in dev_ids]
    dev_base_acc_state_weighted = float(np.mean([correctness_lookup[(r.case_id,r.state_id)]["ANSWER"] for r in dev_states]))
    dev_expert_acc_state_weighted = float(np.mean([correctness_lookup[(r.case_id,r.state_id)]["DEFER"] for r in dev_states]))
    dev_base_acc = float(np.mean([np.mean([correctness_lookup[(r.case_id,r.state_id)]["ANSWER"] for r in dev_states if r.case_id == c]) for c in sorted(dev_ids)]))
    dev_expert_acc = float(np.mean([np.mean([correctness_lookup[(r.case_id,r.state_id)]["DEFER"] for r in dev_states if r.case_id == c]) for c in sorted(dev_ids)]))

    # No genuine base confidence/log probabilities are part of this source cohort.
    confidence_fields = []
    def collect_uncertainty_keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if any(term in str(key).lower() for term in ("confidence", "logprob", "log_prob", "entropy", "margin")):
                    confidence_fields.append(str(key))
                collect_uncertainty_keys(child)
        elif isinstance(value, list):
            for child in value:
                collect_uncertainty_keys(child)
    for tr in raw_trajectories:
        collect_uncertainty_keys(tr)
    missing_uncertainty_reason = "no genuine Base confidence/log-probability/entropy/margin is recorded in source trajectories"
    report = {
        "status": "completed_train_dev_only",
        "experiment_scope": "50 official Train cases only; 35 train / 15 development by case_id",
        "not_independent_generalization_evidence": True,
        "official_validation_internal_test_and_frozen_holdout_opened": False,
        "source_q_sha256": sha256(args.q_dataset), "source_trajectory_sha256": sha256(args.trajectories),
        "split_sha256": sha256(args.split), "feature_groups": feature_groups,
        "seeds": list(SEEDS), "costs_included_in_targets": {"ANSWER":0.0,"ASK":0.01,"DEFER":0.05},
        "evaluation_does_not_deduct_cost_again": True,
        "model_config": {"hidden_dim":args.hidden_dim,"ridge_alpha":args.ridge_alpha,"epochs":args.epochs,
                         "patience":args.patience,"batch_size":args.batch_size,"learning_rate":args.learning_rate,
                         "weight_decay":args.weight_decay,"distillation_temperature":args.temperature},
        "uncertainty_threshold": {"status":"unavailable", "reason":missing_uncertainty_reason,
                                  "observed_key_mentions":sorted(set(confidence_fields))},
        "dev_base_expert_paired": {
            "state_count":dev_pair_count, "base_accuracy_case_weighted":dev_base_acc, "expert_accuracy_case_weighted":dev_expert_acc,
            "base_accuracy_state_weighted":dev_base_acc_state_weighted, "expert_accuracy_state_weighted":dev_expert_acc_state_weighted,
            "paired_accuracy_delta_expert_minus_base":dev_expert_acc-dev_base_acc,
            "case_bootstrap_ci95_delta": [float(np.quantile(boot_pair,.025)),float(np.quantile(boot_pair,.975))],
            "four_cell_counts": {"base_correct_expert_correct":pair_counts[(1,1)],
                                 "base_correct_expert_wrong":pair_counts[(1,0)],
                                 "base_wrong_expert_correct":pair_counts[(0,1)],
                                 "base_wrong_expert_wrong":pair_counts[(0,0)]}},
        "targets": {"absolute_q": "terminal correctness minus one-time action costs",
                    "relative": ["ASK-ANSWER", "DEFER-ANSWER"],
                    "distillation": "masked softmax of costed Q / temperature with cross-entropy"},
        "bootstrap": {"unit":"case_id", "replicates":5000,"seed":20261120,"ci":"paired percentile 95%"},
        "results": results, "policy_summary_mean_sd_and_case_clustered_ci": summaries,
        "notes": ["K=1 action/state labels are single realized outcomes.",
                  "Dev cases are used for early stopping and are not a final test.",
                  "Oracle uses observed dev Q only as an offline upper bound and is not deployable.",
                  "Static L2D chooses ANSWER or DEFER once using training-set mean net advantage; it is not state-conditioned."]
    }
    # Prediction files preserve raw per-state labels locally; do not publish them.
    write_jsonl(args.output_dir / "dev_predictions.jsonl", prediction_rows)
    write_jsonl(args.output_dir / "q_error_diagnostics.jsonl", q_error_rows)
    json_dump(args.output_dir / "training_results.json", report)
    with (args.output_dir / "policy_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["feature","model","seed_count","accuracy_mean","accuracy_seed_sd","utility_mean","utility_seed_sd",
                  "ask_mean","answer_mean","defer_mean","accuracy_state_weighted_mean","accuracy_state_weighted_seed_sd",
                  "utility_state_weighted_mean","utility_state_weighted_seed_sd",
                  "ask_rate_state_weighted_mean","ask_rate_state_weighted_seed_sd",
                  "answer_rate_state_weighted_mean","answer_rate_state_weighted_seed_sd",
                  "defer_rate_state_weighted_mean","defer_rate_state_weighted_seed_sd",
                  "corrections_count","spoilage_count",
                  "correction_rate_given_base_wrong","spoilage_rate_given_base_correct",
                  "q_mae_train_mean","q_mae_dev_mean","relative_advantage_mae_train_mean","relative_advantage_mae_dev_mean",
                  "advantage_auc_ask_dev_mean","advantage_auc_defer_dev_mean",
                  "paired_vs_always_answer_delta_accuracy_ci95","paired_vs_always_answer_delta_utility_ci95"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for item in summaries:
            writer.writerow({k:(json.dumps(item[k]) if isinstance(item.get(k),list) else item.get(k)) for k in fields})
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["feature","model","seed","best_epoch","validation_loss","q_mae_train","q_mae_dev",
                  "relative_advantage_mae_train","relative_advantage_mae_dev","advantage_auc_ask_dev",
                  "advantage_auc_defer_dev","accuracy","utility","ask_rate","answer_rate","defer_rate",
                  "accuracy_state_weighted","utility_state_weighted","ask_rate_state_weighted",
                  "answer_rate_state_weighted","defer_rate_state_weighted",
                  "defer_calls_state_count","corrections_count","spoilage_count"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in results:
            m = row.get("dev_policy") or {}
            writer.writerow({"feature":row["feature"],"model":row["model"],"seed":row.get("seed"),
                             "best_epoch":row.get("best_epoch"),"validation_loss":row.get("validation_loss"),
                             "q_mae_train":row.get("q_mae_train"),"q_mae_dev":row.get("q_mae_dev"),
                             "relative_advantage_mae_train":row.get("relative_advantage_mae_train"),
                             "relative_advantage_mae_dev":row.get("relative_advantage_mae_dev"),
                             "advantage_auc_ask_dev":row.get("advantage_auc_ask_dev"),
                             "advantage_auc_defer_dev":row.get("advantage_auc_defer_dev"),
                             "accuracy":m.get("accuracy"),"utility":m.get("utility"),"ask_rate":m.get("ask"),
                             "answer_rate":m.get("answer"),"defer_rate":m.get("defer"),
                             "accuracy_state_weighted":m.get("accuracy_state_weighted"),
                             "utility_state_weighted":m.get("utility_state_weighted"),
                             "ask_rate_state_weighted":m.get("ask_rate_state_weighted"),
                             "answer_rate_state_weighted":m.get("answer_rate_state_weighted"),
                             "defer_rate_state_weighted":m.get("defer_rate_state_weighted"),
                             "defer_calls_state_count":m.get("defer_calls_state_count"),
                             "corrections_count":m.get("corrections_count"),"spoilage_count":m.get("spoilage_count")})
    print(json.dumps({"status":report["status"], "output_dir":str(args.output_dir.resolve()),
                      "feature_groups":feature_groups, "result_rows":len(results), "summary_rows":len(summaries),
                      "checkpoint_count":len(list((args.output_dir/"checkpoints").rglob("*.pt")))}, indent=2))


if __name__ == "__main__":
    main()
