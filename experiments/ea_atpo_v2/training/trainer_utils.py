"""Shared case-safe feature preparation, optimization, and checkpoint helpers."""
from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .data import QExample, load_public_embedding_cache, split_cases_by_id
from .features import ExistingPublicStateEncoder, Standardizer, feature_matrix


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def prepare_feature_splits(train: Sequence[QExample], validation: Sequence[QExample], feature_file: Path | None,
                           max_features: int) -> tuple[np.ndarray, np.ndarray, ExistingPublicStateEncoder | None, Standardizer, str, dict[str, Any] | None]:
    cache = load_public_embedding_cache(feature_file, expected_examples=[*train, *validation]) if feature_file else None
    vectors = cache.vectors if cache else None
    train_raw, encoder = feature_matrix(train, external_vectors=vectors, encoder=None if vectors else ExistingPublicStateEncoder(max_features))
    val_raw, _ = feature_matrix(validation, external_vectors=vectors, encoder=encoder)
    scaler = Standardizer().fit(train_raw)
    feature_metadata = None
    if cache:
        feature_metadata = {**dict(cache.metadata), "cache_sha256": cache.sha256}
    return (scaler.transform(train_raw), scaler.transform(val_raw), encoder, scaler,
            ("external_public_state_embedding" if vectors else "audited_v1_public_state_tfidf"), feature_metadata)


def split_train_validation(rows: Sequence[QExample], *, validation_fraction: float, seed: int) -> tuple[list[QExample], list[QExample], set[str], set[str]]:
    train_ids, validation_ids = split_cases_by_id(rows, validation_fraction=validation_fraction, seed=seed)
    train = [row for row in rows if row.case_id in train_ids]
    validation = [row for row in rows if row.case_id in validation_ids]
    if {row.case_id for row in train} & {row.case_id for row in validation}:
        raise AssertionError("case_id leakage in internal train/validation split")
    if not any(row.valid_mask.any() for row in train):
        raise ValueError("training case partition has no valid Q targets")
    if not any(row.valid_mask.any() for row in validation):
        raise ValueError("validation case partition has no valid Q targets")
    return train, validation, train_ids, validation_ids


def stack_supervision(rows: Sequence[QExample]):
    target = np.stack([row.q_target for row in rows]).astype(np.float32)
    mask = np.stack([row.valid_mask for row in rows]).astype(bool)
    count = np.stack([row.sample_count for row in rows]).astype(np.float32)
    return (torch.from_numpy(target), torch.from_numpy(mask), torch.from_numpy(count))


def optimize(model: torch.nn.Module, train_x: np.ndarray, train_rows: Sequence[QExample],
             val_x: np.ndarray, val_rows: Sequence[QExample], *, epochs: int, patience: int, batch_size: int,
             learning_rate: float, weight_decay: float, loss_fn: Callable, seed: int) -> dict[str, Any]:
    tx = torch.from_numpy(np.asarray(train_x, dtype=np.float32))
    vx = torch.from_numpy(np.asarray(val_x, dtype=np.float32))
    train_y, train_mask, train_count = stack_supervision(train_rows)
    val_y, val_mask, val_count = stack_supervision(val_rows)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = rng.permutation(len(train_rows))
        losses = []
        for start in range(0, len(order), int(batch_size)):
            idx = torch.as_tensor(order[start:start + int(batch_size)], dtype=torch.long)
            prediction = model(tx[idx])
            loss = loss_fn(prediction, train_y[idx], train_mask[idx], train_count[idx])
            if loss is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
        model.eval()
        with torch.no_grad():
            val_prediction = model(vx)
            val_loss_tensor = loss_fn(val_prediction, val_y, val_mask, val_count)
            if val_loss_tensor is None:
                raise ValueError("validation split has no supervised/comparable rows")
            val_loss = float(val_loss_tensor.item())
        train_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        if not losses:
            raise ValueError("training epoch contained no comparable/supervised batches")
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("early stopping did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return {"best_epoch": best_epoch, "best_validation_loss": best_loss, "history": history}


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        torch.save(payload, temp_path)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != "ea_atpo_v2_training_checkpoint_v1":
        raise ValueError("unknown checkpoint format")
    return payload


def serialize_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
