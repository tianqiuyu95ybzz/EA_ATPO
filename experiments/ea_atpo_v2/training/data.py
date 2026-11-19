"""Strict, case-aware adapters for the V2 Q-dataset contract."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import ACTION_NAMES


@dataclass(frozen=True)
class QExample:
    case_id: str
    state_id: str
    public_state: tuple[dict[str, str], ...]
    q_target: np.ndarray
    valid_mask: np.ndarray
    action_cost: np.ndarray
    sample_count: np.ndarray
    attempted_sample_count: np.ndarray
    return_std: np.ndarray
    costs_included_in_targets: bool
    raw_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PublicEmbeddingCache:
    vectors: dict[tuple[str, str], np.ndarray]
    metadata: Mapping[str, Any]
    sha256: str


def _action_vector(value: Any, *, name: str, default: float | None = None, dtype=float) -> np.ndarray:
    if isinstance(value, Mapping):
        values = [value.get(action, value.get(f"Q_{action}", default)) for action in ACTION_NAMES]
    elif isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != len(ACTION_NAMES):
            raise ValueError(f"{name} must contain {len(ACTION_NAMES)} actions in {ACTION_NAMES} order")
        values = list(value)
    elif value is None:
        values = [default] * len(ACTION_NAMES)
    else:
        values = [value] * len(ACTION_NAMES)
    if dtype is bool:
        if any(not isinstance(x, (bool, np.bool_)) for x in values):
            raise ValueError(f"{name} action values must be JSON booleans")
        return np.asarray([bool(x) for x in values], dtype=bool)
    out = np.asarray([np.nan if x is None else float(x) for x in values], dtype=np.float64)
    finite = np.isfinite(out)
    if name in {"action_cost", "sample_count"} and not finite.all():
        raise ValueError(f"{name} contains a missing or non-finite value")
    if name == "sample_count" and np.any(out < 0):
        raise ValueError("sample_count cannot be negative")
    if name == "action_cost" and np.any(out < 0):
        raise ValueError("action_cost cannot be negative")
    return out


def q_example_from_dict(row: Mapping[str, Any], *, default_action_costs: Mapping[str, float] | None = None) -> QExample:
    metadata = row.get("metadata") or {}
    case_id = str(row.get("case_id", metadata.get("case_id", "")))
    state_id = str(row.get("state_id", metadata.get("state_id", "")))
    if not case_id or not state_id:
        raise ValueError("Q row requires case_id and state_id")
    public_state = row.get("public_state", metadata.get("public_state"))
    if not isinstance(public_state, list) or not public_state:
        raise ValueError(f"{state_id}: public_state must be a non-empty message list")
    visible = []
    for message in public_state:
        role = str(message.get("role", ""))
        if role == "system" or role not in {"user", "assistant"}:
            raise ValueError(f"{state_id}: features may use only visible user/assistant messages")
        visible.append({"role": role, "content": str(message.get("content", ""))})

    q_source = row.get("Q_target", row)
    if isinstance(q_source, Mapping):
        q_values = [q_source.get(f"Q_{action}", q_source.get(action)) for action in ACTION_NAMES]
    elif isinstance(q_source, (list, tuple)):
        if len(q_source) != len(ACTION_NAMES):
            raise ValueError("Q_target list must follow ASK, ANSWER, DEFER order")
        q_values = list(q_source)
    else:
        raise ValueError("Q_target must be a mapping or an action-ordered list")
    q_target = np.asarray([np.nan if value is None else float(value) for value in q_values], dtype=np.float64)
    explicit_mask = row.get("valid_mask")
    if explicit_mask is None:
        valid_mask = np.isfinite(q_target)
    else:
        valid_mask = _action_vector(explicit_mask, name="valid_mask", default=False, dtype=bool)
    if np.any(valid_mask & ~np.isfinite(q_target)):
        raise ValueError(f"{state_id}: valid action target is missing/non-finite")
    # A finite target with a false mask remains masked; it is never made valid implicitly.

    std_values = row.get("return_std")
    return_std = _action_vector(std_values, name="return_std", default=None)
    if np.any(np.isfinite(return_std) & (return_std < 0)):
        raise ValueError("return_std cannot be negative")
    count_values = row.get("sample_count", row.get("valid_sample_count", 1))
    sample_count = _action_vector(count_values, name="sample_count", default=1.0)
    if np.any(valid_mask & (sample_count < 1)):
        raise ValueError(f"{state_id}: valid actions require sample_count >= 1")
    attempted_count = _action_vector(row.get("attempted_sample_count", count_values), name="attempted_sample_count", default=1.0)
    if np.any(attempted_count < sample_count):
        raise ValueError(f"{state_id}: attempted_sample_count cannot be less than valid sample_count")
    cost_values = row.get("action_cost")
    if cost_values is None:
        cost_values = (default_action_costs or {}).get("per_action")
    if cost_values is None:
        cost_values = {"ASK": 0.05, "ANSWER": 0.0, "DEFER": 0.60}
    action_cost = _action_vector(cost_values, name="action_cost", default=0.0)
    raw_cost_flag = row.get("costs_included_in_targets", metadata.get("costs_included_in_targets", False))
    if not isinstance(raw_cost_flag, bool):
        raise ValueError(f"{state_id}: costs_included_in_targets must be a boolean")
    costs_included = raw_cost_flag
    if costs_included and metadata.get("costs_included_in_targets", row.get("costs_included_in_targets")) is not True:
        raise ValueError(f"{state_id}: inconsistent costs_included_in_targets metadata")
    return QExample(
        case_id=case_id,
        state_id=state_id,
        public_state=tuple(visible),
        q_target=q_target,
        valid_mask=valid_mask,
        action_cost=action_cost,
        sample_count=sample_count,
        attempted_sample_count=attempted_count,
        return_std=return_std,
        costs_included_in_targets=costs_included,
        raw_metadata={**dict(metadata), "synthetic_data": bool(row.get("synthetic_data", metadata.get("synthetic_data", False)))},
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def public_state_hash(public_state: Sequence[Mapping[str, str]]) -> str:
    """Stable content hash for the visible messages used by an embedding row."""
    canonical = json.dumps(
        [{"role": str(message["role"]), "content": str(message["content"])} for message in public_state],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_q_dataset(path: Path, *, default_action_costs: Mapping[str, float] | None = None) -> list[QExample]:
    rows = [q_example_from_dict(row, default_action_costs=default_action_costs) for row in read_jsonl(path)]
    keys = [(row.case_id, row.state_id) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate (case_id, state_id) in {path}")
    if not rows:
        raise ValueError(f"empty Q dataset: {path}")
    if not any(row.valid_mask.any() for row in rows):
        raise ValueError(f"no valid Q targets in {path}")
    return rows


def load_public_embedding_cache(
    path: Path,
    *,
    expected_boundary: str = "public_state_before_action_only",
    expected_examples: Sequence[QExample] | None = None,
    expected_config_hash: str | None = None,
) -> PublicEmbeddingCache:
    """Load a versioned embedding cache and bind every vector to its public state."""
    vectors: dict[tuple[str, str], np.ndarray] = {}
    dimension = None
    cache_metadata: dict[str, Any] | None = None
    metadata_keys = (
        "model_id", "model_revision", "pooling", "max_length",
        "prompt_template_hash", "embedding_config_hash",
    )
    allowed_keys = {
        "case_id", "state_id", "feature_boundary", "embedding", "state_hash", "rendered_prompt_hash",
        *metadata_keys,
    }
    for line_number, row in enumerate(read_jsonl(path), 1):
        extras = set(map(str, row)) - allowed_keys
        if extras:
            raise ValueError(f"{path}:{line_number}: unsupported feature-row fields: {sorted(extras)}")
        boundary = row.get("feature_boundary")
        if boundary != expected_boundary:
            raise ValueError(f"{path}:{line_number}: feature_boundary must be {expected_boundary!r}")
        key = (str(row.get("case_id", "")), str(row.get("state_id", "")))
        vector = row.get("embedding")
        if not all(key) or not isinstance(vector, list) or not vector:
            raise ValueError(f"{path}:{line_number}: expected case_id, state_id, and a numeric embedding list")
        row_metadata = {name: row.get(name) for name in metadata_keys}
        if any(value in (None, "") for value in row_metadata.values()):
            raise ValueError(f"{path}:{line_number}: embedding cache lacks immutable model/config metadata")
        if not isinstance(row_metadata["max_length"], int) or row_metadata["max_length"] <= 0:
            raise ValueError(f"{path}:{line_number}: max_length must be a positive integer")
        for name in ("prompt_template_hash", "embedding_config_hash"):
            value = str(row_metadata[name])
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
                raise ValueError(f"{path}:{line_number}: {name} must be a SHA-256 hex digest")
        if cache_metadata is None:
            cache_metadata = row_metadata
        elif cache_metadata != row_metadata:
            raise ValueError(f"{path}:{line_number}: embedding configuration changes within one cache")
        state_hash = str(row.get("state_hash", ""))
        if len(state_hash) != 64 or any(char not in "0123456789abcdef" for char in state_hash.lower()):
            raise ValueError(f"{path}:{line_number}: state_hash must be a SHA-256 hex digest")
        prompt_digest = str(row.get("rendered_prompt_hash", ""))
        if len(prompt_digest) != 64 or any(char not in "0123456789abcdef" for char in prompt_digest.lower()):
            raise ValueError(f"{path}:{line_number}: rendered_prompt_hash must be a SHA-256 hex digest")
        values = np.asarray(vector, dtype=np.float32)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"{path}:{line_number}: feature vector must be finite and one-dimensional")
        dimension = int(values.size) if dimension is None else dimension
        if values.size != dimension:
            raise ValueError(f"{path}:{line_number}: inconsistent feature dimensions")
        if key in vectors:
            raise ValueError(f"{path}:{line_number}: duplicate feature state {key}")
        vectors[key] = values
    if not vectors:
        raise ValueError(f"empty feature file: {path}")
    cache_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"embedding cache manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "ea_atpo_v2_public_embedding_cache_v1":
        raise ValueError("unknown public embedding cache manifest format")
    if manifest.get("output_sha256") != cache_sha256:
        raise ValueError("public embedding cache SHA-256 does not match its manifest")
    if manifest.get("embedding_config_hash") != cache_metadata["embedding_config_hash"]:
        raise ValueError("embedding cache rows and manifest have different configuration hashes")
    if int(manifest.get("state_count", -1)) != len(vectors):
        raise ValueError("embedding cache row count does not match its manifest")
    if expected_config_hash is not None and cache_metadata["embedding_config_hash"] != expected_config_hash:
        raise ValueError("public embedding cache configuration does not match the checkpoint")
    if expected_examples is not None:
        expected = {(row.case_id, row.state_id): public_state_hash(row.public_state) for row in expected_examples}
        missing = set(expected) - set(vectors)
        if missing:
            raise ValueError(f"missing public-state embeddings for {len(missing)} states; first={sorted(missing)[0]}")
        row_hashes = {}
        for row in read_jsonl(path):
            row_hashes[(str(row["case_id"]), str(row["state_id"]))] = str(row["state_hash"])
        mismatched = [key for key, digest in expected.items() if row_hashes.get(key) != digest]
        if mismatched:
            raise ValueError(f"public embedding state hash mismatch for {len(mismatched)} states; first={mismatched[0]}")
    metadata = {
        **cache_metadata,
        "cache_sha256": cache_sha256,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "manifest_path": str(manifest_path.resolve()),
    }
    return PublicEmbeddingCache(vectors=vectors, metadata=metadata, sha256=cache_sha256)


def load_public_embeddings(path: Path, *, expected_boundary: str = "public_state_before_action_only",
                           expected_examples: Sequence[QExample] | None = None) -> dict[tuple[str, str], np.ndarray]:
    """Compatibility wrapper returning vectors from a verified public embedding cache."""
    return load_public_embedding_cache(path, expected_boundary=expected_boundary,
                                       expected_examples=expected_examples).vectors


def require_single_continuation_policy(rows: Iterable[QExample]) -> str:
    """Reject Q targets collected under multiple/unidentified continuation policies."""
    policy_ids = {str(row.raw_metadata.get("continuation_policy_id") or "") for row in rows}
    if "" in policy_ids or len(policy_ids) != 1:
        raise ValueError(f"Q targets require one non-empty continuation_policy_id; found {sorted(policy_ids)}")
    return next(iter(policy_ids))


def q_sampling_summary(rows: Iterable[QExample]) -> dict[str, Any]:
    counts = [int(row.sample_count[index]) for row in rows for index in range(len(ACTION_NAMES)) if row.valid_mask[index]]
    distribution = Counter(counts)
    return {
        "valid_action_target_count": len(counts),
        "valid_rollout_sample_count": sum(counts),
        "rollouts_per_state_action_K_distribution": {str(k): distribution[k] for k in sorted(distribution)},
        "all_valid_targets_are_K1": bool(counts) and set(counts) == {1},
        "return_std_missing_valid_target_count": sum(
            not np.isfinite(row.return_std[index])
            for row in rows for index in range(len(ACTION_NAMES)) if row.valid_mask[index]
        ),
    }


def require_minimum_q_samples(rows: Iterable[QExample], minimum_k: int) -> None:
    if minimum_k < 1:
        raise ValueError("minimum K must be positive")
    short = [
        (row.case_id, row.state_id, ACTION_NAMES[index], int(row.sample_count[index]))
        for row in rows for index in range(len(ACTION_NAMES))
        if row.valid_mask[index] and row.sample_count[index] < minimum_k
    ]
    if short:
        first = short[0]
        raise ValueError(
            f"formal training requires at least K={minimum_k} valid rollouts per observed state/action; "
            f"{len(short)} targets are below the threshold (first={first})"
        )


def validate_q_audit_report(report_path: Path, q_dataset_path: Path) -> dict[str, Any]:
    """Require the latest read-only audit to cover the exact Q JSONL being trained."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(q_dataset_path.read_bytes()).hexdigest()
    if report.get("status") != "passed" or report.get("scope") != "read_only_real_data_integrity_audit":
        raise ValueError("formal training requires a passed read-only Q dataset audit report")
    if (report.get("q_dataset") or {}).get("sha256") != digest:
        raise ValueError("Q dataset changed after the audit report was produced")
    required_true = (
        "all_cases_in_official_train",
        "costs_included_in_q_targets",
        "q_target_matches_mean_valid_raw_return",
        "mask_and_valid_sample_count_match_raw_rollouts",
        "attempted_sample_count_matches_raw_rollouts",
    )
    if any(report.get(key) is not True for key in required_true):
        raise ValueError("Q audit report does not certify train split, masks, or cost accounting")
    if report.get("official_validation_or_internal_test_used") is not False or report.get("frozen_holdout_opened") is not False:
        raise ValueError("Q audit report indicates protected evaluation cases were used")
    if report.get("continuation_policy_mixed_across_dataset") is not False:
        raise ValueError("Q audit report indicates mixed/unknown continuation_policy_id values")
    provenance = report.get("real_run_provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("models"), dict):
        raise ValueError("formal training audit must pin the Base/Expert model revisions from run_manifest")
    if provenance.get("continuation_policy_id") not in (report.get("continuation_policy_ids") or []):
        raise ValueError("run_manifest continuation policy differs from the Q data")
    for role in ("base", "expert"):
        identity = provenance["models"].get(role) or {}
        if not identity.get("model") or not identity.get("model_revision"):
            raise ValueError(f"formal training audit lacks pinned {role} model identity")
    return {
        "report_path": str(report_path.resolve()),
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "q_dataset_sha256": digest,
        "case_count": report.get("case_count"),
        "state_count": report.get("state_count"),
        "valid_action_target_count": report.get("valid_action_target_count"),
        "continuation_policy_ids": report.get("continuation_policy_ids"),
        "real_run_provenance": provenance,
        "rollouts_per_state_action_K_distribution": report.get("rollouts_per_state_action_K_distribution"),
        "return_std_missing_valid_action_count": report.get("return_std_missing_valid_action_count"),
    }


def split_cases_by_id(rows: Iterable[QExample], *, validation_fraction: float = 0.20, seed: int = 20261120) -> tuple[set[str], set[str]]:
    """Deterministically split cases, keeping every state of a case in one side."""
    cases = sorted({row.case_id for row in rows})
    if len(cases) < 2:
        raise ValueError("at least two distinct case_id values are required for train/validation")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    ordered = sorted(cases, key=lambda case: hashlib.sha256(f"{seed}:{case}".encode()).hexdigest())
    n_val = min(len(cases) - 1, max(1, round(len(cases) * validation_fraction)))
    validation_ids = set(ordered[:n_val])
    train_ids = set(cases) - validation_ids
    if train_ids & validation_ids:
        raise AssertionError("internal case split overlap")
    return train_ids, validation_ids


def validate_official_train_cases(rows: Iterable[QExample], case_split_path: Path) -> dict[str, Any]:
    split = json.loads(case_split_path.read_text(encoding="utf-8"))
    if split.get("unit") != "case_id":
        raise ValueError("official case split must declare unit='case_id'")
    allowed = set(map(str, split["train"]))
    val = set(map(str, split.get("validation", [])))
    test = set(map(str, split.get("internal_test", [])))
    seen = {row.case_id for row in rows}
    if not seen or not seen.issubset(allowed):
        raise ValueError(f"Q dataset includes non-train cases: {sorted(seen - allowed)[:5]}")
    if seen & (val | test):
        raise ValueError("validation/internal-test case found in Q training source")
    return {
        "official_split_unit": "case_id",
        "official_train_cases_present": sorted(seen),
        "official_train_case_count": len(allowed),
        "official_train_pool_complete": seen == allowed,
        "heldout_cases_opened": False,
    }
