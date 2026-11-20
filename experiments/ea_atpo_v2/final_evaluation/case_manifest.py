#!/usr/bin/env python3
"""Freeze a case-ID test list after case/history overlap and duplicate audits.

The source is read only to obtain row identity and public prompt fingerprints.
Gold and hidden facts are never used in selection or written to the manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


PUBLIC_ROLES = {"user", "assistant"}
HISTORICAL_SYSTEM = (
    "You are answering a medical multiple-choice benchmark question. Use only the case and options "
    "in the user message. Select the single best option. Do not ask follow-up questions. "
    "Output exactly: Final Answer: <one uppercase option letter A-D>. Do not include an explanation."
)
ID_KEYS = {"case_id", "case_ids", "selected_case_ids", "sampled_case_ids", "case_id_list"}
ID_HASH_KEYS = {"case_id_hash", "case_hash", "case_id_sha256"}
PROMPT_HASH_KEYS = {"prompt_hash", "prompt_sha256", "public_prompt_sha256"}
JSONL_ID_PATTERN = re.compile(
    r'"(?P<key>case_id|case_id_hash|case_hash|case_id_sha256|prompt_hash|prompt_sha256|public_prompt_sha256)"\s*:\s*"(?P<value>[^"]+)"'
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_prompt_hash(messages: list[dict[str, str]]) -> str:
    canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(canonical)


def source_case(row_index: int, row: dict[str, Any]) -> dict[str, Any]:
    """Return ID and prompt hashes using the existing V2 source-row mapping."""
    extra = row.get("extra_info") or {}
    if not isinstance(extra, dict):
        extra = {}
    item_index = extra.get("index", row.get("index", row_index))
    case_id = f"row-{row_index}:item-{item_index}"
    prompt = row.get("prompt") or extra.get("prompt")
    if not isinstance(prompt, list):
        raise ValueError(f"source row {row_index} has no public prompt list")

    public_messages = []
    legacy_messages = [{"role": "system", "content": HISTORICAL_SYSTEM}]
    for message in prompt:
        if not isinstance(message, dict):
            raise ValueError(f"source row {row_index} has a malformed public message")
        role = str(message.get("role", ""))
        if role == "system":
            continue
        if role not in PUBLIC_ROLES:
            raise ValueError(f"source row {row_index} has unsupported public role {role!r}")
        item = {"role": role, "content": str(message.get("content", ""))}
        public_messages.append(item)
        legacy_messages.append(item)
    if not public_messages:
        raise ValueError(f"source row {row_index} has no public user/assistant message")
    return {
        "case_id": case_id,
        "source_row_index": int(row_index),
        "source_item_index": str(item_index),
        "public_prompt_sha256": canonical_prompt_hash(public_messages),
        "historical_direct_eval_prompt_sha256": canonical_prompt_hash(legacy_messages),
    }


def iter_source_cases(path: Path) -> Iterable[dict[str, Any]]:
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for row_index, line in enumerate(stream):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"source JSONL line {row_index + 1} is invalid") from exc
            if not isinstance(row, dict):
                raise ValueError(f"source JSONL line {row_index + 1} is not an object")
            item = source_case(row_index, row)
            if item["case_id"] in seen:
                raise ValueError("source case_id mapping is not unique")
            seen.add(item["case_id"])
            yield item


@dataclass
class HistoryIndex:
    case_ids: set[str] = field(default_factory=set)
    case_id_hashes: set[str] = field(default_factory=set)
    prompt_hashes: set[str] = field(default_factory=set)
    sources: list[dict[str, Any]] = field(default_factory=list)
    source_case_ids: dict[str, set[str]] = field(default_factory=dict)
    source_case_id_hashes: dict[str, set[str]] = field(default_factory=dict)
    source_prompt_hashes: dict[str, set[str]] = field(default_factory=dict)

    def add(self, source_name: str, ids: set[str], id_hashes: set[str], prompt_hashes: set[str],
            source_sha256: str | None = None) -> None:
        self.case_ids.update(ids)
        self.case_id_hashes.update(id_hashes)
        self.prompt_hashes.update(prompt_hashes)
        self.source_case_ids[source_name] = set(ids)
        self.source_case_id_hashes[source_name] = set(id_hashes)
        self.source_prompt_hashes[source_name] = set(prompt_hashes)
        self.sources.append({
            "source": source_name,
            "case_id_count": len(ids),
            "case_id_hash_count": len(id_hashes),
            "prompt_hash_count": len(prompt_hashes),
            "sha256": source_sha256,
        })


def _collect_json_value(value: Any, key: str, ids: set[str], id_hashes: set[str], prompts: set[str]) -> None:
    normalized = key.lower()
    if normalized in ID_KEYS:
        if isinstance(value, str):
            ids.add(value)
        elif isinstance(value, list):
            ids.update(str(item) for item in value if isinstance(item, (str, int)))
    elif normalized in ID_HASH_KEYS and isinstance(value, str):
        id_hashes.add(value)
    elif normalized in PROMPT_HASH_KEYS and isinstance(value, str):
        prompts.add(value)
    if isinstance(value, dict):
        for child_key, child in value.items():
            _collect_json_value(child, str(child_key), ids, id_hashes, prompts)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, dict):
                _collect_json_value(child, "", ids, id_hashes, prompts)


def collect_history_file(path: Path, source_name: str | None = None) -> tuple[set[str], set[str], set[str]]:
    ids: set[str] = set()
    id_hashes: set[str] = set()
    prompt_hashes: set[str] = set()
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                for key, value in row.items():
                    if not value:
                        continue
                    normalized = (key or "").lower()
                    if normalized in ID_KEYS:
                        ids.add(value)
                    elif normalized in ID_HASH_KEYS:
                        id_hashes.add(value)
                    elif normalized in PROMPT_HASH_KEYS:
                        prompt_hashes.add(value)
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                for match in JSONL_ID_PATTERN.finditer(line):
                    key, value = match.group("key").lower(), match.group("value")
                    if key == "case_id":
                        ids.add(value)
                    elif key in ID_HASH_KEYS:
                        id_hashes.add(value)
                    elif key in PROMPT_HASH_KEYS:
                        prompt_hashes.add(value)
    elif path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list) and all(isinstance(item, (str, int)) for item in value):
            ids.update(str(item) for item in value)
        else:
            _collect_json_value(value, "", ids, id_hashes, prompt_hashes)
            if isinstance(value, dict) and value.get("unit") == "case_id":
                for split_name in ("train", "dev", "validation", "internal_test"):
                    split_ids = value.get(split_name)
                    if isinstance(split_ids, list):
                        ids.update(str(item) for item in split_ids if isinstance(item, (str, int)))
    return ids, id_hashes, prompt_hashes


def history_files(root: Path) -> list[Path]:
    skip_parts = {"services", "triton", "torch_compile_cache", "runtime", "pids", "tmp", "confirmation_20261120"}
    selected = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in skip_parts for part in rel_parts):
            continue
        name = path.name.lower()
        if ("case" in name or name.startswith("paired_") or
                name in {"run_manifest.json", "analysis_manifest.json", "expert_reliability_manifest.json"}):
            selected.append(path)
    return selected


def load_history(args: argparse.Namespace) -> HistoryIndex:
    history = HistoryIndex()
    explicit: list[tuple[str, Path]] = []
    if args.historical_case_split:
        explicit.append(("previous_100_case_split", args.historical_case_split))
    if args.current_router_split:
        explicit.append(("current_router_train_dev_split", args.current_router_split))
    if args.v2_recovery_bundle:
        explicit.extend((f"v2_recovery/{p.name}", p) for p in sorted(args.v2_recovery_bundle.glob("batch_*_case_ids.json")))
    if args.historical_direct_results:
        explicit.append(("historical_direct_200_results", args.historical_direct_results))
    seen_paths: set[Path] = set()
    for name, path in explicit:
        if not path.is_file():
            raise FileNotFoundError(path)
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        ids, id_hashes, prompts = collect_history_file(path)
        history.add(name, ids, id_hashes, prompts, sha256_file(path))
        seen_paths.add(resolved)
    for root_index, root in enumerate(args.historical_output_roots):
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in history_files(root):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            ids, id_hashes, prompts = collect_history_file(path)
            if ids or id_hashes or prompts:
                source_name = f"historical_root_{root_index}/{root.name}/{path.relative_to(root).as_posix()}"
                history.add(source_name, ids, id_hashes, prompts, sha256_file(path))
            seen_paths.add(resolved)
    return history


def build_case_bundle(source_path: Path, history: HistoryIndex, *, sample_size: int, selection_seed: str,
                      source_split: str = "official_train") -> tuple[dict[str, Any], dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    records = list(iter_source_cases(source_path))
    all_id_hashes = {sha256_text(item["case_id"]) for item in records}
    exact_overlap = {item["case_id"] for item in records} & history.case_ids
    id_hash_overlap = all_id_hashes & history.case_id_hashes
    candidate_prompt_hashes = {
        digest for item in records
        for digest in (item["public_prompt_sha256"], item["historical_direct_eval_prompt_sha256"])
    }
    prompt_overlap = candidate_prompt_hashes & history.prompt_hashes

    eligible = [item for item in records
                if item["case_id"] not in history.case_ids
                and sha256_text(item["case_id"]) not in history.case_id_hashes
                and item["public_prompt_sha256"] not in history.prompt_hashes
                and item["historical_direct_eval_prompt_sha256"] not in history.prompt_hashes]
    # Collapse exact duplicate public prompts before ranking, keeping the
    # candidate with the smallest stable selection key.
    by_prompt: dict[str, tuple[str, dict[str, Any]]] = {}
    for item in eligible:
        rank = sha256_text(f"{selection_seed}\0{item['case_id']}\0{item['public_prompt_sha256']}")
        previous = by_prompt.get(item["public_prompt_sha256"])
        if previous is None or rank < previous[0]:
            by_prompt[item["public_prompt_sha256"]] = (rank, item)
    ranked = sorted((rank, item) for rank, item in by_prompt.values())
    if len(ranked) < sample_size:
        raise ValueError(f"only {len(ranked)} eligible unique prompts; need {sample_size}")
    selected = [item for _, item in ranked[:sample_size]]
    selected_ids = [item["case_id"] for item in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise AssertionError("frozen test case IDs are not unique")
    selected_id_hashes = {sha256_text(case_id) for case_id in selected_ids}
    selected_prompt_hashes = {
        digest for item in selected
        for digest in (item["public_prompt_sha256"], item["historical_direct_eval_prompt_sha256"])
    }
    overlap_by_source = []
    for detail in history.sources:
        name = detail["source"]
        source_ids = history.source_case_ids.get(name, set())
        source_id_hashes = history.source_case_id_hashes.get(name, set())
        source_prompts = history.source_prompt_hashes.get(name, set())
        overlap_by_source.append({
            **detail,
            "candidate_exact_case_id_overlap": len({item["case_id"] for item in records} & source_ids),
            "selected_test_exact_case_id_overlap": len(set(selected_ids) & source_ids),
            "candidate_case_id_hash_overlap": len(all_id_hashes & source_id_hashes),
            "selected_test_case_id_hash_overlap": len(selected_id_hashes & source_id_hashes),
            "candidate_prompt_hash_overlap": len(candidate_prompt_hashes & source_prompts),
            "selected_test_prompt_hash_overlap": len(selected_prompt_hashes & source_prompts),
        })

    source_sha = sha256_file(source_path)
    case_payload = {
        "schema_version": 1,
        "manifest_id": "ea-atpo-v2-final-test-20261120-v1",
        "unit": "case_id",
        "status": "frozen_case_ids_pending_future_evaluation",
        "source": {
            "dataset_name": "merged_train_dataset.jsonl",
            "declared_split": source_split,
            "record_kind": "single_turn_source_cases; not multi_turn_trajectories",
            "source_sha256": source_sha,
            "source_record_count": len(records),
        },
        "selection": {
            "algorithm": "sort by SHA256(selection_seed + NUL + case_id + NUL + public_prompt_sha256), then take first N",
            "selection_seed": selection_seed,
            "test_case_count": sample_size,
            "deduplicate_exact_public_prompt": True,
            "excluded_historical_case_ids": True,
            "excluded_historical_case_id_hashes": True,
            "excluded_historical_prompt_hashes": True,
            "reserved_from_future_training": True,
        },
        "cases": selected,
        "case_id_list_sha256": sha256_text("\n".join(selected_ids) + "\n"),
    }
    audit = {
        "schema_version": 1,
        "manifest_id": case_payload["manifest_id"],
        "source_record_count": len(records),
        "source_unique_case_id_count": len({item["case_id"] for item in records}),
        "source_unique_public_prompt_count": len({item["public_prompt_sha256"] for item in records}),
        "historical_case_id_count_deduplicated": len(history.case_ids),
        "historical_case_id_hash_count_deduplicated": len(history.case_id_hashes),
        "historical_prompt_hash_count_deduplicated": len(history.prompt_hashes),
        "historical_exact_case_id_overlap_before_exclusion": len(exact_overlap),
        "historical_case_id_hash_overlap_before_exclusion": len(id_hash_overlap),
        "historical_prompt_hash_overlap_before_exclusion": len(prompt_overlap),
        "eligible_after_history_exclusions": len(eligible),
        "eligible_after_exact_public_prompt_deduplication": len(by_prompt),
        "frozen_test_case_count": len(selected),
        "test_overlap_with_historical_case_ids": 0,
        "test_overlap_with_historical_case_id_hashes": 0,
        "test_overlap_with_historical_prompt_hashes": 0,
        "historical_sources": overlap_by_source,
        "frozen_holdout_opened": False,
        "window_a_confirmation_outcomes_opened": False,
        "window_c_code_or_data_modified": False,
        "gold_or_hidden_facts_used_for_selection": False,
        "only_case_ids_and_public_prompt_hashes_written": True,
    }
    return case_payload, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--historical-case-split", type=Path, required=True)
    parser.add_argument("--current-router-split", type=Path)
    parser.add_argument("--v2-recovery-bundle", type=Path, required=True)
    parser.add_argument("--historical-direct-results", type=Path, required=True)
    parser.add_argument("--historical-output-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--selection-seed", default="ea-atpo-v2-final-test-20261120-v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.source_jsonl.is_file():
        raise FileNotFoundError(args.source_jsonl)
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    history = load_history(args)
    case_payload, audit = build_case_bundle(args.source_jsonl, history,
                                            sample_size=args.sample_size,
                                            selection_seed=args.selection_seed)
    audit["source_sha256"] = case_payload["source"]["source_sha256"]
    audit["case_id_list_sha256"] = case_payload["case_id_list_sha256"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "case_manifest.json").write_text(
        json.dumps(case_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "overlap_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "frozen_case_ids",
        "cases": len(case_payload["cases"]),
        "eligible_unique_prompts": audit["eligible_after_exact_public_prompt_deduplication"],
        "historical_case_id_overlap": audit["historical_exact_case_id_overlap_before_exclusion"],
        "historical_prompt_hash_overlap": audit["historical_prompt_hash_overlap_before_exclusion"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
