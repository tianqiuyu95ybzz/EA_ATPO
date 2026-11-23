#!/usr/bin/env python3
"""Inventory Git-untracked files without changing their source worktrees.

The detailed CSV can contain local path names, so write it only to controlled
PFS storage. The script prints aggregate counts, never file contents.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import stat
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PII_KEYS = {
    "patient_name", "first_name", "last_name", "email", "phone",
    "telephone", "address", "street", "mrn", "ssn", "dob",
    "date_of_birth", "birth_date", "medical_record_number", "insurance_id",
    "social_security_number",
}
GOLD_KEYS = {"gold", "gold_option", "gold_answer", "gold_rationale", "ground_truth"}
STRUCTURED_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv"}
TEXT_PATTERNS = {
    "email_matches": re.compile(rb"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    "structured_phone_matches": re.compile(
        rb"(?<!\d)(?:\+?\d{1,3}[ .-])?(?:\(\d{2,4}\)|\d{2,4})"
        rb"[ .-]\d{3,4}[ .-]\d{3,4}(?!\d)"
    ),
    "ssn_matches": re.compile(rb"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "iso_date_matches": re.compile(
        rb"(?<!\d)(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?!\d)"
    ),
}
CACHE_PARTS = {
    "__pycache__", ".cache", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "torchinductor", "triton", "torch_extensions", "cache", "caches", "parallel_cache",
    "kernel_cache", "compile_cache", "build_cache",
}
CACHE_SUFFIXES = {
    ".pyc", ".pyo", ".cubin", ".ptx", ".llir", ".ttir", ".ttgir",
    ".best_config", ".source", ".so.cache",
}
TEMP_SUFFIXES = {".tmp", ".temp", ".swp", ".swo", ".pid", ".part", ".partial"}
CHECKPOINT_SUFFIXES = {".npz", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx"}
CODE_SUFFIXES = {".py", ".sh", ".bash", ".js", ".ts", ".tsx", ".cpp", ".cu", ".h", ".hpp", ".jinja"}
DATA_SUFFIXES = {".jsonl", ".parquet", ".arrow", ".csv", ".tsv", ".xlsx", ".pkl", ".pickle", ".feather"}
CONFIG_SUFFIXES = {".yaml", ".yml", ".toml", ".ini", ".cfg"}
LOG_SUFFIXES = {".log", ".out", ".err"}


def run_git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True
    ).stdout


def untracked_paths(root: Path) -> list[str]:
    data = run_git(root, "status", "--porcelain", "-z", "--untracked-files=all")
    result = []
    for entry in data.split(b"\0"):
        if entry.startswith(b"?? "):
            result.append(os.fsdecode(entry[3:]))
    return result


def sha256_file(path: Path) -> tuple[str, int, str]:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        target = os.fsencode(os.readlink(path))
        return hashlib.sha256(target).hexdigest(), info.st_size, "symlink_text"
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size, "file_bytes"


def category(rel: str) -> str:
    normalized = rel.lower().replace("\\", "/")
    parts = set(normalized.split("/"))
    path = Path(normalized)
    suffix = path.suffix.lower()
    if parts & CACHE_PARTS or suffix in CACHE_SUFFIXES:
        return "cache"
    if suffix == ".so" and parts & {"cache", "caches", ".cache", "torch_extensions", "runtime"}:
        return "cache"
    if suffix in TEMP_SUFFIXES or path.name.endswith("~") or path.name.startswith(".#"):
        return "temporary"
    if suffix in CHECKPOINT_SUFFIXES or any(part in {"checkpoint", "checkpoints", "adapter", "model_weights"} for part in parts):
        return "checkpoint"
    if suffix in LOG_SUFFIXES or "logs" in parts or "log" in parts:
        return "logs"
    if suffix in CODE_SUFFIXES:
        return "code"
    if suffix in DATA_SUFFIXES:
        return "data"
    if suffix in CONFIG_SUFFIXES or (suffix == ".json" and any(word in path.name for word in ("config", "manifest", "lock", "split", "registry"))):
        return "config"
    if suffix in {".md", ".rst", ".pdf", ".html"}:
        return "documentation"
    if suffix == ".txt":
        if any(word in path.name for word in ("log", "stdout", "stderr", "trace", "output")) or parts & {"logs", "log", "traces"}:
            return "logs"
        return "documentation"
    if suffix == ".json":
        return "data"
    return "other"


def collect_flags(path: Path, rel: str) -> dict[str, Any]:
    suffix = Path(rel).suffix.lower()
    flags: dict[str, Any] = {
        "content_scan": "not_applicable",
        "contains_gold_key": "unknown",
        "contains_pii_named_key": "unknown",
        **{name: "unknown" for name in TEXT_PATTERNS},
    }
    if suffix not in STRUCTURED_SUFFIXES or path.is_symlink():
        if suffix in {".parquet", ".arrow", ".pkl", ".pickle"}:
            flags["content_scan"] = "binary_not_scanned"
        return flags
    try:
        raw = path.read_bytes()
    except OSError:
        flags["content_scan"] = "read_error"
        return flags
    for name, pattern in TEXT_PATTERNS.items():
        flags[name] = len(pattern.findall(raw))
    gold_found = False
    pii_found = False
    if suffix in {".json", ".jsonl"}:
        rows = raw.splitlines() if suffix == ".jsonl" else [raw]
        parsed_any = False
        for line in rows:
            try:
                obj = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            parsed_any = True
            stack = [obj]
            while stack:
                current = stack.pop()
                if isinstance(current, dict):
                    for key, value in current.items():
                        key_lower = str(key).lower()
                        gold_found |= key_lower in GOLD_KEYS
                        pii_found |= key_lower in PII_KEYS
                        stack.append(value)
                elif isinstance(current, list):
                    stack.extend(current)
        flags["content_scan"] = "structured_text" if parsed_any else "parse_failed"
    else:
        try:
            header = next(csv.reader(raw.decode("utf-8", errors="replace").splitlines()))
            header_lower = {str(key).strip().lower() for key in header}
            gold_found = bool(header_lower & GOLD_KEYS)
            pii_found = bool(header_lower & PII_KEYS)
            flags["content_scan"] = "structured_text"
        except (StopIteration, csv.Error):
            flags["content_scan"] = "parse_failed"
    flags["contains_gold_key"] = gold_found
    flags["contains_pii_named_key"] = pii_found
    return flags


def parse_repo_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("repo must be LABEL=/absolute/path")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).resolve(strict=True)
    if not label:
        raise argparse.ArgumentTypeError("repo label must not be empty")
    return label, path


def load_archive_hashes(path: Path | None) -> dict[tuple[str, int], list[str]]:
    if path is None:
        return {}
    found: dict[tuple[str, int], list[str]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            found[(row["sha256"], int(row["bytes"]))].append(row["archive"])
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", action="append", type=parse_repo_arg, required=True)
    parser.add_argument("--archive-inventory", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        output.chmod(0o700)
    except OSError:
        pass
    archive_hashes = load_archive_hashes(args.archive_inventory)
    records: list[dict[str, Any]] = []

    for label, root in args.repo:
        branch = run_git(root, "branch", "--show-current").decode().strip()
        head = run_git(root, "rev-parse", "HEAD").decode().strip()
        for rel in untracked_paths(root):
            path = root / rel
            try:
                before = path.lstat()
                digest, size, hash_mode = sha256_file(path)
                after = path.lstat()
                stable = (before.st_size, before.st_mtime_ns, before.st_mode) == (
                    after.st_size, after.st_mtime_ns, after.st_mode
                )
                flags = collect_flags(path, rel)
                archive_matches = archive_hashes.get((digest, size), [])
                records.append({
                    "repo_label": label,
                    "branch": branch,
                    "head": head,
                    "relative_path": rel,
                    "bytes": size,
                    "sha256": digest,
                    "sha_mode": hash_mode,
                    "category": category(rel),
                    "extension": Path(rel).suffix.lower(),
                    "is_symlink": stat.S_ISLNK(before.st_mode),
                    "stable_during_scan": stable,
                    "pfs_archive_match": bool(archive_matches),
                    "matched_archive_names": ";".join(sorted(set(archive_matches))),
                    **flags,
                })
            except FileNotFoundError:
                records.append({
                    "repo_label": label, "branch": branch, "head": head,
                    "relative_path": rel, "bytes": "", "sha256": "",
                    "sha_mode": "missing_during_scan", "category": category(rel),
                    "extension": Path(rel).suffix.lower(), "is_symlink": False,
                    "stable_during_scan": False, "pfs_archive_match": False,
                    "matched_archive_names": "", "content_scan": "missing_during_scan",
                    "contains_gold_key": "unknown", "contains_pii_named_key": "unknown",
                    **{name: "unknown" for name in TEXT_PATTERNS},
                })

    by_hash: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record["sha256"]:
            by_hash[record["sha256"]].append(index)
    for record in records:
        record["same_content_local_copies"] = len(by_hash[record["sha256"]]) if record["sha256"] else 0
        direct_identifier_signal = record.get("contains_pii_named_key") is True or any(
            isinstance(record.get(name), int) and record[name] > 0
            for name in ("email_matches", "structured_phone_matches", "ssn_matches")
        )
        gold_signal = record.get("contains_gold_key") is True
        if direct_identifier_signal or gold_signal:
            record["risk_class"] = "sensitive_candidate_hold_for_review"
            record["archive_recommendation"] = "do_not_copy_until_authorized_privacy_review"
        elif not record["pfs_archive_match"] and record["category"] not in {"cache", "temporary"}:
            record["risk_class"] = "private_or_unknown"
            record["archive_recommendation"] = "review_for_private_archive"
        elif not record["pfs_archive_match"]:
            record["risk_class"] = "regenerable_runtime_artifact"
            record["archive_recommendation"] = "regenerable_or_temporarily_retain"
        else:
            record["risk_class"] = "existing_archive_match"
            record["archive_recommendation"] = "sha256_match_in_existing_archive"

    csv_path = output / "local_untracked_inventory.csv"
    fields = list(records[0].keys()) if records else []
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    csv_path.chmod(0o600)

    missing_path = output / "missing_or_unmatched_assets.csv"
    missing = [record for record in records if not record["pfs_archive_match"]]
    with missing_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(missing)
    missing_path.chmod(0o600)

    category_counts = Counter(record["category"] for record in records)
    category_bytes: Counter[str] = Counter()
    for record in records:
        if isinstance(record["bytes"], int):
            category_bytes[record["category"]] += record["bytes"]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repositories": [
            {"label": label, "path": str(root), "branch": run_git(root, "branch", "--show-current").decode().strip(), "head": run_git(root, "rev-parse", "HEAD").decode().strip()}
            for label, root in args.repo
        ],
        "untracked_file_count": len(records),
        "untracked_bytes": sum(category_bytes.values()),
        "pfs_archive_hash_matches": sum(bool(record["pfs_archive_match"]) for record in records),
        "stable_during_scan": sum(record["stable_during_scan"] is True for record in records),
        "changed_or_missing_during_scan": sum(record["stable_during_scan"] is False for record in records),
        "contains_gold_key_files": sum(record["contains_gold_key"] is True for record in records),
        "contains_pii_named_key_files": sum(record["contains_pii_named_key"] is True for record in records),
        "category_counts": dict(sorted(category_counts.items())),
        "category_bytes": dict(sorted(category_bytes.items())),
        "files_with_strict_identifier_pattern_matches": {
            key: sum(isinstance(record.get(key), int) and record[key] > 0 for record in records)
            for key in TEXT_PATTERNS
        },
        "archive_matches_are_sha256_and_size_matches_against_supplied_inventory": True,
        "source_worktrees_modified": False,
    }
    summary_path = output / "local_untracked_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    summary_path.chmod(0o600)
    print(json.dumps({
        "untracked_file_count": summary["untracked_file_count"],
        "untracked_bytes": summary["untracked_bytes"],
        "pfs_archive_hash_matches": summary["pfs_archive_hash_matches"],
        "stable_during_scan": summary["stable_during_scan"],
        "changed_or_missing_during_scan": summary["changed_or_missing_during_scan"],
        "category_counts": summary["category_counts"],
        "category_bytes": summary["category_bytes"],
        "inventory_path": str(csv_path),
        "summary_path": str(summary_path),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
