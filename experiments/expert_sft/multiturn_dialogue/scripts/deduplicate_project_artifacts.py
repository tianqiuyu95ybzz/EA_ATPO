#!/usr/bin/env python3
"""Replace byte-identical reconstruction copies with links to one local copy.

Only named, hash-verified generated artifacts are touched. Model checkpoints,
runtime caches, unique outputs and source datasets are deliberately excluded.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SMOKE = ROOT / "experiments/ea_atpo_v1_smoke"
REPORT = ROOT / "experiments/expert_sft/multiturn_dialogue/data/dedup_manifest.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def replace_by_symlink(source: Path, target: Path, expected: str) -> None:
    if source.resolve() == target.resolve():
        return
    if source.is_symlink():
        if source.resolve() != target.resolve():
            raise RuntimeError(f"Existing symlink points elsewhere: {source}")
        return
    if not source.is_file() or sha(source) != expected:
        raise RuntimeError(f"Refusing to replace non-matching content: {source}")
    source.unlink()
    source.symlink_to(Path(__import__("os").path.relpath(target, source.parent)))


def record_exact_group(report: list[dict], label: str, paths: list[Path]) -> None:
    available = [path for path in paths if path.is_file()]
    if len(available) < 2:
        return
    hashes = {sha(path) for path in available}
    if len(hashes) != 1:
        raise RuntimeError(f"Expected exact duplicates for {label}, but hashes differ")
    digest = hashes.pop()
    canonical = next((path for path in available if not path.is_symlink()), available[0])
    for path in available:
        if path.resolve() != canonical.resolve():
            replace_by_symlink(path, canonical, digest)
    size = canonical.stat().st_size
    report.append({
        "name": label,
        "sha256": digest,
        "bytes": size,
        "original_copy_count": len(available),
        "bytes_saved": size * (len(available) - 1),
        "canonical": str(canonical.relative_to(ROOT)),
        "preserved_paths": [str(path.relative_to(ROOT)) for path in available],
    })


def main() -> None:
    previous = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    groups = defaultdict(list)
    for attempt_dir in sorted(SMOKE.iterdir()):
        data_dir = attempt_dir / "data"
        if not data_dir.is_dir():
            continue
        for name in ("train.parquet", "test.parquet"):
            path = data_dir / name
            if path.is_file():
                groups[(name, sha(path))].append(path)
    report = {"status": "completed", "policy": "exact_sha256_only; preserve every original path as a relative symlink", "groups": [], "named_identical_outputs": []}
    shared = SMOKE / "shared_data"
    for (name, digest), paths in sorted(groups.items()):
        if len(paths) < 2:
            continue
        shared.mkdir(parents=True, exist_ok=True)
        canonical = shared / f"{Path(name).stem}_{digest[:12]}.parquet"
        if canonical.exists():
            if sha(canonical) != digest:
                raise RuntimeError(f"Canonical data hash mismatch: {canonical}")
        else:
            shutil.copy2(paths[0], canonical)
            if sha(canonical) != digest:
                canonical.unlink(missing_ok=True)
                raise RuntimeError(f"Copy verification failed for {canonical}")
        links = []
        for path in paths:
            replace_by_symlink(path, canonical, digest)
            links.append(str(path.relative_to(ROOT)))
        report["groups"].append({"name": name, "sha256": digest, "bytes": canonical.stat().st_size, "bytes_saved": canonical.stat().st_size * (len(paths) - 1), "original_copy_count": len(paths), "canonical": str(canonical.relative_to(ROOT)), "preserved_paths": links})

    duplicates = [
        (ROOT / "experiments/expert_sft/data/fresh_test_selection_prompt_only_verify.json", ROOT / "experiments/expert_sft/data/fresh_test_selection.json"),
        (ROOT / "experiments/vi_ea_atpo/phase2/smoke_recovery_v2/paired_intervention_metrics.csv", ROOT / "experiments/vi_ea_atpo/phase2/paired_intervention_metrics.csv"),
        (ROOT / "experiments/vi_ea_atpo/phase2/smoke_recovery_v2/voi_pilot_analysis.csv", ROOT / "experiments/vi_ea_atpo/phase2/voi_pilot_analysis.csv"),
    ]
    for duplicate, canonical in duplicates:
        if not duplicate.is_file() or not canonical.is_file():
            continue
        digest = sha(canonical)
        if sha(duplicate) != digest:
            raise RuntimeError(f"Expected byte-identical duplicate differs: {duplicate}")
        replace_by_symlink(duplicate, canonical, digest)
        record_exact_group(report["named_identical_outputs"], str(duplicate.relative_to(ROOT)), [canonical, duplicate])

    attempt09 = SMOKE / "ea_atpo_v1_attempt09_stability/recovery"
    for filename in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "added_tokens.json", "special_tokens_map.json"):
        record_exact_group(report["named_identical_outputs"], f"Attempt09 shared {filename}", [
            attempt09 / "critic_source/huggingface" / filename,
            attempt09 / "hf/attempt09_actor" / filename,
            attempt09 / "hf/attempt09_critic" / filename,
            attempt09 / "hf/original_atpo_attempt20_actor" / filename,
        ])
    for filename in ("config.json", "generation_config.json"):
        record_exact_group(report["named_identical_outputs"], f"Attempt09 shared actor {filename}", [
            attempt09 / "hf/attempt09_actor" / filename,
            attempt09 / "hf/original_atpo_attempt20_actor" / filename,
        ])

    record_exact_group(report["named_identical_outputs"], "ModelScope package-cache duplicate", [
        ROOT / ".task_cache/model_scope/wheels/modelscope-1.40.3-py3-none-any.whl",
        ROOT / ".task_cache/model_scope/pip-cache/http-v2/1/9/1/2/1/19121cedef416dee5e6f5891dcd1b24e8dcbd0ee4fd5c60c07d536d1.body",
    ])

    report["not_removed"] = [
        "All unique model weights and checkpoints; sampled first 8 MiB of Attempt06, Attempt07 and Attempt09 optimizer shards differs.",
        "Runtime CUDA/Triton caches because existing model services may still use them.",
        "Historical reports and outputs with distinct names, even when some payloads match.",
    ]
    for section in ("groups", "named_identical_outputs"):
        current_keys = {(item.get("name"), item.get("sha256"), item.get("canonical")) for item in report[section]}
        for item in previous.get(section, []):
            key = (item.get("name"), item.get("sha256"), item.get("canonical"))
            if key not in current_keys:
                report[section].append(item)
                current_keys.add(key)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bytes_saved = sum(item["bytes_saved"] for item in report["groups"] + report["named_identical_outputs"])
    print(json.dumps({"status": report["status"], "parquet_groups": len(report["groups"]), "named_duplicates": len(report["named_identical_outputs"]), "estimated_bytes_saved": bytes_saved, "manifest": str(REPORT.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
