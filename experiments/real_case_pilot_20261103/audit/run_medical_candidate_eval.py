#!/usr/bin/env python3
"""Paired CPU evaluation of public medical Qwen candidates on a frozen ATPO sample.

Gold labels are read only by the evaluator after response generation. Prompts,
gold labels, raw generations, and per-case traces stay under ignored outputs/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_direct_reliability_eval as direct


AUDIT_DIR = Path(__file__).resolve().parent
PILOT_DIR = AUDIT_DIR.parent
DATA_CANDIDATES = (
    PILOT_DIR / "assets" / "test.pilot.parquet",
    Path("/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/test.pilot.parquet"),
)
DATA_PATH = next((path for path in DATA_CANDIDATES if path.is_file()), DATA_CANDIDATES[0])
MODEL_ROOT = PILOT_DIR / "assets" / "models"
BASE_MODEL_CANDIDATES = (
    MODEL_ROOT / "Qwen3-8B",
    Path("/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B"),
)
BASE_MODEL_PATH = next(
    (path for path in BASE_MODEL_CANDIDATES if (path / "config.json").is_file()),
    BASE_MODEL_CANDIDATES[0],
)
OUTPUT_DIR = AUDIT_DIR / "outputs" / "medical_expert_candidates_20261116"
BASELINE_DIR = Path("/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/audit/outputs/direct_eval")
BASELINE_MANIFEST = BASELINE_DIR / "expert_reliability_manifest.json"
BASELINE_TRACES = BASELINE_DIR / "direct_reliability_trajectories.jsonl"

SEED = 20261104
CASE_COUNT = 200
BASE_ROLE = "base_qwen3_8b"
MODEL_SPECS = (
    {
        "role": BASE_ROLE,
        "name": "Qwen3-8B",
        "repo_id": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "path": BASE_MODEL_PATH,
        "kind": "qwen3",
    },
    {
        "role": "huatuo_gpt_o1_7b",
        "name": "HuatuoGPT-o1-7B",
        "repo_id": "FreedomIntelligence/HuatuoGPT-o1-7B",
        "revision": "6487c8071a3f87702a6816ba7a09f38921b685ed",
        "path": MODEL_ROOT / "medical_candidates" / "huatuo_gpt_o1_7b",
        "kind": "qwen2",
    },
    {
        "role": "qwen3_4b_remedi",
        "name": "Qwen3-4B-ReMeDi",
        "repo_id": "VioletAmethyst/Qwen3-4B-ReMeDi",
        "revision": "dfa418ca1c3c0f8879afd6d356b88083f01ca032",
        "path": MODEL_ROOT / "medical_candidates" / "qwen3_4b_remedi",
        "kind": "qwen3",
    },
    {
        "role": "popit_qwen3_4b_medical_sft",
        "name": "PopiT-Qwen3-4B-Medical-SFT",
        "repo_id": "Hoshino-Yumetsuki/PopiT-Qwen3-4B-Medical-SFT-1128",
        "revision": "f7f95e22f5fd2c8c015abfba9fe249d6b80b6f97",
        "path": MODEL_ROOT / "medical_candidates" / "popit_qwen3_4b_medical_sft",
        "kind": "qwen3",
    },
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=AUDIT_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def selected_cases() -> tuple[list[dict[str, Any]], str]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"test parquet missing: {DATA_PATH}")
    frame = pd.read_parquet(DATA_PATH)
    if len(frame) < CASE_COUNT:
        raise ValueError(f"need {CASE_COUNT} test rows, found {len(frame)}")
    indices = random.Random(SEED).sample(range(len(frame)), CASE_COUNT)
    cases = []
    for source_i in indices:
        row = frame.iloc[source_i].to_dict()
        messages = direct.build_messages(row)
        gold = direct.answer_letter(row.get("ground_truth") or row.get("reward_model", {}).get("ground_truth"))
        if gold not in "ABCD":
            raise ValueError(f"invalid gold at selected source row {source_i}")
        prompt_sha = sha256_bytes(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        cases.append({
            "case_id": f"test-row-{source_i}:item-{row.get('index', source_i)}",
            "source_row_index": int(source_i),
            "gold": gold,
            "messages": messages,
            "prompt_sha256": prompt_sha,
        })
    return cases, sha256_file(DATA_PATH)


def check_previous_frozen_sample(cases: list[dict[str, Any]], data_sha: str) -> None:
    if not BASELINE_MANIFEST.is_file() or not BASELINE_TRACES.is_file():
        raise FileNotFoundError("saved 200-case Qwen3-8B direct-answer baseline is required")
    manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("case_count") != CASE_COUNT or manifest.get("sample_seed") != SEED:
        raise RuntimeError("saved Qwen3-8B baseline is not the frozen 200-case sample")
    if manifest.get("data_sha256") != data_sha:
        raise RuntimeError("test parquet hash differs from the prior frozen baseline")
    old = [
        item for item in load_jsonl(BASELINE_TRACES)
        if item.get("record_type") == "direct_answer_evaluation" and item.get("model_role") == "base_8b"
    ]
    by_case = {str(item["case_id"]): item for item in old}
    if len(by_case) != CASE_COUNT:
        raise RuntimeError(f"expected {CASE_COUNT} saved base rows, found {len(by_case)}")
    if set(by_case) != {item["case_id"] for item in cases}:
        raise RuntimeError("reconstructed fixed sample does not match the saved baseline cases")
    for item in cases:
        prior = by_case[item["case_id"]]
        if prior.get("prompt_sha256") != item["prompt_sha256"] or prior.get("gold") != item["gold"]:
            raise RuntimeError(f"prompt or post-hoc gold mismatch at {item['case_id']}")


def render_input(tokenizer: Any, messages: list[dict[str, str]], model_kind: str) -> dict[str, torch.Tensor]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    if model_kind == "qwen3":
        kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(rendered, "items"):
        tensors = {key: value for key, value in rendered.items() if isinstance(value, torch.Tensor)}
        if tensors:
            return tensors
    if isinstance(rendered, torch.Tensor):
        return {"input_ids": rendered}
    return {"input_ids": rendered}


def bootstrap_ci(values: list[float], seed: int, repeats: int = 10_000) -> tuple[float, float]:
    if len(values) < 2:
        value = values[0] if values else float("nan")
        return value, value
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.mean(rng.choices(values, k=n)) for _ in range(repeats))
    return means[int(.025 * repeats)], means[min(repeats - 1, int(.975 * repeats))]


def evaluate_one(model: Any, tokenizer: Any, case: dict[str, Any], model_kind: str, max_new_tokens: int) -> dict[str, Any]:
    started = time.perf_counter()
    tensors = render_input(tokenizer, case["messages"], model_kind)
    prompt_len = int(tensors["input_ids"].shape[-1])
    tensors = {key: value.to("cpu") for key, value in tensors.items()}
    eos_id = tokenizer.eos_token_id
    generation = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "pad_token_id": eos_id,
    }
    if eos_id is not None:
        generation["eos_token_id"] = eos_id
    with torch.inference_mode():
        generated = model.generate(**tensors, **generation)
    completion_ids = generated[0, prompt_len:]
    raw = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    ended = bool(eos_id is not None and len(completion_ids) and int(completion_ids[-1]) == int(eos_id))
    return {
        "raw_response": raw,
        "predicted": direct.parse_answer(raw),
        "prompt_tokens": prompt_len,
        "completion_tokens": int(completion_ids.numel()),
        "latency_s": time.perf_counter() - started,
        "finish_reason": "stop" if ended else "length_or_model_stop",
        "error": None,
    }


def summarize(records: list[dict[str, Any]], cases: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_case[str(record["case_id"])][str(record["model_role"])] = record
    expected_roles = [str(spec["role"]) for spec in MODEL_SPECS]
    complete_cases = [case["case_id"] for case in cases if all(role in by_case[case["case_id"]] for role in expected_roles)]
    summary: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for case_id in complete_cases:
        values = by_case[case_id]
        base = values[BASE_ROLE]
        for role in expected_roles[1:]:
            candidate = values[role]
            paired_rows.append({
                "case_id": case_id,
                "candidate_role": role,
                "base_correct": bool(base.get("correct")),
                "candidate_correct": bool(candidate.get("correct")),
                "base_valid": bool(base.get("valid")),
                "candidate_valid": bool(candidate.get("valid")),
            })

    def add(metric: str, role: str, value: Any, low: Any = "", high: Any = "", n: int | None = None, note: str = "") -> None:
        summary.append({
            "metric": metric, "model_role": role, "value": value,
            "ci95_low": low, "ci95_high": high,
            "n_cases": len(complete_cases) if n is None else n,
            "notes": note,
        })

    for role in expected_roles:
        rows = [by_case[case_id][role] for case_id in complete_cases]
        if rows:
            outcomes = [float(bool(row.get("correct"))) for row in rows]
            lo, hi = bootstrap_ci(outcomes, SEED + 20)
            add("accuracy", role, statistics.mean(outcomes), lo, hi, note="Case-level percentile bootstrap; invalid formats remain in denominator.")
            add("valid_rate", role, statistics.mean(float(bool(row.get("valid"))) for row in rows))
            add("mean_total_tokens", role, statistics.mean(int(row.get("prompt_tokens", 0)) + int(row.get("completion_tokens", 0)) for row in rows))
            add("mean_completion_tokens", role, statistics.mean(int(row.get("completion_tokens", 0)) for row in rows))
            add("mean_latency_s_cpu_transformers", role, statistics.mean(float(row.get("latency_s", 0.0)) for row in rows), note="CPU request wall time; includes template/tokenization and generation.")

    for role in expected_roles[1:]:
        rows = [row for row in paired_rows if row["candidate_role"] == role]
        if not rows:
            continue
        delta_values = [float(row["candidate_correct"]) - float(row["base_correct"]) for row in rows]
        delta_ci = bootstrap_ci(delta_values, SEED + 21)
        base_error_rows = [row for row in rows if not row["base_correct"]]
        base_correct_rows = [row for row in rows if row["base_correct"]]
        correction_values = [float(row["candidate_correct"]) for row in base_error_rows]
        degradation_values = [float(not row["candidate_correct"]) for row in base_correct_rows]
        correction_ci = bootstrap_ci(correction_values, SEED + 22) if correction_values else ("", "")
        degradation_ci = bootstrap_ci(degradation_values, SEED + 23) if degradation_values else ("", "")
        cells = Counter(("base_correct" if row["base_correct"] else "base_wrong", "candidate_correct" if row["candidate_correct"] else "candidate_wrong") for row in rows)
        add("paired_accuracy_delta_candidate_minus_base", role, statistics.mean(delta_values), delta_ci[0], delta_ci[1])
        add("candidate_accuracy_given_base_error", role, statistics.mean(correction_values) if correction_values else "", correction_ci[0], correction_ci[1], n=len(base_error_rows), note="Candidate correct among cases where same-run Qwen3-8B was wrong.")
        add("degradation_rate_given_base_correct", role, statistics.mean(degradation_values) if degradation_values else "", degradation_ci[0], degradation_ci[1], n=len(base_correct_rows), note="Candidate wrong among cases where same-run Qwen3-8B was correct.")
        for (base_cell, candidate_cell), count in sorted(cells.items()):
            add(f"paired_cell:{base_cell}__{candidate_cell}", role, count, n=len(rows))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "model_role", "value", "ci95_low", "ci95_high", "n_cases", "notes"))
        writer.writeheader(); writer.writerows(summary)
    with (OUTPUT_DIR / "paired_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("case_id", "candidate_role", "base_correct", "candidate_correct", "base_valid", "candidate_valid")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(paired_rows)
    manifest.update({
        "status": "completed" if len(complete_cases) == CASE_COUNT else "partial",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "complete_cases": len(complete_cases),
        "complete_case_ids_sha256": sha256_bytes("\n".join(complete_cases).encode("utf-8")),
        "summary_sha256": sha256_file(OUTPUT_DIR / "summary.csv"),
        "paired_comparison_sha256": sha256_file(OUTPUT_DIR / "paired_comparison.csv"),
    })
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-cases", type=int, default=CASE_COUNT, help="Run the first N cases of the frozen 200-case sample; useful for a timed smoke run.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit_cases <= CASE_COUNT:
        raise ValueError(f"--limit-cases must be in [1, {CASE_COUNT}]")

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(min(8, args.threads))
    except RuntimeError:
        pass
    torch.manual_seed(SEED)
    cases, data_sha = selected_cases()
    check_previous_frozen_sample(cases, data_sha)
    requested_cases = cases[:args.limit_cases]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trajectory_path = OUTPUT_DIR / "candidate_trajectories.jsonl"
    manifest_path = OUTPUT_DIR / "run_manifest.json"
    download_manifest_path = MODEL_ROOT / "medical_candidates" / "download_manifest.json"
    downloaded_models = {}
    if download_manifest_path.is_file():
        downloaded_models = json.loads(download_manifest_path.read_text(encoding="utf-8")).get("models", {})
    manifest = {
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": "same-engine paired direct-answer accuracy and conditional complementarity of public medical Qwen candidates",
        "data_sha256": data_sha,
        "data_rows": 1268,
        "case_count": CASE_COUNT,
        "sample_seed": SEED,
        "sampled_source_rows": [case["source_row_index"] for case in cases],
        "sample_ids_sha256": sha256_bytes("\n".join(case["case_id"] for case in cases).encode("utf-8")),
        "prompt_protocol_sha256": hashlib.sha256(direct.SYSTEM.encode("utf-8")).hexdigest(),
        "models": [
            {
                **{key: (str(value) if isinstance(value, Path) else value) for key, value in spec.items()},
                "config_sha256": sha256_file(Path(spec["path"]) / "config.json") if (Path(spec["path"]) / "config.json").is_file() else None,
                "weight_shards_sha256": downloaded_models.get(spec["role"], {}).get("safetensors_sha256", {}),
            }
            for spec in MODEL_SPECS
        ],
        "generation": {"temperature": 0.0, "do_sample": False, "max_new_tokens": args.max_new_tokens, "qwen3_enable_thinking": False},
        "runtime": {"framework": "Transformers CPU", "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__, "cpu_threads": args.threads, "dtype": "bfloat16"},
        "code_commit": git_commit(),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "download_manifest_sha256": (
            sha256_file(download_manifest_path)
            if download_manifest_path.is_file()
            else None
        ),
        "scoring": "same strict Final Answer A-D parser as frozen baseline; invalid output remains in denominator",
        "gold_context_policy": "Gold and hidden atomic facts are absent from model messages; gold is read by evaluator after generation only.",
        "trajectory_path": str(trajectory_path.resolve()),
    }
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError("--resume requires existing run_manifest.json")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("data_sha256", "sampled_source_rows", "models", "generation", "sample_ids_sha256"):
            if previous.get(key) != manifest.get(key):
                raise RuntimeError(f"resume mismatch in {key}")
        manifest = previous
        manifest["status"] = "running"
        manifest["resume_started_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    elif trajectory_path.exists() or manifest_path.exists():
        raise FileExistsError("output exists; use --resume or change OUTPUT_DIR")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    records = load_jsonl(trajectory_path) if args.resume and trajectory_path.exists() else []
    done = {(str(row["case_id"]), str(row["model_role"])) for row in records}
    for spec in MODEL_SPECS:
        model_path = Path(spec["path"])
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"model config is missing for {spec['name']}: {model_path}")
        print(f"loading {spec['name']} on CPU from pinned local snapshot", flush=True)
        load_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        model.eval()
        model_load_s = time.perf_counter() - load_started
        print(f"loaded {spec['name']} in {model_load_s:.1f}s; running {sum((case['case_id'], spec['role']) not in done for case in requested_cases)} cases", flush=True)
        for idx, case in enumerate(requested_cases, start=1):
            key = (case["case_id"], spec["role"])
            if key in done:
                continue
            try:
                result = evaluate_one(model, tokenizer, case, spec["kind"], args.max_new_tokens)
            except Exception as exc:
                result = {"raw_response": "", "predicted": None, "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0, "finish_reason": None, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            record = {
                "record_type": "medical_expert_candidate_direct_answer",
                "case_id": case["case_id"],
                "source_row_index": case["source_row_index"],
                "model_role": spec["role"],
                "model_name": spec["name"],
                "model_revision": spec["revision"],
                "prompt_sha256": case["prompt_sha256"],
                "gold": case["gold"],
                "predicted": result["predicted"],
                "correct": bool(result["predicted"] == case["gold"]),
                "valid": result["predicted"] is not None and result["error"] is None,
                "raw_response": result["raw_response"],
                "error": result["error"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "latency_s": result["latency_s"],
                "finish_reason": result["finish_reason"],
            }
            with trajectory_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
            records.append(record)
            done.add(key)
            if idx % 5 == 0 or idx == len(requested_cases):
                print(f"{spec['role']}: {idx}/{len(requested_cases)}; predicted={result['predicted']} valid={record['valid']}", flush=True)
        manifest.setdefault("model_load_seconds", {})[spec["role"]] = model_load_s
        del model, tokenizer
        import gc
        gc.collect()
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summarize(records, requested_cases, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "complete_cases": manifest["complete_cases"],
        "output_dir": str(OUTPUT_DIR),
        "summary": str(OUTPUT_DIR / "summary.csv"),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
