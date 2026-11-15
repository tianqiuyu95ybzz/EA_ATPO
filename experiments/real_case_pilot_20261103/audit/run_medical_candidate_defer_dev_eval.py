#!/usr/bin/env python3
"""Paired, leakage-checked DEFER evaluation on the frozen Attempt09 20-case dev set.

The model sees only the public prompt. Gold and atomic facts are used for the
post-hoc leak check and scoring and are never serialized to outputs/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "ea_atpo_stage"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_direct_reliability_eval as direct  # noqa: E402
import run_medical_candidate_eval as candidate_eval  # noqa: E402

_protocol_path = ROOT / "ea_atpo_stage" / "recipe" / "ea_atpo" / "protocol.py"
_protocol_spec = importlib.util.spec_from_file_location("ea_atpo_standalone_protocol", _protocol_path)
if _protocol_spec is None or _protocol_spec.loader is None:
    raise ImportError(f"cannot load protocol module from {_protocol_path}")
_protocol = importlib.util.module_from_spec(_protocol_spec)
sys.modules[_protocol_spec.name] = _protocol
_protocol_spec.loader.exec_module(_protocol)
Action = _protocol.Action
PublicState = _protocol.PublicState
build_expert_messages = _protocol.build_expert_messages
parse_action = _protocol.parse_action
validate_expert_response = _protocol.validate_expert_response
validate_expert_view = _protocol.validate_expert_view
unrevealed_atomic_facts = _protocol.unrevealed_atomic_facts
visible_response = _protocol.visible_response

RUN_DIR = ROOT / "experiments" / "ea_atpo_v1_smoke" / "ea_atpo_v1_attempt09_stability"
DEFAULT_DATA = RUN_DIR / "data" / "test.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "medical_candidate_defer_dev_20261116"
BASE_MODEL = Path(
    "/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def plain(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def answer_letter(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("answer", "")).strip().upper()
    return str(value).strip().upper()


def row_gold(row: dict[str, Any]) -> str:
    raw = row.get("ground_truth")
    if raw is None:
        raw = row.get("reward_model", {}).get("ground_truth")
    gold = answer_letter(raw)
    if gold not in "ABCD":
        raise ValueError(f"invalid gold in dev parquet row {row.get('index')}")
    return gold


def row_facts(row: dict[str, Any]) -> list[str]:
    value = row.get("atomic_facts")
    if value is None:
        value = row.get("extra_info", {}).get("atomic_facts", [])
    value = plain(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [value]
    return [str(x) for x in (value or [])]


def public_state(row: dict[str, Any], index: int) -> tuple[PublicState, str, str, list[str]]:
    prompt = plain(row.get("prompt"))
    if not isinstance(prompt, list):
        prompt = plain(row.get("extra_info", {}).get("prompt"))
    if not isinstance(prompt, list):
        raise ValueError(f"missing public prompt at dev row {index}")
    visible = []
    for message in prompt:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        if role == "system":
            continue
        if role == "assistant":
            try:
                action, payload = parse_action(content)
                if action is Action.ANSWER:
                    content = f"Final Answer: {payload.strip().upper()}"
                elif action is Action.ASK:
                    content = f"Question: {payload.strip()}"
                elif action is Action.CONSULT:
                    content = f"Consult Expert: {payload.strip()}"
                else:
                    content = "Defer to Expert"
            except ValueError:
                content = visible_response(content)
        visible.append({"role": "assistant" if role == "assistant" else "user", "content": content})
    public_payload = json.dumps(visible, ensure_ascii=False, sort_keys=True)
    case_hash = sha256_text(public_payload)
    state = PublicState(sample_id=f"dev-{case_hash[:16]}", initial_messages=tuple(visible))
    facts = row_facts(row)
    gold = row_gold(row)
    messages = build_expert_messages(state, Action.DEFER)
    validate_expert_view(messages, atomic_facts=unrevealed_atomic_facts(state, facts), ground_truth=gold)
    return state, case_hash, gold, facts


def render_input(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, torch.Tensor]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    if hasattr(rendered, "items"):
        tensors = {key: value for key, value in rendered.items() if isinstance(value, torch.Tensor)}
        if tensors:
            return tensors
    if isinstance(rendered, torch.Tensor):
        return {"input_ids": rendered}
    return {"input_ids": rendered}


def infer(model: Any, tokenizer: Any, messages: list[dict[str, str]], max_new_tokens: int, action: Any) -> dict[str, Any]:
    started = time.perf_counter()
    inputs = render_input(tokenizer, messages)
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    generation: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if tokenizer.eos_token_id is not None:
        generation["eos_token_id"] = tokenizer.eos_token_id
    with torch.inference_mode():
        output = model.generate(**inputs, **generation)
    completion = output[0, prompt_tokens:]
    raw = tokenizer.decode(completion, skip_special_tokens=False)
    finish = "stop" if tokenizer.eos_token_id is not None and completion.numel() and int(completion[-1]) == tokenizer.eos_token_id else "length_or_model_stop"
    if action is Action.ANSWER:
        predicted = direct.parse_answer(raw)
        error = None if predicted else "direct_answer_parse_failure"
    else:
        try:
            normalized = validate_expert_response(Action.DEFER, raw)
            match = re.fullmatch(r"Expert Final Answer:\s*([A-D])", normalized, re.IGNORECASE)
            predicted = match.group(1).upper() if match else None
            error = None if predicted else "normalized_answer_parse_failure"
        except ValueError as exc:
            predicted, error = None, str(exc)
    return {
        "predicted": predicted,
        "valid": predicted is not None,
        "error": error,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(completion.numel()),
        "latency_s": time.perf_counter() - started,
        "finish_reason": finish,
    }


def bootstrap_ci(values: list[float], seed: int, repeats: int = 10000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.mean(rng.choices(values, k=n)) for _ in range(repeats))
    return means[int(.025 * repeats)], means[min(repeats - 1, int(.975 * repeats))]


def evaluate_model(spec: dict[str, str], cases: list[dict[str, Any]], max_new_tokens: int, threads: int) -> list[dict[str, Any]]:
    model_path = Path(spec["path"])
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"missing model config for {spec['role']}: {model_path}")
    print(f"loading {spec['name']} on CPU", flush=True)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    print(f"loaded {spec['name']} in {time.perf_counter() - started:.1f}s", flush=True)
    records = []
    for index, case in enumerate(cases, 1):
        try:
            action = Action.ANSWER if spec["message_key"] == "answer_messages" else Action.DEFER
            result = infer(model, tokenizer, case[spec["message_key"]], max_new_tokens, action)
        except Exception as exc:
            result = {
                "predicted": None,
                "valid": False,
                "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_s": 0.0,
                "finish_reason": None,
            }
        records.append({
            "case_hash": case["case_hash"],
            "model_role": spec["role"],
            "predicted": result["predicted"],
            "correct": bool(result["predicted"] == case["gold"]),
            "valid": result["valid"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "latency_s": result["latency_s"],
            "finish_reason": result["finish_reason"],
            "error": result["error"],
        })
        if index % 5 == 0 or index == len(cases):
            print(f"{spec['role']}: {index}/{len(cases)}", flush=True)
    del model, tokenizer
    import gc
    gc.collect()
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--limit-cases", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    frame = pd.read_parquet(args.data)
    if len(frame) != 20 or args.limit_cases != 20:
        raise ValueError(f"frozen Attempt09 dev set must contain exactly 20 rows; found {len(frame)}")
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(min(8, args.threads))
    except RuntimeError:
        pass
    torch.manual_seed(20261115)

    cases = []
    for index, raw_row in frame.iterrows():
        row = raw_row.to_dict()
        state, case_hash, gold, facts = public_state(row, int(index))
        defer_messages = build_expert_messages(state, Action.DEFER)
        answer_messages = direct.build_messages(row)
        # Recheck the exact message payloads immediately before inference.
        hidden_facts = unrevealed_atomic_facts(state, facts)
        validate_expert_view(defer_messages, atomic_facts=hidden_facts, ground_truth=gold)
        validate_expert_view(answer_messages, atomic_facts=hidden_facts, ground_truth=gold)
        cases.append({
            "case_hash": case_hash,
            "gold": gold,
            "answer_messages": answer_messages,
            "defer_messages": defer_messages,
        })
    if len({case["case_hash"] for case in cases}) != len(cases):
        raise ValueError("duplicate public states in the fixed Attempt09 dev set")

    base_answer_spec = {
        "role": "always_answer_qwen3_8b",
        "name": "Qwen3-8B Always Answer",
        "path": str(BASE_MODEL),
        "message_key": "answer_messages",
    }
    base_defer_spec = {
        "role": "base_qwen3_8b_defer_prompt",
        "name": "Qwen3-8B Base under DEFER prompt",
        "path": str(BASE_MODEL),
        "message_key": "defer_messages",
    }
    expert_specs = [
        {
            "role": candidate["role"],
            "name": candidate["name"],
            "path": str(candidate["path"]),
            "repo_id": candidate["repo_id"],
            "revision": candidate["revision"],
            "message_key": "defer_messages",
        }
        for candidate in candidate_eval.MODEL_SPECS[1:]
    ]
    records = evaluate_model(base_answer_spec, cases, args.max_new_tokens, args.threads)
    records += evaluate_model(base_defer_spec, cases, args.max_new_tokens, args.threads)
    for expert_spec in expert_specs:
        records += evaluate_model(expert_spec, cases, args.max_new_tokens, args.threads)

    by_case = {case["case_hash"]: {"gold": case["gold"]} for case in cases}
    for record in records:
        by_case[record["case_hash"]][record["model_role"]] = record
    paired = []
    for case_hash, row in by_case.items():
        base = row[base_answer_spec["role"]]
        base_defer = row[base_defer_spec["role"]]
        paired_row = {
            "case_hash": case_hash,
            "always_answer_prediction": base["predicted"] or "",
            "base_defer_prediction": base_defer["predicted"] or "",
            "always_answer_correct": base["correct"],
            "base_defer_correct": base_defer["correct"],
            "always_answer_valid": base["valid"],
            "base_defer_valid": base_defer["valid"],
            "gold_correctness_used_posthoc_only": True,
        }
        for expert_spec in expert_specs:
            expert = row[expert_spec["role"]]
            paired_row[f"{expert_spec['role']}_prediction"] = expert["predicted"] or ""
            paired_row[f"{expert_spec['role']}_correct"] = expert["correct"]
            paired_row[f"{expert_spec['role']}_valid"] = expert["valid"]
        paired.append(paired_row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "defer_dev_trajectories.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (OUTPUT_DIR / "paired_case_results.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = tuple(paired[0].keys())
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(paired)

    summary: list[dict[str, Any]] = []
    mean_accuracy: dict[str, float] = {}
    specs = (base_answer_spec, base_defer_spec, *expert_specs)
    for spec in specs:
        rows = [record for record in records if record["model_role"] == spec["role"]]
        accuracy = [float(row["correct"]) for row in rows]
        mean_accuracy[spec["role"]] = statistics.mean(accuracy)
        low, high = bootstrap_ci(accuracy, 20261115)
        summary.append({
            "metric": "accuracy", "model_role": spec["role"], "value": statistics.mean(accuracy),
            "ci95_low": low, "ci95_high": high, "n_cases": len(rows),
        })
        summary.append({
            "metric": "valid_rate", "model_role": spec["role"], "value": statistics.mean(float(row["valid"]) for row in rows),
            "ci95_low": "", "ci95_high": "", "n_cases": len(rows),
        })
        summary.append({
            "metric": "mean_total_tokens", "model_role": spec["role"], "value": statistics.mean(row["prompt_tokens"] + row["completion_tokens"] for row in rows),
            "ci95_low": "", "ci95_high": "", "n_cases": len(rows),
        })
    summary.append({"metric": "cost_adjusted_net_value", "model_role": base_answer_spec["role"], "value": mean_accuracy[base_answer_spec["role"]], "ci95_low": "", "ci95_high": "", "n_cases": len(cases)})
    for spec_index, spec in enumerate((base_defer_spec, *expert_specs)):
        role = spec["role"]
        cost = 0.60
        summary.append({"metric": "cost_adjusted_net_value", "model_role": role, "value": mean_accuracy[role] - cost, "ci95_low": "", "ci95_high": "", "n_cases": len(cases)})
        delta_answer = [float(row[f"{role}_correct"]) - float(row["always_answer_correct"]) if role != base_defer_spec["role"] else float(row["base_defer_correct"]) - float(row["always_answer_correct"]) for row in paired]
        delta_answer_low, delta_answer_high = bootstrap_ci(delta_answer, 20261116 + spec_index)
        summary.append({
            "metric": "paired_accuracy_delta_defer_minus_always_answer", "model_role": role,
            "value": statistics.mean(delta_answer), "ci95_low": delta_answer_low, "ci95_high": delta_answer_high, "n_cases": len(paired),
        })
        delta_base_defer = [float(row[f"{role}_correct"]) - float(row["base_defer_correct"]) for row in paired] if role != base_defer_spec["role"] else [0.0] * len(paired)
        delta_base_defer_low, delta_base_defer_high = bootstrap_ci(delta_base_defer, 20261119 + spec_index)
        summary.append({
            "metric": "paired_accuracy_delta_vs_base_defer_prompt", "model_role": role,
            "value": statistics.mean(delta_base_defer), "ci95_low": delta_base_defer_low, "ci95_high": delta_base_defer_high, "n_cases": len(paired),
        })
        if role == base_defer_spec["role"]:
            continue
        correct_key = f"{role}_correct"
        base_errors = [row for row in paired if not row["always_answer_correct"]]
        base_correct = [row for row in paired if row["always_answer_correct"]]
        correction = [float(row[correct_key]) for row in base_errors]
        degradation = [float(not row[correct_key]) for row in base_correct]
        correction_ci = bootstrap_ci(correction, 20261117 + spec_index)
        degradation_ci = bootstrap_ci(degradation, 20261118 + spec_index)
        summary.extend([
            {"metric": "expert_accuracy_given_base_error", "model_role": role, "value": statistics.mean(correction) if correction else "", "ci95_low": correction_ci[0] if correction else "", "ci95_high": correction_ci[1] if correction else "", "n_cases": len(base_errors)},
            {"metric": "expert_degradation_given_base_correct", "model_role": role, "value": statistics.mean(degradation) if degradation else "", "ci95_low": degradation_ci[0] if degradation else "", "ci95_high": degradation_ci[1] if degradation else "", "n_cases": len(base_correct)},
            {"metric": "defer_cost_per_case", "model_role": role, "value": 0.60, "ci95_low": "", "ci95_high": "", "n_cases": len(paired)},
        ])
    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = tuple(summary[0].keys())
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(summary)

    manifest = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "L2D DEFER-action comparison on Attempt09's fixed 20-case development set: Always Answer, Qwen3-8B under the DEFER expert contract, and each public medical Qwen candidate under the same DEFER contract; no policy training.",
        "data_sha256": sha256_file(args.data),
        "case_count": len(cases),
        "case_hashes_sha256": sha256_text("\n".join(case["case_hash"] for case in cases)),
        "models": [
            {"role": base_answer_spec["role"], "repo_id": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218", "path": str(BASE_MODEL), "config_sha256": sha256_file(BASE_MODEL / "config.json")},
            {"role": base_defer_spec["role"], "repo_id": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218", "path": str(BASE_MODEL), "config_sha256": sha256_file(BASE_MODEL / "config.json")},
            *[
                {"role": spec["role"], "repo_id": spec["repo_id"], "revision": spec["revision"], "path": spec["path"], "config_sha256": sha256_file(Path(spec["path"]) / "config.json")}
                for spec in expert_specs
            ],
        ],
        "code_commit": git_commit(),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "generation": {"sampling": "greedy", "do_sample": False, "max_new_tokens": args.max_new_tokens, "dtype": "bfloat16", "runtime": "Transformers CPU", "threads": args.threads},
        "prompt_policy": "Always Answer uses the frozen direct-answer prompt. Both DEFER arms use the exact Attempt09 DEFER system contract and same public case/dialogue. Gold and hidden atomic facts never enter any model messages.",
        "scoring": "Always Answer uses the strict direct-answer parser. DEFER arms use Attempt09 validate_expert_response(DEFER); invalid output scores incorrect and remains in denominator.",
        "action_cost": {"DEFER": 0.60},
        "outputs": {name: sha256_file(OUTPUT_DIR / name) for name in ("defer_dev_trajectories.jsonl", "paired_case_results.csv", "summary.csv")},
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "output_dir": str(OUTPUT_DIR), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
