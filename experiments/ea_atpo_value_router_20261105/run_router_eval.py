#!/usr/bin/env python3
"""Run actual selected action branches on a disjoint official ATPO test sample."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from router_core import ACTIONS, file_sha256, load_jsonl, write_json
from train_router import _load_model, predict


HERE = Path(__file__).resolve().parent
PILOT_DIR = HERE.parent / "real_case_pilot_20261103"
TEST_DATA = PILOT_DIR / "assets/test.pilot.parquet"
DIRECT_MANIFEST = PILOT_DIR / "audit/outputs/direct_eval/expert_reliability_manifest.json"
OUT = HERE / "outputs"
PILOT_TRAINING = PILOT_DIR / "outputs/full/pilot_trajectories.jsonl"
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))


def stable_request_seed(model: str, messages: list[dict[str, str]], temperature: float) -> int:
    payload = json.dumps({"model": model, "messages": messages, "temperature": temperature}, ensure_ascii=False, sort_keys=True)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def plain(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def messages_from_extra(extra: dict[str, Any]) -> list[dict[str, str]]:
    raw = plain(extra.get("prompt"))
    if not isinstance(raw, list):
        raise ValueError("test row lacks public extra_info.prompt message list")
    messages = []
    for message in raw:
        role = str(message.get("role", ""))
        if role not in ("system", "user", "assistant"):
            continue
        messages.append({"role": role, "content": str(message.get("content", ""))})
    if not any(m["role"] == "user" for m in messages):
        raise ValueError("test case has no visible user question")
    return messages


def gold_from_row(row: dict[str, Any]) -> Any:
    value = row.get("ground_truth")
    if value is None or value == {}:
        value = row.get("reward_model", {}).get("ground_truth")
    if value is None or value == {}:
        value = row.get("extra_info", {}).get("ground_truth")
    return value


def value_utilities(predictions: dict[str, dict[str, float]], lambdas: dict[str, float]) -> dict[str, float]:
    result = {}
    for action in ACTIONS:
        costs = predictions[action]["costs"]
        result[action] = (predictions[action]["q"]
                          - lambdas.get("ASK", 0.0) * costs["patient_turns"]
                          - lambdas.get("CONSULT", 0.0) * costs["consult_tokens"]
                          - lambdas.get("DEFER", 0.0) * costs["defer_tokens"])
    return result


def pick_action(values: dict[str, float], allowed=ACTIONS) -> str:
    # Deterministic tie-break favors fewer handoffs: ANSWER, ASK, CONSULT, DEFER.
    return max(allowed, key=lambda a: (values[a], -ACTIONS.index(a)))


def make_policies(preds, model_no_rel, states, train_summary, threshold_cfg):
    costs = {
        "no_cost": {a: 0.0 for a in ACTIONS},
        "reference": {"ANSWER": 0.0, "ASK": 0.01, "CONSULT": 0.02, "DEFER": 0.02},
        "higher_cost": {"ANSWER": 0.0, "ASK": 0.03, "CONSULT": 0.10, "DEFER": 0.10},
    }
    fixed = str(train_summary["fixed_cost_baseline_action"])
    threshold = float(threshold_cfg["equivalent_answer_confidence_threshold"])
    alternate = str(threshold_cfg["alternative_action"])
    actions_by_policy: dict[str, dict[str, str]] = {
        "always_answer": {},
        "uncertainty_threshold": {},
        "fixed_cost_router": {},
        "learned_value_no_cost": {},
        "learned_value_router": {},
        "learned_value_higher_cost": {},
        "ablation_no_expert_reliability": {},
        "ablation_no_consult": {},
        "ablation_no_defer": {},
    }
    pred_by_state = {s["state_id"]: p for s, p in zip(states, preds)}
    nr_by_state = {s["state_id"]: p for s, p in zip(states, model_no_rel)}
    decision_info: dict[str, dict[str, Any]] = {}
    for state in states:
        sid = state["state_id"]
        p = pred_by_state[sid]
        pnr = nr_by_state[sid]
        utilities = {name: value_utilities(p, lam) for name, lam in costs.items()}
        nr_utilities = value_utilities(pnr, costs["reference"])
        actions_by_policy["always_answer"][sid] = "ANSWER"
        actions_by_policy["uncertainty_threshold"][sid] = "ANSWER" if p["ANSWER"]["q"] >= threshold else alternate
        actions_by_policy["fixed_cost_router"][sid] = fixed
        actions_by_policy["learned_value_no_cost"][sid] = pick_action(utilities["no_cost"])
        actions_by_policy["learned_value_router"][sid] = pick_action(utilities["reference"])
        actions_by_policy["learned_value_higher_cost"][sid] = pick_action(utilities["higher_cost"])
        actions_by_policy["ablation_no_expert_reliability"][sid] = pick_action(nr_utilities)
        actions_by_policy["ablation_no_consult"][sid] = pick_action(utilities["reference"], ("ANSWER", "ASK", "DEFER"))
        actions_by_policy["ablation_no_defer"][sid] = pick_action(utilities["reference"], ("ANSWER", "ASK", "CONSULT"))
        decision_info[sid] = {
            "predicted_values": p,
            "predicted_values_no_reliability": pnr,
            "predicted_utilities": utilities,
            "threshold_uncertainty": 1.0 - p["ANSWER"]["q"],
            "threshold": 1.0 - threshold,
            "threshold_alternative_action": alternate,
            "fixed_action": fixed,
        }
    return actions_by_policy, decision_info, costs


def make_case_sample(data_path: Path, strong_rows: set[int], count: int, seed: int) -> tuple[list[dict[str, Any]], list[int]]:
    frame = pd.read_parquet(data_path)
    eligible = [i for i in range(len(frame)) if i not in strong_rows]
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} source rows remain after strong-expert exclusion")
    selected_indices = random.Random(seed).sample(eligible, count)
    cases = []
    for source_i in selected_indices:
        row = frame.iloc[source_i].to_dict()
        extra = row.get("extra_info") or {}
        public_state = messages_from_extra(extra)
        gold = gold_from_row(row)
        facts = [str(x) for x in extra.get("atomic_facts", [])]
        index = extra.get("index", row.get("index", source_i))
        case_id = f"router-test-row-{source_i}:item-{index}"
        digest = hashlib.sha256(json.dumps(public_state, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cases.append({
            "case_id": case_id,
            "source_row_index": int(source_i),
            "state_id": case_id,
            "public_state": public_state,
            "gold": gold,
            "atomic_facts": facts,
            "prompt_sha256": digest,
        })
    return cases, selected_indices


async def run(args: argparse.Namespace) -> None:
    from ea_atpo_stage.recipe.ea_atpo.collect_counterfactuals import APICall, ChatAPI, rollout_action
    from ea_atpo_stage.recipe.ea_atpo.protocol import Action, PublicState

    class StableSeedChatAPI(ChatAPI):
        async def generate(self, messages, *, temperature, max_tokens=768, retries=3, structured_outputs=None):
            request_messages = list(messages)
            payload = {
                "model": self.model,
                "messages": request_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": stable_request_seed(self.model, request_messages, float(temperature)),
            }
            if self.enable_thinking is not None:
                payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
            if structured_outputs is not None:
                payload["structured_outputs"] = dict(structured_outputs)
            last_error = None
            for attempt in range(retries):
                started = time.perf_counter()
                try:
                    async with self.client.post(self.url, json=payload) as response:
                        response.raise_for_status()
                        body = await response.json()
                    usage = body.get("usage") or {}
                    return APICall(
                        content=body["choices"][0]["message"]["content"],
                        prompt_tokens=int(usage.get("prompt_tokens", 0)),
                        completion_tokens=int(usage.get("completion_tokens", 0)),
                        latency_s=time.perf_counter() - started,
                    )
                except Exception as exc:
                    last_error = exc
                    await asyncio.sleep(2 ** attempt)
            raise RuntimeError(f"generation failed after {retries} deterministic retries: {last_error}")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    model_dir = args.artifacts / "checkpoint"
    no_rel_dir = args.artifacts / "checkpoint_no_reliability"
    model = _load_model(model_dir)
    no_rel_model = _load_model(no_rel_dir)
    train_summary = json.loads((args.artifacts / "training_summary.json").read_text(encoding="utf-8"))
    threshold_cfg = json.loads((args.artifacts / "baseline_policy_config.json").read_text(encoding="utf-8"))

    direct_manifest = json.loads(DIRECT_MANIFEST.read_text(encoding="utf-8"))
    service_manifest_path = OUT / "model_service_manifest.json"
    service_manifest = json.loads(service_manifest_path.read_text(encoding="utf-8")) if service_manifest_path.exists() else {}
    strong_rows = set(int(x) for x in direct_manifest["sampled_source_rows"])
    cases, selected_indices = make_case_sample(args.data, strong_rows, args.cases, args.seed)
    state_docs = [{"state_id": x["state_id"], "public_state": x["public_state"]} for x in cases]
    predictions_map = predict(model, state_docs)
    no_rel_predictions_map = predict(no_rel_model, state_docs)
    preds = [predictions_map[x["state_id"]] for x in cases]
    no_rel_preds = [no_rel_predictions_map[x["state_id"]] for x in cases]
    actions_by_policy, decisions, cost_sets = make_policies(preds, no_rel_preds, state_docs, train_summary, threshold_cfg)

    manifest_path = out / "run_manifest.json"
    rollout_path = out / "action_rollouts.jsonl"
    manifest = {
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": "actual selected-action interventional evaluation on independent official ATPO test cases",
        "evaluation_scope": "initial_action_only_diagnostic; ASK/CONSULT continuations use the Base Agent policy",
        "source_data": str(args.data.resolve()),
        "data_sha256": file_sha256(args.data),
        "data_rows": int(pd.read_parquet(args.data, columns=["index"]).shape[0]),
        "test_case_count": len(cases),
        "sample_seed": args.seed,
        "sampled_source_rows": selected_indices,
        "excluded_strong_expert_rows": sorted(strong_rows),
        "strong_expert_sample_seed": int(direct_manifest["sample_seed"]),
        "base_model": args.model_alias,
        "model_revision": args.model_revision,
        "model_roles": {"base": args.model_alias, "patient_simulator": args.model_alias, "expert_simulator": args.model_alias},
        "generation": {"temperature": args.temperature, "role_temperatures": {"base": args.temperature, "expert": args.temperature, "patient_simulator": 0.0}, "seed_strategy": "sha256(model,messages,temperature) per request", "max_turns": args.max_turns, "max_consults": args.max_consults, "enable_thinking": False},
        "model_service": service_manifest,
        "model_service_manifest_sha256": file_sha256(service_manifest_path) if service_manifest_path.exists() else None,
        "cost_sets": cost_sets,
        "threshold_baseline": threshold_cfg,
        "fixed_cost_baseline_action": train_summary["fixed_cost_baseline_action"],
        "policies": {k: "case-to-action frozen before model calls" for k in actions_by_policy},
        "input_boundary": "router uses only public_state user/assistant text and derived structural features; gold and atomic_facts are accessed only by the evaluator and Patient Simulator respectively",
        "pilot_training_sha256": file_sha256(PILOT_TRAINING),
        "checkpoint_sha256": file_sha256(model_dir / "router_checkpoint.npz"),
        "no_reliability_checkpoint_sha256": file_sha256(no_rel_dir / "router_checkpoint.npz"),
        "rollout_path": str(rollout_path.resolve()),
    }
    if manifest_path.exists() and args.resume:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in ("data_sha256", "sampled_source_rows", "model_revision", "cost_sets", "checkpoint_sha256", "generation"):
            if old.get(field) != manifest.get(field):
                raise ValueError(f"resume configuration mismatch for {field}")
    elif rollout_path.exists():
        raise FileExistsError(f"{rollout_path} exists; use --resume or choose a new output directory")
    write_json(manifest_path, manifest)

    done: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume and rollout_path.exists():
        for row in load_jsonl(rollout_path):
            done[(str(row["case_id"]), str(row["action"]))] = row

    needed = sorted({(case["case_id"], action) for case in cases for policy, mapping in actions_by_policy.items() for action in [mapping[case["state_id"]]]})
    base = StableSeedChatAPI(args.base_url, args.model_alias, timeout_s=args.timeout_s, enable_thinking=False)
    patient = StableSeedChatAPI(args.base_url, args.model_alias, timeout_s=args.timeout_s, enable_thinking=False)
    expert = StableSeedChatAPI(args.base_url, args.model_alias, timeout_s=args.timeout_s, enable_thinking=False)
    started_all = time.perf_counter()
    try:
        with rollout_path.open("a", encoding="utf-8") as writer:
            for index, (case_id, action_name) in enumerate(needed, 1):
                if (case_id, action_name) in done:
                    continue
                case = next(x for x in cases if x["case_id"] == case_id)
                state = PublicState(case_id, tuple(case["public_state"]), ())
                started = time.perf_counter()
                result = await rollout_action(
                    Action(action_name), state, case["atomic_facts"], case["gold"], base, patient, expert,
                    temperature=args.temperature, max_turns=args.max_turns, max_consults=args.max_consults,
                )
                trace = result["trace"]
                consult_expert_tokens = sum(
                    int(x.get("prompt_tokens", 0) or 0) + int(x.get("completion_tokens", 0) or 0)
                    for x in trace if x.get("actor") == "expert" and x.get("action") == "CONSULT"
                )
                defer_expert_tokens = sum(
                    int(x.get("prompt_tokens", 0) or 0) + int(x.get("completion_tokens", 0) or 0)
                    for x in trace if x.get("actor") == "expert" and x.get("action") == "DEFER"
                )
                expert_calls = sum(x.get("actor") == "expert" for x in trace)
                patient_turns = int(result["patient_turns"])
                row = {
                    "case_id": case_id,
                    "source_row_index": case["source_row_index"],
                    "state_id": case["state_id"],
                    "prompt_sha256": case["prompt_sha256"],
                    "action": action_name,
                    "gold": result["gold"],
                    "final_answer": result["answer"],
                    "correctness": float(result["outcome"]),
                    "valid": bool(result["valid"]),
                    "failure_reason": result["error"],
                    "consult_question": next((x.get("consult_question") for x in result["trace"] if x.get("actor") == "base" and x.get("action") == "CONSULT"), None),
                    "expert_advice": next((x.get("response") for x in result["trace"] if x.get("actor") == "expert" and x.get("action") == "CONSULT"), None),
                    "patient_response": next((x.get("response") for x in result["trace"] if x.get("actor") == "patient"), None),
                    "patient_turns": patient_turns,
                    "consult_expert_tokens": int(consult_expert_tokens),
                    "defer_expert_tokens": int(defer_expert_tokens),
                    "expert_calls": int(expert_calls),
                    "expert_tokens": int(result["expert_prompt_tokens"] + result["expert_completion_tokens"]),
                    "total_tokens": int(result["total_prompt_tokens"] + result["total_completion_tokens"]),
                    "total_prompt_tokens": int(result["total_prompt_tokens"]),
                    "total_completion_tokens": int(result["total_completion_tokens"]),
                    "model_inference_latency_s": float(result["total_latency_s"]),
                    "elapsed_wall_s": time.perf_counter() - started,
                    "cost_components": {
                        "patient_turns": patient_turns,
                        "consult_tokens_per_1000": consult_expert_tokens / 1000.0,
                        "defer_tokens_per_1000": defer_expert_tokens / 1000.0,
                    },
                    "trace": trace,
                }
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                writer.flush()
                done[(case_id, action_name)] = row
                if index % 20 == 0 or index == len(needed):
                    print(f"rollouts {index}/{len(needed)} | elapsed {time.perf_counter()-started_all:.1f}s", flush=True)
    finally:
        await base.close(); await patient.close(); await expert.close()

    # Each policy gets one real branch per case. If two policies select the
    # same deterministic action on a case, that single identical branch is
    # shared, preserving paired inputs while avoiding duplicate inference.
    episodes_path = out / "policy_episodes.jsonl"
    rollout_by_key = done
    with episodes_path.open("w", encoding="utf-8") as writer:
        for policy_name, mapping in actions_by_policy.items():
            for case in cases:
                sid = case["state_id"]
                action = mapping[sid]
                rollout = rollout_by_key[(case["case_id"], action)]
                decision = decisions[sid]
                record = {
                    "policy": policy_name,
                    "case_id": case["case_id"],
                    "source_row_index": case["source_row_index"],
                    "state_id": sid,
                    "action": action,
                    "gold": rollout["gold"],
                    "final_answer": rollout["final_answer"],
                    "correctness": rollout["correctness"],
                    "valid": rollout["valid"],
                    "failure_reason": rollout["failure_reason"],
                    "consult_question": rollout["consult_question"],
                    "expert_advice": rollout["expert_advice"],
                    "patient_response": rollout["patient_response"],
                    "patient_turns": rollout["patient_turns"],
                    "consult_expert_tokens": rollout["consult_expert_tokens"],
                    "defer_expert_tokens": rollout["defer_expert_tokens"],
                    "expert_calls": rollout["expert_calls"],
                    "expert_tokens": rollout["expert_tokens"],
                    "total_tokens": rollout["total_tokens"],
                    "total_prompt_tokens": rollout["total_prompt_tokens"],
                    "total_completion_tokens": rollout["total_completion_tokens"],
                    "model_inference_latency_s": rollout["model_inference_latency_s"],
                    "elapsed_wall_s": rollout["elapsed_wall_s"],
                    "cost_components": rollout["cost_components"],
                    "rollout_key": f"{case['case_id']}::{action}",
                    "decision": decision,
                    "trace": rollout["trace"],
                }
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest.update({
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "unique_case_action_rollouts": len(done),
        "policy_episode_count": len(actions_by_policy) * len(cases),
        "elapsed_wall_s": time.perf_counter() - started_all,
        "action_rollouts_sha256": file_sha256(rollout_path),
        "policy_episodes_sha256": file_sha256(episodes_path),
    })
    write_json(manifest_path, manifest)
    print(f"completed {len(cases)} cases, {len(done)} unique action rollouts", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=TEST_DATA)
    parser.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--cases", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--base-url", default="http://127.0.0.1:30131/v1")
    parser.add_argument("--model-alias", default="qwen3-8b-router")
    parser.add_argument("--model-revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-consults", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
