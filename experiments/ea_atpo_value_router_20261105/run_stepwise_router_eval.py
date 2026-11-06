#!/usr/bin/env python3
"""Run leakage-controlled, per-turn Router policies on an independent test set.

Unlike the initial-action diagnostic, each policy is re-evaluated after every
public ASK/CONSULT response and chooses the next action from its own value model.
Gold is touched only after the episode terminates.
"""

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

import pandas as pd

from router_core import ACTIONS, file_sha256, write_json
from train_router import _load_model, predict

HERE = Path(__file__).resolve().parent
PILOT_DIR = HERE.parent / "real_case_pilot_20261103"
DEFAULT_DATA = PILOT_DIR / "assets/test.pilot.parquet"
DIRECT_MANIFEST = PILOT_DIR / "audit/outputs/direct_eval/expert_reliability_manifest.json"
DEFAULT_OUT = HERE / "outputs/stepwise_test"
PILOT_TRAINING = PILOT_DIR / "outputs/full/pilot_trajectories.jsonl"
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))


def stable_request_seed(model: str, messages: list[dict[str, str]], temperature: float) -> int:
    body = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature},
        ensure_ascii=False,
        sort_keys=True,
    )
    return int(hashlib.sha256(body.encode("utf-8")).hexdigest()[:8], 16)


def public_state_from_row(row: dict[str, Any], source_index: int) -> dict[str, Any]:
    extra = row.get("extra_info") or {}
    raw = extra.get("prompt")
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, list):
        raise ValueError(f"test row {source_index} has no public prompt")
    messages = [
        {"role": str(x.get("role", "")), "content": str(x.get("content", ""))}
        for x in raw if x.get("role") in ("system", "user", "assistant")
    ]
    if not any(x["role"] == "user" for x in messages):
        raise ValueError(f"test row {source_index} has no visible user question")
    gold = row.get("ground_truth")
    if gold is None or gold == {}:
        gold = (row.get("reward_model") or {}).get("ground_truth")
    if gold is None or gold == {}:
        gold = (extra or {}).get("ground_truth")
    return {
        "case_id": f"stepwise-test-row-{source_index}:item-{extra.get('index', row.get('index', source_index))}",
        "source_row_index": int(source_index),
        "initial_messages": messages,
        "atomic_facts": [str(x) for x in extra.get("atomic_facts", [])],
        "gold": gold,
        "prompt_sha256": hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def select_action(policy: str, pred, no_rel_pred, threshold_cfg, train_summary, used_consults: int, final_turn: bool) -> tuple[str, dict[str, float]]:
    costs = {
        "no_cost": {a: 0.0 for a in ACTIONS},
        "reference": {"ANSWER": 0.0, "ASK": 0.01, "CONSULT": 0.02, "DEFER": 0.02},
        "higher_cost": {"ANSWER": 0.0, "ASK": 0.03, "CONSULT": 0.10, "DEFER": 0.10},
    }
    if policy == "always_answer":
        lambdas = costs["reference"]
    elif policy == "learned_value_no_cost":
        lambdas = costs["no_cost"]
    elif policy == "learned_value_higher_cost":
        lambdas = costs["higher_cost"]
    else:
        lambdas = costs["reference"]

    allowed = list(ACTIONS)
    if used_consults >= 1:
        allowed.remove("CONSULT")
    if policy == "ablation_no_consult" and "CONSULT" in allowed:
        allowed.remove("CONSULT")
    if policy == "ablation_no_defer":
        allowed.remove("DEFER")
    if final_turn:
        allowed = ["ANSWER"]

    def utilities(values, lambda_set):
        return {
            a: values[a]["q"]
            - lambda_set.get("ASK", 0.0) * values[a]["costs"]["patient_turns"]
            - lambda_set.get("CONSULT", 0.0) * values[a]["costs"]["consult_tokens"]
            - lambda_set.get("DEFER", 0.0) * values[a]["costs"]["defer_tokens"]
            for a in ACTIONS
        }

    u = utilities(pred, lambdas)
    if final_turn:
        action = "ANSWER"
    elif policy == "always_answer":
        action = "ANSWER"
    elif policy == "uncertainty_threshold":
        if pred["ANSWER"]["q"] >= float(threshold_cfg["equivalent_answer_confidence_threshold"]):
            action = "ANSWER"
        else:
            alternative = str(threshold_cfg["alternative_action"])
            if alternative not in allowed:
                means = train_summary["fixed_cost_training_mean_utility"]
                candidates = [a for a in allowed if a != "ANSWER"]
                alternative = max(candidates, key=lambda a: (means[a], -ACTIONS.index(a))) if candidates else "ANSWER"
            action = alternative
    elif policy == "fixed_cost_router":
        fixed = str(train_summary["fixed_cost_baseline_action"])
        if fixed in allowed:
            action = fixed
        else:
            means = train_summary["fixed_cost_training_mean_utility"]
            action = max(allowed, key=lambda a: (means[a], -ACTIONS.index(a)))
    elif policy == "ablation_no_expert_reliability":
        u = utilities(no_rel_pred, costs["reference"])
        action = max(allowed, key=lambda a: (u[a], -ACTIONS.index(a)))
    else:
        action = max(allowed, key=lambda a: (u[a], -ACTIONS.index(a)))
    return action, u


async def run_one_policy(case, policy, model, no_rel_model, threshold_cfg, train_summary, base, patient, expert, temperature, max_turns):
    from ea_atpo_stage.recipe.ea_atpo.collect_counterfactuals import (
        ACTION_REGEX, _extend_state, _forced_messages,
        _patient_messages, _public_action_text,
    )
    from ea_atpo_stage.recipe.ea_atpo.protocol import (
        Action, PublicState, build_expert_messages, parse_action,
        unrevealed_atomic_facts, validate_expert_response, validate_expert_view,
        visible_response,
    )

    current = PublicState(case["case_id"], tuple(case["initial_messages"]), ())
    atomic_facts = case["atomic_facts"]
    trace: list[dict[str, Any]] = []
    actions_taken: list[str] = []
    answer = None
    error = None
    used_consults = 0
    try:
        for decision_index in range(max_turns):
            final_turn = decision_index == max_turns - 1
            state_doc = {"state_id": current.state_id, "public_state": current.messages}
            pred = predict(model, [state_doc])[current.state_id]
            no_rel_pred = predict(no_rel_model, [state_doc])[current.state_id]
            action_name, utilities = select_action(
                policy, pred, no_rel_pred, threshold_cfg, train_summary, used_consults, final_turn
            )
            actions_taken.append(action_name)
            trace.append({
                "actor": "router",
                "decision_index": decision_index,
                "state_id": current.state_id,
                "action": action_name,
                "predicted_q": {a: float(pred[a]["q"]) for a in ACTIONS},
                "predicted_costs": {a: pred[a]["costs"] for a in ACTIONS},
                "predicted_utility": {a: float(utilities[a]) for a in ACTIONS},
                "expert_reliability": float(pred["ANSWER"]["expert_reliability"]),
                "forced_terminal_answer": bool(final_turn),
            })
            action = Action(action_name)
            if action is Action.DEFER:
                expert_messages = build_expert_messages(current, Action.DEFER)
                validate_expert_view(expert_messages, atomic_facts=unrevealed_atomic_facts(current, atomic_facts))
                expert_call = await expert.generate(expert_messages, temperature=temperature, max_tokens=384)
                event = {"actor": "expert", "action": "DEFER",
                         "prompt_tokens": expert_call.prompt_tokens,
                         "completion_tokens": expert_call.completion_tokens,
                         "latency_s": expert_call.latency_s}
                trace.append(event)
                defer_response = validate_expert_response(Action.DEFER, expert_call.content)
                answer = defer_response.split(":", 1)[1].strip().upper()
                event["response"] = f"Expert Final Answer: {answer}"
                visible_defer = {"role": "assistant", "content": "Defer to Expert"}
                current = _extend_state(current, visible_defer,
                                        {"role": "user", "content": f"Expert Final Answer: {answer}"})
                break
            content = None
            call = None
            last_action = "invalid"
            for strict in (False, True):
                for attempt in range(2):
                    base_call = await base.generate(
                        _forced_messages(current, action, strict=strict),
                        temperature=temperature,
                        max_tokens=16 if action is Action.ANSWER else 256,
                        structured_outputs={"regex": ACTION_REGEX[action]},
                    )
                    try:
                        parsed_action, parsed_content = parse_action(base_call.content)
                        last_action = parsed_action.value
                    except ValueError:
                        parsed_action, parsed_content = None, ""
                        last_action = "invalid_format"
                    if parsed_action is action:
                        content, call = parsed_content, base_call
                        break
                    trace.append({
                        "actor": "base", "action": action.value,
                        "error": "forced_action_mismatch", "observed_action": last_action,
                        "retry_attempt": attempt, "strict_prompt": strict,
                        "prompt_tokens": base_call.prompt_tokens,
                        "completion_tokens": base_call.completion_tokens,
                        "latency_s": base_call.latency_s,
                    })
                if call is not None:
                    break
            if call is None or content is None:
                raise ValueError(f"forced {action.value}, model emitted {last_action}")
            visible = {"role": "assistant", "content": _public_action_text(action, content)}
            trace.append({
                "actor": "base", "action": action.value,
                "consult_question": content if action is Action.CONSULT else None,
                "response": visible["content"],
                "prompt_tokens": call.prompt_tokens,
                "completion_tokens": call.completion_tokens,
                "latency_s": call.latency_s,
            })
            if action is Action.ANSWER:
                answer = content.strip().upper()
                current = _extend_state(current, visible)
                break
            if action is Action.ASK:
                patient_call = await patient.generate(
                    _patient_messages(atomic_facts, content), temperature=0.0, max_tokens=256
                )
                from ea_atpo_stage.recipe.ea_atpo.protocol import visible_response
                response = visible_response(patient_call.content)
                trace.append({"actor": "patient", "response": response,
                              "prompt_tokens": patient_call.prompt_tokens,
                              "completion_tokens": patient_call.completion_tokens,
                              "latency_s": patient_call.latency_s})
                current = _extend_state(current, visible, {"role": "user", "content": response})
                continue
            if action is Action.CONSULT:
                expert_messages = build_expert_messages(current, Action.CONSULT, consult_question=content)
                validate_expert_view(expert_messages, atomic_facts=unrevealed_atomic_facts(current, atomic_facts))
                expert_call = await expert.generate(expert_messages, temperature=temperature, max_tokens=384)
                event = {"actor": "expert", "action": "CONSULT", "consult_question": content,
                         "prompt_tokens": expert_call.prompt_tokens,
                         "completion_tokens": expert_call.completion_tokens,
                         "latency_s": expert_call.latency_s}
                trace.append(event)
                advice = validate_expert_response(Action.CONSULT, expert_call.content)
                event["response"] = advice
                current = _extend_state(current, visible, {"role": "user", "content": advice})
                used_consults += 1
                continue
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    # Gold is first read after the generation loop; it never enters Router,
    # Base Agent, Patient, or Expert messages.
    from ea_atpo_stage.recipe.ea_atpo.collect_counterfactuals import _gold_letter
    gold = _gold_letter(case["gold"])
    def usage(actor):
        selected = [x for x in trace if x.get("actor") == actor]
        return {
            "prompt": sum(int(x.get("prompt_tokens", 0)) for x in selected),
            "completion": sum(int(x.get("completion_tokens", 0)) for x in selected),
            "latency": sum(float(x.get("latency_s", 0.0)) for x in selected),
        }
    base_usage, patient_usage, expert_usage = usage("base"), usage("patient"), usage("expert")
    consult_tokens = sum(x["prompt_tokens"] + x["completion_tokens"] for x in trace
                         if x.get("actor") == "expert" and x.get("action") == "CONSULT")
    defer_tokens = sum(x["prompt_tokens"] + x["completion_tokens"] for x in trace
                       if x.get("actor") == "expert" and x.get("action") == "DEFER")
    return {
        "policy": policy,
        "case_id": case["case_id"],
        "source_row_index": case["source_row_index"],
        "prompt_sha256": case["prompt_sha256"],
        "initial_public_state": case["initial_messages"],
        "decision_actions": actions_taken,
        "first_action": actions_taken[0] if actions_taken else None,
        "final_answer": answer,
        "gold": gold,
        "correctness": float(answer == gold) if answer else 0.0,
        "valid": answer is not None and error is None,
        "failure_reason": error,
        "ASK_count": actions_taken.count("ASK"),
        "CONSULT_count": actions_taken.count("CONSULT"),
        "DEFER_count": actions_taken.count("DEFER"),
        "patient_turns": sum(x.get("actor") == "patient" for x in trace),
        "consult_expert_tokens": consult_tokens,
        "defer_expert_tokens": defer_tokens,
        "expert_calls": sum(x.get("actor") == "expert" for x in trace),
        "expert_tokens": expert_usage["prompt"] + expert_usage["completion"],
        "total_prompt_tokens": base_usage["prompt"] + patient_usage["prompt"] + expert_usage["prompt"],
        "total_completion_tokens": base_usage["completion"] + patient_usage["completion"] + expert_usage["completion"],
        "model_inference_latency_s": base_usage["latency"] + patient_usage["latency"] + expert_usage["latency"],
        "trace": trace,
    }


async def run(args):
    from ea_atpo_stage.recipe.ea_atpo.collect_counterfactuals import APICall, ChatAPI

    class StableSeedChatAPI(ChatAPI):
        async def generate(self, messages, *, temperature, max_tokens=768, retries=3, structured_outputs=None):
            request_messages = list(messages)
            payload = {"model": self.model, "messages": request_messages, "temperature": temperature,
                       "max_tokens": max_tokens,
                       "seed": stable_request_seed(self.model, request_messages, float(temperature))}
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
                    usage_data = body.get("usage") or {}
                    return APICall(body["choices"][0]["message"]["content"],
                                   int(usage_data.get("prompt_tokens", 0)),
                                   int(usage_data.get("completion_tokens", 0)),
                                   time.perf_counter() - started)
                except Exception as exc:
                    last_error = exc
                    await asyncio.sleep(2 ** attempt)
            raise RuntimeError(f"generation failed after {retries} deterministic retries: {last_error}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(args.data)
    direct = json.loads(DIRECT_MANIFEST.read_text(encoding="utf-8"))
    excluded = {int(x) for x in direct["sampled_source_rows"]}
    eligible = [i for i in range(len(data)) if i not in excluded]
    if len(eligible) < args.cases:
        raise ValueError("not enough official test cases after excluding the strong-expert subset")
    selected = random.Random(args.seed).sample(eligible, args.cases)
    cases = [public_state_from_row(data.iloc[i].to_dict(), i) for i in selected]
    model_dir = args.artifacts / "checkpoint"
    no_rel_dir = args.artifacts / "checkpoint_no_reliability"
    model, no_rel_model = _load_model(model_dir), _load_model(no_rel_dir)
    train_summary = json.loads((args.artifacts / "training_summary.json").read_text(encoding="utf-8"))
    threshold_cfg = json.loads((args.artifacts / "baseline_policy_config.json").read_text(encoding="utf-8"))
    service_path = HERE / "outputs/model_service_manifest.json"
    service = json.loads(service_path.read_text(encoding="utf-8")) if service_path.exists() else {}
    policies = ["always_answer", "uncertainty_threshold", "fixed_cost_router", "learned_value_no_cost",
                "learned_value_router", "learned_value_higher_cost", "ablation_no_expert_reliability",
                "ablation_no_consult", "ablation_no_defer"]
    manifest_path = args.output_dir / "run_manifest.json"
    episodes_path = args.output_dir / "policy_episodes.jsonl"
    manifest = {
        "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": "actual stepwise action selection by the frozen Router on independent official test cases",
        "source_data": str(args.data.resolve()), "data_sha256": file_sha256(args.data),
        "case_count": len(cases), "selected_source_rows": selected, "excluded_direct_expert_rows": sorted(excluded),
        "seed": args.seed, "pilot_training_sha256": file_sha256(PILOT_TRAINING),
        "checkpoint_sha256": file_sha256(model_dir / "router_checkpoint.npz"),
        "no_reliability_checkpoint_sha256": file_sha256(no_rel_dir / "router_checkpoint.npz"),
        "model_roles": {"base": args.model_alias, "patient_simulator": args.model_alias, "expert_simulator": args.model_alias},
        "model_revision": args.model_revision, "model_service": service,
        "generation": {"base_temperature": args.temperature, "expert_temperature": args.temperature,
                       "patient_temperature": 0.0, "max_decisions": args.max_turns,
                       "max_consults": 1, "seed_strategy": "sha256(model,messages,temperature) per request"},
        "policy_names": policies,
        "input_boundary": "Router sees only current public user/assistant dialogue and structural features; hidden atomic_facts only enter Patient Simulator; expert sees public dialogue and consult question; gold is evaluated after episode",
        "reference_cost": {"ASK": 0.01, "CONSULT": 0.02, "DEFER": 0.02},
        "rollout_file": str(episodes_path.resolve()),
    }
    if manifest_path.exists() and args.resume:
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in ("data_sha256", "selected_source_rows", "checkpoint_sha256",
                      "no_reliability_checkpoint_sha256", "model_revision", "generation"):
            if old_manifest.get(field) != manifest.get(field):
                raise ValueError(f"resume configuration mismatch for {field}")
    elif episodes_path.exists():
        raise FileExistsError(f"{episodes_path} exists; pass --resume or choose another output directory")
    write_json(manifest_path, manifest)
    done = {}
    if args.resume and episodes_path.exists():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[(row["case_id"], row["policy"])] = row
    elif episodes_path.exists():
        raise FileExistsError(f"{episodes_path} exists; pass --resume or use another output directory")

    base = StableSeedChatAPI(args.base_url, args.model_alias, timeout_s=args.timeout_s, enable_thinking=False)
    patient = StableSeedChatAPI(args.base_url, args.model_alias, timeout_s=args.timeout_s, enable_thinking=False)
    expert = StableSeedChatAPI(args.base_url, args.model_alias, timeout_s=args.timeout_s, enable_thinking=False)
    started_all = time.perf_counter()
    try:
        with episodes_path.open("a", encoding="utf-8") as writer:
            total = len(cases) * len(policies)
            index = 0
            for case in cases:
                for policy in policies:
                    index += 1
                    key = (case["case_id"], policy)
                    if key in done:
                        continue
                    started = time.perf_counter()
                    row = await run_one_policy(case, policy, model, no_rel_model, threshold_cfg,
                                               train_summary, base, patient, expert,
                                               args.temperature, args.max_turns)
                    row["elapsed_wall_s"] = time.perf_counter() - started
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                    writer.flush()
                    done[key] = row
                    if index % 20 == 0 or index == total:
                        print(f"stepwise episodes {index}/{total}; elapsed {time.perf_counter()-started_all:.1f}s", flush=True)
    finally:
        await base.close(); await patient.close(); await expert.close()
    manifest.update({"status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "episode_count": len(done), "elapsed_wall_s": time.perf_counter() - started_all,
                     "episodes_sha256": file_sha256(episodes_path)})
    write_json(manifest_path, manifest)
    print(f"completed {len(done)} policy/case episodes", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifacts", type=Path, default=HERE / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cases", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--base-url", default="http://127.0.0.1:30131/v1")
    parser.add_argument("--model-alias", default="qwen3-8b-router")
    parser.add_argument("--model-revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
