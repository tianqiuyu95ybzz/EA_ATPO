#!/usr/bin/env python3
"""Restore Attempt09 checkpoints and run a matched 20-case policy evaluation.

No prompts, gold answers, atomic facts, or simulator text are written to the
result artifacts. The case-level file contains only a hash, answers, actions,
resource counts, and post-hoc correctness.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ea_atpo_stage"))
sys.path.insert(0, str(ROOT / "ATPO-official/codes"))
from recipe.ea_atpo.ea_agent_loop import (  # noqa: E402
    PATIENT_SYSTEM,
    _decision_observation,
    _install_action_protocol,
    _render_public_initial,
)
from recipe.ea_atpo.protocol import (  # noqa: E402
    Action,
    PublicState,
    build_expert_messages,
    parse_action,
    validate_expert_response,
    visible_response,
)


RUN_DIR = ROOT / "experiments/ea_atpo_v1_smoke/ea_atpo_v1_attempt09_stability"
RECOVERY = RUN_DIR / "recovery"
ATTEMPT09_ACTOR = RECOVERY / "hf/attempt09_actor"
ATTEMPT09_CRITIC = RECOVERY / "hf/attempt09_critic"
ORIGINAL_ACTOR = RECOVERY / "hf/original_atpo_attempt20_actor"
BASE_MODEL = Path(
    "/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B"
)
OUT_DIR = RUN_DIR / "paired_evaluation"
GPU = "cuda:0"
MAX_NEW_TOKENS = 512
MAX_ACTOR_ACTIONS = 3
MAX_ASKS = 1
MAX_CONSULTS = 1
COSTS = {"ASK": 0.05, "CONSULT": 0.20, "DEFER": 0.60}
ACTION_PREFIXES = {
    "ANSWER": "Final Answer:",
    "ASK": "Question:",
    "CONSULT": "Consult Expert:",
    "DEFER": "Defer to Expert",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value: Any):
    """Make diagnostic manifests robust to sets and scalar wrapper types."""
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def hash_model_files(path: Path) -> dict[str, str]:
    keep = {
        "config.json", "generation_config.json", "model.safetensors.index.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "vocab.json", "merges.txt",
    }
    files = [
        p for p in path.rglob("*")
        if p.is_file() and not p.is_symlink()
        and (p.name.endswith(".safetensors") or p.name in keep)
    ]
    return {str(p.relative_to(path)): sha256_file(p) for p in sorted(files)}


def hash_fsdp_weights(path: Path) -> dict[str, str]:
    files = list(path.glob("model_world_size_*_rank_*.pt"))
    files += [p for p in (path / "huggingface").glob("*") if p.is_file() and p.name in {"config.json", "generation_config.json"}]
    fsdp_cfg = path / "fsdp_config.json"
    if fsdp_cfg.exists():
        files.append(fsdp_cfg)
    return {str(p.relative_to(path)): sha256_file(p) for p in sorted(files)}


def checkpoint_inventory(path: Path) -> dict[str, Any]:
    files = sorted(p for p in path.rglob("*") if p.is_file() and not p.is_symlink())
    return {
        "path": str(path.relative_to(ROOT)),
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "files": [{"path": str(p.relative_to(path)), "bytes": p.stat().st_size} for p in files],
    }


def hash_tokenizer_files(path: Path) -> dict[str, str]:
    names = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt"}
    return {p.name: sha256_file(p) for p in sorted(path.iterdir()) if p.is_file() and p.name in names}


def case_hash(index: Any) -> str:
    return hashlib.sha256(str(index).encode()).hexdigest()[:24]


def parse_gold(row: dict[str, Any]) -> str:
    obj = row.get("reward_model", {}).get("ground_truth", row.get("ground_truth", {}))
    if isinstance(obj, dict):
        obj = obj.get("answer", "")
    return str(obj).strip().upper()


def normalize_prompt(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [{"role": str(m.get("role", "user")), "content": str(m.get("content", ""))} for m in value]


def normalize_facts(value: Any) -> list[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [str(x) for x in value]


def model_load(path: Path, device: str = GPU):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval().to(device)
    return model


def tokenizer_load(path: Path):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


@torch.inference_mode()
def generate_ids(model, prompt_ids: Sequence[int], *, max_new_tokens: int = MAX_NEW_TOKENS):
    ids = torch.tensor([list(map(int, prompt_ids))], dtype=torch.long, device=GPU)
    started = time.perf_counter()
    generated = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=max(1, int(max_new_tokens)),
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
    )
    elapsed = time.perf_counter() - started
    response = generated.sequences[0, ids.shape[1] :].detach().cpu().tolist()
    token_logprobs: list[float] = []
    for step, scores in enumerate(generated.scores):
        token_id = int(generated.sequences[0, ids.shape[1] + step].item())
        token_logprobs.append(float(torch.log_softmax(scores[0].float(), dim=-1)[token_id].item()))
    return response, token_logprobs, elapsed


def template_ids(tokenizer, messages: Sequence[dict[str, str]], *, thinking: bool) -> list[int]:
    kwargs = {"add_generation_prompt": True, "tokenize": True}
    if not thinking:
        kwargs["enable_thinking"] = False
    value = tokenizer.apply_chat_template(list(messages), **kwargs)
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    return [int(x) for x in value]


def observation_ids(tokenizer, content: str, *, thinking: bool) -> list[int]:
    extra = tokenizer.encode("\n", add_special_tokens=False)
    return list(map(int, extra)) + template_ids(
        tokenizer, [{"role": "user", "content": content}], thinking=thinking
    )


def response_finish_reason(model, response_ids: Sequence[int]) -> str:
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    return "eos" if response_ids and response_ids[-1] in eos_ids else "length_or_stop"


def format_action(action: Action, content: str) -> str:
    if action is Action.ANSWER:
        return f"Final Answer: {content.strip().upper()}"
    if action is Action.ASK:
        return f"Question: {content.strip()}"
    if action is Action.CONSULT:
        return f"Consult Expert: {content.strip()}"
    return "Defer to Expert"


def answer_from_text(text: str) -> str | None:
    try:
        action, content = parse_evaluation_action(text)
    except ValueError:
        return None
    return content.strip().upper() if action is Action.ANSWER else None


def parse_evaluation_action(text: str) -> tuple[Action, str]:
    """Shared answer/action parser; accepts a bare single letter for every policy."""
    try:
        return parse_action(text)
    except ValueError:
        tail = visible_response(text)
        tail = re.sub(r"(?:<\|im_end\|>|<\|endoftext\|>|<\|fim_suffix\|>|</s>)+\s*$", "", tail).strip()
        if re.fullmatch(r"[A-D]", tail, re.IGNORECASE):
            return Action.ANSWER, tail.upper()
        raise


@torch.inference_mode()
def prefix_likelihoods(model, tokenizer, prompt_ids: Sequence[int]) -> dict[str, float]:
    """Likelihood of each literal action label, normalized over those labels.

    This is a prefix-only relative likelihood, not a categorical action head and
    not a probability over all possible phrasings of an action.
    """
    context = torch.tensor([list(map(int, prompt_ids))], dtype=torch.long, device=GPU)
    context_len = context.shape[1]
    loglik: dict[str, float] = {}
    for action, literal in ACTION_PREFIXES.items():
        token_ids = tokenizer.encode(literal, add_special_tokens=False)
        if not token_ids:
            loglik[action] = float("-inf")
            continue
        prefix = torch.tensor([token_ids], dtype=torch.long, device=GPU)
        # The model input needs every candidate token except its final target.
        model_input = torch.cat([context, prefix[:, :-1]], dim=1)
        logits = model(input_ids=model_input, use_cache=False).logits[0]
        target_positions = torch.arange(context_len - 1, context_len - 1 + len(token_ids), device=GPU)
        target_ids = prefix[0]
        selected = torch.log_softmax(logits[target_positions].float(), dim=-1).gather(
            1, target_ids.unsqueeze(1)
        )
        loglik[action] = float(selected.sum().item())
    max_log = max(loglik.values())
    exps = {k: math.exp(v - max_log) if math.isfinite(v) else 0.0 for k, v in loglik.items()}
    denom = sum(exps.values()) or 1.0
    return {k: v / denom for k, v in exps.items()}


@torch.inference_mode()
def complete_text(model, tokenizer, messages: Sequence[dict[str, str]], *, thinking: bool,
                  max_new_tokens: int = MAX_NEW_TOKENS) -> dict[str, Any]:
    prompt_ids = template_ids(tokenizer, messages, thinking=thinking)
    response_ids, logprobs, latency = generate_ids(model, prompt_ids, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(response_ids, skip_special_tokens=False)
    return {
        "text": text,
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(response_ids),
        "response_logprob_sum": sum(logprobs),
        "response_probability": math.exp(max(-745.0, min(0.0, sum(logprobs)))),
        "latency_s": latency,
        "finish_reason": response_finish_reason(model, response_ids),
    }


def public_state_from_prompt(prompt: Sequence[dict[str, str]], sample_id: str) -> PublicState:
    return PublicState(
        sample_id=sample_id,
        initial_messages=_render_public_initial(prompt),
        remaining_interaction_budget=MAX_ACTOR_ACTIONS,
    )


def patient_response(base_model, tokenizer, facts: Sequence[str], question: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": PATIENT_SYSTEM},
        {"role": "user", "content": "Patient atomic facts:\n" + json.dumps(list(facts), ensure_ascii=False) + "\n\nDoctor question:\n" + question},
    ]
    result = complete_text(base_model, tokenizer, messages, thinking=False, max_new_tokens=256)
    result["text"] = visible_response(result["text"])
    return result


def expert_response(base_model, tokenizer, state: PublicState, action: Action,
                    question: str = "") -> dict[str, Any]:
    messages = build_expert_messages(state, action, consult_question=question)
    total_prompt = total_completion = attempts = 0
    total_latency = 0.0
    last_error = None
    for attempt in range(2):
        current = list(messages)
        if attempt:
            correction = (
                "Format correction: return exactly one `Expert Advice: ...` suggestion; do not name an option, state a correct answer, or give a final answer."
                if action is Action.CONSULT
                else "Format correction: choose one option and return exactly `Expert Final Answer: A` (replace A with your choice). No rationale."
            )
            current.append({"role": "user", "content": correction})
        result = complete_text(base_model, tokenizer, current, thinking=False, max_new_tokens=128)
        total_prompt += result["prompt_tokens"]
        total_completion += result["completion_tokens"]
        total_latency += result["latency_s"]
        attempts += 1
        try:
            normalized = validate_expert_response(action, result["text"])
            return {
                "text": normalized,
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "latency_s": total_latency,
                "attempts": attempts,
                "format_failure": False,
            }
        except ValueError as exc:
            last_error = str(exc)
    return {
        "text": "",
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "latency_s": total_latency,
        "attempts": attempts,
        "format_failure": True,
        "failure_category": "consult_protocol_error" if action is Action.CONSULT else "defer_protocol_error",
        "parse_reason_code": "answer_leak_or_format" if "CONSULT" in str(last_error) else "unparseable_expert_output",
    }


def run_always_answer(model, tokenizer, row: dict[str, Any]) -> dict[str, Any]:
    prompt = normalize_prompt(row["prompt"])
    messages = [dict(m) for m in prompt]
    system_idx = next((i for i, m in enumerate(messages) if m["role"] == "system"), None)
    directive = (
        "\n\nMatched evaluation baseline: do not ask the patient, do not consult or defer. "
        "Choose one option now. Output exactly `Final Answer: <A|B|C|D>` and nothing else."
    )
    if system_idx is None:
        messages.insert(0, {"role": "system", "content": directive.strip()})
    else:
        messages[system_idx]["content"] += directive
    result = complete_text(model, tokenizer, messages, thinking=False)
    answer = answer_from_text(result["text"])
    return policy_record(row, answer, ["ANSWER"], [result], [None], invalid=answer is None)


def run_original_atpo(model, base_model, tokenizer, row: dict[str, Any]) -> dict[str, Any]:
    prompt = normalize_prompt(row["prompt"])
    facts = normalize_facts(row["extra_info"]["atomic_facts"])
    # Use the same non-thinking chat-template setting as Always Answer and
    # Attempt09. The prompt still uses Original ATPO's native ASK/ANSWER
    # protocol; only the template generation flag is standardized.
    budget_rule = (
        "\n\nMatched evaluation constraints: this Original ATPO baseline supports only "
        "ASK and ANSWER, has at most 3 actor decisions and at most 1 patient question. "
        "Do not consult or defer. Follow the latest decision-state limits."
    )
    system_idx = next((i for i, m in enumerate(prompt) if m["role"] == "system"), None)
    if system_idx is None:
        prompt.insert(0, {"role": "system", "content": budget_rule.strip()})
    else:
        prompt[system_idx]["content"] += budget_rule
    actor_prompt_ids = template_ids(tokenizer, prompt, thinking=False)
    generated: list[dict[str, Any]] = []
    actions: list[str] = []
    answer = None
    invalid = False
    patient_calls = expert_calls = 0
    extra_tokens = 0
    extra_latency = 0.0
    cumulative_response_tokens = 0
    for _ in range(MAX_ACTOR_ACTIONS):
        prompt_ids = actor_prompt_ids
        room = max(1, MAX_NEW_TOKENS - cumulative_response_tokens)
        response_ids, logprobs, latency = generate_ids(model, prompt_ids, max_new_tokens=room)
        cumulative_response_tokens += len(response_ids)
        text = tokenizer.decode(response_ids, skip_special_tokens=False)
        generated.append({
            "prompt_tokens": len(prompt_ids), "completion_tokens": len(response_ids),
            "response_logprob_sum": sum(logprobs),
            "response_probability": math.exp(max(-745.0, min(0.0, sum(logprobs)))),
            "latency_s": latency,
            "finish_reason": response_finish_reason(model, response_ids),
        })
        try:
            action, content = parse_evaluation_action(text)
        except ValueError:
            invalid = True
            actions.append("INVALID")
            break
        if action not in (Action.ANSWER, Action.ASK):
            invalid = True
            actions.append("INVALID")
            break
        actions.append(action.value)
        if action is Action.ANSWER:
            answer = content.strip().upper()
            break
        if patient_calls >= MAX_ASKS or len(actions) >= MAX_ACTOR_ACTIONS:
            # The action was sampled but cannot execute under the common budget.
            actions[-1] = "INVALID"
            invalid = True
            break
        patient_calls += 1
        patient = patient_response(base_model, tokenizer, facts, content)
        extra_tokens += int(patient["prompt_tokens"] + patient["completion_tokens"])
        extra_latency += float(patient["latency_s"])
        patient_msg = str(patient["text"]) + (
            "\n\nEvaluation decision state: actor actions used="
            f"{len(actions)}/{MAX_ACTOR_ACTIONS}; remaining={MAX_ACTOR_ACTIONS - len(actions)}; "
            f"ASK used={patient_calls}/{MAX_ASKS}. Legal action: Final Answer only."
        )
        obs_ids = observation_ids(tokenizer, patient_msg, thinking=False)
        cumulative_response_tokens += len(obs_ids)
        if cumulative_response_tokens >= MAX_NEW_TOKENS:
            invalid = True
            break
        actor_prompt_ids = list(actor_prompt_ids) + list(response_ids) + obs_ids
    return policy_record(row, answer, actions, generated, [None] * len(generated), invalid=invalid,
                         patient_calls=patient_calls, expert_calls=expert_calls,
                         extra_tokens=extra_tokens, extra_latency=extra_latency)


def run_ea_atpo(model, base_model, tokenizer, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    prompt = normalize_prompt(row["prompt"])
    facts = normalize_facts(row["extra_info"]["atomic_facts"])
    sample_id = case_hash(row["extra_info"].get("index", row.get("index")))
    messages = _install_action_protocol(prompt, budget=MAX_ACTOR_ACTIONS, max_asks=MAX_ASKS, max_consults=MAX_CONSULTS)
    actor_prompt_ids = template_ids(tokenizer, messages, thinking=False)
    state = public_state_from_prompt(prompt, sample_id)
    actor_turns: list[dict[str, Any]] = []
    actions: list[str] = []
    token_total = 0
    latency_total = 0.0
    patient_calls = expert_calls = expert_retries = protocol_failures = 0
    answer = None
    invalid = False
    consult_counterfactual = None
    cumulative_response_tokens = 0

    for _ in range(MAX_ACTOR_ACTIONS):
        room = max(1, MAX_NEW_TOKENS - cumulative_response_tokens)
        response_ids, logprobs, latency = generate_ids(model, actor_prompt_ids, max_new_tokens=room)
        text = tokenizer.decode(response_ids, skip_special_tokens=False)
        probs = prefix_likelihoods(model, tokenizer, actor_prompt_ids)
        event = {
            "prompt_tokens": len(actor_prompt_ids), "completion_tokens": len(response_ids),
            "response_logprob_sum": sum(logprobs),
            "response_probability": math.exp(max(-745.0, min(0.0, sum(logprobs)))),
            "prefix_likelihoods": probs,
            "latency_s": latency,
            "finish_reason": response_finish_reason(model, response_ids),
        }
        actor_turns.append(event)
        token_total += event["prompt_tokens"] + event["completion_tokens"]
        latency_total += latency
        cumulative_response_tokens += len(response_ids)
        try:
            action, content = parse_evaluation_action(text)
        except ValueError:
            invalid = True
            actions.append("INVALID")
            break
        actions.append(action.value)
        action_turn = {"role": "assistant", "content": format_action(action, content)}
        if action is Action.ANSWER:
            answer = content.strip().upper()
            state = PublicState(
                sample_id=state.sample_id,
                initial_messages=state.initial_messages,
                dialogue=(*state.dialogue, action_turn),
                asked_count=state.asked_count,
                consulted_count=state.consulted_count,
                remaining_interaction_budget=max(0, MAX_ACTOR_ACTIONS - len(actions)),
                actor_actions_used=len(actions),
            )
            break
        if action is Action.ASK:
            if state.asked_count >= MAX_ASKS or len(actions) >= MAX_ACTOR_ACTIONS:
                invalid = True
                break
            patient = patient_response(base_model, tokenizer, facts, content)
            patient_calls += 1
            token_total += patient["prompt_tokens"] + patient["completion_tokens"]
            latency_total += patient["latency_s"]
            patient_text = str(patient["text"])
            state = PublicState(
                sample_id=state.sample_id,
                initial_messages=state.initial_messages,
                dialogue=(*state.dialogue, action_turn, {"role": "user", "content": patient_text}),
                asked_count=state.asked_count + 1,
                consulted_count=state.consulted_count,
                remaining_interaction_budget=max(0, MAX_ACTOR_ACTIONS - len(actions)),
                actor_actions_used=len(actions),
            )
            observation = _decision_observation(
                "Patient", patient_text, actor_actions_used=len(actions),
                remaining=max(0, MAX_ACTOR_ACTIONS - len(actions)), asked=state.asked_count,
                max_asks=MAX_ASKS, consulted=state.consulted_count, max_consults=MAX_CONSULTS,
            )
        elif action is Action.CONSULT:
            if state.consulted_count >= MAX_CONSULTS or len(actions) >= MAX_ACTOR_ACTIONS:
                invalid = True
                break
            if consult_counterfactual is None:
                # Same exact pre-consult token context, adding only a final-answer
                # counterfactual instruction. The consult question and advice are
                # absent from this run.
                cf_instruction = (
                    "Counterfactual measurement: answer this visible state now. "
                    "Do not ask, consult, or defer. Output exactly `Final Answer: <A|B|C|D>`."
                )
                cf_prompt_ids = list(actor_prompt_ids) + observation_ids(
                    tokenizer, cf_instruction, thinking=False
                )
                cf_ids, cf_logprobs, cf_latency = generate_ids(model, cf_prompt_ids)
                cf_text = tokenizer.decode(cf_ids, skip_special_tokens=False)
                consult_counterfactual = {
                    "answer": answer_from_text(cf_text),
                    "prompt_tokens": len(cf_prompt_ids),
                    "completion_tokens": len(cf_ids),
                    "latency_s": cf_latency,
                }
            public_after_question = PublicState(
                sample_id=state.sample_id,
                initial_messages=state.initial_messages,
                dialogue=(*state.dialogue, action_turn),
                asked_count=state.asked_count,
                consulted_count=state.consulted_count + 1,
                remaining_interaction_budget=max(0, MAX_ACTOR_ACTIONS - len(actions)),
                actor_actions_used=len(actions),
            )
            expert = expert_response(base_model, tokenizer, public_after_question, Action.CONSULT, question=content)
            expert_calls += 1
            expert_retries += max(0, expert["attempts"] - 1)
            protocol_failures += int(expert["format_failure"])
            token_total += expert["prompt_tokens"] + expert["completion_tokens"]
            latency_total += expert["latency_s"]
            if expert["format_failure"]:
                invalid = True
                break
            advice = expert["text"]
            state = PublicState(
                sample_id=state.sample_id,
                initial_messages=state.initial_messages,
                dialogue=(*state.dialogue, action_turn, {"role": "user", "content": advice}),
                asked_count=state.asked_count,
                consulted_count=state.consulted_count + 1,
                remaining_interaction_budget=max(0, MAX_ACTOR_ACTIONS - len(actions)),
                actor_actions_used=len(actions),
            )
            observation = _decision_observation(
                "Expert advice", advice, actor_actions_used=len(actions),
                remaining=max(0, MAX_ACTOR_ACTIONS - len(actions)), asked=state.asked_count,
                max_asks=MAX_ASKS, consulted=state.consulted_count, max_consults=MAX_CONSULTS,
            )
        else:  # DEFER
            if len(actions) >= MAX_ACTOR_ACTIONS:
                invalid = True
                break
            expert = expert_response(base_model, tokenizer, state, Action.DEFER)
            expert_calls += 1
            expert_retries += max(0, expert["attempts"] - 1)
            protocol_failures += int(expert["format_failure"])
            token_total += expert["prompt_tokens"] + expert["completion_tokens"]
            latency_total += expert["latency_s"]
            if expert["format_failure"]:
                invalid = True
                break
            m = re.search(r"Expert Final Answer:\s*([A-D])\b", expert["text"], re.IGNORECASE)
            answer = m.group(1).upper() if m else None
            state = PublicState(
                sample_id=state.sample_id,
                initial_messages=state.initial_messages,
                dialogue=(*state.dialogue, action_turn, {"role": "user", "content": expert["text"]}),
                asked_count=state.asked_count,
                consulted_count=state.consulted_count,
                remaining_interaction_budget=max(0, MAX_ACTOR_ACTIONS - len(actions)),
                actor_actions_used=len(actions),
            )
            break

        obs_ids = observation_ids(tokenizer, observation, thinking=False)
        cumulative_response_tokens += len(obs_ids)
        if cumulative_response_tokens >= MAX_NEW_TOKENS:
            invalid = True
            break
        actor_prompt_ids = list(actor_prompt_ids) + list(response_ids) + obs_ids

    record = policy_record(row, answer, actions, actor_turns, [None] * len(actor_turns), invalid=invalid,
                           patient_calls=patient_calls, expert_calls=expert_calls,
                           extra_tokens=token_total - sum(t["prompt_tokens"] + t["completion_tokens"] for t in actor_turns),
                           extra_latency=latency_total - sum(t["latency_s"] for t in actor_turns),
                           expert_retries=expert_retries, expert_protocol_failures=protocol_failures)
    if consult_counterfactual is not None:
        record["consult_counterfactual_answer"] = consult_counterfactual["answer"]
        record["consult_counterfactual_correct"] = (
            consult_counterfactual["answer"] == parse_gold(row) if consult_counterfactual["answer"] else False
        )
        record["consult_counterfactual_tokens"] = consult_counterfactual["prompt_tokens"] + consult_counterfactual["completion_tokens"]
        record["consult_counterfactual_latency_s"] = consult_counterfactual["latency_s"]
    return record, consult_counterfactual


def policy_record(row: dict[str, Any], answer: str | None, actions: list[str], turns: list[dict[str, Any]],
                  _unused, *, invalid: bool = False, patient_calls: int = 0, expert_calls: int = 0,
                  extra_tokens: int = 0, extra_latency: float = 0.0, expert_retries: int = 0,
                  expert_protocol_failures: int = 0) -> dict[str, Any]:
    gold = parse_gold(row)
    counts = Counter(actions)
    actor_tokens = sum(int(t.get("prompt_tokens", 0)) + int(t.get("completion_tokens", 0)) for t in turns)
    total_tokens = actor_tokens + int(extra_tokens)
    total_latency = sum(float(t.get("latency_s", 0.0)) for t in turns) + float(extra_latency)
    return {
        "case_id_sha256": case_hash(row["extra_info"].get("index", row.get("index"))),
        "final_answer": answer,
        "correct": bool(answer and answer == gold),
        "parseable": bool(answer),
        "invalid_or_incomplete": bool(invalid or not answer),
        "actions": actions,
        "answer_count": counts["ANSWER"],
        "ask_count": counts["ASK"],
        "consult_count": counts["CONSULT"],
        "defer_count": counts["DEFER"],
        "invalid_action_count": counts["INVALID"],
        "actor_decisions": len(actions),
        "patient_calls": int(patient_calls),
        "expert_calls": int(expert_calls),
        "expert_retries": int(expert_retries),
        "expert_protocol_failures": int(expert_protocol_failures),
        "total_tokens": int(total_tokens),
        "total_latency_s": float(total_latency),
        "clinical_action_cost": sum(COSTS.get(a, 0.0) for a in actions),
        "action_response_probabilities": [t.get("response_probability") for t in turns],
        "action_prefix_likelihoods": [t.get("prefix_likelihoods") for t in turns],
        "actor_finish_reasons": [t.get("finish_reason") for t in turns],
    }


def paired_bootstrap(rows: list[dict[str, Any]], left: str, right: str, samples: int = 20000):
    import numpy as np
    rng = np.random.default_rng(20261115)
    diffs = np.array([int(r[left]) - int(r[right]) for r in rows], dtype=float)
    boot = np.empty(samples, dtype=float)
    for i in range(samples):
        boot[i] = rng.choice(diffs, size=len(diffs), replace=True).mean()
    return {
        "mean_accuracy_difference": float(diffs.mean()),
        "ci95_percentile": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "wins": int((diffs > 0).sum()),
        "ties": int((diffs == 0).sum()),
        "losses": int((diffs < 0).sum()),
    }


def summarize_method(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    denom = max(1, n)
    return {
        "cases": n,
        "correct": sum(int(r["correct"]) for r in records),
        "accuracy": sum(int(r["correct"]) for r in records) / denom,
        "parseable": sum(int(r["parseable"]) for r in records),
        "action_events": {
            name: sum(r[f"{name.lower()}_count"] for r in records)
            for name in ("ANSWER", "ASK", "CONSULT", "DEFER")
        },
        "action_rate_per_case": {
            name: sum(r[f"{name.lower()}_count"] for r in records) / denom
            for name in ("ANSWER", "ASK", "CONSULT", "DEFER")
        },
        "mean_tokens": sum(r["total_tokens"] for r in records) / denom,
        "mean_actor_decisions": sum(r["actor_decisions"] for r in records) / denom,
        "mean_latency_s": sum(r["total_latency_s"] for r in records) / denom,
        "patient_calls": sum(r["patient_calls"] for r in records),
        "expert_calls": sum(r["expert_calls"] for r in records),
        "expert_retries": sum(r["expert_retries"] for r in records),
        "expert_protocol_failures": sum(r["expert_protocol_failures"] for r in records),
        "mean_clinical_action_cost": sum(r["clinical_action_cost"] for r in records) / denom,
    }


def load_critic_and_probe(tokenizer):
    critic = AutoModelForTokenClassification.from_pretrained(
        ATTEMPT09_CRITIC,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval().to("cpu")
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Value probe. Return a scalar value."}],
        add_generation_prompt=True,
        tokenize=True,
    )
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.inference_mode():
        logits = critic(input_ids=ids).logits
    values = logits[0, :, 0]
    return {
        "loaded_class": type(critic).__name__,
        "parameter_count": sum(p.numel() for p in critic.parameters()),
        "probe_value_finite": bool(torch.isfinite(values).all()),
        "probe_value_last": float(values[-1]),
        "critic_score_head_shape": list(critic.score.weight.shape),
    }


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for required in (ATTEMPT09_ACTOR / "config.json", ATTEMPT09_CRITIC / "config.json", ORIGINAL_ACTOR / "config.json", BASE_MODEL / "config.json"):
        if not required.exists():
            raise SystemExit(f"Required recovered model file is missing: {required}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; paired GPU inference was not run")
    data_path = RUN_DIR / "data/test.parquet"
    df = pd.read_parquet(data_path)
    if len(df) != 20:
        raise SystemExit(f"Expected the fixed 20-case dev set; found {len(df)}")
    rows = [row for _, row in df.iterrows()]
    for row in rows:
        if row.get("prompt") is None or row.get("extra_info") is None:
            raise SystemExit("A dev case is missing prompt/extra_info")

    base_tok = tokenizer_load(BASE_MODEL)
    attempt_tok = tokenizer_load(ATTEMPT09_ACTOR)
    orig_tok = tokenizer_load(ORIGINAL_ACTOR)
    if len(base_tok) != len(attempt_tok) or len(base_tok) != len(orig_tok):
        raise SystemExit("Recovered model tokenizers have different vocabulary sizes")
    tokenizer_hashes = {
        "base_tokenizer": hash_tokenizer_files(BASE_MODEL),
        "attempt09_tokenizer": hash_tokenizer_files(ATTEMPT09_ACTOR),
        "original_attempt20_tokenizer": hash_tokenizer_files(ORIGINAL_ACTOR),
    }

    critic_probe = load_critic_and_probe(attempt_tok)

    # Keep the shared Patient/Expert simulator loaded beside one candidate Actor.
    base_model = model_load(BASE_MODEL)
    always_records = [run_always_answer(base_model, base_tok, row) for row in rows]

    original_model = model_load(ORIGINAL_ACTOR)
    original_records = [run_original_atpo(original_model, base_model, orig_tok, row) for row in rows]
    del original_model
    torch.cuda.empty_cache()

    attempt_model = model_load(ATTEMPT09_ACTOR)
    ea_records = []
    for row in rows:
        record, _ = run_ea_atpo(attempt_model, base_model, attempt_tok, row)
        ea_records.append(record)
    del attempt_model, base_model
    torch.cuda.empty_cache()

    by_method = {
        "always_answer": always_records,
        "original_atpo_attempt20": original_records,
        "ea_atpo_attempt09": ea_records,
    }
    case_rows = []
    maps = {name: {r["case_id_sha256"]: r for r in records} for name, records in by_method.items()}
    hashes = sorted(maps["always_answer"])
    for case_id in hashes:
        item: dict[str, Any] = {"case_id_sha256": case_id}
        for name, records in maps.items():
            r = records[case_id]
            item[f"{name}_answer"] = r["final_answer"] or ""
            item[f"{name}_correct"] = int(r["correct"])
            item[f"{name}_actions"] = ";".join(r["actions"])
            item[f"{name}_tokens"] = r["total_tokens"]
            item[f"{name}_rounds"] = r["actor_decisions"]
        ea = maps["ea_atpo_attempt09"][case_id]
        item["ea_consult_counterfactual_answer"] = ea.get("consult_counterfactual_answer") or ""
        item["ea_consult_counterfactual_correct"] = int(ea.get("consult_counterfactual_correct", False))
        case_rows.append(item)

    with (OUT_DIR / "paired_case_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(case_rows)

    # Per-case detail stores only action labels and numeric telemetry. It omits
    # public case text, hidden facts, gold, patient text, and Expert text.
    with (OUT_DIR / "policy_telemetry.jsonl").open("w") as f:
        for method, records in by_method.items():
            for record in records:
                f.write(json.dumps({"method": method, **record}, sort_keys=True) + "\n")

    summaries = {name: summarize_method(records) for name, records in by_method.items()}
    paired = {
        "ea_vs_always_answer": paired_bootstrap(case_rows, "ea_atpo_attempt09_correct", "always_answer_correct"),
        "ea_vs_original_atpo_attempt20": paired_bootstrap(case_rows, "ea_atpo_attempt09_correct", "original_atpo_attempt20_correct"),
        "original_atpo_attempt20_vs_always_answer": paired_bootstrap(case_rows, "original_atpo_attempt20_correct", "always_answer_correct"),
    }

    consulted = [r for r in ea_records if r["consult_count"] > 0]
    cf_pairs = [r for r in consulted if r.get("consult_counterfactual_answer")]
    consult_analysis = {
        "cases_with_consult": len(consulted),
        "cases_with_parseable_same_state_answer_only_counterfactual": len(cf_pairs),
        "corrected_wrong_to_right": sum(not r["consult_counterfactual_correct"] and r["correct"] for r in cf_pairs),
        "degraded_right_to_wrong": sum(r["consult_counterfactual_correct"] and not r["correct"] for r in cf_pairs),
        "unchanged_correctness": sum(r["consult_counterfactual_correct"] == r["correct"] for r in cf_pairs),
        "consult_counterfactual_unparseable": len(consulted) - len(cf_pairs),
    }
    consult_analysis["correction_rate_given_consult"] = (
        consult_analysis["corrected_wrong_to_right"] / len(cf_pairs) if cf_pairs else None
    )
    consult_analysis["degradation_rate_given_consult"] = (
        consult_analysis["degraded_right_to_wrong"] / len(cf_pairs) if cf_pairs else None
    )

    # Compare the restored Attempt09 model with its saved SGLang validation trace.
    prior_trace_path = RUN_DIR / "traces/validation.jsonl"
    prior = {}
    for line in prior_trace_path.read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            prior[obj["case_id_sha256"]] = obj
    consistency = {"saved_sglang_validation_cases": len(prior), "reproduced_cases": len(ea_records), "action_sequence_matches": 0, "final_correctness_matches": 0, "final_answer_matches": 0, "per_case": []}
    for r in ea_records:
        old = prior.get(r["case_id_sha256"], {})
        old_actions = [e.get("action") or "INVALID" for e in old.get("action_trace", [])]
        action_match = old_actions == r["actions"]
        consistency["action_sequence_matches"] += int(action_match)
        consistency["final_correctness_matches"] += int(bool(old.get("evaluation", {}).get("correct")) == r["correct"])
        consistency["final_answer_matches"] += int((old.get("final_answer") or "") == (r["final_answer"] or ""))
        consistency["per_case"].append({
            "case_id_sha256": r["case_id_sha256"],
            "action_sequence_matches": action_match,
            "final_correctness_matches": bool(old.get("evaluation", {}).get("correct")) == r["correct"],
            "final_answer_matches": (old.get("final_answer") or "") == (r["final_answer"] or ""),
        })

    action_prefix_summary: dict[str, Any] = {}
    action_names = ("ANSWER", "ASK", "CONSULT", "DEFER")
    for action in action_names:
        occurrences = [
            (r, i) for r in ea_records for i, a in enumerate(r["actions"]) if a == action
        ]
        values = [r["action_prefix_likelihoods"][i].get(action) for r, i in occurrences
                  if i < len(r["action_prefix_likelihoods"]) and r["action_prefix_likelihoods"][i]]
        action_prefix_summary[action] = {
            "observed_events": len(occurrences),
            "mean_prefix_only_softmax_on_selected_action": sum(values) / len(values) if values else None,
            "note": "Conditional likelihood of the exact literal label prefix, softmaxed over four labels; not a four-way Actor head probability.",
        }

    manifest = {
        "evaluation_date": "2026-11-15",
        "attempt09_run_manifest_sha256": sha256_file(RUN_DIR / "run_manifest.json"),
        "attempt09_git_commit": json.loads((RUN_DIR / "run_manifest.json").read_text())["runtime_code_snapshot_commit"],
        "data": {"path": str(data_path.relative_to(ROOT)), "sha256": sha256_file(data_path), "cases": len(rows), "case_hashes": hashes},
        "checkpoints": {
            "attempt09_actor_fsdp_step10": checkpoint_inventory(RUN_DIR / "checkpoints/global_step_10/actor"),
            "attempt09_critic_fsdp_step10": checkpoint_inventory(RUN_DIR / "checkpoints/global_step_10/critic"),
            "attempt09_actor_hf_merged": {"path": str(ATTEMPT09_ACTOR.relative_to(ROOT)), "sha256_by_file": hash_model_files(ATTEMPT09_ACTOR)},
            "attempt09_critic_hf_merged": {"path": str(ATTEMPT09_CRITIC.relative_to(ROOT)), "sha256_by_file": hash_model_files(ATTEMPT09_CRITIC)},
            "original_atpo_attempt20_actor_fsdp_step2": checkpoint_inventory(ROOT / "experiments/ea_atpo_v1_smoke/original_atpo_attempt20/checkpoints/global_step_2/actor"),
            "original_atpo_attempt20_actor_hf_merged": {"path": str(ORIGINAL_ACTOR.relative_to(ROOT)), "sha256_by_file": hash_model_files(ORIGINAL_ACTOR)},
        },
        "model_recovery": {
            "attempt09_critic_probe": critic_probe,
            "tokenizer_files_sha256": tokenizer_hashes,
            "base_model_config_sha256": sha256_file(BASE_MODEL / "config.json"),
        },
        "generation": {
            "do_sample": False, "temperature": 0.0, "top_p": 1.0,
            "actor_episode_response_observation_token_budget": MAX_NEW_TOKENS,
            "actor_token_budget_note": "Cumulative generated responses plus observation tokens; each subsequent Actor call is capped to the remaining budget.",
            "patient_max_new_tokens": 256, "expert_max_new_tokens": 128, "actor_actions_budget": MAX_ACTOR_ACTIONS,
            "max_asks": MAX_ASKS, "max_consults": MAX_CONSULTS, "torch_dtype": "bfloat16",
            "attention_implementation": "sdpa", "device": "single allowed GPU (physical GPU4)",
            "attempt09_actor_chat_template": "enable_thinking=False, matching EAATPOAgentLoop",
            "original_atpo_chat_template": "enable_thinking=False, standardized with the other evaluated policies",
            "always_answer_chat_template": "enable_thinking=False; direct-answer system directive",
        },
        "evaluation_protocol": {
            "action_budget_by_method": {
                "always_answer": {"actor_decisions": 1, "ask": 0, "consult": 0, "defer": 0},
                "original_atpo_attempt20": {"actor_decisions": MAX_ACTOR_ACTIONS, "ask": MAX_ASKS, "consult": 0, "defer": 0},
                "ea_atpo_attempt09": {"actor_decisions": MAX_ACTOR_ACTIONS, "ask": MAX_ASKS, "consult": MAX_CONSULTS, "defer": 1, "defer_is_terminal": True},
            },
            "answer_parser": "Shared parse_evaluation_action: native action markers, with a bare single A-D letter fallback; unparseable result counts incorrect.",
            "raw_generation_text_saved": False,
        },
        "simulator": {"patient_model": "Qwen3-8B base checkpoint", "expert_model": "same Qwen3-8B base checkpoint", "greedy": True,
                      "patient_prompt": "Attempt09 PATIENT_SYSTEM contract", "expert_prompt": "Attempt09 build_expert_messages contract"},
        "clinical_action_costs_for_reporting_only": COSTS,
        "summaries": summaries,
        "paired_accuracy_differences": paired,
        "consult_same_state_counterfactual": consult_analysis,
        "attempt09_sglang_vs_hf_recovery_consistency": consistency,
        "attempt09_action_prefix_relative_likelihoods": action_prefix_summary,
        "original_atpo_checkpoint_caveat": "Attempt20 Original ATPO checkpoint is a 2-update engineering smoke trained on 2 cases with one assistant turn; its same-case comparison is descriptive and not a budget-matched baseline.",
        "privacy": {"case level contains hashes, action labels, answer letters, correctness, and numeric telemetry only; no prompt, gold, atomic facts, patient text, or expert advice is stored."},
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=json_default) + "\n")

    # Secondary JSON summary makes the report generator and audit reproducible.
    (OUT_DIR / "summary.json").write_text(json.dumps({"methods": summaries, "paired": paired,
                                                       "consult": consult_analysis,
                                                       "attempt09_consistency": consistency,
                                                       "action_prefix_summary": action_prefix_summary}, indent=2,
                                                      default=json_default) + "\n")
    print(json.dumps({"summaries": summaries, "paired": paired, "consult": consult_analysis,
                      "attempt09_consistency": consistency, "critic_probe": critic_probe}, indent=2))


if __name__ == "__main__":
    main()
