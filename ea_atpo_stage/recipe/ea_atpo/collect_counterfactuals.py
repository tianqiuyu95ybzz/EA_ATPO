"""Collect paired four-action counterfactual rollouts from OpenAI-compatible APIs.

This is a feasibility collector, not the final learned router.  Each public state
is evaluated under ANSWER, ASK, CONSULT, and DEFER while hidden patient facts are
kept outside the expert request path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import aiohttp
import pandas as pd
import yaml

from .protocol import (
    Action,
    PublicState,
    build_expert_messages,
    parse_action,
    unrevealed_atomic_facts,
    validate_expert_response,
    validate_expert_view,
)
from .utility import UtilityConfig, select_action


BASE_FORCE = {
    Action.ANSWER: "For this counterfactual branch, answer now. End with exactly: Final Answer: <one option letter>.",
    Action.ASK: "For this counterfactual branch, ask exactly one useful patient question. End with exactly: Question: <one question>.",
}

PATIENT_SYSTEM = """You are a patient simulator. Answer the doctor's question using
only the supplied patient atomic facts. Do not add medical inference or outside
knowledge. If no supplied fact directly answers it, reply exactly: The patient
cannot answer this question. Give only the concise patient response."""


@dataclass
class APICall:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class ChatAPI:
    def __init__(self, base_url: str, model: str, timeout_s: float = 180.0, enable_thinking: bool | None = None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.enable_thinking = enable_thinking
        self.client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s))

    async def close(self) -> None:
        await self.client.close()

    async def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        max_tokens: int = 768,
        retries: int = 3,
    ) -> APICall:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        last_error: Exception | None = None
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
            except Exception as exc:  # API errors include transient server startup failures.
                last_error = exc
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"generation failed after {retries} attempts: {last_error}")


def _jsonable_messages(value: Any) -> tuple[dict[str, str], ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return tuple({"role": str(x["role"]), "content": str(x["content"])} for x in value)


def _gold_letter(ground_truth: Any) -> str:
    if isinstance(ground_truth, Mapping):
        return str(ground_truth.get("answer", "")).strip().upper()
    return str(ground_truth).strip().upper()


def _answer_letter(text: str, *, expert: bool = False) -> str | None:
    prefix = r"Expert Final Answer" if expert else r"Final Answer"
    match = re.search(prefix + r":\s*([A-D])\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def _forced_messages(state: PublicState, action: Action, *, strict: bool = False) -> list[dict[str, str]]:
    messages = state.messages
    constraint = (
        "HIGHEST-PRIORITY COUNTERFACTUAL EVALUATION CONSTRAINT: "
        + BASE_FORCE[action]
        + " Do not choose a different action in this branch.\n\n"
    )
    system_index = next((i for i, m in enumerate(messages) if m["role"] == "system"), None)
    if strict:
        public_messages = [m for m in messages if m["role"] != "system"]
        return [{"role": "system", "content": constraint}] + public_messages
    if system_index is None:
        messages.insert(0, {"role": "system", "content": constraint})
    else:
        messages[system_index] = {
            "role": "system",
            "content": constraint + messages[system_index]["content"],
        }
    return messages


def _patient_messages(atomic_facts: Sequence[str], question: str) -> list[dict[str, str]]:
    hidden = json.dumps([str(x) for x in atomic_facts], ensure_ascii=False)
    return [
        {"role": "system", "content": PATIENT_SYSTEM},
        {"role": "user", "content": f"Patient atomic facts:\n{hidden}\n\nDoctor question:\n{question}"},
    ]


async def _base_action(api: ChatAPI, state: PublicState, action: Action, temperature: float) -> tuple[str, APICall]:
    last: tuple[Action, APICall] | None = None
    for strict in (False, True):
        for _ in range(2):
            call = await api.generate(_forced_messages(state, action, strict=strict), temperature=0.0)
            parsed, content = parse_action(call.content)
            if parsed is action:
                return content, call
            last = (parsed, call)
    raise ValueError(f"forced {action.value}, model emitted {last[0].value if last else 'invalid'}")


async def _continue_base(
    base: ChatAPI,
    patient: ChatAPI,
    state: PublicState,
    atomic_facts: Sequence[str],
    temperature: float,
    max_turns: int,
) -> tuple[str | None, list[dict[str, Any]]]:
    messages = state.messages
    trace: list[dict[str, Any]] = []
    for _ in range(max_turns):
        call = await base.generate(messages, temperature=temperature)
        try:
            action, content = parse_action(call.content)
        except ValueError:
            trace.append({"actor": "base", "error": "invalid_format", **asdict(call)})
            return None, trace
        trace.append({"actor": "base", "action": action.value, **asdict(call)})
        messages.append({"role": "assistant", "content": call.content})
        if action is Action.ANSWER:
            return content.strip().upper(), trace
        if action is not Action.ASK:
            return None, trace
        patient_call = await patient.generate(_patient_messages(atomic_facts, content), temperature=0.0, max_tokens=256)
        trace.append({"actor": "patient", **asdict(patient_call)})
        messages.append({"role": "user", "content": patient_call.content})
    return None, trace


async def rollout_action(
    action: Action,
    state: PublicState,
    atomic_facts: Sequence[str],
    ground_truth: Mapping[str, Any] | str,
    base: ChatAPI,
    patient: ChatAPI,
    expert: ChatAPI,
    *,
    temperature: float,
    max_turns: int,
) -> dict[str, Any]:
    gold = _gold_letter(ground_truth)
    trace: list[dict[str, Any]] = []
    answer: str | None = None
    error: str | None = None
    try:
        if action in (Action.ANSWER, Action.ASK):
            content, call = await _base_action(base, state, action, temperature)
            trace.append({"actor": "base", "action": action.value, **asdict(call)})
            if action is Action.ANSWER:
                answer = content.strip().upper()
            else:
                patient_call = await patient.generate(_patient_messages(atomic_facts, content), temperature=0.0, max_tokens=256)
                trace.append({"actor": "patient", **asdict(patient_call)})
                next_state = PublicState(
                    state.sample_id,
                    state.initial_messages,
                    (*state.dialogue, {"role": "assistant", "content": call.content}, {"role": "user", "content": patient_call.content}),
                )
                answer, continued = await _continue_base(base, patient, next_state, atomic_facts, temperature, max_turns - 1)
                trace.extend(continued)
        elif action is Action.CONSULT:
            expert_messages = build_expert_messages(state, action)
            validate_expert_view(expert_messages, atomic_facts=unrevealed_atomic_facts(state, atomic_facts), ground_truth=ground_truth)
            expert_call = await expert.generate(expert_messages, temperature=temperature, max_tokens=384)
            trace.append({"actor": "expert", "action": action.value, **asdict(expert_call)})
            advice = validate_expert_response(action, expert_call.content)
            next_state = PublicState(
                state.sample_id,
                state.initial_messages,
                (*state.dialogue, {"role": "user", "content": advice}),
            )
            answer, continued = await _continue_base(base, patient, next_state, atomic_facts, temperature, max_turns)
            trace.extend(continued)
        else:
            expert_messages = build_expert_messages(state, action)
            validate_expert_view(expert_messages, atomic_facts=unrevealed_atomic_facts(state, atomic_facts), ground_truth=ground_truth)
            expert_call = await expert.generate(expert_messages, temperature=temperature, max_tokens=384)
            trace.append({"actor": "expert", "action": action.value, **asdict(expert_call)})
            validated = validate_expert_response(action, expert_call.content)
            answer = _answer_letter(validated, expert=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    return {
        "state_id": state.state_id,
        "sample_id": state.sample_id,
        "dialogue_turns": len(state.dialogue),
        "public_messages": state.messages,
        "action": action.value,
        "answer": answer,
        "gold": gold,
        "outcome": float(answer == gold) if answer else 0.0,
        "valid": answer is not None and error is None,
        "error": error,
        "expert_prompt_tokens": sum(x.get("prompt_tokens", 0) for x in trace if x["actor"] == "expert"),
        "expert_completion_tokens": sum(x.get("completion_tokens", 0) for x in trace if x["actor"] == "expert"),
        "expert_latency_s": sum(x.get("latency_s", 0.0) for x in trace if x["actor"] == "expert"),
        "trace": trace,
    }


async def generate_states(
    rows: Sequence[Mapping[str, Any]],
    base: ChatAPI,
    patient: ChatAPI,
    *,
    limit: int,
    temperature: float,
    max_turns: int,
) -> list[tuple[PublicState, list[str], Mapping[str, Any] | str]]:
    states: list[tuple[PublicState, list[str], Mapping[str, Any] | str]] = []
    for row in rows:
        extra = row["extra_info"]
        initial = _jsonable_messages(extra["prompt"])
        facts = [str(x) for x in extra["atomic_facts"]]
        ground_truth = row["reward_model"]["ground_truth"]
        dialogue: tuple[Mapping[str, str], ...] = ()
        for _ in range(max_turns):
            state = PublicState(str(extra.get("index", len(states))), initial, dialogue)
            states.append((state, facts, ground_truth))
            if len(states) >= limit:
                return states
            call = await base.generate(state.messages, temperature=temperature)
            try:
                action, content = parse_action(call.content)
            except ValueError:
                break
            if action is Action.ANSWER:
                break
            if action is not Action.ASK:
                break
            patient_call = await patient.generate(_patient_messages(facts, content), temperature=0.0, max_tokens=256)
            dialogue = (*dialogue, {"role": "assistant", "content": call.content}, {"role": "user", "content": patient_call.content})
    return states


async def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config).read_text())
    random.seed(config["seed"])
    frame = pd.read_parquet(args.data)
    indices = list(range(len(frame)))
    random.shuffle(indices)
    rows = [frame.iloc[i].to_dict() for i in indices]

    base = ChatAPI(config["endpoints"]["base"], config["models"]["base"], enable_thinking=True)
    patient = ChatAPI(config["endpoints"]["patient"], config["models"]["patient"], enable_thinking=False)
    expert = ChatAPI(config["endpoints"]["expert"], config["models"]["expert"], enable_thinking=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        states = await generate_states(
            rows,
            base,
            patient,
            limit=args.num_states,
            temperature=config["temperature"],
            max_turns=config["max_assistant_turns"],
        )
        records: list[dict[str, Any]] = []
        with output.open("w", encoding="utf-8") as handle:
            for state, facts, ground_truth in states:
                for rollout_index in range(args.rollouts_per_action):
                    # Branches share the exact state but are evaluated independently.
                    branches = await asyncio.gather(*[
                        rollout_action(
                            action, state, facts, ground_truth, base, patient, expert,
                            temperature=config["temperature"],
                            max_turns=config["max_assistant_turns"],
                        ) for action in Action
                    ])
                    for record in branches:
                        record["rollout_index"] = rollout_index
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        records.append(record)

        costs = {Action(k): float(v) for k, v in config["cost"].items()}
        ucfg = UtilityConfig(resource_cost=costs, lambda_cost=float(config["lambda_cost"]), lambda_harm=float(config["lambda_harm"]))
        grouped: dict[str, dict[Action, list[float]]] = {}
        for record in records:
            grouped.setdefault(record["state_id"], {a: [] for a in Action})[Action(record["action"])].append(record["outcome"])
        winners = {a: 0 for a in Action}
        state_summaries = []
        for state_id, values in grouped.items():
            q_out = {a: sum(xs) / len(xs) for a, xs in values.items()}
            winner, utilities = select_action(q_out, ucfg)
            winners[winner] += 1
            ordered = sorted(utilities.values(), reverse=True)
            state_summaries.append({
                "state_id": state_id,
                "q_out": {a.value: q_out[a] for a in Action},
                "utility": {a.value: utilities[a] for a in Action},
                "winner": winner.value,
                "utility_gap": ordered[0] - ordered[1],
            })
        summary = {
            "num_states": len(grouped),
            "rollouts_per_action": args.rollouts_per_action,
            "winner_counts": {a.value: winners[a] for a in Action},
            "valid_rate": sum(r["valid"] for r in records) / max(1, len(records)),
            "states": state_summaries,
        }
        output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "states"}, indent=2))
    finally:
        await asyncio.gather(base.close(), patient.close(), expert.close())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="recipe/ea_atpo/config/feasibility.yaml")
    parser.add_argument("--data", default="data/train.parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-states", type=int, default=12)
    parser.add_argument("--rollouts-per-action", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
