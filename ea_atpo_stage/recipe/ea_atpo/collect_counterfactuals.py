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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .protocol import (
    Action,
    PublicState,
    build_expert_messages,
    parse_action,
    unrevealed_atomic_facts,
    validate_expert_response,
    validate_expert_view,
    visible_response,
)
from .utility import UtilityConfig, select_action


BASE_FORCE = {
    Action.ANSWER: "For this counterfactual branch, answer now. End with exactly: Final Answer: <one option letter>.",
    Action.ASK: "For this counterfactual branch, ask exactly one useful patient question. End with exactly: Question: <one question>.",
    Action.CONSULT: "For this counterfactual branch, request one focused expert hint based only on the visible case. End with exactly: Consult Expert: <one specific diagnostic question>.",
}

# vLLM's structured-output regex constrains syntax only; the model still
# generates the clinical question, consultation focus, or selected option.
FINAL_ANSWER_REGEX = r"^Final Answer: [A-D]$"
ACTION_REGEX = {
    Action.ANSWER: FINAL_ANSWER_REGEX,
    Action.ASK: r"^Question: [^\r\n]{8,320}$",
    Action.CONSULT: r"^Consult Expert: [^\r\n]{8,320}$",
}
CONTINUATION_REGEX = (
    r"^(?:Final Answer: [A-D]|Question: [^\r\n]{8,320}|"
    r"Consult Expert: [^\r\n]{8,320}|Defer to Expert)$"
)
CONTINUATION_NO_CONSULT_REGEX = (
    r"^(?:Final Answer: [A-D]|Question: [^\r\n]{8,320}|Defer to Expert)$"
)

CONTINUATION_SYSTEM = """Continue the visible medical dialogue. After any private reasoning,
output exactly one next action in this format:
- Final Answer: <one option letter>
- Question: <one question for the patient>
- Consult Expert: <one focused question for the expert>
- Defer to Expert
Never include private reasoning in the visible action. A CONSULT asks for a short hint
and then you continue reasoning. DEFER ends the episode and transfers control to the
expert. Treat 'The patient cannot answer this question.' as UNKNOWN, not as a negative
clinical finding."""

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


def _call_metrics(call: APICall) -> dict[str, Any]:
    """Keep usage data while excluding raw private model reasoning from JSONL."""
    return {
        "prompt_tokens": call.prompt_tokens,
        "completion_tokens": call.completion_tokens,
        "latency_s": call.latency_s,
    }


class ChatAPI:
    def __init__(self, base_url: str, model: str, timeout_s: float = 180.0, enable_thinking: bool | None = None):
        try:
            import aiohttp
        except ImportError as exc:
            raise RuntimeError("ChatAPI requires aiohttp; install the ATPO runtime dependencies") from exc
        self._aiohttp = aiohttp
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.enable_thinking = enable_thinking
        self.client = self._aiohttp.ClientSession(timeout=self._aiohttp.ClientTimeout(total=timeout_s))

    async def close(self) -> None:
        await self.client.close()

    async def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        max_tokens: int = 768,
        retries: int = 3,
        structured_outputs: Mapping[str, Any] | None = None,
    ) -> APICall:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        if structured_outputs is not None:
            payload["structured_outputs"] = dict(structured_outputs)
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


def _public_action_text(action: Action, content: str) -> str:
    """Return only the visible action, never the model's private <think> text."""
    if action is Action.ANSWER:
        return f"Final Answer: {content.strip().upper()}"
    if action is Action.ASK:
        return f"Question: {content.strip()}"
    if action is Action.CONSULT:
        return f"Consult Expert: {content.strip()}"
    return "Defer to Expert"


def _extend_state(state: PublicState, *turns: Mapping[str, str]) -> PublicState:
    return PublicState(state.sample_id, state.initial_messages, (*state.dialogue, *turns))


def _continuation_messages(state: PublicState, *, consult_available: bool) -> list[dict[str, str]]:
    """Add a per-turn action contract without exposing it as dialogue to experts."""
    messages = state.messages
    prompt = CONTINUATION_SYSTEM
    if not consult_available:
        prompt = prompt.replace("- Consult Expert: <one focused question for the expert>\n", "")
        prompt += "\nThe consultation budget is exhausted; do not choose CONSULT."
    system_index = next((i for i, message in enumerate(messages) if message["role"] == "system"), None)
    if system_index is None:
        return [{"role": "system", "content": prompt}, *messages]
    messages[system_index] = {**messages[system_index], "content": prompt + "\n\n" + messages[system_index]["content"]}
    return messages


async def _consult_expert(
    expert: ChatAPI,
    state: PublicState,
    question: str,
    atomic_facts: Sequence[str],
    temperature: float,
) -> tuple[str, APICall]:
    expert_messages = build_expert_messages(state, Action.CONSULT, consult_question=question)
    validate_expert_view(
        expert_messages,
        atomic_facts=unrevealed_atomic_facts(state, atomic_facts),
    )
    call = await expert.generate(expert_messages, temperature=temperature, max_tokens=384)
    return validate_expert_response(Action.CONSULT, call.content), call


async def _defer_to_expert(
    expert: ChatAPI,
    state: PublicState,
    atomic_facts: Sequence[str],
    temperature: float,
) -> tuple[str, APICall]:
    expert_messages = build_expert_messages(state, Action.DEFER)
    validate_expert_view(
        expert_messages,
        atomic_facts=unrevealed_atomic_facts(state, atomic_facts),
    )
    call = await expert.generate(expert_messages, temperature=temperature, max_tokens=384)
    validated = validate_expert_response(Action.DEFER, call.content)
    answer = _answer_letter(validated, expert=True)
    if answer is None:
        raise ValueError("validated DEFER response did not contain a final option")
    return answer, call


async def _base_action(api: ChatAPI, state: PublicState, action: Action, temperature: float) -> tuple[str, APICall]:
    last: tuple[Action, APICall] | None = None
    for strict in (False, True):
        for _ in range(2):
            call = await api.generate(
                _forced_messages(state, action, strict=strict),
                temperature=temperature,
                max_tokens=256 if action is not Action.ANSWER else 16,
                structured_outputs={"regex": ACTION_REGEX[action]},
            )
            parsed, content = parse_action(call.content)
            if parsed is action:
                return content, call
            last = (parsed, call)
    raise ValueError(f"forced {action.value}, model emitted {last[0].value if last else 'invalid'}")


async def _continue_base(
    base: ChatAPI,
    patient: ChatAPI,
    expert: ChatAPI,
    state: PublicState,
    atomic_facts: Sequence[str],
    temperature: float,
    max_turns: int,
    *,
    consults_used: int = 0,
    max_consults: int = 2,
) -> tuple[str | None, list[dict[str, Any]], PublicState]:
    current_state = state
    trace: list[dict[str, Any]] = []
    for turn_index in range(max_turns):
        terminal_turn = turn_index == max_turns - 1
        if terminal_turn:
            messages = _forced_messages(current_state, Action.ANSWER, strict=True)
            structured_regex = FINAL_ANSWER_REGEX
            token_limit = 16
        else:
            messages = _continuation_messages(
                current_state, consult_available=consults_used < max_consults
            )
            structured_regex = (
                CONTINUATION_REGEX
                if consults_used < max_consults
                else CONTINUATION_NO_CONSULT_REGEX
            )
            token_limit = 256
        call = await base.generate(
            messages,
            temperature=temperature,
            max_tokens=token_limit,
            structured_outputs={"regex": structured_regex},
        )
        try:
            action, content = parse_action(call.content)
        except ValueError:
            trace.append({"actor": "base", "error": "invalid_format", **_call_metrics(call)})
            return None, trace, current_state
        trace.append({
            "actor": "base",
            "action": action.value,
            "response": _public_action_text(action, content),
            **_call_metrics(call),
        })
        if terminal_turn and action is not Action.ANSWER:
            trace.append({"actor": "router", "error": "terminal_answer_constraint_violated"})
            return None, trace, current_state
        visible_action = {"role": "assistant", "content": _public_action_text(action, content)}
        if action is Action.ANSWER:
            return content.strip().upper(), trace, _extend_state(current_state, visible_action)
        if action is Action.ASK:
            patient_call = await patient.generate(
                _patient_messages(atomic_facts, content), temperature=0.0, max_tokens=256
            )
            patient_response = visible_response(patient_call.content)
            trace.append({"actor": "patient", "response": patient_response, **_call_metrics(patient_call)})
            current_state = _extend_state(
                current_state,
                visible_action,
                {"role": "user", "content": patient_response},
            )
            continue
        if action is Action.CONSULT:
            if consults_used >= max_consults:
                trace.append({"actor": "router", "error": "consult_budget_exhausted"})
                return None, trace, current_state
            advice, expert_call = await _consult_expert(
                expert, current_state, content, atomic_facts, temperature
            )
            trace.append({
                "actor": "expert",
                "action": Action.CONSULT.value,
                "consult_question": content,
                "response": advice,
                **_call_metrics(expert_call),
            })
            current_state = _extend_state(
                current_state,
                visible_action,
                {"role": "user", "content": advice},
            )
            consults_used += 1
            continue

        # DEFER is terminal. The expert sees the public state before the agent's
        # deferral action, so it cannot be anchored by private reasoning or a guess.
        answer, expert_call = await _defer_to_expert(
            expert, current_state, atomic_facts, temperature
        )
        trace.append({
            "actor": "expert",
            "action": Action.DEFER.value,
            "response": f"Expert Final Answer: {answer}",
            **_call_metrics(expert_call),
        })
        current_state = _extend_state(
            current_state,
            visible_action,
            {"role": "user", "content": f"Expert Final Answer: {answer}"},
        )
        return answer, trace, current_state
    return None, trace, current_state


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
    max_consults: int = 2,
) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []
    answer: str | None = None
    error: str | None = None
    final_state = state
    try:
        if action in (Action.ANSWER, Action.ASK, Action.CONSULT):
            content, call = await _base_action(base, state, action, temperature)
            visible_action = {"role": "assistant", "content": _public_action_text(action, content)}
            trace.append({
                "actor": "base",
                "action": action.value,
                **({"consult_question": content} if action is Action.CONSULT else {}),
                "response": visible_action["content"],
                **_call_metrics(call),
            })
            if action is Action.ANSWER:
                answer = content.strip().upper()
                final_state = _extend_state(state, visible_action)
            else:
                if action is Action.ASK:
                    patient_call = await patient.generate(
                        _patient_messages(atomic_facts, content), temperature=0.0, max_tokens=256
                    )
                    patient_response = visible_response(patient_call.content)
                    trace.append({"actor": "patient", "response": patient_response, **_call_metrics(patient_call)})
                    next_state = _extend_state(
                        state,
                        visible_action,
                        {"role": "user", "content": patient_response},
                    )
                    consults_used = 0
                else:
                    advice, expert_call = await _consult_expert(
                        expert, state, content, atomic_facts, temperature
                    )
                    trace.append({
                        "actor": "expert",
                        "action": Action.CONSULT.value,
                        "consult_question": content,
                        "response": advice,
                        **_call_metrics(expert_call),
                    })
                    next_state = _extend_state(
                        state,
                        visible_action,
                        {"role": "user", "content": advice},
                    )
                    consults_used = 1
                answer, continued, final_state = await _continue_base(
                    base,
                    patient,
                    expert,
                    next_state,
                    atomic_facts,
                    temperature,
                    max_turns - 1,
                    consults_used=consults_used,
                    max_consults=max_consults,
                )
                trace.extend(continued)
        else:
            answer, expert_call = await _defer_to_expert(
                expert, state, atomic_facts, temperature
            )
            trace.append({
                "actor": "expert",
                "action": Action.DEFER.value,
                "response": f"Expert Final Answer: {answer}",
                **_call_metrics(expert_call),
            })
            final_state = _extend_state(
                state,
                {"role": "assistant", "content": "Defer to Expert"},
                {"role": "user", "content": f"Expert Final Answer: {answer}"},
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    if answer is None and error is None:
        traced_error = next((item.get("error") for item in reversed(trace) if item.get("error")), None)
        error = traced_error or "no_terminal_answer"

    # Gold is first read after all model calls have finished; it is used only by
    # this post-hoc evaluator and never enters a prompt or pre-call validator.
    gold = _gold_letter(ground_truth)
    def actor_usage(actor: str) -> dict[str, float | int]:
        calls = [event for event in trace if event.get("actor") == actor]
        return {
            "prompt_tokens": sum(int(event.get("prompt_tokens", 0)) for event in calls),
            "completion_tokens": sum(int(event.get("completion_tokens", 0)) for event in calls),
            "latency_s": sum(float(event.get("latency_s", 0.0)) for event in calls),
        }

    base_usage = actor_usage("base")
    patient_usage = actor_usage("patient")
    expert_usage = actor_usage("expert")
    return {
        "state_id": state.state_id,
        "sample_id": state.sample_id,
        "dialogue_turns": len(state.dialogue),
        "public_messages": final_state.messages,
        "action": action.value,
        "answer": answer,
        "gold": gold,
        "outcome": float(answer == gold) if answer else 0.0,
        "valid": answer is not None and error is None,
        "error": error,
        "expert_prompt_tokens": sum(x.get("prompt_tokens", 0) for x in trace if x["actor"] == "expert"),
        "expert_completion_tokens": sum(x.get("completion_tokens", 0) for x in trace if x["actor"] == "expert"),
        "expert_latency_s": sum(x.get("latency_s", 0.0) for x in trace if x["actor"] == "expert"),
        "base_prompt_tokens": base_usage["prompt_tokens"],
        "base_completion_tokens": base_usage["completion_tokens"],
        "base_latency_s": base_usage["latency_s"],
        "patient_prompt_tokens": patient_usage["prompt_tokens"],
        "patient_completion_tokens": patient_usage["completion_tokens"],
        "patient_latency_s": patient_usage["latency_s"],
        "patient_turns": sum(x.get("actor") == "patient" for x in trace),
        "total_prompt_tokens": sum(int(x.get("prompt_tokens", 0)) for x in trace),
        "total_completion_tokens": sum(int(x.get("completion_tokens", 0)) for x in trace),
        "total_latency_s": sum(float(x.get("latency_s", 0.0)) for x in trace),
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
            patient_response = visible_response(patient_call.content)
            dialogue = (
                *dialogue,
                {"role": "assistant", "content": _public_action_text(action, content)},
                {"role": "user", "content": patient_response},
            )
    return states


async def run(args: argparse.Namespace) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("The counterfactual collector requires pandas to read Parquet data") from exc
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
