"""Lightweight, model-independent DEFER case builder shared by audit runners."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / "ea_atpo_stage/recipe/ea_atpo/protocol.py"


def _load_protocol():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("qwen32_router_protocol", PROTOCOL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load protocol from {PROTOCOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_protocol()
Action = protocol.Action
PublicState = protocol.PublicState
build_expert_messages = protocol.build_expert_messages
parse_action = protocol.parse_action
visible_response = protocol.visible_response
unrevealed_atomic_facts = protocol.unrevealed_atomic_facts
validate_expert_response = protocol.validate_expert_response
validate_expert_view = protocol.validate_expert_view


def plain(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def answer_letter(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("answer", "")).strip().upper()
    return str(value).strip().upper()


def row_gold(row: dict[str, Any]) -> str:
    raw = row.get("ground_truth")
    if raw is None:
        raw = (row.get("reward_model") or {}).get("ground_truth")
    gold = answer_letter(raw)
    if gold not in "ABCD":
        raise ValueError("invalid ground truth in paired pilot row")
    return gold


def row_facts(row: dict[str, Any]) -> list[str]:
    value = row.get("atomic_facts")
    if value is None:
        value = (row.get("extra_info") or {}).get("atomic_facts", [])
    value = plain(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [value]
    return [str(item) for item in (value or [])]


def public_state(row: dict[str, Any], index: int):
    prompt = plain(row.get("prompt"))
    if not isinstance(prompt, list):
        prompt = plain((row.get("extra_info") or {}).get("prompt"))
    if not isinstance(prompt, list):
        raise ValueError(f"missing public prompt at pilot row {index}")
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
    payload = json.dumps(visible, ensure_ascii=False, sort_keys=True)
    digest = sha_text(payload)
    state = PublicState(sample_id=f"dev-{digest[:16]}", initial_messages=tuple(visible))
    facts, gold = row_facts(row), row_gold(row)
    messages = build_expert_messages(state, Action.DEFER)
    validate_expert_view(messages, atomic_facts=unrevealed_atomic_facts(state, facts), ground_truth=gold)
    return state, digest, gold, facts


def case_key(row: dict[str, Any], index: int, direct_module):
    state, digest, gold, facts = public_state(row, index)
    defer_messages = build_expert_messages(state, Action.DEFER)
    base_messages = direct_module.build_messages(row)
    hidden = unrevealed_atomic_facts(state, facts)
    validate_expert_view(defer_messages, atomic_facts=hidden, ground_truth=gold)
    validate_expert_view(base_messages, atomic_facts=hidden, ground_truth=gold)
    return digest, gold, facts, base_messages, defer_messages
