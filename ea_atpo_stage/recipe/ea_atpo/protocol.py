"""Leakage-safe state and expert protocols for EA-ATPO.

The patient simulator may see hidden ``atomic_facts``.  The expert never may.
Both CONSULT and DEFER receive only the public case statement and dialogue.
CONSULT provides bounded advice; DEFER takes control and returns a final option.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class Action(str, Enum):
    ANSWER = "ANSWER"
    ASK = "ASK"
    CONSULT = "CONSULT"
    DEFER = "DEFER"


@dataclass(frozen=True)
class PublicState:
    sample_id: str
    initial_messages: tuple[Mapping[str, str], ...]
    dialogue: tuple[Mapping[str, str], ...] = ()

    @property
    def messages(self) -> list[dict[str, str]]:
        return [dict(x) for x in (*self.initial_messages, *self.dialogue)]

    @property
    def state_id(self) -> str:
        payload = json.dumps(self.messages, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"{self.sample_id}:{digest}"


CONSULT_SYSTEM_PROMPT = """You are a senior medical consultant assisting another clinician.
You receive only the currently visible case and dialogue. Give one short, actionable
diagnostic suggestion that helps the clinician reason or choose the next information
to seek. Do not invent unavailable patient facts. Do not state a final option letter,
do not give a 'Final Answer', and do not claim access to privileged supervision.
Return exactly: Expert Advice: <one concise suggestion>"""


DEFER_SYSTEM_PROMPT = """You are the senior clinician taking over this case.
Independently review only the currently visible case and dialogue, then choose the
single best option now, even when the evidence is incomplete. This is a forced
transfer-of-control evaluation: never ask another question and never request more
information. Do not claim access to hidden patient facts or privileged supervision.
Return only: Expert Final Answer: <one option letter>"""


def build_expert_messages(state: PublicState, action: Action, consult_question: str = "") -> list[dict[str, str]]:
    if action not in (Action.CONSULT, Action.DEFER):
        raise ValueError(f"expert protocol is undefined for {action}")
    system = CONSULT_SYSTEM_PROMPT if action is Action.CONSULT else DEFER_SYSTEM_PROMPT
    visible = json.dumps(state.messages, ensure_ascii=False)
    request = f"Visible case and dialogue:\n{visible}"
    if action is Action.CONSULT:
        request += "\nConsultation focus: " + (consult_question.strip() or "What is the most useful next diagnostic consideration?")
    else:
        request += "\nMake the terminal decision now. Do not ask a question. Output only: Expert Final Answer: <one option letter>"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": request}]
    validate_expert_view(messages)
    return messages


def _flatten_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


def unrevealed_atomic_facts(state: PublicState, atomic_facts: Iterable[str]) -> list[str]:
    """Return facts whose exact content has not already entered the public state."""
    public_text = _flatten_messages(state.messages)
    return [str(fact) for fact in atomic_facts if str(fact).strip() and str(fact).strip() not in public_text]


def validate_expert_view(
    messages: Sequence[Mapping[str, Any]],
    *,
    atomic_facts: Iterable[str] | None = None,
    ground_truth: Mapping[str, Any] | str | None = None,
) -> None:
    """Fail closed if privileged fields or their values reach the expert prompt."""
    text = _flatten_messages(messages)
    lowered = text.lower()
    forbidden_keys = ("atomic_facts", "answer_idx", "ground_truth", "answer key")
    found = [key for key in forbidden_keys if key in lowered]

    if atomic_facts:
        for fact in atomic_facts:
            fact = str(fact).strip()
            if len(fact) >= 12 and fact in text:
                found.append("hidden atomic fact value")
                break

    if ground_truth is not None:
        if isinstance(ground_truth, Mapping):
            # Option letters and answer descriptions are part of the public
            # multiple-choice prompt, so their values cannot diagnose leakage.
            # Only an explicit serialized supervision object is forbidden.
            rendered = json.dumps(dict(ground_truth), ensure_ascii=False)
            if rendered in text:
                found.append("serialized ground truth")
        elif len(str(ground_truth).strip()) >= 8 and str(ground_truth).strip() in text:
            found.append("ground-truth value")

    if found:
        raise ValueError("privileged information in expert view: " + ", ".join(sorted(set(found))))


def parse_action(text: str) -> tuple[Action, str]:
    tail = text.split("</think>")[-1].strip()
    patterns = (
        (Action.ANSWER, r"Final Answer:\s*([A-D])\b"),
        (Action.ASK, r"Question:\s*(.+)"),
        (Action.CONSULT, r"Consult(?: Expert)?:\s*(.+)"),
        (Action.DEFER, r"Defer(?: to Expert)?:?\s*(.*)"),
    )
    for action, pattern in patterns:
        match = re.search(pattern, tail, re.IGNORECASE | re.DOTALL)
        if match:
            return action, match.group(1).strip()
    raise ValueError(f"unrecognized action format: {tail[:160]!r}")


def validate_expert_response(action: Action, text: str) -> str:
    # Qwen-family reasoning models may wrap private reasoning in think tags. Only
    # the post-thinking answer is exposed to the collaborator and validated.
    value = text.split("</think>")[-1].strip()
    if action is Action.CONSULT:
        if not re.fullmatch(r"Expert Advice:\s*.+", value, re.IGNORECASE | re.DOTALL):
            raise ValueError("CONSULT response must use 'Expert Advice:' format")
        if re.search(r"(?:Final Answer|correct (?:answer|option)|option\s+[A-D])", value, re.IGNORECASE):
            raise ValueError("CONSULT leaked or asserted a final answer")
    elif action is Action.DEFER:
        if not re.fullmatch(r"Expert Final Answer:\s*[A-D]", value, re.IGNORECASE):
            raise ValueError("DEFER response must contain exactly one final option")
    else:
        raise ValueError(f"not an expert action: {action}")
    return value
