"""Three-action V2 protocol. Kept separate from the historical V1 protocol."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class Action(str, Enum):
    ASK = "ASK"
    ANSWER = "ANSWER"
    DEFER = "DEFER"


ACTION_ORDER = (Action.ASK, Action.ANSWER, Action.DEFER)
FINAL_ANSWER_REGEX = r"^Final Answer: [A-D]$"
ASK_QUESTION_REGEX = r"^Question: [^\r\n]{8,320}$"


@dataclass(frozen=True)
class PublicState:
    case_id: str
    state_id: str
    state_index: int
    initial_messages: tuple[Mapping[str, str], ...]
    dialogue: tuple[Mapping[str, str], ...] = ()
    asked_count: int = 0
    max_asks: int = 3

    @property
    def messages(self) -> list[dict[str, str]]:
        return [dict(message) for message in (*self.initial_messages, *self.dialogue)]

    @property
    def visible_messages(self) -> list[dict[str, str]]:
        # The source system prompt is a legacy action protocol, not patient information.
        return [message for message in self.messages if message["role"] != "system"]

    @property
    def public_text(self) -> str:
        return "\n".join(message["content"] for message in self.visible_messages)

    @property
    def digest(self) -> str:
        payload = json.dumps(self.visible_messages, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def legal_actions(self) -> tuple[Action, ...]:
        if self.asked_count >= self.max_asks:
            return (Action.ANSWER, Action.DEFER)
        return ACTION_ORDER


def _with_contract(state: PublicState, contract: str) -> list[dict[str, str]]:
    # The old V1 system prompt may advertise CONSULT or other four-action rules.
    # Exclude it while retaining the original user case and public dialogue.
    return [{"role": "system", "content": contract}, *state.visible_messages]


def build_base_messages(state: PublicState, action: Action, *, stage: str = "initial") -> list[dict[str, str]]:
    if action not in (Action.ASK, Action.ANSWER):
        raise ValueError(f"Base cannot execute {action.value}")
    if action not in state.legal_actions():
        raise ValueError(f"illegal action {action.value} for state {state.state_id}")
    if stage not in ("initial", "after_ask"):
        raise ValueError(f"unknown Base stage: {stage}")
    if stage == "after_ask" and action is not Action.ANSWER:
        raise ValueError("the frozen continuation policy allows only terminal ANSWER after ASK")

    if action is Action.ASK:
        contract = (
            "You are a medical diagnostic assistant. EA-ATPO V2 forced counterfactual action: ASK. "
            "Based only on the visible case, the question being investigated, and the dialogue, "
            "ask exactly one specific question about missing clinical information that could "
            "help answer this case. Ground the question in the visible symptoms or history; "
            "do not introduce an unrelated measurement or repeat information already visible. "
            "Do not answer, defer, request an expert, or repeat information already visible. "
            "Output exactly: Question: <one question>"
        )
    else:
        stage_text = "after receiving one patient reply" if stage == "after_ask" else "now"
        contract = (
            f"You are a medical diagnostic assistant. EA-ATPO V2 forced counterfactual action: "
            f"ANSWER {stage_text}. Use only the visible "
            "case and dialogue. This is a forced single-choice item: choose exactly one of the "
            "listed options A, B, C, or D, even if uncertain. Do not abstain, answer 'none', "
            "or repeat the option text. Do not ask, defer, or request an expert. "
            "Output only: Final Answer: <A-D>"
        )
    return _with_contract(state, contract)


def build_patient_messages(atomic_facts: Sequence[str], question: str) -> list[dict[str, str]]:
    """Only the Patient role receives hidden atomic facts."""
    hidden = json.dumps([str(fact) for fact in atomic_facts], ensure_ascii=False)
    return [
        {
            "role": "system",
            "content": (
                "You are a patient simulator. Answer the doctor's question using only the supplied "
                "patient atomic facts. Match facts by meaning, including ordinary paraphrases; do "
                "not require the question to repeat the fact's exact wording. For a broad question, "
                "report relevant facts that are known and clearly say when a requested detail is "
                "not stated. Do not add medical inference or outside knowledge. If none of the "
                "supplied facts answers any part of the question, reply exactly: The patient cannot "
                "answer this question. Give only the concise patient response."
            ),
        },
        {"role": "user", "content": f"Patient facts:\n{hidden}\n\nDoctor question:\n{question}"},
    ]


def build_defer_messages(state: PublicState) -> list[dict[str, str]]:
    visible = json.dumps(state.visible_messages, ensure_ascii=False)
    return [
        {
            "role": "system",
            "content": (
                "You are the independent Expert taking over this case. Use only the visible case "
                "and dialogue below. This is a forced single-choice item: choose exactly one of "
                "the listed options A, B, C, or D, even if uncertain. Do not abstain, answer 'none', "
                "repeat the option text, ask a question, or mention hidden facts. "
                "Return only: Final Answer: <A-D>"
            ),
        },
        {"role": "user", "content": f"Visible case and dialogue:\n{visible}"},
    ]


def parse_question(text: str) -> str:
    match = re.fullmatch(r"\s*Question:\s*(.{8,500}?)\s*", str(text), re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("ASK output must be one non-empty Question: line")
    question = match.group(1).strip()
    if not question or "\nQuestion:" in question:
        raise ValueError("ASK output contains multiple questions")
    return question


def parse_answer(text: str) -> str:
    match = re.fullmatch(r"\s*Final Answer:\s*([A-D])\s*", str(text), re.IGNORECASE)
    if not match:
        raise ValueError("answer must match 'Final Answer: <A-D>' exactly")
    return match.group(1).upper()


def valid_ask_transition(before: PublicState, after: PublicState, question: str,
                         patient_reply: str) -> bool:
    """Check structural state update; this does not claim clinical information gain."""
    expected_tail = (
        {"role": "assistant", "content": f"Question: {question}"},
        {"role": "user", "content": patient_reply.strip()},
    )
    return (
        bool(patient_reply.strip())
        and before.digest != after.digest
        and after.asked_count == before.asked_count + 1
        and len(after.dialogue) == len(before.dialogue) + 2
        and tuple(after.dialogue[-2:]) == expected_tail
    )


def assert_no_privileged_payload(messages: Sequence[Mapping[str, Any]], *, public_state: PublicState,
                                 unrevealed_facts: Sequence[str]) -> None:
    """Fail closed on source-label fields or exact unrevealed facts in Base/Expert prompts."""
    serialized = json.dumps(list(messages), ensure_ascii=False).lower()
    forbidden_fields = ("ground_truth", "answer_idx", "atomic_facts", "gold")
    present = [field for field in forbidden_fields if f'"{field}"' in serialized]
    if present:
        raise ValueError("privileged field leaked into Base/Expert prompt: " + ", ".join(present))
    public_text = public_state.public_text
    if not public_text:
        raise ValueError("empty public state")
    for fact in unrevealed_facts:
        value = str(fact).strip()
        if len(value) >= 12 and value.casefold() in serialized:
            raise ValueError("an unrevealed atomic fact value leaked into a Base/Expert prompt")


def public_state_from_row(row: Mapping[str, Any], *, max_asks: int) -> PublicState:
    initial = tuple(
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in row["initial_messages"]
    )
    dialogue = tuple(
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in row.get("dialogue", [])
    )
    asked = sum(
        message["role"] == "assistant" and message["content"].lstrip().lower().startswith("question:")
        for message in dialogue
    )
    return PublicState(
        case_id=str(row["case_id"]),
        state_id=str(row["state_id"]),
        state_index=int(row["state_index"]),
        initial_messages=initial,
        dialogue=dialogue,
        asked_count=asked,
        max_asks=int(max_asks),
    )


def extend_after_ask(state: PublicState, question: str, patient_reply: str) -> PublicState:
    return PublicState(
        case_id=state.case_id,
        state_id=state.state_id,
        state_index=state.state_index,
        initial_messages=state.initial_messages,
        dialogue=(
            *state.dialogue,
            {"role": "assistant", "content": f"Question: {question}"},
            {"role": "user", "content": patient_reply.strip()},
        ),
        asked_count=state.asked_count + 1,
        max_asks=state.max_asks,
    )


def action_net_return(correctness: float, *, ask_count: int, defer_count: int,
                      ask_cost: float, defer_cost: float) -> dict[str, float]:
    ask_charge = float(ask_cost) * int(ask_count)
    defer_charge = float(defer_cost) * int(defer_count)
    return {
        "accuracy_reward": float(correctness),
        "ask_cost": ask_charge,
        "defer_cost": defer_charge,
        "return": float(correctness) - ask_charge - defer_charge,
    }
