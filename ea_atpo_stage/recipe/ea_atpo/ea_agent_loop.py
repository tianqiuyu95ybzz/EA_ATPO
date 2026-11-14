"""Four-action EA-ATPO loop integrated with the pinned ATPO Agent/Critic stack.

This module is added outside ATPO-official. The actor-generated action and its
content remain ordinary assistant tokens, while Patient/Expert observations are
appended as unmasked human/tool content. The inherited ATPO tree backup then
propagates the same net action rewards into Critic returns and Actor advantages.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, convert_to_messages, convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult

from recipe.atpo.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from recipe.atpo.chat_model import ChatModel, MaxTokenExceededError, convert_to_agent_output
from recipe.atpo.user_assistant_agent_loop import TreeSearchRunner as ATPOTreeSearchRunner
from recipe.atpo.user_assistant_agent_loop import call_critic_model
from recipe.atpo.tree_search_manager import TreeNode

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


PATIENT_SYSTEM = """You are a patient simulator. Answer the doctor's question using only
these patient facts. Do not infer, diagnose, or add outside information. If the
facts do not directly answer the question, say exactly: The patient cannot answer
this question. Return a concise patient response."""

ACTION_PROTOCOL = """EA-ATPO action protocol:
At each turn output exactly one action, and do not output other action formats:
- Final Answer: <A|B|C|D>
- Question: <one specific question for the patient>
- Consult Expert: <one specific question for the expert>
- Defer to Expert (the expert independently selects the final answer)
ASK and CONSULT return one observation and then you continue deciding. DEFER is
terminal: the expert independently selects the answer. Only the visible case and
messages are available to you; never claim to know unasked patient facts. Keep the
answer concise. Treat DEFER as an ordinary legal option when specialist judgment
is appropriate. The episode has a fixed action budget; obey the latest decision
state and select only from the listed legal actions."""


def _legal_actions(
    *, actor_actions_used: int, budget: int, asked: int, max_asks: int,
    consulted: int, max_consults: int,
) -> list[str]:
    remaining = max(0, budget - actor_actions_used)
    actions = ["ANSWER", "DEFER"]
    if remaining > 1:
        if asked < max_asks:
            actions.insert(1, "ASK")
        if consulted < max_consults:
            actions.insert(-1, "CONSULT")
    return actions


def _failure_category(reason: str | None) -> str:
    if not reason:
        return "none"
    if reason.startswith("invalid_actor_action"):
        return "protocol_error"
    if reason.startswith((
        "actor_generation_failed", "patient_call_failed", "consult_call_failed", "defer_call_failed"
    )):
        return "environment_error"
    if reason in {
        "ask_budget_exhausted", "consult_budget_exhausted", "no_remaining_continuation_budget"
    }:
        return "policy_constraint_violation"
    return "other_failure"


class ExpertProtocolError(ValueError):
    """An Expert response remained invalid after one format-only retry."""

    def __init__(self, message: str, *, request_count: int, prompt_tokens: int,
                 completion_tokens: int, latency_s: float, raw_response: str):
        super().__init__(message)
        self.request_count = request_count
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.latency_s = latency_s
        self.raw_response = raw_response


def _masked_failure_placeholder(prompt_ids: Sequence[int], tokenizer: Any) -> AgentLoopOutput:
    """Keep one failed sample from aborting valid sibling rollouts.

    The upstream worker expects one ``AgentLoopOutput`` per input and lets an
    exception from any sample abort its whole ``asyncio.gather``. This row has
    a zero response mask, so it contributes no Actor or Critic tokens. The
    actual protocol/environment failures remain in the persisted episode
    traces; the placeholder is not a strategy trajectory.
    """
    terminal_id = getattr(tokenizer, "eos_token_id", None)
    if terminal_id is None:
        terminal_id = getattr(tokenizer, "pad_token_id", None)
    if terminal_id is None:
        raise RuntimeError("Cannot build an all-masked failure placeholder without an EOS or PAD token")
    return AgentLoopOutput(
        prompt_ids=[int(token) for token in prompt_ids],
        response_ids=[int(terminal_id)],
        response_mask=[0],
        response_logprobs=[0.0],
        advantages=[0.0],
        returns=[0.0],
        value_response_mask=[0.0],
        reward_score=0.0,
        num_turns=0,
        metrics=AgentLoopMetrics(),
        verifier_responses=[],
        statistics={
            "ea_batch_placeholder": True,
            "q_value_variance_list": [],
            "mdp_value_list": [],
            "critic_value_list": [],
        },
    )


def _expert_failure_usage(exc: Exception) -> dict[str, Any]:
    request_count = int(getattr(exc, "request_count", 1))
    return {
        "expert_request_count": request_count,
        "expert_retry_count": max(0, request_count - 1),
        "expert_prompt_tokens": int(getattr(exc, "prompt_tokens", 0)),
        "expert_completion_tokens": int(getattr(exc, "completion_tokens", 0)),
        "expert_latency_s": float(getattr(exc, "latency_s", 0.0)),
        "expert_protocol_error_response": str(getattr(exc, "raw_response", ""))[:500],
    }


def _gold_letter(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("answer", "")
    return str(value).strip().upper()


def _public_action(action: Action, content: str) -> str:
    if action is Action.ANSWER:
        return f"Final Answer: {content.strip().upper()}"
    if action is Action.ASK:
        return f"Question: {content.strip()}"
    if action is Action.CONSULT:
        return f"Consult Expert: {content.strip()}"
    return "Defer to Expert"


def _render_public_initial(prompt: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], ...]:
    result: list[dict[str, str]] = []
    for message in prompt:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        if role == "system":
            continue
        if role == "assistant":
            try:
                action, payload = parse_action(content)
                content = _public_action(action, payload)
            except ValueError:
                content = visible_response(content)
        result.append({"role": "assistant" if role == "assistant" else "user", "content": content})
    return tuple(result)


def _install_action_protocol(prompt: Sequence[Mapping[str, Any]], *, budget: int, max_asks: int, max_consults: int) -> list[dict[str, str]]:
    messages = [{"role": str(m.get("role", "user")), "content": str(m.get("content", ""))} for m in prompt]
    legal_actions = _legal_actions(
        actor_actions_used=0, budget=budget, asked=0, max_asks=max_asks,
        consulted=0, max_consults=max_consults,
    )
    rule = (
        ACTION_PROTOCOL
        + f"\nInitial decision state: actor actions used=0/{budget}; remaining={budget}; "
        + f"ASK used=0/{max_asks}; CONSULT used=0/{max_consults}. "
        + f"Legal actions: {', '.join(legal_actions)}."
    )
    system_idx = next((i for i, message in enumerate(messages) if message["role"] == "system"), None)
    if system_idx is None:
        messages.insert(0, {"role": "system", "content": rule})
    else:
        messages[system_idx] = {**messages[system_idx], "content": rule + "\n\n" + messages[system_idx]["content"]}
    return messages


def _decision_observation(
    role: str,
    response: str,
    *,
    actor_actions_used: int,
    remaining: int,
    asked: int,
    max_asks: int,
    consulted: int,
    max_consults: int,
) -> str:
    legal_actions = _legal_actions(
        actor_actions_used=actor_actions_used,
        budget=actor_actions_used + remaining,
        asked=asked,
        max_asks=max_asks,
        consulted=consulted,
        max_consults=max_consults,
    )
    return (
        f"{role} response:\n{response}\n\n"
        "EA-ATPO decision state (follow these limits): "
        f"actor actions used={actor_actions_used}; remaining={remaining}; "
        f"ASK used={asked}/{max_asks}; CONSULT used={consulted}/{max_consults}. "
        f"Legal actions: {', '.join(legal_actions)}. Choose exactly one of them."
    )


class EAChatModel(ChatModel):
    """ChatModel that preserves sampled token log-probabilities for action traces."""

    async def preprocess(self, messages, **kwargs):
        if not messages or messages[-1].type != "human":
            raise ValueError("EA-ATPO actor prompt must end with a human message")
        loop = asyncio.get_running_loop()
        prior_ai = next((m for m in reversed(messages) if m.type == "ai"), None)

        # The upstream Critic worker requires the canonical Qwen3 assistant
        # prefix as its five-token value-query response. Re-tokenize the exact
        # visible transcript with that template; Actor generations use the
        # no-thinking template below to keep action responses bounded.
        if kwargs.get("is_critic", False):
            prompt_ids = await loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    convert_to_openai_messages(messages),
                    add_generation_prompt=True,
                    tokenize=True,
                ),
            )
            request_id = (
                prior_ai.response_metadata.get("request_id")
                if prior_ai is not None
                else None
            )
            return request_id or str(uuid4()), list(prompt_ids), []

        has_prior_rollout = prior_ai is not None and "prompt_ids" in prior_ai.response_metadata
        if not has_prior_rollout:
            prompt_ids = await loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    convert_to_openai_messages(messages),
                    add_generation_prompt=True,
                    tokenize=True,
                    enable_thinking=False,
                ),
            )
            return str(uuid4()), list(prompt_ids), []

        metadata = prior_ai.response_metadata
        prior_mask = list(metadata["response_mask"])
        prior_prompt_ids = list(metadata["prompt_ids"])
        human_messages = convert_to_openai_messages(messages[messages.index(prior_ai) + 1 :])
        human_ids = await loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode("\n")
            + self.tokenizer.apply_chat_template(
                human_messages,
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=False,
            ),
        )
        if len(prior_mask) + len(human_ids) >= self.max_tokens:
            raise MaxTokenExceededError(f"Max response length {self.max_tokens} exceeded")

        request_id = metadata.pop("request_id")
        prompt_ids = metadata.pop("prompt_ids")
        response_mask = metadata.pop("response_mask")
        prompt_ids = list(prompt_ids) + list(human_ids)
        response_mask = list(response_mask) + [0] * len(human_ids)
        return request_id, prompt_ids, response_mask

    async def _agenerate(self, messages, stop=None, **kwargs):
        n = int(kwargs.get("n", 1))
        prior_ai = next((m for m in reversed(messages) if m.type == "ai"), None)
        prior_logprobs = []
        if prior_ai is not None:
            prior_logprobs = list(prior_ai.response_metadata.get("response_logprobs", []))
        request_id_base, prompt_ids, response_mask = await self.preprocess(messages, **kwargs)
        sampling_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }
        if "sampling_params" in kwargs:
            sampling_params.update(kwargs["sampling_params"])
        remaining_response_tokens = max(1, self.max_tokens - len(response_mask))
        sampling_params["max_new_tokens"] = min(
            int(sampling_params.get("max_new_tokens", remaining_response_tokens)),
            remaining_response_tokens,
        )
        tasks = [
            self.client.generate(
                request_id=f"{request_id_base}-{i}",
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
            for i in range(n)
        ]
        started = time.perf_counter()
        outputs = await asyncio.gather(*tasks)
        # n candidates share one batched request window. Dividing by n would
        # report an artificially low per-candidate wall latency.
        elapsed = time.perf_counter() - started
        generations = []
        for i, output in enumerate(outputs):
            request_id = f"{request_id_base}-{i}"
            message = await self._postprocess(request_id, prompt_ids, response_mask, output.token_ids, **kwargs)
            token_logprobs = list(output.log_probs) if output.log_probs is not None else None
            message.response_metadata["current_response_ids"] = list(output.token_ids)
            message.response_metadata["current_logprobs"] = token_logprobs
            message.response_metadata["current_prompt_token_count"] = len(prompt_ids)
            message.response_metadata["current_latency_s"] = elapsed
            if token_logprobs is not None:
                prefix = prior_logprobs + [0.0] * max(0, len(response_mask) - len(prior_logprobs))
                if len(prefix) > len(response_mask):
                    prefix = prefix[: len(response_mask)]
                message.response_metadata["response_logprobs"] = prefix + token_logprobs
                message.response_metadata["sampled_response_logprob_sum"] = float(sum(token_logprobs))
            generations.append(ChatGeneration(message=message))
        return ChatResult(generations=generations)


class EAATPOSearchRunner(ATPOTreeSearchRunner):
    """ATPO tree search with ASK, CONSULT, DEFER and EA net action rewards."""

    def __init__(self, initial_messages, params, model, config, reward_coefficients, statistic_metrics, public_state, costs):
        super().__init__(initial_messages, params, model, config, reward_coefficients, statistic_metrics)
        self.max_actor_actions = int(config["configurable"]["max_actor_actions"])
        self.max_asks = int(config["configurable"]["max_asks"])
        self.max_consults = int(config["configurable"]["max_consults"])
        self.costs = dict(costs)
        self.correctness_reward = float(config["configurable"]["correctness_reward"])
        self.root.ea_public_state = public_state
        self.root.ea_actor_actions_used = 0
        self.root.ea_asked_count = 0
        self.root.ea_consulted_count = 0
        self.root.ea_action = None
        self.root.ea_event = None
        self.root.ea_terminal_answer = None
        self.root.ea_failure_reason = None

    def _state_after(self, state: PublicState, *turns: Mapping[str, str], action: Action | None = None) -> PublicState:
        actions_used = state.actor_actions_used + (1 if action is not None else 0)
        asked = state.asked_count + int(action is Action.ASK)
        consulted = state.consulted_count + int(action is Action.CONSULT)
        return PublicState(
            sample_id=state.sample_id,
            initial_messages=state.initial_messages,
            dialogue=(*state.dialogue, *turns),
            asked_count=asked,
            consulted_count=consulted,
            remaining_interaction_budget=max(0, self.max_actor_actions - actions_used),
            actor_actions_used=actions_used,
        )

    async def _role_generate(self, role_messages: Sequence[Mapping[str, str]], *, max_new_tokens: int) -> dict[str, Any]:
        tokenizer = self.model.tokenizer
        client = self.config["configurable"]["server_manager"]
        loop = asyncio.get_running_loop()
        prompt_ids = await loop.run_in_executor(
            None,
            lambda: tokenizer.apply_chat_template(
                list(role_messages),
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=False,
            ),
        )
        prompt_ids = [int(x) for x in prompt_ids]
        started = time.perf_counter()
        output = await client.generate(
            request_id=f"ea-{uuid4()}",
            prompt_ids=prompt_ids,
            sampling_params={"temperature": 0.0, "top_p": 1.0, "max_new_tokens": int(max_new_tokens)},
        )
        elapsed = time.perf_counter() - started
        token_ids = list(output.token_ids)
        response = await loop.run_in_executor(None, lambda: tokenizer.decode(token_ids, skip_special_tokens=False))
        return {
            "content": str(response),
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(token_ids),
            "latency_s": float(elapsed),
        }

    async def _patient_response(self, facts: Sequence[str], question: str) -> dict[str, Any]:
        request = json.dumps([str(x) for x in facts], ensure_ascii=False)
        messages = [
            {"role": "system", "content": PATIENT_SYSTEM},
            {"role": "user", "content": f"Patient atomic facts:\n{request}\n\nDoctor question:\n{question}"},
        ]
        result = await self._role_generate(messages, max_new_tokens=256)
        result["content"] = visible_response(result["content"])
        return result

    async def _expert_response(
        self,
        state: PublicState,
        action: Action,
        *,
        question: str = "",
        facts: Sequence[str],
        gold: Any,
    ) -> dict[str, Any]:
        base_messages = build_expert_messages(state, action, consult_question=question)
        prompt_tokens = completion_tokens = request_count = 0
        latency_s = 0.0
        raw_response = ""
        last_error: ValueError | None = None
        for attempt in range(2):
            messages = list(base_messages)
            if attempt:
                if action is Action.CONSULT:
                    correction = (
                        "Format correction: the prior draft violated the CONSULT contract. Discard it. "
                        "Return exactly one `Expert Advice: ...` suggestion. Do not name an option letter, "
                        "state which option is correct, or provide a final answer."
                    )
                else:
                    correction = (
                        "Format correction: the prior draft was not a parseable DEFER result. Discard it. "
                        "Choose one option and return exactly `Expert Final Answer: A` (replace A with your choice). "
                        "Do not include rationale or any other text."
                    )
                messages.append({"role": "user", "content": correction})
            validate_expert_view(
                messages,
                atomic_facts=unrevealed_atomic_facts(state, facts),
                ground_truth=gold,
            )
            request_count += 1
            result = await self._role_generate(messages, max_new_tokens=128)
            prompt_tokens += int(result["prompt_tokens"])
            completion_tokens += int(result["completion_tokens"])
            latency_s += float(result["latency_s"])
            raw_response = str(result["content"])
            try:
                result["content"] = validate_expert_response(action, raw_response)
                result.update({
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency_s": latency_s,
                    "request_count": request_count,
                    "retry_count": request_count - 1,
                })
                return result
            except ValueError as exc:
                last_error = exc
        raise ExpertProtocolError(
            str(last_error or "invalid expert response"),
            request_count=request_count,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency_s,
            raw_response=raw_response,
        )

    def _action_cost(self, action: Action) -> float:
        if action is Action.ASK:
            return -float(self.costs["lambda_ask"]) * float(self.costs["ask_cost"])
        if action is Action.CONSULT:
            return -float(self.costs["lambda_consult"]) * float(self.costs["consult_cost"])
        if action is Action.DEFER:
            return -float(self.costs["lambda_defer"]) * float(self.costs["defer_cost"])
        return 0.0

    def _sample_metadata(self, message: AIMessage) -> dict[str, Any]:
        metadata = message.response_metadata
        logprobs = metadata.get("current_logprobs")
        logprob_sum = float(sum(logprobs)) if logprobs is not None else None
        probability = math.exp(max(-745.0, min(0.0, logprob_sum))) if logprob_sum is not None else None
        return {
            "joint_response_log_probability": logprob_sum,
            "joint_response_probability": probability,
            "actor_prompt_tokens": int(metadata.get("current_prompt_token_count", 0)),
            "actor_completion_tokens": len(metadata.get("current_response_ids", [])),
            "actor_latency_s": float(metadata.get("current_latency_s", 0.0)),
        }

    async def _expand_leaves_with_assistant(self):
        """Expand leaves while converting continuation failures into terminal rollouts.

        A failure before the first Actor token cannot form a valid PPO sample, so
        it is raised. A failure after an ASK/CONSULT edge ends that partial episode
        with zero terminal correctness and keeps already incurred costs.
        """
        tasks = {}
        for leaf in self.active_leaves:
            num_candidates = self.params["N_candidates"]
            if len(self.completed_trajectories) + len(self.active_leaves) * num_candidates > self.params["M_trajectories"]:
                num_candidates = 1
            tasks[leaf] = self.model.agenerate(
                [leaf.messages],
                n=num_candidates,
                sampling_params=self.config["configurable"]["sampling_params"],
            )

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        expansions = {}
        updated_active_leaves = []
        for leaf, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                if leaf.parent is None:
                    raise RuntimeError(f"Actor generation failed before any action: {type(result).__name__}: {result}") from result
                leaf.is_terminal = True
                leaf.ea_failure_reason = f"actor_generation_failed: {type(result).__name__}: {result}"
                leaf.critic_state_value = 0.0
                leaf.q_value = float(leaf.action_reward or 0.0)
                if leaf.ea_event is not None:
                    leaf.ea_event.update({
                        "terminated": True,
                        "termination_reason": "actor_generation_failed",
                        "failure_reason": leaf.ea_failure_reason,
                        "critic_value": 0.0,
                        "q_value": leaf.q_value,
                    })
                self.completed_trajectories.append(leaf)
            else:
                expansions[leaf] = [generation.message for generation in result.generations[0]]
                updated_active_leaves.append(leaf)
        self.active_leaves = updated_active_leaves
        return expansions

    async def _process_assistant_expansions(self, expansions):
        all_new_nodes: list[TreeNode] = []
        facts = self.config["configurable"]["atomic_facts"]
        gold = self.config["configurable"]["ground_truth"]
        for parent, assistant_messages in expansions.items():
            for assistant_message in assistant_messages:
                node = TreeNode(
                    messages=parent.messages + [assistant_message],
                    parent=parent,
                    assistant_message=assistant_message,
                    level=parent.level + 1,
                )
                parent.children.append(node)
                prior_state: PublicState = parent.ea_public_state
                node.ea_actor_actions_used = parent.ea_actor_actions_used + 1
                node.ea_asked_count = parent.ea_asked_count
                node.ea_consulted_count = parent.ea_consulted_count
                node.ea_terminal_answer = None
                node.ea_failure_reason = None
                node.ea_expert_final = None
                node.ea_patient_response = None
                node.ea_expert_advice = None
                node.ea_consult_question = None
                node.ea_action_cost = 0.0
                node.ea_sample = self._sample_metadata(assistant_message)

                try:
                    action, content = parse_action(assistant_message.content)
                except ValueError as exc:
                    node.ea_action = None
                    node.ea_public_state = prior_state
                    node.ea_failure_reason = f"invalid_actor_action: {exc}"
                    node.is_terminal = True
                    node.ea_event = {
                        "action": None,
                        "action_valid": False,
                        "terminated": True,
                        "termination_reason": "invalid_actor_action",
                        "actor_text": visible_response(assistant_message.content),
                        **node.ea_sample,
                    }
                    all_new_nodes.append(node)
                    continue

                node.ea_action = action.value
                node.ea_asked_count += int(action is Action.ASK)
                node.ea_consulted_count += int(action is Action.CONSULT)
                node.ea_action_cost = self._action_cost(action)
                remaining_before = max(0, self.max_actor_actions - parent.ea_actor_actions_used)
                invalid_reason = None
                if action is Action.ASK and parent.ea_asked_count >= self.max_asks:
                    invalid_reason = "ask_budget_exhausted"
                elif action is Action.CONSULT and parent.ea_consulted_count >= self.max_consults:
                    invalid_reason = "consult_budget_exhausted"
                elif remaining_before <= 1 and action in (Action.ASK, Action.CONSULT):
                    invalid_reason = "no_remaining_continuation_budget"

                actor_text = _public_action(action, content)
                action_turn = {"role": "assistant", "content": actor_text}
                node.ea_consult_question = content if action is Action.CONSULT else None
                node.ea_event = {
                    "action": action.value,
                    "action_valid": invalid_reason is None,
                    "actor_text": actor_text,
                    "remaining_budget_before": remaining_before,
                    "asked_count_before": parent.ea_asked_count,
                    "consulted_count_before": parent.ea_consulted_count,
                    "termination_reason": None,
                    **node.ea_sample,
                }

                if invalid_reason is not None:
                    # An out-of-budget action makes no external call and incurs no resource cost.
                    node.ea_action_cost = 0.0
                    node.ea_failure_reason = invalid_reason
                    node.ea_public_state = self._state_after(prior_state, action_turn, action=action)
                    node.is_terminal = True
                    node.ea_event.update({
                        "action_valid": False,
                        "terminated": True,
                        "termination_reason": invalid_reason,
                        "action_cost": 0.0,
                    })
                    all_new_nodes.append(node)
                    continue

                if action is Action.ANSWER:
                    node.ea_public_state = self._state_after(prior_state, action_turn, action=action)
                    node.ea_terminal_answer = content.strip().upper()
                    node.is_terminal = True
                    node.ea_event.update({
                        "terminated": True,
                        "termination_reason": "base_answer",
                        "action_cost": 0.0,
                        "final_answer": node.ea_terminal_answer,
                    })
                    all_new_nodes.append(node)
                    continue

                if action is Action.ASK:
                    try:
                        patient = await self._patient_response(facts, content)
                        response = patient["content"]
                        node.ea_patient_response = response
                        node.ea_public_state = self._state_after(
                            prior_state,
                            action_turn,
                            {"role": "user", "content": response},
                            action=action,
                        )
                        observation = _decision_observation(
                            "Patient",
                            response,
                            actor_actions_used=node.ea_actor_actions_used,
                            remaining=max(0, self.max_actor_actions - node.ea_actor_actions_used),
                            asked=node.ea_asked_count,
                            max_asks=self.max_asks,
                            consulted=node.ea_consulted_count,
                            max_consults=self.max_consults,
                        )
                        node.messages.append(HumanMessage(
                            content=observation,
                            additional_kwargs={"ea_role": "patient", "verifier_response": ""},
                        ))
                        node.is_terminal = False
                        node.ea_event.update({
                            "terminated": False,
                            "termination_reason": None,
                            "action_cost": node.ea_action_cost,
                            "patient_response": response,
                            "patient_prompt_tokens": patient["prompt_tokens"],
                            "patient_completion_tokens": patient["completion_tokens"],
                            "patient_latency_s": patient["latency_s"],
                        })
                    except Exception as exc:
                        node.ea_public_state = self._state_after(prior_state, action_turn, action=action)
                        node.ea_failure_reason = f"patient_call_failed: {type(exc).__name__}: {exc}"
                        node.is_terminal = True
                        node.ea_event.update({
                            "terminated": True,
                            "termination_reason": "patient_call_failed",
                            "failure_reason": node.ea_failure_reason,
                            "action_cost": node.ea_action_cost,
                        })
                    all_new_nodes.append(node)
                    continue

                if action is Action.CONSULT:
                    public_after_question = self._state_after(prior_state, action_turn, action=action)
                    try:
                        expert = await self._expert_response(
                            public_after_question,
                            Action.CONSULT,
                            question=content,
                            facts=facts,
                            gold=gold,
                        )
                        advice = expert["content"]
                        node.ea_expert_advice = advice
                        node.ea_public_state = PublicState(
                            sample_id=public_after_question.sample_id,
                            initial_messages=public_after_question.initial_messages,
                            dialogue=(*public_after_question.dialogue, {"role": "user", "content": advice}),
                            asked_count=node.ea_asked_count,
                            consulted_count=node.ea_consulted_count,
                            remaining_interaction_budget=max(0, self.max_actor_actions - node.ea_actor_actions_used),
                            actor_actions_used=node.ea_actor_actions_used,
                        )
                        observation = _decision_observation(
                            "Expert advice",
                            advice,
                            actor_actions_used=node.ea_actor_actions_used,
                            remaining=max(0, self.max_actor_actions - node.ea_actor_actions_used),
                            asked=node.ea_asked_count,
                            max_asks=self.max_asks,
                            consulted=node.ea_consulted_count,
                            max_consults=self.max_consults,
                        )
                        node.messages.append(HumanMessage(
                            content=observation,
                            additional_kwargs={"ea_role": "expert_advice", "verifier_response": ""},
                        ))
                        node.is_terminal = False
                        node.ea_event.update({
                            "terminated": False,
                            "termination_reason": None,
                            "action_cost": node.ea_action_cost,
                            "consult_question": content,
                            "expert_advice": advice,
                            "expert_prompt_tokens": expert["prompt_tokens"],
                            "expert_completion_tokens": expert["completion_tokens"],
                            "expert_latency_s": expert["latency_s"],
                            "expert_request_count": expert["request_count"],
                            "expert_retry_count": expert["retry_count"],
                        })
                    except Exception as exc:
                        node.ea_public_state = public_after_question
                        node.ea_failure_reason = f"consult_call_failed: {type(exc).__name__}: {exc}"
                        node.is_terminal = True
                        node.ea_event.update({
                            "terminated": True,
                            "termination_reason": "consult_call_failed",
                            "failure_reason": node.ea_failure_reason,
                            "action_cost": node.ea_action_cost,
                            "consult_question": content,
                            **_expert_failure_usage(exc),
                        })
                    all_new_nodes.append(node)
                    continue

                # DEFER is terminal. The Expert receives the public state before the
                # transfer action and returns the final answer independently.
                try:
                    expert = await self._expert_response(
                        prior_state,
                        Action.DEFER,
                        facts=facts,
                        gold=gold,
                    )
                    answer_match = re.search(r"Expert Final Answer:\s*([A-D])\b", expert["content"], re.IGNORECASE)
                    if answer_match is None:
                        raise ValueError("validated DEFER result has no option letter")
                    answer = answer_match.group(1).upper()
                    node.ea_expert_final = f"Expert Final Answer: {answer}"
                    node.ea_terminal_answer = answer
                    node.ea_public_state = self._state_after(
                        prior_state,
                        action_turn,
                        {"role": "user", "content": node.ea_expert_final},
                        action=action,
                    )
                    node.messages.append(HumanMessage(
                        content=node.ea_expert_final,
                        additional_kwargs={"ea_role": "expert_final"},
                    ))
                    node.is_terminal = True
                    node.ea_event.update({
                        "terminated": True,
                        "termination_reason": "expert_deferred_answer",
                        "action_cost": node.ea_action_cost,
                        "expert_final_answer": node.ea_expert_final,
                        "expert_prompt_tokens": expert["prompt_tokens"],
                        "expert_completion_tokens": expert["completion_tokens"],
                        "expert_latency_s": expert["latency_s"],
                        "expert_request_count": expert["request_count"],
                        "expert_retry_count": expert["retry_count"],
                    })
                except Exception as exc:
                    node.ea_public_state = self._state_after(prior_state, action_turn, action=action)
                    node.ea_failure_reason = f"defer_call_failed: {type(exc).__name__}: {exc}"
                    node.is_terminal = True
                    node.ea_event.update({
                        "terminated": True,
                        "termination_reason": "defer_call_failed",
                        "failure_reason": node.ea_failure_reason,
                        "action_cost": node.ea_action_cost,
                        **_expert_failure_usage(exc),
                    })
                all_new_nodes.append(node)
        return all_new_nodes

    async def _evaluate_nodes(self, nodes: list[TreeNode]):
        critic_values: list[float] = []
        if self.params["call_critic_enabled"]:
            critic_values = await asyncio.gather(*[
                call_critic_model(node.messages, node.is_terminal, self.config)
                for node in nodes
            ])
        gold = _gold_letter(self.config["configurable"]["ground_truth"])
        gamma = float(self.params.get("gamma", 1.0))
        for idx, node in enumerate(nodes):
            answer = node.ea_terminal_answer
            correctness = self.correctness_reward * float(answer is not None and answer == gold)
            node.ea_terminal_correctness_reward = correctness
            node.action_reward = float(node.ea_action_cost) + correctness
            if node.is_terminal:
                node.critic_state_value = 0.0
            else:
                node.critic_state_value = float(critic_values[idx]) if self.params["call_critic_enabled"] else None
            node.q_value = (
                float(node.action_reward) + gamma * float(node.critic_state_value)
                if node.critic_state_value is not None
                else None
            )
            node.ea_event.update({
                "terminal_correctness_reward": correctness,
                "net_action_reward": float(node.action_reward),
                "critic_value": node.critic_state_value,
                "q_value": node.q_value,
                "terminal_answer": answer,
                "correct": bool(answer is not None and answer == gold),
            })

    def episode_trace(self, leaf: TreeNode) -> dict[str, Any]:
        path: list[TreeNode] = []
        current = leaf
        while current.parent is not None:
            path.append(current)
            current = current.parent
        path.reverse()
        events = []
        total_cost = 0.0
        token_usage = {
            "actor_prompt_tokens": 0,
            "actor_completion_tokens": 0,
            "patient_prompt_tokens": 0,
            "patient_completion_tokens": 0,
            "expert_prompt_tokens": 0,
            "expert_completion_tokens": 0,
        }
        measured_latency_s = 0.0
        for node in path:
            event = dict(node.ea_event or {})
            before = node.parent.ea_public_state if node.parent is not None else node.ea_public_state
            event.update({
                "public_state_id_before": before.state_id,
                "public_state_before": before.messages,
                "public_state_id_after": node.ea_public_state.state_id,
                "public_state_after": node.ea_public_state.messages,
                "actor_actions_used": node.ea_actor_actions_used,
                "asked_count": node.ea_asked_count,
                "consulted_count": node.ea_consulted_count,
                "remaining_interaction_budget": max(0, self.max_actor_actions - node.ea_actor_actions_used),
                "action_reward": node.action_reward,
                "return": node.returns,
                "advantage": node.advantage,
            })
            if node.ea_action_cost is not None:
                total_cost += -float(node.ea_action_cost)
            for key in token_usage:
                token_usage[key] += int(event.get(key, 0) or 0)
            for key in ("actor_latency_s", "patient_latency_s", "expert_latency_s"):
                measured_latency_s += float(event.get(key, 0.0) or 0.0)
            events.append(event)
        gold = _gold_letter(self.config["configurable"]["ground_truth"])
        answer = leaf.ea_terminal_answer
        correctness = self.correctness_reward * float(answer is not None and answer == gold)
        failure_category = _failure_category(leaf.ea_failure_reason)
        return {
            "sample_id": leaf.ea_public_state.sample_id,
            "public_initial_state": list(leaf.ea_public_state.initial_messages),
            "action_trace": events,
            "final_answer": answer,
            "evaluation": {
                "gold": gold,
                "correct": bool(answer is not None and answer == gold),
                "terminal_correctness_reward": correctness,
                "resource_cost_total": total_cost,
                "net_episode_reward": sum(float(n.action_reward or 0.0) for n in path),
                "failure_reason": leaf.ea_failure_reason,
                "failure_category": failure_category,
                "training_eligible": failure_category not in {"protocol_error", "environment_error"},
            },
            "resource_usage": {
                **token_usage,
                "total_tokens": sum(token_usage.values()),
                "measured_serial_latency_s": measured_latency_s,
                "patient_calls": sum(1 for event in events if event.get("patient_response") is not None),
                "expert_calls": sum(1 for event in events if event.get("expert_advice") is not None or event.get("expert_final_answer") is not None),
                "expert_requests": sum(int(event.get("expert_request_count", 0) or 0) for event in events),
                "expert_retries": sum(int(event.get("expert_retry_count", 0) or 0) for event in events),
            },
        }


class EAATPOAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        return None

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        extra_info = kwargs["extra_info"]
        prompt = extra_info["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        facts = [str(x) for x in extra_info["atomic_facts"]]
        gold = kwargs["reward_model"]["ground_truth"]
        rollout = self.config.actor_rollout_ref.rollout
        ea_config = self.config.get("ea_atpo", {})
        tree_params = dict(kwargs["tree_search_params"])
        tree_params["gamma"] = float(self.config.algorithm.gamma)
        tree_params["lam"] = float(self.config.algorithm.lam)
        max_actions = int(ea_config.get("max_actor_actions", rollout.multi_turn.max_assistant_turns or 4))
        max_asks = int(ea_config.get("max_asks", 2))
        max_consults = int(ea_config.get("max_consults", 2))
        costs = {
            "ask_cost": float(ea_config.get("ask_cost", 0.05)),
            "consult_cost": float(ea_config.get("consult_cost", 0.20)),
            "defer_cost": float(ea_config.get("defer_cost", 0.60)),
            "lambda_ask": float(ea_config.get("lambda_ask", 1.0)),
            "lambda_consult": float(ea_config.get("lambda_consult", 1.0)),
            "lambda_defer": float(ea_config.get("lambda_defer", 1.0)),
        }
        correctness_reward = float(ea_config.get("correctness_reward", self.config.get("reward_coefficients", {}).get("correctness_score", 1.0)))
        actor_messages = _install_action_protocol(prompt, budget=max_actions, max_asks=max_asks, max_consults=max_consults)
        public_state = PublicState(
            sample_id=str(extra_info.get("index", extra_info.get("case_id", "unknown"))),
            initial_messages=_render_public_initial(prompt),
            remaining_interaction_budget=max_actions,
        )

        model = EAChatModel(
            model=self.config.actor_rollout_ref.model_type,
            client=self.server_manager,
            tokenizer=self.tokenizer,
            max_tokens=int(rollout.response_length),
        )
        sampling = dict(sampling_params)
        sampling["logprobs"] = True
        config = {
            "configurable": {
                "model": model,
                "server_manager": self.server_manager,
                "tokenizer": self.tokenizer,
                "critic_worker": kwargs.get("critic_worker"),
                "sampling_params": sampling,
                "max_assistant_turns": max_actions,
                "max_actor_actions": max_actions,
                "max_asks": max_asks,
                "max_consults": max_consults,
                "correctness_reward": correctness_reward,
                "atomic_facts": facts,
                "ground_truth": gold,
            }
        }
        initial_messages = convert_to_messages(actor_messages)
        runner = EAATPOSearchRunner(
            initial_messages,
            tree_params,
            model,
            config,
            self.config.get("reward_coefficients", {}),
            kwargs.get("statistic_metrics", {}),
            public_state,
            costs,
        )
        trajectories = await runner.run()
        outputs: list[AgentLoopOutput] = []
        trace_path = str(ea_config.get("trace_path", ""))
        for leaf in trajectories:
            episode = runner.episode_trace(leaf)
            if trace_path:
                self._append_jsonl(trace_path, episode)
            if not episode["evaluation"]["training_eligible"]:
                continue
            output = convert_to_agent_output(leaf, int(rollout.response_length))
            self._append_terminal_observation(output, leaf, int(rollout.response_length))
            last_ai = next((m for m in reversed(leaf.messages) if m.type == "ai"), None)
            if last_ai is not None:
                logprobs = last_ai.response_metadata.get("response_logprobs")
                if logprobs is not None:
                    valid_len = len(output.response_mask)
                    output.response_logprobs = list(logprobs[:valid_len]) + [0.0] * max(0, valid_len - len(logprobs))
            output.statistics = output.statistics or {}
            output.statistics["ea_episode"] = episode
            output.reward_score = float(episode["evaluation"]["net_episode_reward"])
            output.statistics["q_value_variance_list"] = output.statistics.get("q_value_variance_list", [])
            output.statistics["mdp_value_list"] = output.statistics.get("mdp_value_list", [])
            output.statistics["critic_value_list"] = output.statistics.get("critic_value_list", [])
            outputs.append(output)
        if not outputs:
            # A sibling sample may have valid strategy trajectories. Returning
            # a zero-mask row lets the upstream per-sample gather preserve those
            # trajectories instead of failing the entire worker batch. If every
            # sample in the global PPO batch is a placeholder, EARewardManager
            # rejects the all-zero Actor mask before any optimizer update.
            return [_masked_failure_placeholder(prompt, self.tokenizer)]
        return outputs

    def _append_terminal_observation(self, output: AgentLoopOutput, leaf: TreeNode, response_length: int) -> None:
        last_ai_index = next((i for i in range(len(leaf.messages) - 1, -1, -1) if leaf.messages[i].type == "ai"), None)
        if last_ai_index is None:
            return
        trailing = leaf.messages[last_ai_index + 1 :]
        if not trailing:
            return
        human_messages = convert_to_openai_messages(trailing)
        observation_ids = self.tokenizer.encode("\n") + self.tokenizer.apply_chat_template(
            human_messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
        remaining = max(0, response_length - len(output.response_ids))
        observation_ids = [int(x) for x in observation_ids[:remaining]]
        if not observation_ids:
            return
        count = len(observation_ids)
        output.response_ids = list(output.response_ids) + observation_ids
        output.response_mask = list(output.response_mask) + [0] * count
        if output.advantages is not None:
            output.advantages = list(output.advantages) + [0.0] * count
        if output.returns is not None:
            output.returns = list(output.returns) + [0.0] * count
        if output.value_response_mask is not None:
            output.value_response_mask = list(output.value_response_mask) + [0.0] * count
        if output.response_logprobs is not None:
            output.response_logprobs = list(output.response_logprobs) + [0.0] * count

    @staticmethod
    def _append_jsonl(path: str, payload: dict[str, Any]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
