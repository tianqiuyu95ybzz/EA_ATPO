#!/usr/bin/env python3
"""Extract reproducible, frozen Base hidden-state vectors from public V2 states.

This module reads only the Q-dataset's pre-action public_state, uses a pinned
Transformers model/tokenizer revision, and writes vectors plus a content manifest.
It does not start a generation service or inspect Q labels while encoding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Sequence

from .data import public_state_hash, read_jsonl


FORMAT_VERSION = "ea_atpo_v2_public_embedding_cache_v1"
SERIALIZATION_RULE = "tokenizer.apply_chat_template(add_generation_prompt=false, chat_template_kwargs); tokenizer(add_special_tokens=false)"


@dataclass(frozen=True)
class PublicState:
    case_id: str
    state_id: str
    public_state: tuple[dict[str, str], ...]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def template_hash(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        raise ValueError("tokenizer must define chat_template to serialize public conversation states")
    return json_hash(template)


def render_public_state(public_state: Sequence[dict[str, str]], tokenizer: Any,
                        chat_template_kwargs: dict[str, Any] | None = None) -> str:
    if any(message.get("role") not in {"user", "assistant"} for message in public_state):
        raise ValueError("embedding input may contain only public user/assistant messages")
    try:
        return str(tokenizer.apply_chat_template(
            list(public_state), tokenize=False, add_generation_prompt=False,
            **(chat_template_kwargs or {}),
        ))
    except Exception as exc:
        raise ValueError(f"failed to apply the tokenizer chat template: {type(exc).__name__}") from exc


def pool_hidden_states(hidden: Any, attention_mask: Any, *, normalize_l2: bool = False):
    """Mean-pool only unpadded final-layer tokens; optionally L2-normalize."""
    import torch

    if hidden.ndim != 3 or attention_mask.ndim != 2 or hidden.shape[:2] != attention_mask.shape:
        raise ValueError("hidden states must be [batch, tokens, dim] and match the attention mask")
    weights = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    counts = weights.sum(dim=1).clamp_min(1.0)
    pooled = (hidden * weights).sum(dim=1) / counts
    if normalize_l2:
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    if not torch.isfinite(pooled).all():
        raise ValueError("pooled embeddings contain non-finite values")
    return pooled


def build_embedding_config(*, model_id: str, revision: str, tokenizer_id: str,
                           tokenizer_revision: str, prompt_template_sha256: str,
                           pooling: str, max_length: int, truncation_side: str,
                           normalize_l2: bool, dtype: str, device_name: str,
                           batch_size: int, torch_version: str,
                           transformers_version: str,
                           chat_template_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    config = {
        "format_version": FORMAT_VERSION,
        "model_id": model_id,
        "model_revision": revision,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "prompt_template_hash": prompt_template_sha256,
        "serialization_rule": SERIALIZATION_RULE,
        "chat_template_kwargs": chat_template_kwargs or {},
        "pooling": pooling,
        "max_length": int(max_length),
        "truncation_side": truncation_side,
        "padding_side": "right",
        "batch_size": int(batch_size),
        "normalize_l2": bool(normalize_l2),
        "dtype": dtype,
        "device_name": device_name,
        "feature_boundary": "public_state_before_action_only",
        "torch_version": torch_version,
        "transformers_version": transformers_version,
    }
    config["embedding_config_hash"] = json_hash(config)
    return config


def unique_public_states(states_path: Path, case_split_path: Path, split_name: str):
    """Read only case/state/public_state fields; all labels and outcomes are ignored."""
    by_key: dict[tuple[str, str], PublicState] = {}
    for line_number, row in enumerate(read_jsonl(states_path), 1):
        metadata = row.get("metadata") or {}
        case_id = str(row.get("case_id", metadata.get("case_id", "")))
        state_id = str(row.get("state_id", metadata.get("state_id", "")))
        messages = row.get("public_state", metadata.get("public_state"))
        if not case_id or not state_id or not isinstance(messages, list) or not messages:
            raise ValueError(f"{states_path}:{line_number}: expected case_id, state_id, and public_state")
        visible = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
                raise ValueError(f"{states_path}:{line_number}: features may use only public user/assistant messages")
            visible.append({"role": str(message["role"]), "content": str(message.get("content", ""))})
        example = PublicState(case_id, state_id, tuple(visible))
        key = (case_id, state_id)
        if key in by_key and public_state_hash(by_key[key].public_state) != public_state_hash(example.public_state):
            raise ValueError(f"{states_path}:{line_number}: public state differs across repeated rollout rows for {key}")
        by_key.setdefault(key, example)
    examples = [by_key[key] for key in sorted(by_key)]
    if not examples:
        raise ValueError(f"no public states found in {states_path}")
    split = json.loads(case_split_path.read_text(encoding="utf-8"))
    if split.get("unit") != "case_id" or split_name not in {"train", "validation", "internal_test"}:
        raise ValueError("case split must use case_id and split must be train/validation/internal_test")
    allowed_cases = set(map(str, split.get(split_name, [])))
    observed_cases = {row.case_id for row in examples}
    if not observed_cases or not observed_cases.issubset(allowed_cases):
        raise ValueError(f"states include cases outside official {split_name}: {len(observed_cases-allowed_cases)}")
    return examples, {
        "official_split": split_name,
        "official_split_unit": "case_id",
        "case_count": len(observed_cases),
        "state_count": len(examples),
        "no_gold_or_outcome_fields_used": True,
    }


def extract_batches(model: Any, tokenizer: Any, examples: Sequence[Any], *, device: str,
                    max_length: int, batch_size: int, normalize_l2: bool,
                    chat_template_kwargs: dict[str, Any] | None = None):
    import torch

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs a pad_token_id or eos_token_id for batching")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    prompts = [render_public_state(row.public_state, tokenizer, chat_template_kwargs) for row in examples]
    prompt_hashes = [sha256_bytes(prompt.encode("utf-8")) for prompt in prompts]
    output_vectors = []
    for start in range(0, len(prompts), batch_size):
        prompt_batch = prompts[start:start + batch_size]
        tokens = tokenizer(
            prompt_batch,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        with torch.inference_mode():
            result = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = result.last_hidden_state
            pooled = pool_hidden_states(hidden, attention_mask, normalize_l2=normalize_l2)
        values = pooled.detach().float().cpu().numpy()
        output_vectors.extend(values.astype("float32").tolist())
    return output_vectors, prompt_hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, required=True, help="Q JSONL or raw trajectory JSONL; only public_state fields are read")
    parser.add_argument("--split", choices=("train", "validation", "internal_test"), default="train")
    parser.add_argument("--case-split", type=Path, required=True, help="official case_id split; only train cases are allowed")
    parser.add_argument("--model-path", required=True, help="local HF snapshot or model identifier")
    parser.add_argument("--model-id", required=True, help="stable logical model id to store in the cache")
    parser.add_argument("--revision", required=True, help="immutable 40-hex model revision")
    parser.add_argument("--tokenizer-path", help="defaults to --model-path")
    parser.add_argument("--tokenizer-revision", help="defaults to --revision")
    parser.add_argument("--output", type=Path, required=True, help="new JSONL embedding cache path")
    parser.add_argument("--device", default="cpu", help="cpu by default; cuda:N must be explicitly selected")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--pooling", choices=("mean_last_hidden_state",), default="mean_last_hidden_state")
    parser.add_argument("--chat-template-kwargs-json", default='{"enable_thinking":false}',
                        help="JSON kwargs passed to the model-native chat template")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--truncation-side", choices=("left", "right"), default="left")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--normalize-l2", action="store_true")
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        parser.error("--revision must be an immutable 40-character commit hash")
    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer_revision = args.tokenizer_revision or args.revision
    if not re.fullmatch(r"[0-9a-fA-F]{40}", tokenizer_revision):
        parser.error("--tokenizer-revision must be an immutable 40-character commit hash")
    if args.max_length < 8 or args.batch_size < 1:
        parser.error("--max-length must be at least 8 and --batch-size must be positive")
    try:
        chat_template_kwargs = json.loads(args.chat_template_kwargs_json)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid --chat-template-kwargs-json: {exc}")
    if not isinstance(chat_template_kwargs, dict):
        parser.error("--chat-template-kwargs-json must be a JSON object")
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".manifest.json").exists():
        raise FileExistsError("refusing to overwrite an embedding cache or its manifest")

    examples, split_audit = unique_public_states(args.states, args.case_split, args.split)
    for row in examples:
        if row.raw_metadata.get("gold_or_future_fields_in_features") is True:
            raise ValueError("Q dataset marks privileged fields in features")

    # Set deterministic CUDA environment before torch is imported by Transformers.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoModel, AutoTokenizer
    transformers_version = __import__("transformers").__version__

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.device.startswith("cuda"):
        device_index = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(f"requested {args.device}, but only {torch.cuda.device_count()} CUDA devices are visible")
        torch.cuda.set_device(device_index)
        device_name = torch.cuda.get_device_name(device_index)
    elif args.device == "cpu":
        device_name = "cpu"
    else:
        parser.error("--device must be cpu or cuda:N")

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, revision=tokenizer_revision, trust_remote_code=False,
    )
    tokenizer.truncation_side = args.truncation_side
    prompt_template_sha256 = template_hash(tokenizer)
    model = AutoModel.from_pretrained(
        args.model_path, revision=args.revision, torch_dtype=torch_dtype,
        trust_remote_code=False, low_cpu_mem_usage=True, attn_implementation="eager",
    ).to(args.device)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False

    config = build_embedding_config(
        model_id=args.model_id,
        revision=args.revision.lower(),
        tokenizer_id=tokenizer_path,
        tokenizer_revision=tokenizer_revision.lower(),
        prompt_template_sha256=prompt_template_sha256,
        pooling=args.pooling,
        max_length=args.max_length,
        truncation_side=args.truncation_side,
        normalize_l2=args.normalize_l2,
        dtype=args.dtype,
        device_name=device_name,
        batch_size=args.batch_size,
        torch_version=torch.__version__,
        transformers_version=transformers_version,
        chat_template_kwargs=chat_template_kwargs,
    )
    vectors, prompt_hashes = extract_batches(
        model, tokenizer, examples, device=args.device, max_length=args.max_length,
        batch_size=args.batch_size, normalize_l2=args.normalize_l2,
        chat_template_kwargs=chat_template_kwargs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("w", encoding="utf-8") as stream:
            for row, vector, prompt_digest in zip(examples, vectors, prompt_hashes):
                record = {
                    "case_id": row.case_id,
                    "state_id": row.state_id,
                    "feature_boundary": "public_state_before_action_only",
                    "state_hash": public_state_hash(row.public_state),
                    "rendered_prompt_hash": prompt_digest,
                    "embedding": vector,
                    "model_id": args.model_id,
                    "model_revision": args.revision.lower(),
                    "pooling": args.pooling,
                    "max_length": args.max_length,
                    "prompt_template_hash": prompt_template_sha256,
                    "embedding_config_hash": config["embedding_config_hash"],
                }
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        temp_path.replace(args.output)
    finally:
        temp_path.unlink(missing_ok=True)

    manifest = {
        "format": FORMAT_VERSION,
        **config,
        "chat_template_kwargs": chat_template_kwargs,
        "source_states_sha256": sha256_bytes(args.states.read_bytes()),
        "case_split_sha256": sha256_bytes(args.case_split.read_bytes()),
        "case_split_audit": split_audit,
        "state_count": len(examples),
        "embedding_dimension": len(vectors[0]),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_bytes(args.output.read_bytes()),
        "device": args.device,
        "seed": args.seed,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "privileged_fields_used": False,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "completed",
        "state_count": len(examples),
        "embedding_dimension": len(vectors[0]),
        "model_id": args.model_id,
        "model_revision": args.revision.lower(),
        "feature_boundary": "public_state_before_action_only",
        "output": str(args.output.resolve()),
        "manifest": str(manifest_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
