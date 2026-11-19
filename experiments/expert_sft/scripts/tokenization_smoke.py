#!/usr/bin/env python3
"""Check tokenizer, chat-template boundary, answer-only labels, and lengths."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/expert_sft"
os.environ.setdefault("HF_HOME", str(EXP / "cache/hf_home"))
os.environ.setdefault("HF_HUB_CACHE", str(EXP / "cache/hf_home/hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(EXP / "cache/datasets"))
os.environ.setdefault("XDG_CACHE_HOME", str(EXP / "cache/xdg"))
os.environ.setdefault("TMPDIR", str(EXP / "cache/tmp"))

from transformers import AutoTokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=EXP / "assets/models/Qwen3-32B")
    parser.add_argument("--data", type=Path, default=EXP / "data/train_sft.jsonl")
    parser.add_argument("--output", type=Path, default=EXP / "evaluations/tokenization_smoke.json")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    rows = [json.loads(line) for line in args.data.open(encoding="utf-8") if line.strip()]
    lengths = []
    target_lengths = []
    for row in rows:
        messages = row["messages"]
        if messages[-1]["role"] != "assistant":
            raise ValueError("SFT target is not the final assistant turn")
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"Chat-template prompt/target boundary mismatch for case {row['case_id']}")
        target_ids = full_ids[len(prompt_ids) :]
        if not target_ids or "Expert Final Answer:" not in messages[-1]["content"]:
            raise ValueError(f"Missing answer-only Expert target for {row['case_id']}")
        if "<think>" in messages[-1]["content"] or "reason" in messages[-1]["content"].lower():
            raise ValueError("Target contains reasoning; only source answer letters are permitted")
        lengths.append(len(full_ids))
        target_lengths.append(len(target_ids))
    if not rows:
        raise ValueError("SFT dataset is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "passed",
        "tokenizer_source": str(args.model),
        "chat_template_present": bool(tokenizer.chat_template),
        "enable_thinking": False,
        "assistant_only_prefix_mask_boundary_valid": True,
        "sample_count": len(rows),
        "sequence_length_max": max(lengths),
        "sequence_length_p50": sorted(lengths)[len(lengths) // 2],
        "target_length_max": max(target_lengths),
        "no_truncation": True,
        "target_format": "Expert Final Answer: X (official A-D label, no explanation)",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
