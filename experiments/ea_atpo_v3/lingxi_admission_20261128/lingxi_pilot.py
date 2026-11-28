import hashlib
import json
from pathlib import Path
import re
import sys

import pyarrow.parquet as parquet

ROOT = Path(__file__).resolve().parent
LABELS = {"Depression": "A", "Anxiety": "B", "Mixed": "C", "Others": "D"}
DATA = ROOT / "lingxi"
ROLES = re.compile(r"(?:^|\s)(医生|患者)\s*[:：]")
SALT = "ea-atpo-v3-lingxi-pilot-20261128"


def read_split(split):
    return parquet.read_table(
        DATA / f"{split}-00000-of-00001.parquet",
        columns=["patient_id", "cleaned_text", "four_class_label"],
    ).to_pylist()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def clean(text):
    return re.sub(r"\s+", "", text or "")


def paired(text):
    matches = list(ROLES.finditer(text or ""))
    turns = [(m[1], text[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip())
             for i, m in enumerate(matches)]
    if turns and turns[0][0] == "医生":
        turns = turns[1:]
    if len(turns) < 3 or [role for role, _ in turns[:3]] != ["患者", "医生", "患者"]:
        return None
    first, question, response = (message for _, message in turns[:3])
    if not all((first, question, response)) or not re.search(r"[?？]", question):
        return None
    if clean(first) == clean(response):
        return None
    return [first, question, response]


def prepare():
    train, validation = read_split("train"), read_split("validation")
    train_ids = {r["patient_id"] for r in train if r["patient_id"]}
    val_ids = {r["patient_id"] for r in validation if r["patient_id"]}
    train_text = {digest(clean(r["cleaned_text"])) for r in train if clean(r["cleaned_text"])}
    val_text = {digest(clean(r["cleaned_text"])) for r in validation if clean(r["cleaned_text"])}
    train_first = {digest(clean(pair[0])) for row in train if (pair := paired(row["cleaned_text"]))}
    used_ids, used_text, used_first = set(), set(), set()
    eligible = []
    ranked = sorted(validation, key=lambda r: digest(SALT + str(r["patient_id"])))
    for row in ranked:
        pid, label = row["patient_id"], row["four_class_label"]
        pair = paired(row["cleaned_text"])
        if not pid or pid in train_ids or pid in used_ids or label not in LABELS or not pair:
            continue
        text_key = digest(clean(row["cleaned_text"]))
        first_key = digest(clean(pair[0]))
        if text_key in train_text or text_key in used_text or first_key in train_first or first_key in used_first:
            continue
        used_ids.add(pid)
        used_text.add(text_key)
        used_first.add(first_key)
        eligible.append({"sample": digest(SALT + pid), "label": label, "turns": pair})
    selected = [row for label in LABELS for row in [r for r in eligible if r["label"] == label][:2]]
    meta = {
        "train_rows": len(train), "validation_rows": len(validation),
        "train_unique_ids": len(train_ids), "validation_unique_ids": len(val_ids),
        "cross_split_shared_ids": len(train_ids & val_ids),
        "cross_split_shared_dialogues_normalized_exact": len(train_text & val_text),
        "eligible": {label: sum(r["label"] == label for r in eligible) for label in LABELS},
        "selected": {label: sum(r["label"] == label for r in selected) for label in LABELS},
    }
    print(json.dumps(meta), flush=True)
    assert len(selected) == 8
    (ROOT / "pilot-private.json").write_text(json.dumps(selected, ensure_ascii=False), encoding="utf-8")


def infer():
    from time import perf_counter
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    weights = Path(sys.argv[2])
    torch.manual_seed(0)
    assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    free, total = torch.cuda.mem_get_info()
    assert free >= 28 * 2**30
    torch.cuda.set_per_process_memory_fraction(24 * 2**30 / total)
    tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        weights, local_files_only=True, use_safetensors=True,
        dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager",
    ).eval()
    prefix = "这是研究用的合成精神科病例。只按给定对话选择基准分类：A 抑郁，B 焦虑，C 抑郁与焦虑混合，D 其他。只输出一个大写字母。\n"
    rows = json.loads((ROOT / "pilot-private.json").read_text(encoding="utf-8"))
    traces = []
    result = {}
    for row in rows:
        first, question, response = row["turns"]
        contexts = {"ANSWER": f"患者：{first}", "ASK_REPLAY": f"患者：{first}\n医生：{question}\n患者：{response}"}
        scores = {}
        for action, context in contexts.items():
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prefix + context}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False,
                return_dict=True, return_tensors="pt",
            ).to("cuda")
            torch.cuda.synchronize()
            start = perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=4, do_sample=False)
            torch.cuda.synchronize()
            new = output[0, inputs.input_ids.shape[1]:]
            text = tokenizer.decode(new, skip_special_tokens=True).strip()
            prediction = text if text in LABELS.values() else None
            correct = prediction == LABELS[row["label"]]
            scores[action] = correct
            metrics = result.setdefault(action, {"correct": 0, "parse_failed": 0, "seconds": 0.0})
            metrics["correct"] += correct
            metrics["parse_failed"] += prediction is None
            metrics["seconds"] += perf_counter() - start
            traces.append({"sample": row["sample"], "action": action, "output": text, "correct": correct})
        transition = (scores["ANSWER"], scores["ASK_REPLAY"])
        key = "ties" if transition[0] == transition[1] else "improved" if transition[1] else "worsened"
        result[key] = result.get(key, 0) + 1
    result["cases"] = len(rows)
    result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() // 2**20
    (ROOT / "pilot-outputs-private.json").write_text(json.dumps(traces, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    elif sys.argv[1] == "infer":
        infer()
    else:
        raise ValueError("expected prepare or infer")
