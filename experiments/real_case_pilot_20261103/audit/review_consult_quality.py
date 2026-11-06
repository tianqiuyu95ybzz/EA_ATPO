#!/usr/bin/env python3
"""Sample Pilot CONSULT branches and collect explicitly model-judged post-hoc labels."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd


AUDIT_DIR = Path(__file__).resolve().parent
PILOT_DIR = AUDIT_DIR.parent
TRAJECTORY_PATH = PILOT_DIR / "outputs/full/pilot_trajectories.jsonl"
TRAIN_PATH = PILOT_DIR / "assets/train.pilot.parquet"
JUDGE_URL = "http://127.0.0.1:30131/v1/chat/completions"
JUDGE_ALIAS = "qwen3-8b-audit-judge"
SAMPLE_SEED = 20261104
SAMPLE_SIZE = 150

JUDGE_SYSTEM = """You are conducting a post-hoc research annotation of an LLM-to-LLM medical consultation trace. You are not a physician and your labels are not clinical adjudication. Use only visible case/dialogue, option text, and benchmark gold answer/rationale; never infer hidden patient facts. Distinguish the labels carefully: supports_gold means the advice makes a medically relevant, answer-key-consistent point; conflicts_gold means it explicitly asserts a medical claim incompatible with the keyed answer or directs toward an identifiable wrong option. Generic, vague, off-focus, incomplete, or unverifiable advice is neutral_or_unverifiable, NOT conflicts_gold merely because it fails to name the keyed answer. If the stem/options are inconsistent, an advice that flags that inconsistency is neutral_or_unverifiable, not conflicting. Return only valid JSON with an `annotations` array and one object per supplied sample, preserving `sample_id` exactly. Required fields: sample_id; advice_alignment (supports_gold, conflicts_gold, neutral_or_unverifiable, unassessable); specificity (specific_actionable, generic_or_repetitive, no_advice); repeats_public_info (true/false; only when it restates a fact already explicitly present); question_quality (specific, vague, malformed); rationale (at most 35 words). Mark unassessable if no advice was captured. A recommendation is not specific/actionable if it only echoes the consultation question or uses generic wording such as 'consider symptoms/risk factors' without a case-relevant discriminator."""


def plain(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def extract_question(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(x.get("content", "")) for x in messages if x.get("role") == "user")


def find_question(row: dict[str, Any]) -> str:
    for event in row.get("trace", []):
        if event.get("actor") == "base" and event.get("action") == "CONSULT":
            text = str(event.get("consult_question", ""))
            if text:
                return text
    return ""


def find_advice(row: dict[str, Any]) -> str:
    for event in row.get("trace", []):
        if event.get("actor") == "expert" and event.get("action") == "CONSULT":
            return str(event.get("response", ""))
    return ""


def file_sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def prepare_sample(output_dir: Path, size: int, seed: int, suffix: str = "") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trajectories=[x for x in load_records(TRAJECTORY_PATH) if x.get("record_type")=="trajectory"]
    consults=[x for x in trajectories if x.get("action")=="CONSULT"]
    grouped: dict[tuple[str,str],list[dict[str,Any]]]=defaultdict(list)
    for x in consults: grouped[(str(x["case_id"]),str(x["state_id"]))].append(x)
    rng=random.Random(seed)
    state_keys=sorted(grouped)
    if len(state_keys)<size: raise ValueError(f"only {len(state_keys)} distinct CONSULT states; need {size}")
    chosen_keys=rng.sample(state_keys,size)
    selected=[rng.choice(grouped[k]) for k in chosen_keys]
    selected.sort(key=lambda x:(str(x["case_id"]),str(x["state_id"]),int(x.get("rollout_index",0))))
    frame=pd.read_parquet(TRAIN_PATH)
    source_rows=sorted({int(x["source_row_index"]) for x in selected})
    case_metadata={}
    for idx in source_rows:
        row=frame.iloc[idx].to_dict()
        extra=row.get("extra_info",{})
        extra=extra if isinstance(extra,dict) else {}
        case_metadata[idx]={"ground_truth":plain(extra.get("ground_truth",{}))}
    by_pair={(str(x["case_id"]),str(x["state_id"]),int(x["rollout_index"]),str(x["action"])):x for x in trajectories}
    sample=[]
    for serial,x in enumerate(selected):
        key=(str(x["case_id"]),str(x["state_id"]),int(x["rollout_index"]))
        answer=by_pair.get((key[0],key[1],key[2],"ANSWER"))
        gold_metadata=case_metadata[int(x["source_row_index"])].get("ground_truth") or {}
        sample.append({
            "sample_id":f"consult-{serial:03d}","case_id":x["case_id"],"state_id":x["state_id"],"rollout_index":x["rollout_index"],
            "source_row_index":x["source_row_index"],"public_case_and_dialogue":extract_question(x.get("public_state",[])),
            "consult_question":find_question(x),"expert_advice":find_advice(x),"gold_option":str(x.get("gold","")),
            "gold_rationale":str(gold_metadata.get("answer_info", "")) if isinstance(gold_metadata,dict) else "",
            "consult_final_answer":x.get("final_answer"),"consult_correct":bool(x.get("valid") and float(x.get("outcome") or 0)==1),
            "consult_valid":bool(x.get("valid")),"consult_failure_reason":x.get("failure_reason"),
            "paired_answer":(answer or {}).get("final_answer"),"paired_answer_correct":bool((answer or {}).get("valid") and float((answer or {}).get("outcome") or 0)==1),
            "prompt_sha256":hashlib.sha256(extract_question(x.get("public_state",[])).encode()).hexdigest(),
        })
    manifest={"created_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"source":"Existing full Pilot only; no inference changes the Pilot result.","source_trajectory_sha256":file_sha(TRAJECTORY_PATH),"total_initial_consult_trajectories":len(consults),"eligible_distinct_case_state_units":len(state_keys),"sample_size":len(sample),"sample_seed":seed,"sampling":"One randomly selected repeat per randomly selected distinct case/state; case clusters retained for intervals.","judge_model":"Qwen/Qwen3-8B revision b968826d9c46dd6066d109eabc6255188de91218 (same model family as Pilot; exploratory LLM judge, not independent clinical adjudicator).","gold_and_rationale_usage":"Added only after rollout for post-hoc annotation; neither was given to the Pilot Expert/Base calls.","categories_are_nonexclusive":True}
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/f"consult_quality_sample_manifest{suffix}.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    with (output_dir/"consult_quality_sample.jsonl").open("w",encoding="utf-8") as f:
        for x in sample: f.write(json.dumps(x,ensure_ascii=False)+"\n")
    return sample,manifest


def parse_json_response(text: str) -> list[dict[str, Any]]:
    clean=text.strip()
    clean=re.sub(r"^```(?:json)?\s*|\s*```$","",clean,flags=re.I)
    start=clean.find("{"); end=clean.rfind("}")
    if start<0 or end<start: raise ValueError("judge did not return a JSON object")
    obj=json.loads(clean[start:end+1])
    values=obj.get("annotations") if isinstance(obj,dict) else obj
    if not isinstance(values,list): raise ValueError("judge JSON lacks annotations array")
    return values


async def main_async(args: argparse.Namespace) -> None:
    out=args.output_dir
    sample,manifest=prepare_sample(out,args.sample_size,args.seed,args.suffix)
    judgment_path=out/f"consult_quality_judgments{args.suffix}.jsonl"
    summary_path=out/f"consult_quality_summary{args.suffix}.csv"
    done={}
    if judgment_path.exists() and args.resume:
        for row in load_records(judgment_path): done[row["sample_id"]]=row
    elif judgment_path.exists():
        raise FileExistsError("consult_quality_judgments.jsonl exists; use --resume or a new output directory")
    timeout=aiohttp.ClientTimeout(total=240)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get("http://127.0.0.1:30131/v1/models") as response:
            response.raise_for_status()
            model_ids=[str(x.get("id")) for x in (await response.json()).get("data",[])]
        if JUDGE_ALIAS not in model_ids:
            raise RuntimeError(f"quality-judge model alias not served: {model_ids}")
        remaining=[x for x in sample if x["sample_id"] not in done]
        for start in range(0,len(remaining),3):
            batch=remaining[start:start+3]
            user=json.dumps({"items":[{k:x[k] for k in ("sample_id","public_case_and_dialogue","consult_question","expert_advice","gold_option","gold_rationale","consult_final_answer","consult_correct","paired_answer","paired_answer_correct","consult_valid","consult_failure_reason")} for x in batch]},ensure_ascii=False)
            payload={"model":JUDGE_ALIAS,"messages":[{"role":"system","content":JUDGE_SYSTEM},{"role":"user","content":user}],"temperature":0.0,"top_p":1.0,"max_tokens":900,"chat_template_kwargs":{"enable_thinking":False}}
            started=time.perf_counter()
            try:
                async with session.post(JUDGE_URL,json=payload) as response:
                    response.raise_for_status(); body=await response.json()
                raw=str(body["choices"][0]["message"].get("content",""))
                annotations=parse_json_response(raw)
                by_id={str(x.get("sample_id")):x for x in annotations}
                for item in batch:
                    ann=by_id.get(item["sample_id"],{})
                    record={"record_type":"consult_quality_annotation","sample_id":item["sample_id"],"case_id":item["case_id"],"state_id":item["state_id"],"rollout_index":item["rollout_index"],"judge_model":JUDGE_ALIAS,"annotation":ann,"batch_latency_s":time.perf_counter()-started,"error":None if ann else "missing_annotation"}
                    done[record["sample_id"]]=record
                    with judgment_path.open("a",encoding="utf-8") as f: f.write(json.dumps(record,ensure_ascii=False)+"\n"); f.flush()
            except Exception as exc:
                for item in batch:
                    record={"record_type":"consult_quality_annotation","sample_id":item["sample_id"],"case_id":item["case_id"],"state_id":item["state_id"],"rollout_index":item["rollout_index"],"judge_model":JUDGE_ALIAS,"annotation":{},"batch_latency_s":time.perf_counter()-started,"error":f"{type(exc).__name__}: {exc}"}
                    done[record["sample_id"]]=record
                    with judgment_path.open("a",encoding="utf-8") as f: f.write(json.dumps(record,ensure_ascii=False)+"\n"); f.flush()
            if len(done)%15==0: print(f"judged {len(done)}/{len(sample)} sampled consultations",flush=True)
    merged=[]
    for item in sample:
        merged.append({**item,**done.get(item["sample_id"],{})})
    # Write a compact, explicit label table. Labels overlap and are proportions
    # over the complete sample unless the row says the advice was unassessable.
    fields=("category","count","denominator","proportion","ci95_low","ci95_high","notes")
    categories={
        "correct_and_helpful_final_correct":lambda x:x.get("annotation",{}).get("advice_alignment")=="supports_gold" and x.get("annotation",{}).get("specificity")=="specific_actionable" and x.get("consult_correct"),
        "gold_aligned_but_final_wrong":lambda x:x.get("annotation",{}).get("advice_alignment")=="supports_gold" and not x.get("consult_correct"),
        "advice_conflicts_with_gold":lambda x:x.get("annotation",{}).get("advice_alignment")=="conflicts_gold",
        "repeats_public_information":lambda x:x.get("annotation",{}).get("repeats_public_info") is True,
        "unclear_consult_question":lambda x:x.get("annotation",{}).get("question_quality") in ("vague","malformed"),
    }
    assessable=[x for x in merged if x.get("annotation",{}).get("advice_alignment")!="unassessable" and x.get("annotation",{}).get("specificity")!="no_advice" and not x.get("error")]
    # Case-clustered percentile bootstrap for each sample indicator, with all
    # sampled states from a case resampled together.
    cases=sorted({str(x["case_id"]) for x in merged})
    by_case={c:[x for x in merged if str(x["case_id"])==c] for c in cases}
    result_rows=[]
    brng=random.Random(args.seed+1)
    for name,predicate in categories.items():
        point=sum(bool(predicate(x)) for x in merged)/len(merged)
        values=[]
        for _ in range(4000):
            selected_cases=brng.choices(cases,k=len(cases))
            sample_rows=[x for c in selected_cases for x in by_case[c]]
            values.append(sum(bool(predicate(x)) for x in sample_rows)/len(sample_rows) if sample_rows else 0)
        values.sort(); low=values[int(.025*len(values))]; high=values[min(len(values)-1,int(.975*len(values)))]
        result_rows.append({"category":name,"count":sum(bool(predicate(x)) for x in merged),"denominator":len(merged),"proportion":point,"ci95_low":low,"ci95_high":high,"notes":"Overlapping exploratory labels from same-family Qwen3-8B LLM judge; not physician-adjudicated."})
    result_rows.append({"category":"judge_assessable_advice","count":len(assessable),"denominator":len(merged),"proportion":len(assessable)/len(merged),"ci95_low":"","ci95_high":"","notes":"Advice present and judge returned a non-unassessable label."})
    with summary_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(result_rows)
    manifest.update({"annotation_suffix":args.suffix or "v1_initial_rubric","judge_completed":datetime.now(timezone.utc).isoformat(timespec="seconds"),"judged_count":len(done),"judge_error_count":sum(bool(x.get("error")) for x in done.values()),"judgment_file":str(judgment_path),"summary_csv":str(summary_path),"judgment_file_sha256":file_sha(judgment_path),"sample_file_sha256":file_sha(out/"consult_quality_sample.jsonl"),"summary_csv_sha256":file_sha(summary_path),"label_method":"Qwen3-8B post-hoc rubric labels, conditional on benchmark gold; outcome-based degradation is separately recomputed for all 873 paired CONSULT/ANSWER rollouts."})
    (out/f"consult_quality_sample_manifest{args.suffix}.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"sampled":len(sample),"judge_errors":manifest["judge_error_count"],"assessable":len(assessable),"categories":{x["category"]:[x["count"],x["denominator"],x["proportion"]] for x in result_rows}},ensure_ascii=False,indent=2))


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=AUDIT_DIR/"outputs")
    parser.add_argument("--sample-size",type=int,default=SAMPLE_SIZE)
    parser.add_argument("--seed",type=int,default=SAMPLE_SEED)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--suffix",default="",help="Optional filename suffix, for example _v2, to preserve prior rubric outputs.")
    asyncio.run(main_async(parser.parse_args()))


if __name__=="__main__":main()
