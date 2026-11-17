#!/usr/bin/env python3
"""Paired Qwen3-32B direct/DEFER test and predecision-only router dataset capture."""
from __future__ import annotations

import argparse, asyncio, csv, hashlib, importlib.util, json, random, re, subprocess, sys, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT/"ea_atpo_stage"))
import run_direct_reliability_eval as direct  # noqa: E402
import run_medical_candidate_eval as candidate  # noqa: E402
from run_l2d_remedi_confirmation import case_key, DATA_PATH as TEST_DATA  # noqa: E402

_spec=importlib.util.spec_from_file_location("defer_helper", HERE/"run_medical_candidate_defer_dev_eval.py")
if _spec is None or _spec.loader is None: raise ImportError("DEFER protocol helper missing")
defer_helper=importlib.util.module_from_spec(_spec); sys.modules[_spec.name]=defer_helper; _spec.loader.exec_module(defer_helper)

OUT=HERE/"outputs/qwen32_awq_complementarity_20261117_gpu7"
TRAIN_DATA=candidate.DATA_PATH.parent/"train.pilot.parquet"
BASE_URL="http://127.0.0.1:30147/v1"
EXPERT_URL="http://127.0.0.1:30146/v1"
BASE_ALIAS="qwen3-8b-complementarity-base"
EXPERT_ALIAS="qwen3-32b-awq-complementarity"
SPLIT_SEED=20261117
TRAIN_N=200
VAL_N=100
TEST_N=200
BASE_ROLE="base_qwen3_8b"
EXPERT_ROLE="qwen3_32b_awq_defer"
DIRECT_ROLE="candidate_expert_32b"
AWQ_REV="0499c3ac83fdef8810b907a23894ba91e95eddd8"

def sha256_file(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def sha256_text(s:str)->str: return hashlib.sha256(s.encode()).hexdigest()
def plain(x:Any)->Any: return x.tolist() if hasattr(x,'tolist') else x

def public_features(messages:list[dict[str,str]])->dict[str,float]:
 text="\n".join(str(m.get("content","")) for m in messages if m.get("role")!="system")
 words=re.findall(r"\b\w+\b",text,re.UNICODE)
 option_lines=re.findall(r"(?m)^\s*[A-D]\s*[).:]\s*(.*?)\s*$",text)
 lower=text.lower()
 clinical=("pain","fever","cough","blood","dose","drug","symptom","diagnosis","treatment","history","onset","patient","exam","test","surgery","infection","pressure","heart","lung","kidney","pregnan")
 return {
  "log_public_chars":float(__import__('math').log1p(len(text))),
  "log_public_words":float(__import__('math').log1p(len(words))),
  "option_count":float(len(option_lines)),
  "mean_option_chars":float(sum(map(len,option_lines))/len(option_lines)) if option_lines else 0.0,
  "question_marks":float(text.count("?")+text.count("？")),
  "digit_count":float(sum(c.isdigit() for c in text)),
  "clinical_lexicon_count":float(sum(lower.count(w) for w in clinical)),
 }

def make_case(row:dict[str,Any], source_index:int, split:str, case_id:str|None=None)->dict[str,Any]:
 messages=direct.build_messages(row)
 prompt_hash=sha256_text(json.dumps(messages,ensure_ascii=False,sort_keys=True))
 state,_,gold,facts=defer_helper.public_state(row,source_index)
 defer_messages=defer_helper.build_expert_messages(state,defer_helper.Action.DEFER)
 hidden=defer_helper.unrevealed_atomic_facts(state,facts)
 defer_helper.validate_expert_view(messages,atomic_facts=hidden,ground_truth=gold)
 defer_helper.validate_expert_view(defer_messages,atomic_facts=hidden,ground_truth=gold)
 return {"case_id":case_id or f"{split}-row-{source_index}","source_row_index":int(source_index),"split":split,"case_hash":prompt_hash,"prompt_hash":prompt_hash,"gold":gold,"base_messages":messages,"defer_messages":defer_messages,"features":public_features(messages)}

def load_frozen_test()->tuple[list[dict[str,Any]],dict[str,dict[str,Any]],dict[str,dict[str,Any]],dict[str,Any]]:
 cases, data_sha=candidate.selected_cases()
 frame=pd.read_parquet(candidate.DATA_PATH)
 manifest_path=Path("/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/ea_atpo_value_router_20261105/outputs/strong_expert_awq/run_manifest.json")
 manifest=json.loads(manifest_path.read_text())
 if manifest.get("data_sha256")!=data_sha or manifest.get("case_count")!=TEST_N: raise RuntimeError("Frozen 200-case Qwen3-32B direct sample manifest mismatch")
 expected=set(map(int,manifest["sampled_source_rows"]))
 if expected!={c["source_row_index"] for c in cases}: raise RuntimeError("Frozen sample source indices mismatch")
 direct_path=Path("/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/ea_atpo_value_router_20261105/outputs/strong_expert_awq/qwen3_32b_direct_answers.jsonl")
 direct_rows={r["case_id"]:r for r in (json.loads(s) for s in direct_path.open())}
 base_path=Path(manifest["base_baseline_path"])
 base_rows={r["case_id"]:r for r in (json.loads(s) for s in base_path.open()) if r.get("model_role")=="base_8b"}
 if set(direct_rows)!=set(c["case_id"] for c in cases) or set(base_rows)!=set(direct_rows): raise RuntimeError("Frozen baseline case IDs are not fully paired")
 for c in cases:
  d=direct_rows[c["case_id"]]; b=base_rows[c["case_id"]]
  if d.get("prompt_sha256")!=c["prompt_sha256"] or b.get("prompt_sha256")!=c["prompt_sha256"] or d.get("gold")!=c["gold"] or b.get("gold")!=c["gold"]:
   raise RuntimeError(f"Prompt/gold audit mismatch for {c['case_id']}")
  row=frame.iloc[int(c["source_row_index"])].to_dict()
  state,_,gold,facts=defer_helper.public_state(row,int(c["source_row_index"]))
  c["defer_messages"]=defer_helper.build_expert_messages(state,defer_helper.Action.DEFER)
  defer_helper.validate_expert_view(c["defer_messages"],atomic_facts=defer_helper.unrevealed_atomic_facts(state,facts),ground_truth=gold)
  c["gold"]=gold
  c["split"]="test"; c["case_hash"]=c["prompt_sha256"]; c["prompt_hash"]=c["prompt_sha256"]
  c["features"]=public_features(c["messages"])
  c["base_frozen"]={"predicted":b.get("predicted"),"correct":bool(b.get("correct")),"valid":bool(b.get("valid")),"prompt_tokens":int(b.get("prompt_tokens",0)),"completion_tokens":int(b.get("completion_tokens",0)),"latency_s":float(b.get("latency_s",0.0))}
  c["direct_frozen"]={"predicted":d.get("predicted"),"correct":bool(d.get("correct")),"valid":bool(d.get("valid")),"prompt_tokens":int(d.get("prompt_tokens",0)),"completion_tokens":int(d.get("completion_tokens",0)),"latency_s":float(d.get("latency_s",0.0))}
 return cases,base_rows,direct_rows,{"test_data_sha256":data_sha,"test_seed":20261104,"test_source_indices_sha256":sha256_text("\n".join(str(i) for i in manifest["sampled_source_rows"])),"qwen32_run_manifest_sha256":sha256_file(manifest_path),"qwen32_direct_trajectory_sha256":sha256_file(direct_path),"base_direct_trajectory_sha256":sha256_file(base_path)}

def load_train_val(test_cases:list[dict[str,Any]])->tuple[list[dict[str,Any]],str]:
 if not TRAIN_DATA.is_file(): raise FileNotFoundError(TRAIN_DATA)
 frame=pd.read_parquet(TRAIN_DATA)
 rng=random.Random(SPLIT_SEED); order=list(range(len(frame))); rng.shuffle(order)
 excluded={c["prompt_hash"] for c in test_cases}; seen=set(); selected=[]
 for ix in order:
  row=frame.iloc[ix].to_dict(); c=make_case(row,int(ix),"train_pool")
  if c["prompt_hash"] in excluded or c["prompt_hash"] in seen: continue
  seen.add(c["prompt_hash"]); selected.append(c)
  if len(selected)==TRAIN_N+VAL_N: break
 if len(selected)!=TRAIN_N+VAL_N: raise RuntimeError(f"Only {len(selected)} unique train/validation cases found")
 for i,c in enumerate(selected): c["split"]="train" if i<TRAIN_N else "validation"
 return selected,sha256_file(TRAIN_DATA)

async def health(session:aiohttp.ClientSession,url:str,alias:str)->None:
 async with session.get(url+"/models") as r:
  r.raise_for_status(); ids=[x.get("id") for x in (await r.json()).get("data",[])]
 if alias not in ids: raise RuntimeError(f"Expected service {alias}; available models: {ids}")

async def infer(session:aiohttp.ClientSession,url:str,alias:str,messages:list[dict[str,str]],action:str)->dict[str,Any]:
 body={"model":alias,"messages":messages,"temperature":0.0,"top_p":1.0,"max_tokens":64 if action=="ANSWER" else 128,"chat_template_kwargs":{"enable_thinking":False}}
 started=time.perf_counter()
 async with session.post(url+"/chat/completions",json=body) as r: r.raise_for_status(); response=await r.json()
 choice=response["choices"][0]; raw=str(choice["message"].get("content", "")); usage=response.get("usage",{})
 try:
  if action=="ANSWER":
   pred=direct.parse_answer(raw)
  else:
   normalized=defer_helper.validate_expert_response(defer_helper.Action.DEFER,raw)
   match=re.fullmatch(r"Expert Final Answer:\s*([A-D])",normalized,re.IGNORECASE)
   pred=match.group(1).upper() if match else None
  error=None if pred else "strict_answer_parse_failure"
 except (ValueError,TypeError) as e: pred=None; error=f"protocol_parse_failure:{type(e).__name__}"
 return {"predicted":pred,"valid":pred is not None,"parse_error":error,"prompt_tokens":int(usage.get("prompt_tokens",0)),"completion_tokens":int(usage.get("completion_tokens",0)),"latency_s":time.perf_counter()-started,"finish_reason":choice.get("finish_reason"),"response_sha256":sha256_text(raw)}

def load_jsonl(path:Path)->list[dict[str,Any]]:
 return [json.loads(s) for s in path.open() if s.strip()] if path.is_file() else []
def append(path:Path,row:dict[str,Any])->None:
 with path.open("a",encoding="utf-8") as f: f.write(json.dumps(row,ensure_ascii=False)+"\n"); f.flush()

async def evaluate(cases:list[dict[str,Any]],resume:bool)->list[dict[str,Any]]:
 OUT.mkdir(parents=True,exist_ok=True)
 base_path=OUT/"base_train_validation_trajectories.jsonl"; defer_path=OUT/"qwen32_defer_trajectories.jsonl"
 old_base=load_jsonl(base_path) if resume else []; old_defer=load_jsonl(defer_path) if resume else []
 if not resume and (base_path.exists() or defer_path.exists()): raise FileExistsError("Output exists; pass --resume")
 base_done={r["case_hash"] for r in old_base}; defer_done={r["case_hash"] for r in old_defer}
 timeout=aiohttp.ClientTimeout(total=300)
 async with aiohttp.ClientSession(timeout=timeout) as session:
  await health(session,BASE_URL,BASE_ALIAS); await health(session,EXPERT_URL,EXPERT_ALIAS)
  for c in cases:
   if c["split"]=="test" or c["case_hash"] in base_done: continue
   try: result=await infer(session,BASE_URL,BASE_ALIAS,c["messages"],"ANSWER")
   except Exception as e: result={"predicted":None,"valid":False,"parse_error":f"request:{type(e).__name__}","prompt_tokens":0,"completion_tokens":0,"latency_s":0.0,"finish_reason":"error","response_sha256":""}
   row={"case_id":c["case_id"],"case_hash":c["case_hash"],"split":c["split"],"model_role":BASE_ROLE,"action":"ANSWER","predicted":result["predicted"],"correct":bool(result["predicted"]==c["gold"]),"valid":result["valid"],"parse_error":result["parse_error"],"prompt_tokens":result["prompt_tokens"],"completion_tokens":result["completion_tokens"],"latency_s":result["latency_s"],"finish_reason":result["finish_reason"],"response_sha256":result["response_sha256"],"features":c["features"]}
   append(base_path,row); old_base.append(row); base_done.add(c["case_hash"])
   if len(base_done)%20==0: print(f"Base train/validation: {len(base_done)}/{TRAIN_N+VAL_N}",flush=True)
  for i,c in enumerate(cases,1):
   if c["case_hash"] in defer_done: continue
   try: result=await infer(session,EXPERT_URL,EXPERT_ALIAS,c["defer_messages"],"DEFER")
   except Exception as e: result={"predicted":None,"valid":False,"parse_error":f"request:{type(e).__name__}","prompt_tokens":0,"completion_tokens":0,"latency_s":0.0,"finish_reason":"error","response_sha256":""}
   row={"case_id":c["case_id"],"case_hash":c["case_hash"],"split":c["split"],"model_role":EXPERT_ROLE,"action":"DEFER","predicted":result["predicted"],"correct":bool(result["predicted"]==c["gold"]),"valid":result["valid"],"parse_error":result["parse_error"],"prompt_tokens":result["prompt_tokens"],"completion_tokens":result["completion_tokens"],"latency_s":result["latency_s"],"finish_reason":result["finish_reason"],"response_sha256":result["response_sha256"],"features":c["features"]}
   append(defer_path,row); old_defer.append(row); defer_done.add(c["case_hash"])
   if i%20==0 or i==len(cases): print(f"Qwen3-32B DEFER: {i}/{len(cases)}",flush=True)
 return old_base+old_defer

def main()->None:
 p=argparse.ArgumentParser(); p.add_argument("--resume",action="store_true"); args=p.parse_args()
 test,_,_,test_manifest=load_frozen_test(); trainval,train_sha=load_train_val(test); cases=trainval+test
 selection={"train_rows":TRAIN_N,"validation_rows":VAL_N,"test_rows":TEST_N,"train_seed":SPLIT_SEED,"test_seed":20261104,"train_data_sha256":train_sha,**test_manifest,"trainval_case_hashes_sha256":sha256_text("\n".join(c["case_hash"] for c in trainval)),"all_case_hashes_sha256":sha256_text("\n".join(c["case_hash"] for c in cases))}
 manifest_path=OUT/"run_manifest.json"; trajectory=OUT/"qwen32_defer_trajectories.jsonl"
 if manifest_path.exists() and args.resume:
  old=json.loads(manifest_path.read_text());
  if old.get("selection")!=selection: raise RuntimeError("Resume sample/split differs from manifest")
 elif (manifest_path.exists() or trajectory.exists()) and not args.resume: raise FileExistsError("Output exists; use --resume")
 else:
  manifest={"status":"running","created_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"purpose":"Qwen3-32B-AWQ direct vs DEFER paired candidate evaluation plus train/validation-only public-feature router data capture; no PPO/RL.","selection":selection,"model":{"repo_id":"Qwen/Qwen3-32B-AWQ","revision":AWQ_REV,"direct_result_source":"frozen same 200-case direct evaluation","defer_service_manifest":str(OUT/"model_service_manifest.json")},"base_model":{"repo_id":"Qwen/Qwen3-8B","revision":"b968826d9c46dd6066d109eabc6255188de91218","service_manifest":str(OUT/"base_service_manifest.json")},"generation":{"temperature":0,"top_p":1,"enable_thinking":False,"base_max_tokens":64,"defer_max_tokens":128},"information_policy":"Models receive only their public prompts. No Gold or hidden atomic facts are included. The evaluator scores after generation. Traces store only predictions, correctness, hashes and telemetry.","predecision_features":{"source":"public initial prompt only","fields":["log_public_chars","log_public_words","option_count","mean_option_chars","question_marks","digit_count","clinical_lexicon_count"]},"split_policy":"200 ATPO train rows for router fitting, 100 different ATPO train rows for threshold selection, frozen 200-case independent test parquet sample for one final evaluation.","code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"runner_sha256":sha256_file(Path(__file__).resolve())}
  OUT.mkdir(parents=True,exist_ok=True); manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
 records=asyncio.run(evaluate(cases,args.resume))
 # Join without persisting Gold, prompts, hidden facts, or raw completions.
 base={r["case_hash"]:r for r in records if r["model_role"]==BASE_ROLE}; expert={r["case_hash"]:r for r in records if r["model_role"]==EXPERT_ROLE}
 rows=[]
 for c in cases:
  b=c.get("base_frozen") or base.get(c["case_hash"]); e=expert.get(c["case_hash"])
  if b is None or e is None: continue
  out={"case_id":c["case_id"],"case_hash":c["case_hash"],"split":c["split"],"base_predicted":b.get("predicted"),"base_correct":bool(b.get("correct")),"base_valid":bool(b.get("valid")),"base_prompt_tokens":int(b.get("prompt_tokens",0)),"base_completion_tokens":int(b.get("completion_tokens",0)),"base_latency_s":float(b.get("latency_s",0.0)),"defer32_predicted":e.get("predicted"),"defer32_correct":bool(e.get("correct")),"defer32_valid":bool(e.get("valid")),"defer32_prompt_tokens":int(e.get("prompt_tokens",0)),"defer32_completion_tokens":int(e.get("completion_tokens",0)),"defer32_latency_s":float(e.get("latency_s",0.0)),"defer32_parse_error":e.get("parse_error"),"features":c["features"]}
  if c.get("direct_frozen"):
   d=c["direct_frozen"]; out.update({"direct32_predicted":d.get("predicted"),"direct32_correct":bool(d.get("correct")),"direct32_valid":bool(d.get("valid")),"direct32_prompt_tokens":int(d.get("prompt_tokens",0)),"direct32_completion_tokens":int(d.get("completion_tokens",0)),"direct32_latency_s":float(d.get("latency_s",0.0))})
  rows.append(out)
 with (OUT/"case_level_results.jsonl").open("w",encoding="utf-8") as f:
  for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
 m=json.loads(manifest_path.read_text()); m.update({"status":"completed" if len(rows)==TRAIN_N+VAL_N+TEST_N else "partial","completed_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"case_level_rows":len(rows),"base_train_validation_rows":len(base),"defer_rows":len(expert),"case_level_results_sha256":sha256_file(OUT/"case_level_results.jsonl"),"base_trajectory_sha256":sha256_file(OUT/"base_train_validation_trajectories.jsonl"),"defer_trajectory_sha256":sha256_file(OUT/"qwen32_defer_trajectories.jsonl")})
 manifest_path.write_text(json.dumps(m,indent=2,ensure_ascii=False)+"\n")
 print(json.dumps({"status":m["status"],"case_level_rows":len(rows),"output":str(OUT)},indent=2))

if __name__=="__main__": main()
