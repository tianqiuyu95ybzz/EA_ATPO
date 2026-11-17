#!/usr/bin/env python3
"""Recompute paired expert cells and normalized cost curves without pooling cohorts."""
from __future__ import annotations
import csv, hashlib, json, random, statistics
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/"results/expert_complementarity_20261117"
AUDIT=HERE/"outputs"
V1=Path("/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1")
STRONG=Path("/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/ea_atpo_value_router_20261105/outputs/strong_expert_awq")
COSTS=(0.0,0.15,0.30,0.45,0.60,0.75,0.90,1.00)

def read_jsonl(p:Path): return [json.loads(s) for s in p.open() if s.strip()]
def sha_file(p:Path):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def ci(values,seed,repeats=10000):
 if not values:return (float('nan'),float('nan'))
 if len(values)==1:return (float(values[0]),float(values[0]))
 rng=random.Random(seed); n=len(values); means=[]
 for _ in range(repeats): means.append(sum(rng.choices(values,k=n))/n)
 means.sort(); return means[int(.025*repeats)],means[min(repeats-1,int(.975*repeats))]
def add_pair(pairs,cohort,action,expert,base_rows,expert_rows,key):
 for cid,b in base_rows.items():
  e=expert_rows.get(cid)
  if e is None: continue
  pairs.append({"cohort":cohort,"action":action,"expert":expert,"case_id_hash":hashlib.sha256(str(cid).encode()).hexdigest(),"base_correct":bool(b.get('correct')),"expert_correct":bool(e.get('correct')),"expert_valid":bool(e.get('valid',False)),"expert_tokens":int(e.get('prompt_tokens',0))+int(e.get('completion_tokens',0)),"expert_latency_s":float(e.get('latency_s',0.0))})

def compare(pairs,cohort,action,expert,base_rows,expert_rows):
 common=sorted(set(base_rows)&set(expert_rows))
 sub=[]
 for cid in common:
  b,e=base_rows[cid],expert_rows[cid]
  sub.append({"cohort":cohort,"action":action,"expert":expert,"case_id_hash":hashlib.sha256(str(cid).encode()).hexdigest(),"base_correct":int(bool(b.get('correct'))),"expert_correct":int(bool(e.get('correct'))),"expert_valid":int(bool(e.get('valid',False))),"expert_tokens":int(e.get('prompt_tokens',0))+int(e.get('completion_tokens',0)),"expert_latency_s":float(e.get('latency_s',0.0))})
 n=len(sub)
 if not n:return [],[]
 b=[x['base_correct'] for x in sub]; e=[x['expert_correct'] for x in sub]; d=[y-x for x,y in zip(b,e)]
 b_err=[float(x['expert_correct']) for x in sub if not x['base_correct']]
 b_ok=[float(not x['expert_correct']) for x in sub if x['base_correct']]
 cells={(1,1):0,(1,0):0,(0,1):0,(0,0):0}
 for x in sub:cells[(x['base_correct'],x['expert_correct'])]+=1
 dci=ci(d,seed_hash(cohort+expert+action)); cci=ci(b_err,seed_hash(cohort+expert+'corr')); gci=ci(b_ok,seed_hash(cohort+expert+'degr'))
 summary={"cohort":cohort,"action":action,"expert":expert,"n_cases":n,"base_accuracy":statistics.mean(b),"expert_accuracy":statistics.mean(e),"paired_delta":statistics.mean(d),"delta_ci_low":dci[0],"delta_ci_high":dci[1],"base_correct_expert_correct":cells[(1,1)],"base_correct_expert_wrong":cells[(1,0)],"base_wrong_expert_correct":cells[(0,1)],"base_wrong_expert_wrong":cells[(0,0)],"correction_rate_given_base_error":statistics.mean(b_err) if b_err else '',"correction_ci_low":cci[0] if b_err else '',"correction_ci_high":cci[1] if b_err else '',"degradation_rate_given_base_correct":statistics.mean(b_ok) if b_ok else '',"degradation_ci_low":gci[0] if b_ok else '',"degradation_ci_high":gci[1] if b_ok else '',"expert_valid_rate":statistics.mean([x['expert_valid'] for x in sub]),"mean_expert_tokens":statistics.mean([x['expert_tokens'] for x in sub]),"mean_expert_latency_s":statistics.mean([x['expert_latency_s'] for x in sub])}
 return [summary],sub

def seed_hash(s): return int(hashlib.sha256(s.encode()).hexdigest()[:8],16)

def histories():
 pairs=[]; summaries=[]
 direct200=read_jsonl(AUDIT/"medical_expert_candidates_20261116/candidate_trajectories.jsonl")
 groups=defaultdict(dict)
 for x in direct200: groups[x['model_role']][x['case_id']]=x
 b=groups['base_qwen3_8b']
 for role in ('huatuo_gpt_o1_7b','qwen3_4b_remedi','popit_qwen3_4b_medical_sft'):
  s,rows=compare(pairs,'direct_candidates_200_seed20261104','direct_candidate_call',role,b,groups[role]); summaries+=s; pairs+=rows
 # Use the exact frozen Base trace paired with the official 32B direct run; it differs by one answer from another 8B rerun.
 man=json.loads((STRONG/'run_manifest.json').read_text())
 t32={x['case_id']:x for x in read_jsonl(STRONG/'qwen3_32b_direct_answers.jsonl')}
 b32={x['case_id']:x for x in read_jsonl(Path(man['base_baseline_path'])) if x.get('model_role')=='base_8b'}
 s,rows=compare(pairs,'qwen32_direct_frozen_200_seed20261104','direct_candidate_call','qwen3_32b_awq_direct',b32,t32); summaries+=s; pairs+=rows
 # 20-case candidate DEFER development cohort.
 dev=defaultdict(dict)
 for x in read_jsonl(AUDIT/"medical_candidate_defer_dev_20261116/defer_dev_trajectories.jsonl"): dev[x['model_role']][x['case_hash']]=x
 b=dev['always_answer_qwen3_8b']
 for role in ('base_qwen3_8b_defer_prompt','qwen3_4b_remedi','popit_qwen3_4b_medical_sft','huatuo_gpt_o1_7b'):
  s,rows=compare(pairs,'defer_dev20_20261116','DEFER',role,b,dev[role]); summaries+=s; pairs+=rows
 # Keep each fresh 100-case collection separate.
 cohorts=[('defer_confirmation100_seed20261117','l2d_remedi_confirmation_20261117_gpu6_full',['base_qwen3_8b_defer','qwen3_4b_remedi_defer']),('defer_candidate_screen100_seed20261118','l2d_prior_candidates_defer_20261118_gpu5',['base_qwen3_8b_defer','qwen3_4b_remedi_defer','popit_qwen3_4b_medical_sft_defer','huatuo_gpt_o1_7b_defer'])]
 for cohort,dirname,roles in cohorts:
  rows=read_jsonl(AUDIT/f'{dirname}/trajectories.jsonl'); by=defaultdict(dict)
  for x in rows: by[x['model_role']][x['case_hash']]=x
  b=by['always_answer_qwen3_8b']
  for role in roles:
   if role not in by: continue
   s,pp=compare(pairs,cohort,'DEFER',role,b,by[role]); summaries+=s; pairs+=pp
 return summaries,pairs

def cost_rows(summaries,pairs):
 results=[]
 by=defaultdict(list)
 for p in pairs:by[(p['cohort'],p['action'],p['expert'])].append(p)
 for (cohort,action,expert),rows in by.items():
  n=len(rows); base=[r['base_correct'] for r in rows]; cand=[r['expert_correct'] for r in rows]
  for lam in COSTS:
   choices={
    'always_answer':[(b,0) for b in base],
    'always_expert':[(e,1) for e in cand],
    'oracle_actual_expert_selective':[(1 if (not b and e and lam<1) else (b if not (not b and e and lam<1) else 1),1 if (not b and e and lam<1) else 0) for b,e in zip(base,cand)],
    'oracle_perfect_selective_defer':[(1 if (not b and lam<1) else b,1 if (not b and lam<1) else 0) for b in base],
   }
   for strategy,arr in choices.items():
    acc=[a for a,_ in arr]; calls=[c for _,c in arr]; utilities=[a-lam*c for a,c in arr]
    aci=ci(acc,seed_hash(cohort+expert+strategy+str(lam)+'acc')); uci=ci(utilities,seed_hash(cohort+expert+strategy+str(lam)+'util'))
    results.append({'cohort':cohort,'action':action,'expert':expert,'lambda_cost':lam,'strategy':strategy,'n_cases':n,'accuracy':statistics.mean(acc),'accuracy_ci_low':aci[0],'accuracy_ci_high':aci[1],'expert_call_rate':statistics.mean(calls),'mean_cost':lam*statistics.mean(calls),'net_utility':statistics.mean(utilities),'utility_ci_low':uci[0],'utility_ci_high':uci[1]})
   correction_budget=(sum(1 for b,e in zip(base,cand) if not b and e)/n) if lam<1 else 0.0
   random_acc=[b+correction_budget*(e-b) for b,e in zip(base,cand)]
   random_util=[a-lam*correction_budget for a in random_acc]
   aci=ci(random_acc,seed_hash(cohort+expert+'random'+str(lam)+'acc')); uci=ci(random_util,seed_hash(cohort+expert+'random'+str(lam)+'util'))
   results.append({'cohort':cohort,'action':action,'expert':expert,'lambda_cost':lam,'strategy':'random_same_budget_as_oracle_actual','n_cases':n,'accuracy':statistics.mean(random_acc),'accuracy_ci_low':aci[0],'accuracy_ci_high':aci[1],'expert_call_rate':correction_budget,'mean_cost':lam*correction_budget,'net_utility':statistics.mean(random_util),'utility_ci_low':uci[0],'utility_ci_high':uci[1]})
 return results

def consult20_rows():
 path=AUDIT/"l2d_consult_hint_pilot20_20261118_cpu/trajectories.jsonl"
 records=read_jsonl(path); by=defaultdict(dict)
 for r in records: by[r["case_hash"]][(r["model_role"],r["record_type"])]=r
 out=[]
 for case_hash,items in by.items():
  base=items.get(("always_answer","baseline"))
  if not base: continue
  q=items.get(("consult_question","question"),{})
  for role in ("consult_base_hint","consult_remedi_hint","consult_popit_hint"):
   hint=items.get((role,"hint"),{}); cont=items.get((role,"continuation"),{})
   if not hint and not cont: continue
   out.append({"case_hash":case_hash,"method":role,"base_correct":bool(base.get("correct")),"consult_final_correct":bool(cont.get("correct",False)),"question_valid":bool(q.get("valid",False)),"advice_format_valid":bool(hint.get("valid",False)),"advice_present":bool(hint.get("hint")),"base_tokens":int(base.get("prompt_tokens",0))+int(base.get("completion_tokens",0)),"consult_tokens":sum(int(x.get("prompt_tokens",0))+int(x.get("completion_tokens",0)) for x in (hint,cont)),"consult_latency_s":sum(float(x.get("latency_s",0.0)) for x in (hint,cont))})
 return out

def consult_cost_rows():
 rows=[]
 sources=[
  ("consult_advice_utilization_200_states","PILOT_ANSWER",AUDIT/"advice_utilization/advice_utilization_summary.csv",AUDIT/"advice_utilization/advice_utilization_cost_summary.csv"),
  ("consult_hint_exploration20","always_answer",AUDIT/"l2d_consult_hint_pilot20_20261118_cpu/summary.csv",AUDIT/"l2d_consult_hint_pilot20_20261118_cpu/summary.csv"),
 ]
 for cohort,baseline,summary_path,cost_path in sources:
  summary=list(csv.DictReader(summary_path.open()))
  if cohort=="consult_advice_utilization_200_states":
   by={(r["metric"],r["group"]):r for r in summary}
   methods=[("CONSULT_NAIVE","CONSULT_NAIVE"),("CONSULT_RETAIN","CONSULT_RETAIN"),("CONSULT_CRITICAL","CONSULT_CRITICAL")]
   cost={}
   for r in csv.DictReader(cost_path.open()):
    if r["metric"]=="full_chain_total_tokens": cost[r["method"]]=float(r["mean"])
   for method,group in methods:
    r=by[("accuracy",group)]; acc=float(r["value"]); low=float(r["ci95_low"]); high=float(r["ci95_high"])
    for lam in (0.0,0.05,0.10,0.20,0.40,0.60): rows.append({"cohort":cohort,"method":method,"lambda_consult":lam,"n_cases":r["n_states"],"accuracy":acc,"accuracy_ci_low":low,"accuracy_ci_high":high,"expert_call_rate":1.0,"mean_cost":lam,"net_utility":acc-lam,"utility_ci_low":low-lam,"utility_ci_high":high-lam,"mean_full_chain_tokens":cost.get(method,'')})
  else:
   by={(r["metric"],r["method"]):r for r in summary}
   roles=("consult_base_hint","consult_remedi_hint","consult_popit_hint")
   for method in roles:
    accrow=by[("accuracy",method)]; acc=float(accrow["value"]); low=float(accrow["ci95_low"]); high=float(accrow["ci95_high"])
    tokens=float(by[("mean_branch_tokens",method)]["value"])
    for lam in (0.0,0.05,0.10,0.20,0.40,0.60): rows.append({"cohort":cohort,"method":method,"lambda_consult":lam,"n_cases":int(accrow["n"]),"accuracy":acc,"accuracy_ci_low":low,"accuracy_ci_high":high,"expert_call_rate":1.0,"mean_cost":lam,"net_utility":acc-lam,"utility_ci_low":low-lam,"utility_ci_high":high-lam,"mean_full_chain_tokens":tokens})
 return rows

def write(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); w.writeheader(); w.writerows(rows)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 summary,pairs=histories(); costs=cost_rows(summary,pairs)
 write(OUT/'paired_expert_summary.csv',summary); write(OUT/'expert_paired_case_results.csv',pairs); write(OUT/'expert_cost_sensitivity.csv',costs)
 consult=consult20_rows(); consult_cost=consult_cost_rows(); write(OUT/'consult20_case_results.csv',consult); write(OUT/'consult_cost_sensitivity.csv',consult_cost)
 input_paths=[AUDIT/'medical_expert_candidates_20261116/candidate_trajectories.jsonl',AUDIT/'medical_candidate_defer_dev_20261116/defer_dev_trajectories.jsonl',AUDIT/'l2d_remedi_confirmation_20261117_gpu6_full/trajectories.jsonl',AUDIT/'l2d_prior_candidates_defer_20261118_gpu5/trajectories.jsonl',STRONG/'qwen3_32b_direct_answers.jsonl',STRONG/'run_manifest.json',AUDIT/'l2d_consult_hint_pilot20_20261118_cpu/trajectories.jsonl',AUDIT/'advice_utilization/advice_utilization_summary.csv',AUDIT/'advice_utilization/advice_utilization_cost_summary.csv',AUDIT/'advice_utilization/advice_utilization_paired_differences.csv',AUDIT/'consult_quality_summary_v2.csv']
 manifest={'analysis':'case-paired expert complementarity and normalized cost reanalysis','cost_grid':list(COSTS),'bootstrap_replicates':10000,'bootstrap_unit':'case, within each saved cohort','cohort_policy':'Every source experiment remains a distinct cohort; no pooled accuracy across unmatched samples.','input_sha256':{str(p):sha_file(p) for p in input_paths},'outputs':{p.name:sha_file(p) for p in (OUT/'paired_expert_summary.csv',OUT/'expert_paired_case_results.csv',OUT/'expert_cost_sensitivity.csv',OUT/'consult20_case_results.csv',OUT/'consult_cost_sensitivity.csv')},'med_qwen_and_new_32b_defer_not_included':'med_qwen had no retrievable checkpoint; new Qwen3-32B-AWQ DEFER run was blocked before service startup. See the separate resource attempt manifest.'}
 (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({'summary_rows':len(summary),'case_rows':len(pairs),'cost_rows':len(costs),'consult_case_rows':len(consult),'consult_cost_rows':len(consult_cost),'output':str(OUT)},indent=2))
if __name__=='__main__': main()
