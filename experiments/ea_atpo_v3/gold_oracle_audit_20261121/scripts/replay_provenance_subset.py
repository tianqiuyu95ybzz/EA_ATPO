#!/usr/bin/env python3
"""Exploratory replay of prior Dev; filtering only by public-source textual provenance."""
import hashlib,importlib.util,json,pathlib
P=pathlib.Path(__file__).resolve().parents[1];ROOT=P.parents[2];mapfile=P/'private/private_129_gold_provenance_v3.jsonl'; state=pathlib.Path('/pfs/pfs-spMCeu/image_team/workspace-copy/EA-ATPO_Oracle_20261121/run_08/private_state_results.jsonl')
provenance=[json.loads(s) for s in mapfile.open() if s.strip()];cases={}
for x in provenance:
 cs=x['candidates']
 if len(cs)==1 and cs[0]['problem_text_exact_substring'] and cs[0]['mapped_option']==x['benchmark_answer_key']:cases[x['case_id']]=x['atpo_row_index']
rows=[json.loads(l) for l in state.open() if l.strip()];rows=[r for r in rows if r['case_id'] in cases]
if len(cases)!=11 or len({r['case_id'] for r in rows})!=11:raise ValueError('unexpected provenance subset')
file=ROOT/'experiments/ea_atpo_v3/oracle_expert/scripts/evaluate_oracle_router.py';spec=importlib.util.spec_from_file_location('oracle_replay',file);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
result=mod.evaluate(rows,costs=[.05,.10,.20,.60]);result['status']='exploratory_historical_dev_subset_NOT_INDEPENDENT';result['estimand']='case macro for 11 previously selected Dev cases; inclusion by unique four-option match plus problem fragment substring';result['gold_interpretation']='author-archive option-text agreement only, original question is truncated; no clinical gold validation';result['bootstrap']['interpretation']='exploratory, only 11 reused case clusters';result['sample']={'cases':len(cases),'states':len(rows),'action_results':len(rows)*3,'selection_depends_on_outcomes':False,'selected_after_historical_dev_results_seen':True,'provenance_mapping_sha256':hashlib.sha256(mapfile.read_bytes()).hexdigest(),'private_input_sha256':hashlib.sha256(state.read_bytes()).hexdigest()}
# Counts are state-weighted, averaging the three frozen seeds for learned methods.
for cost, methods in result['cost_sensitivity'].items():
 for method, metrics in methods.items():
  seeds=mod.SEEDS if method in ('learned_actual','learned_oracle') else [None]
  corrected=[]; degraded=[]
  for seed in seeds:
   rr=[mod.row_result(row,method,seed,float(cost)) for row in rows]
   corrected.append(sum(not q['base_correct'] and q['final_correct'] for q in rr))
   degraded.append(sum(q['base_correct'] and not q['final_correct'] for q in rr))
  metrics['state_net_gain_count_mean_across_seeds']=sum(a-b for a,b in zip(corrected,degraded))/len(seeds)
  metrics['state_corrected_count_mean_across_seeds']=sum(corrected)/len(seeds)
  metrics['state_degraded_count_mean_across_seeds']=sum(degraded)/len(seeds)
out=P/'data/historical_subset_replay_aggregate_v2.json'
with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'sample':result['sample'],'costs':{c:{m:{k:v[k] for k in ['accuracy','utility','base_correct_retention','base_error_correction','defer_precision_useful_corrections_per_call','defer_rate','net_gain_vs_always_answer_utility','paired_ci95_utility_vs_always_answer']} for m,v in methods.items()} for c,methods in result['cost_sensitivity'].items()}},indent=2))
