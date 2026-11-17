#!/usr/bin/env python3
"""Fit a small public-feature ridge router after paired train/validation/test capture."""
from __future__ import annotations
import csv, hashlib, json, random, statistics
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
RESULTS=HERE/"results/expert_complementarity_20261117"
CAPTURE=HERE/"outputs/qwen32_awq_complementarity_20261117_gpu7/case_level_results.jsonl"
FEATURES=("log_public_chars","log_public_words","option_count","mean_option_chars","question_marks","digit_count","clinical_lexicon_count")
COSTS=(0.0,0.15,0.30,0.45,0.60,0.75,0.90,1.00)
ALPHA=10.0
BOOTSTRAPS=10000

def sha_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ci(values,seed):
 if not values:return (float('nan'),float('nan'))
 if len(values)==1:return (values[0],values[0])
 rng=random.Random(seed); n=len(values); samples=[]
 for _ in range(BOOTSTRAPS):samples.append(sum(rng.choices(values,k=n))/n)
 samples.sort();return samples[int(.025*BOOTSTRAPS)],samples[min(BOOTSTRAPS-1,int(.975*BOOTSTRAPS))]
def write_csv(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
 if not CAPTURE.is_file():raise FileNotFoundError(f"paired rollout capture missing: {CAPTURE}")
 rows=[json.loads(s) for s in CAPTURE.open() if s.strip()]
 by_split={s:[r for r in rows if r['split']==s] for s in ('train','validation','test')}
 required={'train':200,'validation':100,'test':200}
 if any(len(by_split[k])!=n for k,n in required.items()):raise ValueError(f"Expected disjoint 200/100/200 split, got { {k:len(v) for k,v in by_split.items()} }")
 all_hashes=[r['case_hash'] for r in rows]
 if len(all_hashes)!=len(set(all_hashes)):raise ValueError('A public state appears in more than one split or more than once.')
 if any(('features' not in r or any(f not in r['features'] for f in FEATURES)) for r in rows):raise ValueError('Missing frozen predecision features.')
 train,val,test=(by_split['train'],by_split['validation'],by_split['test'])
 X=np.asarray([[float(r['features'][f]) for f in FEATURES] for r in train],dtype=np.float64)
 y=np.asarray([float(r['defer32_correct'])-float(r['base_correct']) for r in train],dtype=np.float64)
 mean=X.mean(0);scale=X.std(0);scale[scale<1e-8]=1.0;Z=(X-mean)/scale
 ymean=float(y.mean());beta=np.linalg.solve(Z.T@Z+ALPHA*np.eye(len(FEATURES)),Z.T@(y-ymean));intercept=ymean
 def predict(sample):
  x=np.asarray([[float(r['features'][f]) for f in FEATURES] for r in sample],dtype=np.float64)
  return ((x-mean)/scale)@beta+intercept
 val_pred=predict(val);test_pred=predict(test)
 def choose_threshold(pred,sample,lam):
  choices=[float('-inf'),*sorted(set(map(float,pred))),float('inf')]
  scored=[]
  for t in choices:
   calls=[float(p>t) for p in pred]
   final=[float(r['defer32_correct']) if c else float(r['base_correct']) for r,c in zip(sample,calls)]
   utility=[a-lam*c for a,c in zip(final,calls)]
   scored.append((statistics.mean(utility),-statistics.mean(calls),t,statistics.mean(final),statistics.mean(calls)))
  best=max(scored,key=lambda z:(z[0],z[1],z[2]))
  return {'threshold':best[2],'val_utility':best[0],'val_call_rate':best[4],'val_accuracy':best[3]}
 def outcome_vectors(sample,pred,threshold,lam):
  calls=[float(p>threshold) for p in pred]
  final=[float(r['defer32_correct']) if c else float(r['base_correct']) for r,c in zip(sample,calls)]
  utility=[a-lam*c for a,c in zip(final,calls)]
  base=[float(r['base_correct']) for r in sample]
  deltas=[f-b for f,b in zip(final,base)]
  return calls,final,utility,base,deltas
 curve=[]; test_case=[]; thresholds=[]
 seed=20261117
 for i,lam in enumerate(COSTS):
  selected=choose_threshold(val_pred,val,lam); thresholds.append({'lambda_defer':lam,**selected})
  calls,final,utility,base,deltas=outcome_vectors(test,test_pred,selected['threshold'],lam)
  ci_acc=ci(final,seed+i); ci_util=ci(utility,seed+20+i); ci_delta=ci(deltas,seed+40+i)
  curve.append({'lambda_defer':lam,'policy':'ridge_public_feature_router','threshold':selected['threshold'],'n_test':len(test),'accuracy':statistics.mean(final),'accuracy_ci_low':ci_acc[0],'accuracy_ci_high':ci_acc[1],'expert_call_rate':statistics.mean(calls),'mean_cost':lam*statistics.mean(calls),'net_utility':statistics.mean(utility),'utility_ci_low':ci_util[0],'utility_ci_high':ci_util[1],'paired_accuracy_delta_vs_answer':statistics.mean(deltas),'delta_ci_low':ci_delta[0],'delta_ci_high':ci_delta[1]})
  b=[float(r['base_correct']) for r in test]; e=[float(r['defer32_correct']) for r in test]
  actual_calls=[1.0 if (not bb and ee and lam<1) else 0.0 for bb,ee in zip(b,e)]
  actual_acc=[1.0 if c else bb for bb,c in zip(b,actual_calls)]
  perfect_calls=[1.0 if (not bb and lam<1) else 0.0 for bb in b]
  perfect_acc=[1.0 if c else bb for bb,c in zip(b,perfect_calls)]
  fixed_policies=(
   ('always_answer',b,[0.0]*len(test),b),
   ('always_expert',e,[1.0]*len(test),[a-lam for a in e]),
   ('oracle_actual_expert_selective',actual_acc,actual_calls,[a-lam*c for a,c in zip(actual_acc,actual_calls)]),
   ('oracle_perfect_selective_defer',perfect_acc,perfect_calls,[a-lam*c for a,c in zip(perfect_acc,perfect_calls)]),
  )
  for strategy,acc,cost,utils in fixed_policies:
   # Fixed and oracle policies are evaluated after outcomes; oracle rows are upper bounds only.
   vals=list(map(float,acc)); callv=list(map(float,cost)); utilityv=list(map(float,utils))
   curve.append({'lambda_defer':lam,'policy':strategy,'threshold':'','n_test':len(test),'accuracy':statistics.mean(vals),'accuracy_ci_low':ci(vals,seed+100+i)[0],'accuracy_ci_high':ci(vals,seed+100+i)[1],'expert_call_rate':statistics.mean(callv),'mean_cost':lam*statistics.mean(callv),'net_utility':statistics.mean(utilityv),'utility_ci_low':ci(utilityv,seed+120+i)[0],'utility_ci_high':ci(utilityv,seed+120+i)[1],'paired_accuracy_delta_vs_answer':statistics.mean([a-bb for a,bb in zip(vals,b)]),'delta_ci_low':ci([a-bb for a,bb in zip(vals,b)],seed+140+i)[0],'delta_ci_high':ci([a-bb for a,bb in zip(vals,b)],seed+140+i)[1]})
  callrate=statistics.mean(calls)
  rand_acc=[bb+callrate*(ee-bb) for bb,ee in zip(b,e)]; rand_util=[a-lam*callrate for a in rand_acc]
  curve.append({'lambda_defer':lam,'policy':'random_same_test_budget_as_router','threshold':'','n_test':len(test),'accuracy':statistics.mean(rand_acc),'accuracy_ci_low':ci(rand_acc,seed+160+i)[0],'accuracy_ci_high':ci(rand_acc,seed+160+i)[1],'expert_call_rate':callrate,'mean_cost':lam*callrate,'net_utility':statistics.mean(rand_util),'utility_ci_low':ci(rand_util,seed+180+i)[0],'utility_ci_high':ci(rand_util,seed+180+i)[1],'paired_accuracy_delta_vs_answer':statistics.mean([a-bb for a,bb in zip(rand_acc,b)]),'delta_ci_low':ci([a-bb for a,bb in zip(rand_acc,b)],seed+200+i)[0],'delta_ci_high':ci([a-bb for a,bb in zip(rand_acc,b)],seed+200+i)[1]})
  if lam==0.60:
   for r,pred,c in zip(test,test_pred,calls):test_case.append({'case_hash':r['case_hash'],'predicted_advantage':float(pred),'router_calls_defer':int(c),'base_correct_posthoc':bool(r['base_correct']),'defer32_correct_posthoc':bool(r['defer32_correct'])})
 RESULTS.mkdir(parents=True,exist_ok=True)
 write_csv(RESULTS/'router_validation_thresholds.csv',thresholds);write_csv(RESULTS/'router_test_cost_curve.csv',curve);write_csv(RESULTS/'router_test_case_results_cost060.csv',test_case)
 model={'model':'standardized ridge regression','alpha':ALPHA,'features':list(FEATURES),'feature_mean':mean.tolist(),'feature_scale':scale.tolist(),'coefficients_standardized':beta.tolist(),'intercept':intercept,'training_label':'defer32_correct - base_correct; used only during offline fitting','runtime_input_policy':'Public prompt-derived numeric features only; no Gold, hidden facts, expert output, or outcomes.'}
 (RESULTS/'public_feature_router_model.json').write_text(json.dumps(model,indent=2)+'\n')
 manifest={'status':'completed','train_validation_test_cases':required,'capture_sha256':sha_file(CAPTURE),'model':model,'thresholds_chosen_on_validation_only':True,'test_used_for_threshold_selection':False,'bootstrap_replicates':BOOTSTRAPS,'bootstrap_unit':'case','outputs':{p.name:sha_file(p) for p in (RESULTS/'router_validation_thresholds.csv',RESULTS/'router_test_cost_curve.csv',RESULTS/'router_test_case_results_cost060.csv',RESULTS/'public_feature_router_model.json')}}
 (RESULTS/'router_analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'status':'completed','fit_cases':len(train),'threshold_cases':len(val),'test_cases':len(test),'outputs':list(manifest['outputs'])},indent=2))
if __name__=='__main__':main()
