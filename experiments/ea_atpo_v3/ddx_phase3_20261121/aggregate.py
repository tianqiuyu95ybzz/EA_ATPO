#!/usr/bin/env python3
"""Aggregate private matched outputs; no patient text, predictions or labels exported."""
import argparse,collections,hashlib,json,pathlib,random

def load(p):return {x['case_id']:x for x in map(json.loads,p.read_text().splitlines())}
p=argparse.ArgumentParser();p.add_argument('--base',type=pathlib.Path,required=True);p.add_argument('--ask',type=pathlib.Path,required=True);a=p.parse_args();b=load(a.base);q=load(a.ask)
if len(b)!=50 or set(b)!=set(q):raise ValueError('incomplete or non-paired 50-case inputs')
ids=sorted(b);n=len(ids);bc=[];ac=[];stats=collections.Counter()
for k in ids:
 x=b[k];y=q[k];assert x['gold_pathology']==y['gold_pathology'] and x['base_prediction']==y['base_initial'];assert len(y['ask_trace'])==2 and y['ask_trace'][-1]['cost_units']==1
 base=x['base_correct'];after=y['base_correct_after_ask'];bc.append(int(base));ac.append(int(after));stats['base_correct']+=base;stats['ask_correct']+=after;stats['base_invalid']+=x['parsing_status']!='exact';stats['ask_invalid']+=y['parsing_status']!='exact';stats['ask_corrected']+=(not base and after);stats['ask_degraded']+=(base and not after)
accuracy=sum(bc)/n;asked=sum(ac)/n;r=random.Random(20261121);diff=[]
for _ in range(10000):
 idx=[r.randrange(n) for _ in range(n)];diff.append(sum(ac[i]-bc[i] for i in idx)/n)
diff.sort()
summary={'sample_n':n,'sample':'first 50 official validation rows; exploratory, not random and not independent certification','base_accuracy':accuracy,'base_confidence_available':False,'always_defer_A_accuracy':1.0,'always_defer_B_accuracy':1.0,'defer_cost_free_accuracy_gain_A':1-accuracy,'defer_cost_free_accuracy_gain_B':1-accuracy,'base_correct_preserved_A_and_B':sum(bc),'base_errors_corrected_A_and_B':n-sum(bc),'base_correct_destroyed_A':0,'base_correct_destroyed_B':0,'always_ask_accuracy':asked,'ask_accuracy_difference':asked-accuracy,'ask_accuracy_difference_paired_bootstrap_ci95':[diff[249],diff[9749]],'always_answer_utility':accuracy,'always_ask_utility':{str(c):asked-c for c in (0.01,0.05,0.10)},'always_defer_A_and_B_utility':{str(c):1-c for c in (0.05,0.10,0.20,0.60)},'q_guided_router':'unavailable: no DDX-specific trained policy or frozen compatible embeddings','counts':dict(stats),'parse_policy':'strict exact official disease label; invalid counted wrong for base accuracy','diagnostic_scope':'synthetic pathology classification only','base_output_sha256':hashlib.sha256(a.base.read_bytes()).hexdigest(),'ask_output_sha256':hashlib.sha256(a.ask.read_bytes()).hexdigest()}
print(json.dumps(summary,indent=2))
