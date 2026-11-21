#!/usr/bin/env python3
"""Map existing Oracle actions to author archives without treating option-only matches as proof."""
import ast,collections,csv,difflib,hashlib,json,os,pathlib,re,unicodedata,zipfile
P=pathlib.Path(__file__).resolve().parents[1];D=P/'private';SOURCE=pathlib.Path('/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1/experiments/expert_sft/data/source/dataset/merged_train_dataset.jsonl');STATE=pathlib.Path('/pfs/pfs-spMCeu/image_team/workspace-copy/EA-ATPO_Oracle_20261121/run_08/private_state_results.jsonl');TRAJ=pathlib.Path('/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1_router_training/experiments/ea_atpo_v2/recovery_bundle_20261120/pilot_trajectories.jsonl')
def norm(x):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(x))).strip().lower()
def tokens(x):return set(re.findall(r'[a-z0-9]+',norm(x)))
src=[json.loads(l) for l in SOURCE.open() if l.strip()]; states=[json.loads(l) for l in STATE.open() if l.strip()];ids=sorted({x['source_row_index'] for x in states});prompts={};opts={}
for i in ids:
 prompt=next(m['content'] for m in src[i]['prompt'] if m['role']=='user'); line=next(l for l in prompt.splitlines() if l.startswith('Options:'));opts[i]=ast.literal_eval(line.split(':',1)[1].strip());prompts[i]=norm(prompt.split('Options:',1)[0])
found=collections.defaultdict(list)
for name,files in [('medqa',[('data_clean/questions/US/'+s+'.jsonl',s) for s in ['train','dev','test']]),('medmcqa',[(s+'.json',s) for s in ['train','dev','test']])]:
 with zipfile.ZipFile(D/(name+'.zip')) as z:
  for file,split in files:
   for j,line in enumerate(z.open(file),1):
    o=json.loads(line);raw=o.get('options') or {k:o.get('op'+k.lower(),'') for k in 'ABCD'};vals={norm(v) for v in raw.values() if v};q=norm(o.get('question',''))
    for i in ids:
     if src[i]['data_source'] != ('medqa' if name=='medqa' else 'mcqa'):continue
     cur={norm(v) for v in opts[i].values()};
     if len(cur)<4 or not cur.issubset(vals):continue
     if name=='medqa': correct=o.get('answer');official_option=o.get('answer_idx')
     else:
      cop=o.get('cop'); official_option='ABCD'[cop-1] if isinstance(cop,int) and 1<=cop<=4 else None; correct=raw.get(official_option) if official_option else None
     mapped=[k for k,v in opts[i].items() if correct and norm(v)==norm(correct)]
     t=tokens(prompts[i]);u=tokens(q); inter=len(t&u);overlap=inter/max(1,len(t));
     found[i].append({'dataset':name,'official_split':split,'official_locator':o.get('id') or file+':'+str(j),'official_answer_option':official_option,'official_correct_text':correct,'mapped_option':mapped[0] if len(mapped)==1 else None,'all_4_options_exact':True,'prompt_exact_to_original_question':prompts[i]==q,'problem_text_exact_substring':norm(prompts[i].split('problem:',1)[-1]) in q,'prompt_token_overlap_fraction':round(overlap,4),'source_line':j,'license_repository':'MIT (repository license; data rights unverified)','source_url':('https://github.com/jind11/MedQA' if name=='medqa' else 'https://github.com/medmcqa/medmcqa'),'archive_sha256':('1c2ca8130b3d86d9a99a432ab9bef14f3bb9807bef20facd9ac86ba36960f629' if name=='medqa' else '16c1fbc6f47d548d2af7837b18e893aa45f45c0be9bda0a9adfff3c625bf9262')})
keys={(x['case_id'],x['state_id']):x for x in states};actions=[json.loads(l) for l in TRAJ.open() if l.strip()];counts=collections.Counter();records=[];percase={}
for row in actions:
 key=(str(row['case_id']),str(row['state_id']))
 if key not in keys or row['action'] not in ('ASK','ANSWER','DEFER'):continue
 st=keys[key];i=st['source_row_index']; matches=found[i];benchmark=src[i]['ground_truth']['answer'];sig=[x for x in matches if x['prompt_exact_to_original_question']];status='EXACT_QUESTION_AND_OPTIONS' if len(sig)==1 else 'UNIQUE_OPTION_SET_ONLY' if len(matches)==1 else 'AMBIGUOUS_OPTION_SET' if matches else 'NO_MATCH'
 counts[status]+=1
 for candidate in matches:candidate['answer_agrees_with_atpo_key'] = candidate['mapped_option']==benchmark
 records.append({'case_id':key[0],'state_id':key[1],'action':row['action'],'atpo_row_index':i,'atpo_source':src[i]['data_source'],'benchmark_answer_key':benchmark,'mapping_status':status,'candidate_count':len(matches),'same_case_source_row_count':1,'cross_case_duplicate_status':'not established; see aggregate and limitations','candidates':matches})
 percase[key[0]]=i
if len(records)!=129 or len(keys)!=43:raise ValueError('unexpected action coverage')
path=D/'private_129_gold_provenance_v3.jsonl'
with path.open('x',encoding='utf-8') as f:
 os.chmod(path,0o600)
 for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
unique={i:found[i] for i in ids}; duplicate_option_sets={i:len(v) for i,v in unique.items() if len(v)>1}
comparisons=collections.Counter()
for i,vs in unique.items():
 if len(vs)==1:comparisons['unique_agree' if vs[0]['mapped_option']==src[i]['ground_truth']['answer'] else 'unique_disagree_or_unmapped']+=1
 else:
  answers={v['mapped_option'] for v in vs};comparisons['ambiguous_all_agree' if answers=={src[i]['ground_truth']['answer']} else 'ambiguous_not_all_agree']+=1
summary={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'states':len(states),'cases':len(percase),'action_records':len(records),'case_source_indices_unique':len(set(percase.values())),'mapping_action_counts':dict(counts),'mapping_case_counts':dict(collections.Counter('EXACT_QUESTION_AND_OPTIONS' if len([x for x in v if x['prompt_exact_to_original_question']])==1 else 'UNIQUE_OPTION_SET_ONLY' if len(v)==1 else 'AMBIGUOUS_OPTION_SET' if v else 'NO_MATCH' for v in unique.values())),'official_correct_option_comparison_case_counts':dict(comparisons),'cases_problem_exact_substring':sum(any(c['problem_text_exact_substring'] for c in v) for v in unique.values()),'candidate_official_split_counts':dict(collections.Counter(x['official_split'] for v in unique.values() for x in v)),'multiple_candidate_source_indices':duplicate_option_sets,'private_mapping_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'private_mapping_path':str(path),'interpretation':'OPTION SET matches are candidates only; transformed prompts do not prove case equivalence. No official answer or raw text in this aggregate.'}
(P/'data/gold_provenance_summary_v3.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
