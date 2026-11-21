#!/usr/bin/env python3
"""Prepare private paired blind review; no scores are invented."""
import hashlib,json,os,pathlib
p=pathlib.Path(__file__).parent;cases={x['case_id']:x for x in map(json.loads,(p/'private'/'cases.jsonl').read_text().splitlines())};results=[json.loads(x) for x in (p/'private'/'model_trajectories_retry.jsonl').read_text().splitlines()]
if len(results)<20:raise ValueError('need at least 20 completed cases')
selected=results[:20];queue=[];key=[]
for row in selected:
 c=cases[row['case_id']];ident=hashlib.sha256(('review-v1:'+row['case_id']).encode()).hexdigest()[:16];flip=int(ident[0],16)%2
 opts=[{'answer':row['always_answer'],'exposure':'initial_context'},{'answer':row['always_ask_replay'],'exposure':'initial_plus_recorded_question_and_patient_reply'}]
 names=['always_answer','historical_replay_ask']
 if flip:opts.reverse();names.reverse()
 queue.append({'review_case':ident,'initial_history':c['initial'],'recorded_followup_question':c['doctor_question'],'recorded_patient_response':c['patient_feedback'],'blind_answers':opts,'scores':None,'safety_findings':None})
 key.append({'review_case':ident,'case_id':row['case_id'],'answer_order':names})
for name,items in [('blind_review_queue.jsonl',queue),('blind_review_key.jsonl',key)]:
 fd=os.open(p/'private'/name,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
 with os.fdopen(fd,'w') as f:
  for x in items:f.write(json.dumps(x,ensure_ascii=False)+'\n')
print(json.dumps({'review_cases':len(queue),'review_status':'PENDING_HUMAN','scored_cases':0,'independent_reviewer':None}))
