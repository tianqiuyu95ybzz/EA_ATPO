#!/usr/bin/env python3
"""Validate paired pilot and create a private unscored human review packet."""
import collections,csv,hashlib,json,os,pathlib
p=pathlib.Path(__file__).resolve().parent;old=p.parent/'private'
read=lambda f:[json.loads(l) for l in f.open(encoding='utf8')]
rows=read(old/'model_trajectories_retry.jsonl');cases=read(old/'cases.jsonl');queue=read(old/'blind_review_queue.jsonl');keys=read(old/'blind_review_key.jsonl')
assert len(rows)==len(cases)==50 and len(queue)==len(keys)==20
byid={r['case_id']:r for r in rows};cby={x['case_id']:x for x in cases};keyby={x['review_case']:x for x in keys}
assert len(byid)==len(cby)==50 and set(byid)==set(cby) and len(keyby)==20
assert len({q['review_case'] for q in queue})==20 and set(keyby)=={q['review_case'] for q in queue}
packets=[]; review_rows=[]
for q in queue:
 k=keyby[q['review_case']];cid=k['case_id'];r=byid[cid];c=cby[cid];assert q['initial_history']==c['initial'] and q['recorded_followup_question']==c['doctor_question'] and q['recorded_patient_response']==c['patient_feedback'];assert set(k['answer_order'])=={'always_answer','historical_replay_ask'}
 assert r['initial_eos'] and r['replay_eos'] and r['always_answer'].strip() and r['always_ask_replay'].strip()
 assert len(q['blind_answers'])==2
 variants=[]
 for i,name in enumerate(k['answer_order']):
  expected=r['always_answer'] if name=='always_answer' else r['always_ask_replay'];assert q['blind_answers'][i]['answer']==expected
  ctx=list(c['initial']);
  if name=='historical_replay_ask':ctx+= [{'role':'doctor','text':c['doctor_question']},{'role':'patient','text':c['patient_feedback']}]
  variants.append({'option':chr(65+i),'visible_context':ctx,'answer':expected})
  review_rows.append({'review_case':q['review_case'],'option':chr(65+i),'medical_correctness_1_5':'','completeness_1_5':'','safety_1_5':'','serious_error_yes_no_unclear':'','reviewer_id':'','reviewed_at':'','notes':''})
 packets.append({'review_case':q['review_case'],'variants':variants,'additional_information_effective_yes_no_unclear':'','diagnostic_or_advice_change_yes_no_unclear':'','paired_notes':''})
# IDs are pseudonyms; source text is not guaranteed de-identified.
assert len({x['review_case'] for x in review_rows})==20 and len(review_rows)==40
for name,items in [('review_packets.jsonl',packets),('review_key.jsonl',keys)]:
 fd=os.open(p/'private'/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w',encoding='utf8') as f:
  for item in items:f.write(json.dumps(item,ensure_ascii=False)+'\n')
fd=os.open(p/'private'/'blind_review_scores.csv',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(review_rows[0]),lineterminator='\n');w.writeheader();w.writerows(review_rows)
fd=os.open(p/'private'/'paired_review_scores.csv',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w',encoding='utf8',newline='') as f:
 fields=['review_case','added_information_useful_yes_no_unclear','diagnosis_or_advice_changed_yes_no_unclear','comparison_notes','reviewer_id','reviewed_at']
 w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
 for packet in packets:w.writerow({'review_case':packet['review_case']})
ask=[r for r in rows if r['router_action']=='ASK'];assert len(ask)==1
x=ask[0];c=cby[x['case_id']];assert x['router_marker'] and x['router_result']==x['always_ask_replay'] and c['initial'][-1]['role']=='patient'
needles=('建议','可以','考虑','需要','最好','不用','不需要','服用','就医','医院');advice=[c for c in cases if any(s in c['doctor_question'] for s in needles)]
result={'pilot_pairs':50,'review_cases':20,'review_answers':40,'unique_review_cases':len({x['review_case'] for x in review_rows}),'paired_output_integrity':'PASS: source identity, chronological contexts, option key, nonempty and EOS','human_scores_present':False,'rubric_status':'PENDING_HUMAN','name_blinding':'policy names and answer order hidden; context exposure differs and can reveal condition','privacy':'pseudonymous IDs only; source text may contain personal information and must be reviewed before sharing','unique_router_ask':{'trigger':'explicit patient uncertainty marker and retrospective replay availability','historical_question_and_feedback':'source-matched and chronological','additional_information':'feedback contains previous treatment status and frequency not stated at same precision in initial history','answer_changed':x['always_answer']!=x['always_ask_replay'],'answer_change_summary':'initial answer recommends seeking care; replay answer includes self-care or treatment suggestions and a conditional recommendation to seek care','clinical_quality_or_safety':'NOT_REVIEWED; no causal attribution to feedback'},'same_state_defer':{'original_pilot_scored':0,'candidate_note':'one original next-doctor turn contains both a question and conditional advice; potentially usable as an unverified reference only after manual review','screening':'keyword-based screening of 50 next-doctor turns; not a complete clinical annotation','keyword_screen_hits':len(advice),'full_paired_defer_cohort':'not established'},'patient_ids_and_transcripts_published':False}
(p/'quality_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
