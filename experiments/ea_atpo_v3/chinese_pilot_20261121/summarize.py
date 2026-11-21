#!/usr/bin/env python3
"""Aggregate completed cases without writing dialogue, answers, or Gold."""
import collections,csv,hashlib,json,pathlib
p=pathlib.Path(__file__).parent;input_path=p/'private'/'model_trajectories_retry.jsonl'
rows=[json.loads(x) for x in input_path.read_text().splitlines()];n=len(rows);assert len({x['case_id'] for x in rows})==n
counts=collections.Counter(x['router_action'] for x in rows)
summary={'planned_cases':50,'completed_paired_cases':n,'always_answer_n':n,'historical_ask_replay_n':n,'rule_router_n':n,'same_initial_state_defer_n':0,'post_feedback_doctor_proxy_available_n':n,'nonempty_answer_n':sum(bool(x['always_answer'].strip()) for x in rows),'nonempty_replay_answer_n':sum(bool(x['always_ask_replay'].strip()) for x in rows),'initial_eos_n':sum(bool(x['initial_eos']) for x in rows),'replay_eos_n':sum(bool(x['replay_eos']) for x in rows),'router_action_counts':dict(counts),'router_ask_rate':counts['ASK']/n if n else None,'router_defer_rate':counts['DEFER']/n if n else None,'average_ask_turns_router':counts['ASK']/n if n else None,'average_ask_turns_always_ask':1.0 if n else None,'answer_quality':None,'useful_ask_rate':None,'critical_error_rate':None,'utility':None,'reason_unavailable':'no trustworthy independent review; historical post-feedback doctor response is not verified Gold and is not comparable to initial-state DEFER','router_limit':'retrospective replay availability; uncertainty keyword not validated as clinically important','private_results_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest()}
(p/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
with (p/'results.csv').open('w',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['case_hash','answer_completed','ask_replay_completed','router_action','same_state_defer_available','quality_reviewed'])
 for x in rows:w.writerow([hashlib.sha256(('pilot-v1:'+x['case_id']).encode()).hexdigest(),bool(x['always_answer'].strip()),bool(x['always_ask_replay'].strip()),x['router_action'],False,False])
print(json.dumps(summary,indent=2))
