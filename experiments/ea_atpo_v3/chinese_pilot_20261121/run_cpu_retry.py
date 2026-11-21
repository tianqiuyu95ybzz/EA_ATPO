#!/usr/bin/env python3
"""Exploratory CPU paired replay. Never send private text to remote services."""
import hashlib,json,os,pathlib,time
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only')
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
p=pathlib.Path(__file__).parent; model=pathlib.Path('/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B')
if hashlib.sha256((model/'model.safetensors.index.json').read_bytes()).hexdigest()!='f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc':raise ValueError('wrong model')
torch.set_num_threads(8);tok=AutoTokenizer.from_pretrained(model,local_files_only=True);net=AutoModelForCausalLM.from_pretrained(model,torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True).eval();start=time.monotonic()
out=p/'private'/'model_trajectories_retry.jsonl';fd=os.open(out,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
system='请根据已知信息审慎回答。最多两句话、60个汉字；不知道就说不确定，不要编造。'
def run(turns):
 messages=[{'role':'system','content':system}]+[{'role':'user' if t['role']=='patient' else 'assistant','content':t['text']} for t in turns]
 encoded=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True)['input_ids']; length=encoded.shape[-1]
 if length>3072:raise ValueError('context exceeds limit; no silent truncation')
 with torch.inference_mode():generated=net.generate(encoded,max_new_tokens=96,do_sample=False,pad_token_id=tok.eos_token_id)
 return tok.decode(generated[0,length:],skip_special_tokens=True),length,int(generated.shape[-1]-length),bool(generated[0,-1].item()==tok.eos_token_id)
with os.fdopen(fd,'w') as f:
 for n,line in enumerate((p/'private'/'cases.jsonl').open(),1):
  case=json.loads(line);before=case['initial'];response,tok_n,n_before,eos_before=run(before)
  after=before+[{'role':'doctor','text':case['doctor_question']},{'role':'patient','text':case['patient_feedback']}];follow,tok_after,n_after,eos_after=run(after)
  # Limited retrospective proxy: explicit uncertainty marker only, with observed replay available.
  last_patient=next(t['text'] for t in reversed(before) if t['role']=='patient');markers=['不知道','不清楚','不确定'];matched=[x for x in markers if x in last_patient];action='ASK' if matched else 'ANSWER'
  r={'case_id':case['case_id'],'always_answer':response,'always_ask_replay':follow,'router_action':action,'router_reason':'explicit_uncertainty_marker_and_recorded_replay_available' if matched else 'no_verified_missing_background_or_high_risk','router_marker':matched,'router_result':follow if action=='ASK' else response,'doctor_reference_after_feedback':case['doctor_after_feedback'],'defer_reference_comparable_to_initial':False,'ask_count':1 if action=='ASK' else 0,'initial_tokens':tok_n,'replay_tokens':tok_after,'generated_initial':n_before,'generated_replay':n_after,'initial_eos':eos_before,'replay_eos':eos_after}
  f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno());print(json.dumps({'completed':n,'initial_nonempty':bool(response.strip()),'replay_nonempty':bool(follow.strip()),'route':action,'seconds':round(time.monotonic()-start,1)}),flush=True)
