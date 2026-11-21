#!/usr/bin/env python3
"""Bounded CPU smoke, pinned project Qwen3-8B. Results are private."""
import argparse,hashlib,json,os,pathlib,time
from ddx import load_meta,validation_rows,Simulator,format_initial,parse,paired
PIN='b968826d9c46dd6066d109eabc6255188de91218'
MODEL='/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-8B'
INDEX_SHA='f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc'
def main(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only; set CUDA_VISIBLE_DEVICES to empty')
    if 'private' not in a.out.parts:raise ValueError('private output required')
    if a.limit<1 or a.limit>50:raise ValueError('CPU smoke maximum 50')
    model=pathlib.Path(MODEL)
    if hashlib.sha256((model/'model.safetensors.index.json').read_bytes()).hexdigest()!=INDEX_SHA:raise ValueError('wrong checkpoint index')
    onto,labels=load_meta(a.evidences,a.conditions)
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(a.threads)
    t=time.monotonic();tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    net=AutoModelForCausalLM.from_pretrained(model,torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True).eval()
    a.out.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    fd=os.open(a.out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:
        for cid,row in validation_rows(a.zip,a.limit):
            sim=Simulator(row,onto);view=sim.view();prompt=format_initial(view,onto,labels)
            messages=[{'role':'user','content':prompt}]
            encoded=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True)['input_ids']
            with torch.inference_mode():generated=net.generate(encoded,max_new_tokens=24,do_sample=False,pad_token_id=tok.eos_token_id)
            raw=tok.decode(generated[0,encoded.shape[-1]:],skip_special_tokens=True);pred,status=parse(raw,labels)
            record={'case_id':cid,'initial_evidence':view['observed'],'gold_pathology':row['PATHOLOGY'],'base_prediction':pred,'base_confidence':None,'base_correct':paired(pred,row['PATHOLOGY'],labels)['base_correct'],'raw_output':raw,'parsing_status':status,'base_revision':PIN}
            f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno());print('completed',cid,'status',status,'elapsed_seconds',round(time.monotonic()-t,1),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',type=pathlib.Path,required=True);p.add_argument('--evidences',type=pathlib.Path,required=True);p.add_argument('--conditions',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--limit',type=int,default=50);p.add_argument('--threads',type=int,default=8);main(p.parse_args())
