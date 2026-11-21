#!/usr/bin/env python3
"""Paired one-ASK CPU smoke on same 50 cases; no labels in prompt."""
import argparse,hashlib,json,os,pathlib,time
from ddx import load_meta,validation_rows,Simulator,format_initial,parse,paired
from run_cpu_base import MODEL,INDEX_SHA,PIN

def main(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'private' not in a.out.parts:raise ValueError('CPU and private output required')
    if hashlib.sha256((pathlib.Path(MODEL)/'model.safetensors.index.json').read_bytes()).hexdigest()!=INDEX_SHA:raise ValueError('checkpoint index mismatch')
    onto,labels=load_meta(a.evidences,a.conditions)
    originals={x['case_id']:x for x in map(json.loads,a.base.read_text().splitlines())}
    if len(originals)!=50:raise ValueError('requires complete paired 50 Base predictions')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(8);tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    net=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True).eval();start=time.monotonic()
    fd=os.open(a.out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:
        for cid,row in validation_rows(a.zip,50):
            prior=originals[cid];sim=Simulator(row,onto)
            if sim.view()['observed']!=prior['initial_evidence']:raise ValueError('state mismatch')
            # Predefined fixed order based only on public ontology and previously observed keys.
            key=next(k for k in sorted(onto) if onto[k]['code_question']==k and k not in sim.view()['observed'])
            sim.ask(key);view=sim.view();prompt=format_initial(view,onto,labels)
            ids=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True)['input_ids']
            with torch.inference_mode():gen=net.generate(ids,max_new_tokens=24,do_sample=False,pad_token_id=tok.eos_token_id)
            raw=tok.decode(gen[0,ids.shape[-1]:],skip_special_tokens=True);pred,status=parse(raw,labels)
            record={'case_id':cid,'ask_trace':view['trace'],'gold_pathology':row['PATHOLOGY'],'base_initial':prior['base_prediction'],'base_after_ask':pred,'base_correct_after_ask':paired(pred,row['PATHOLOGY'],labels)['base_correct'],'raw_output':raw,'parsing_status':status,'base_revision':PIN}
            f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno());print('completed',cid,'elapsed_seconds',round(time.monotonic()-start,1),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',type=pathlib.Path,required=True);p.add_argument('--evidences',type=pathlib.Path,required=True);p.add_argument('--conditions',type=pathlib.Path,required=True);p.add_argument('--base',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);main(p.parse_args())
