#!/usr/bin/env python3
"""Convert observed original-turn samples to chat format; does not train or authorize use."""
import argparse,hashlib,json,os,pathlib,pickle
class Restricted(pickle.Unpickler):
 def find_class(self,m,n):raise pickle.UnpicklingError('forbidden global')
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['remedi','meddg'],required=True);p.add_argument('--input',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--max-dialogues',type=int,default=0);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
if 'private' not in a.output.parts:raise ValueError('output must be inside private directory')
if a.dataset=='remedi':
 x=json.loads(a.input.read_text()); docs=[[(t.get('role'),t.get('sentence')) for t in d['information']] for d in x]
else:
 with a.input.open('rb') as f:x=Restricted(f).load()
 docs=[[({'Patients':'patient','Doctor':'doctor'}.get(t.get('id')),t.get('Sentence')) for t in d] for d in x]
if a.max_dialogues:docs=docs[:a.max_dialogues]
count=0;seen=set()
with a.output.open('x',encoding='utf-8') as out:
 os.chmod(a.output,0o600)
 for i,doc in enumerate(docs):
  if any(role not in ('patient','doctor') or not isinstance(text,str) or not text.strip() for role,text in doc):continue
  fingerprint=hashlib.sha256(json.dumps(doc,ensure_ascii=False).encode()).hexdigest()
  if fingerprint in seen:continue
  seen.add(fingerprint);bucket=int(fingerprint[:8],16)%100;split='train' if bucket<80 else 'validation' if bucket<90 else 'test';messages=[]
  for role,text in doc:
   messages.append({'role':'user' if role=='patient' else 'assistant','content':text})
   if role=='doctor' and any(m['role']=='user' for m in messages):
    out.write(json.dumps({'source':a.dataset,'case_hash':fingerprint,'split':split,'messages':messages},ensure_ascii=False)+'\n');count+=1
print(json.dumps({'output_rows':count,'unique_dialogues':len(seen),'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()}))
