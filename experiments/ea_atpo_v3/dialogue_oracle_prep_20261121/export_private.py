#!/usr/bin/env python3
"""Create local-only state and exclusion audit files; output directory must be new."""
import argparse, hashlib, json, os, pathlib
from pipeline import load, normalize
p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['remedi','meddg'],required=True);p.add_argument('--input',type=pathlib.Path,required=True);p.add_argument('--output-dir',type=pathlib.Path,required=True);a=p.parse_args()
if 'private' not in a.output_dir.parts: raise ValueError('output directory must be named within private')
a.output_dir.mkdir(mode=0o700,parents=False,exist_ok=False)
seen={}
with (a.output_dir/'states.jsonl').open('x',encoding='utf-8') as states, (a.output_dir/'audit.jsonl').open('x',encoding='utf-8') as audit:
 os.chmod(states.name,0o600);os.chmod(audit.name,0o600)
 for i,sid,turns in load(a.dataset,a.input):
  fp=hashlib.sha256(json.dumps(turns,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
  reason=None
  if fp in seen: reason='exact_duplicate_of_record_'+str(seen[fp])
  else:
   seen[fp]=i
   if not all(r in ('patient','doctor') and isinstance(t,str) and t.strip() for r,t in turns): reason='invalid_role_or_empty_text'
   elif {r for r,t in turns}!={'patient','doctor'}: reason='missing_role'
  audit.write(json.dumps({'source_record_index':i,'source_dialogue_id':sid,'content_sha256':fp,'excluded_reason':reason})+'\n')
  if reason is None:
   for s in normalize(a.dataset,i,sid,turns): states.write(json.dumps(s,ensure_ascii=False)+'\n')
print('local sensitive output created; do not upload:',a.output_dir)
