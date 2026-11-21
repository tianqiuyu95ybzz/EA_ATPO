#!/usr/bin/env python3
"""Construct non-Gold historical replay candidates, outputs private."""
import json,hashlib,pathlib,os,collections
source=pathlib.Path('/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1_public_medical_data_20261121/experiments/ea_atpo_v3/public_medical_20261121/private/remedi_base.json')
out=pathlib.Path(__file__).parent/'private'/'cases.jsonl'
if hashlib.sha256(source.read_bytes()).hexdigest()!='b27b6b1222255cc87cbc6cc0fb6ce7de98e0b6341f5ea61892bb92f247383458':raise ValueError('source hash mismatch')
rows=json.loads(source.read_text()); seen=set();candidates=[];audit=collections.Counter()
for index,record in enumerate(rows):
 turns=[(t.get('role'),t.get('sentence')) for t in record['information']]
 if not all(r in ('doctor','patient') and isinstance(x,str) and x.strip() for r,x in turns):audit['invalid']+=1;continue
 h=hashlib.sha256(json.dumps(turns,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
 if h in seen:audit['duplicate']+=1;continue
 seen.add(h)
 for i in range(1,len(turns)-2):
  role,text=turns[i];patient=turns[i+1];after=turns[i+2]
  if turns[i-1][0]!='patient' or role!='doctor' or not any(c in text for c in '？?') or patient[0]!='patient' or after[0]!='doctor':continue
  if not any(r=='patient' for r,t in turns[:i]):continue
  cid=str(record['dialogue']);candidates.append({'case_id':'remedi:'+cid,'source_index':index,'question_turn':i,'initial':[{'role':r,'text':t} for r,t in turns[:i]],'doctor_question':text,'patient_feedback':patient[1],'doctor_after_feedback':after[1],'doctor_proxy_semantics':'historical doctor turn after feedback; not initial-state answer, not Gold','source_hash':h});break
candidates.sort(key=lambda x:hashlib.sha256(('chinese-pilot-v1:'+x['case_id']).encode()).digest())
candidates=candidates[:50]
assert len(candidates)==50 and len({x['case_id'] for x in candidates})==50
fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as f:
 for c in candidates:f.write(json.dumps(c,ensure_ascii=False)+'\n')
print(json.dumps({'selected':len(candidates),'selection':'fixed hash of source dialogue id among chronological question/patient/doctor eligible records','source':'ReMeDi base','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'case_ids_sha256':hashlib.sha256(('\n'.join(x['case_id'] for x in candidates)+'\n').encode()).hexdigest(),'private_cases_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'audit':dict(audit)},ensure_ascii=False,indent=2))
