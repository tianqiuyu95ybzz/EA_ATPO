#!/usr/bin/env python3
"""Aggregate sampled dialogue structure; no dialogue text leaves the private directory."""
import collections,csv,hashlib,json,pickle,pathlib,re,statistics,unicodedata
P=pathlib.Path(__file__).resolve().parents[1];D=P/'private';ROOT=P.parents[1]
class Restricted(pickle.Unpickler):
 def find_class(self,module,name):raise pickle.UnpicklingError('forbidden global: '+module+'.'+name)
def loadpk(path):
 with path.open('rb') as f:return Restricted(f).load()
def jl(path):return [json.loads(s) for s in path.read_text().splitlines()]
def norm(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',s)).lower()
def lines(v):
 if isinstance(v,str):yield v
 elif isinstance(v,list):
  for x in v:yield from lines(x)
 elif isinstance(v,dict):
  for k,x in v.items():
   if k in ('content','text','message','patient_message','question','prompt'):yield from lines(x)
   elif isinstance(x,(dict,list)):yield from lines(x)
atpo=ROOT/'real_case_pilot_20261103/outputs/full/pilot_public_states.jsonl'
refs=set()
if atpo.exists():
 for row in jl(atpo):
  for s in lines(row):
   if len(norm(s))>=10:refs.add(hashlib.sha256(norm(s).encode()).digest())
items={}
x=json.loads((D/'remedi_base.json').read_text());items['ReMeDi']=[[(t.get('role',''),t.get('sentence','')) for t in z['information']] for z in x]
x=loadpk(D/'meddg_train.pk');items['MedDG']=[[({'Patients':'patient','Doctor':'doctor'}.get(t.get('id'),t.get('id','')),t.get('Sentence','')) for t in z] for z in x]
for name,file in [('DISC-Med-SFT','disc_sample.jsonl'),('AquilaMed-Instruct','aquila_sample.jsonl')]:
 result=[]
 for x in jl(D/file):
  row=x['row']; turns=[]
  if name.startswith('DISC'):
   turns=[({'user':'patient','assistant':'doctor'}.get(t.get('role'),t.get('role','')),t.get('content','')) for t in row['conversation']]
  else:
   for pair in row.get('history',[]):
    if isinstance(pair,list) and len(pair)==2:turns.extend([('patient',pair[0]),('doctor',pair[1])])
   turns.extend([('patient',(row.get('instruction','')+' '+row.get('input','')).strip()),('doctor',row.get('output',''))])
  result.append(turns)
 items[name]=result
items['MedDialog']=[]
for row in jl(D/'meddialog_sample.jsonl'):
 turns=[]
 for j,t in enumerate(row):
  if not isinstance(t,str):turns.append(('unknown',''));continue
  role='doctor' if t.startswith(('医生：','医生:')) else 'patient' if j==0 or t.startswith(('病人：','病人:','患者：','患者:')) else 'unknown'
  turns.append((role,re.sub(r'^(医生|病人|患者)[:：]','',t)))
 items['MedDialog'].append(turns)
rows=[];hist={}
for name,docs in items.items():
 lens=[];valid=multi=zh=dup=anomaly=substantive=overlap=medical=ends_doctor=0;seen=set();role=collections.Counter()
 for turns in docs:
  lens.append(len(turns));role.update(r for r,t in turns);patients=[t for r,t in turns if r=='patient' and isinstance(t,str) and t.strip()];doctors=[t for r,t in turns if r=='doctor' and isinstance(t,str) and t.strip()]
  valid+=bool(patients and doctors);multi+=len(patients)>=2 and len(doctors)>=2
  zh+=bool(re.search(r'[\u4e00-\u9fff]',''.join(str(t) for r,t in turns)))
  anomaly+=bool(not patients or not doctors or any(r not in ('patient','doctor') or not isinstance(t,str) or not t.strip() for r,t in turns))
  substantive+=bool(doctors and any(len(norm(t))>=20 for t in doctors))
  medical+=bool(doctors and any(re.search(r"症状|检查|诊断|疾病|治疗|药|医院|服用|疼痛|建议|感染|炎症|血压|手术",t) for t in doctors))
  ends_doctor+=bool(turns and turns[-1][0]=="doctor")
  key=hashlib.sha256(json.dumps([(r,norm(t) if isinstance(t,str) else '') for r,t in turns],ensure_ascii=False).encode()).digest();dup+=key in seen;seen.add(key)
  overlap+=any(hashlib.sha256(norm(t).encode()).digest() in refs for r,t in turns if isinstance(t,str) and len(norm(t))>=10)

 n=len(docs);hist[name]=dict(sorted(collections.Counter(lens).items()));rows.append({'dataset':name,'observed_records':n,'patient_and_doctor':valid,'multi_dialogues_ge_2_each':multi,'turn_median':statistics.median(lens),'turn_p90':sorted(lens)[min(n-1,int(n*.9))],'chinese_any_cjk':zh,'exact_duplicate_dialogues':dup,'structural_anomalies':anomaly,'doctor_reply_ge_20_chars':substantive,'doctor_medical_keyword_heuristic':medical,'last_role_doctor':ends_doctor,'exact_utterance_overlap_atpo_public_pilot':overlap if refs else 'UNAVAILABLE','roles':json.dumps(dict(role),ensure_ascii=False)})
out=P/'DATASET_SAMPLE_STATISTICS.csv'
with out.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
(P/'turn_distributions.json').write_text(json.dumps(hist,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'atpo_public_reference_unique_strings':len(refs),'rows':rows},ensure_ascii=False,indent=2))
