#!/usr/bin/env python3
"""Finish partial public-source sampling; never read a pickle or log dialogue text."""
import hashlib,json,os,pathlib,time,requests
P=pathlib.Path(__file__).resolve().parents[1]; D=P/'private'; manifest=P/'download_manifest.json'
m=json.loads(manifest.read_text())
def save(name,b):
 p=D/name
 with p.open('xb') as f:f.write(b)
 os.chmod(p,0o600)
 return {'path':str(p.relative_to(P)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def get(url,**kwargs):
 for attempt in range(4):
  try:
   r=requests.get(url,timeout=90,**kwargs);r.raise_for_status();return r
  except requests.RequestException:
   if attempt==3:raise
   time.sleep(2**attempt)
try:
 url='https://datasets-server.huggingface.co/rows';params={'dataset':'Flmc/DISC-Med-SFT','config':'default','split':'train','offset':0,'length':100}
 r=get(url,params=params).json();total=r['num_rows_total']; rows=r['rows']; offsets=[0]
 for off in [max(0,total-100)]:
  try:params['offset']=off;rows+=get(url,params=params).json()['rows'];offsets.append(off)
  except Exception:pass
 b=('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n').encode()
 m['datasets']['disc']={'status':'downloaded','repo':params['dataset'],'revision_observed':get('https://huggingface.co/api/datasets/'+params['dataset']).json()['sha'],'snapshot_revision_proven':False,'total_reported':total,'offsets':offsets,'sample_rows':len(rows),**save('disc_sample.jsonl',b)}
except Exception as e:m['datasets']['disc']['retry_error']=str(e)[:300]
try:
 url='https://drive.usercontent.google.com/download';r=get(url,params={'id':'1PXwM1P1Cz3ajEVB9GpO6KBDG2U5bWOtZ','export':'download','confirm':'t'})
 if 'octet-stream' not in r.headers.get('content-type',''):raise ValueError('unexpected type')
 m['datasets']['meddg_dialogues']={'status':'downloaded_uninspected_pickle','source':'official MedDG train.pk','url':url,'file_id':'1PXwM1P1Cz3ajEVB9GpO6KBDG2U5bWOtZ',**save('meddg_train.pk',r.content)}
except Exception as e:m['datasets']['meddg_dialogues']={'status':'blocked','error':str(e)[:300]}
manifest.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
print('DISC and MedDG retry complete; run download_meddialog_sample.py separately')
