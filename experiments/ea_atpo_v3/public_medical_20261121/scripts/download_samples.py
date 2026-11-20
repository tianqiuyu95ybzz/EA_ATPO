#!/usr/bin/env python3
"""Download public sample artifacts to a private, immutable PFS run directory."""
import argparse, hashlib, json, os, pathlib, requests

REMEDI='f94bb9abfba48f467669d2454771bbe263cc4ae5'
DATASETS={'disc':('Flmc/DISC-Med-SFT','b10eedfc23530ebe257a149b00db73c4162fbfc3'), 'aquila':('BAAI/AquilaMed-Instruct','7f21bf31bdbc5ce991ce6fec85f82bba42a43fdb')}

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
 if a.out.name!='private' or a.out.exists():raise ValueError('use a new private directory')
 a.out.mkdir(parents=True,mode=0o700);os.chmod(a.out,0o700)
 manifest={'datasets':{},'note':'sample text remains private; HF rows endpoint is a snapshot, not a revision-pinned API'}
 def save(name,content):
  f=a.out/name
  with f.open('xb') as h:h.write(content)
  os.chmod(f,0o600)
  return {'path':str(f),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
 url=f'https://raw.githubusercontent.com/yanguojun123/Medical-Dialogue/{REMEDI}/data/ReMeDi-base.json'
 try:
  r=requests.get(url,timeout=90);r.raise_for_status();json.loads(r.content)
  manifest['datasets']['remedi']={'status':'downloaded','revision':REMEDI,'url':url,**save('remedi_base.json',r.content)}
 except Exception as e:manifest['datasets']['remedi']={'status':'blocked','error':str(e)[:300],'url':url}
 for key,(repo,revision) in DATASETS.items():
  try:
   url='https://datasets-server.huggingface.co/rows'; params={'dataset':repo,'config':'default','split':'train','offset':0,'length':1}
   r=requests.get(url,params=params,timeout=45);r.raise_for_status();total=r.json()['num_rows_total']
   offsets=sorted(set(max(0,min(total-100,int(total*f))) for f in (0,.25,.5,.75,1)))
   records=[]
   for offset in offsets:
    params.update(offset=offset,length=100);r=requests.get(url,params=params,timeout=60);r.raise_for_status();records.extend(r.json()['rows'])
   content=('\n'.join(json.dumps(x,ensure_ascii=False) for x in records)+'\n').encode()
   meta=requests.get('https://huggingface.co/api/datasets/'+repo,timeout=30);meta.raise_for_status(); observed=meta.json()['sha']
   manifest['datasets'][key]={'status':'downloaded','repo':repo,'revision_observed':observed,'revision_expected':revision,'snapshot_revision_proven':False,'total_reported':total,'offsets':offsets,'sample_rows':len(records),**save(key+'_sample.jsonl',content)}
  except Exception as e:manifest['datasets'][key]={'status':'blocked','repo':repo,'error':str(e)[:300]}
 # Official MedDG folder's gen_train.pk: do not unpickle untrusted bytes during download.
 url='https://drive.google.com/uc?export=download&id=1u9lmFtmKqBoHz_jJfAAHt2IIoQdAm9pB'
 try:
  r=requests.get(url,timeout=90);r.raise_for_status()
  if 'text/html' in r.headers.get('content-type','').lower() or len(r.content)<1000:raise ValueError('Drive returned HTML/short response, not dataset bytes')
  manifest['datasets']['meddg']={'status':'downloaded_uninspected_pickle','url':url,**save('meddg_gen_train.pk',r.content)}
 except Exception as e:manifest['datasets']['meddg']={'status':'blocked','url':url,'error':str(e)[:300]}
 manifest['datasets']['meddialog']={'status':'blocked','url':'https://github.com/UCSD-AI4H/Medical-Dialogue-System','reason':'official repository returned 404; UCSD26 dataset viewer disabled and no data files in repository'}
 out=a.out.parent/'download_manifest.json'
 with out.open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({k:{x:v for x,v in d.items() if x in ('status','sample_rows','total_reported','error','reason')} for k,d in manifest['datasets'].items()},ensure_ascii=False))
if __name__=='__main__':main()
