#!/usr/bin/env python3
"""Fetch byte-identical author-linked archives into a new private directory."""
import hashlib,json,os,pathlib,requests
p=pathlib.Path(__file__).resolve().parents[1];d=p/'private';d.mkdir(mode=0o700,exist_ok=False);os.chmod(d,0o700)
manifest={}
for name,id in [('medqa','1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw'),('medmcqa','15VkJdq5eyWIkfb_aoD3oS8i4tScbHYky')]:
 u='https://drive.usercontent.google.com/download';r=requests.get(u,params={'id':id,'export':'download','confirm':'t'},stream=True,timeout=(30,180));r.raise_for_status();f=d/(name+'.zip');h=hashlib.sha256();n=0
 with f.open('xb') as o:
  os.chmod(f,0o600)
  for block in r.iter_content(1024*1024):o.write(block);h.update(block);n+=len(block)
 manifest[name]={'source_url':'https://drive.google.com/uc?export=download&id='+id,'bytes':n,'sha256':h.hexdigest(),'path':str(f.relative_to(p))}
 print(name,n,h.hexdigest(),flush=True)
(p/'data/official_archives_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
