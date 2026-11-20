#!/usr/bin/env python3
"""Stream a bounded prefix from the official Chinese train array; no full corpus download."""
import codecs,hashlib,json,os,pathlib,requests
p=pathlib.Path(__file__).resolve().parents[1]; dest=p/'private/meddialog_sample.jsonl'; manifest=p/'download_manifest.json'
if dest.exists():raise FileExistsError(dest)
r=requests.get('https://drive.usercontent.google.com/download',params={'id':'1AaDJoHaiHAwEZwtskRH8oL1UP4FRgmgx','export':'download','confirm':'t'},stream=True,timeout=90);r.raise_for_status()
utf=codecs.getincrementaldecoder('utf-8')();dec=json.JSONDecoder();buf='';rows=[];started=False;read=0
for chunk in r.iter_content(chunk_size=262144):
 read+=len(chunk);buf+=utf.decode(chunk)
 if not started:
  i=buf.find('[')
  if i<0:continue
  buf=buf[i+1:];started=True
 while len(rows)<200:
  buf=buf.lstrip(' \r\n\t,')
  try:obj,end=dec.raw_decode(buf)
  except json.JSONDecodeError:break
  rows.append(obj);buf=buf[end:]
 if len(rows)==200:break
 if read>50000000:raise ValueError('prefix exceeded 50 MB')
r.close()
if len(rows)<100:raise ValueError('fewer than 100 rows')
b=('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n').encode()
with dest.open('xb') as f:f.write(b)
os.chmod(dest,0o600)
m=json.loads(manifest.read_text());m['datasets']['meddialog']={'status':'downloaded_prefix_sample','source':'UCSD26 official zh train','source_url':'https://drive.google.com/uc?export=download&id=1AaDJoHaiHAwEZwtskRH8oL1UP4FRgmgx','source_bytes':int(r.headers.get('Content-Length','0')),'stream_bytes_read':read,'sample_rows':len(rows),'sampling':'first 200 array items, not random','path':str(dest.relative_to(p)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};manifest.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
print('sample rows',len(rows),'stream bytes',read)
