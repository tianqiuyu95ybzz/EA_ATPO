#!/usr/bin/env python3
import hashlib,json,pathlib,sys
p=pathlib.Path(__file__).resolve().parents[1];m=json.loads((p/'download_manifest.json').read_text());bad=[]
for key,item in m['datasets'].items():
 if 'sha256' not in item:continue
 f=p/'private'/pathlib.Path(item['path']).name
 if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=item['sha256']:bad.append(key)
 print(key,'FAIL' if key in bad else 'OK')
sys.exit(bool(bad))
