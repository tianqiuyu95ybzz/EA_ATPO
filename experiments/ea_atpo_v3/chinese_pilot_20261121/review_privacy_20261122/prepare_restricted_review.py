#!/usr/bin/env python3
"""Private, preliminary redaction. Never assume automated scanning makes text anonymous."""
import collections,csv,hashlib,json,os,pathlib,re,secrets
p=pathlib.Path(__file__).resolve().parent;src=p.parent/'quality_audit_20261121/private';packets=[json.loads(l) for l in (src/'review_packets.jsonl').open()];assert len(packets)==20
patterns=[('phone',re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)|(?<!\d)\d{3,4}[-－]\d{7,8}(?!\d)'),'[联系方式已隐藏]'),('email',re.compile(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}'),'[联系方式已隐藏]'),('identity_number',re.compile(r'(?<!\d)\d{17}[\dXx](?!\d)'),'[证件号码已隐藏]'),('url',re.compile(r'https?://[^\s]+|www\.[^\s]+'),'[网址已隐藏]'),('calendar_date',re.compile(r'(?:19|20)\d{2}年\d{1,2}月(?:\d{1,2}[日号])?|\d{1,2}月\d{1,2}[日号]|(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}|[一二三四五六七八九十]{1,3}月份|中秋节'),'[具体日历时间已隐藏]')]
age=re.compile(r'(?<!\d)(\d{2,3})\s*(岁|周岁)'); age_paren=re.compile(r'([（(][男女]\s*[,，]\s*)(\d{2,3})(\s*[）)])');counts=collections.Counter()
def redact(text):
 for name,pat,repl in patterns:
  text,n=pat.subn(repl,text);counts[name]+=n
 def a(m):
  num=int(m.group(1))
  if num<18:return m.group()
  counts['adult_age_generalized']+=1;return '老年' if num>=60 else '成年'
 text=age.sub(a,text)
 def b(m):
  num=int(m.group(2))
  if num<18:return m.group()
  counts['adult_age_generalized']+=1;return m.group(1)+('老年' if num>=60 else '成年')+m.group(3)
 return age_paren.sub(b,text)
# Source reviewed locally: no actual personal names, contact details or named locations
# were identified; these patterns cannot prove that none remain.
output=[];mapping=[];scores=[];pairrows=[];ids=set()
for item in packets:
 rid=secrets.token_hex(12)
 while rid in ids:rid=secrets.token_hex(12)
 ids.add(rid);opts=[]
 for option in item['variants']:
  # Model outputs remain byte-for-byte identical. Fail closed if a direct-ID pattern hits.
  answer=option['answer']
  for name,pat,_ in patterns:
   if name!='calendar_date' and pat.search(answer):raise ValueError('possible direct identifier in model answer; manual review required')
  context=[{'role':t['role'],'text':redact(t['text'])} for t in option['visible_context']]
  opts.append({'option':option['option'],'visible_context':context,'answer':answer})
  scores.append({'case_id':rid,'option':option['option'],'medical_correctness_1_5':'','completeness_1_5':'','safety_1_5':'','serious_error_yes_no_unclear':'','reviewer_id':'','reviewed_at':'','notes':''})
 output.append({'case_id':rid,'variants':opts});mapping.append({'case_id':rid,'previous_review_id':item['review_case']});pairrows.append({'case_id':rid,'added_information_useful_yes_no_unclear':'','diagnosis_or_advice_changed_yes_no_unclear':'','comparison_notes':'','reviewer_id':'','reviewed_at':''})
def write_jsonl(name,items):
 fd=os.open(p/'private'/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w',encoding='utf8') as f:
  for x in items:f.write(json.dumps(x,ensure_ascii=False)+'\n')
def write_csv(name,items):
 fd=os.open(p/'private'/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(items[0]),lineterminator='\n');w.writeheader();w.writerows(items)
write_jsonl('restricted_review_packets.jsonl',output);write_jsonl('private_case_mapping.jsonl',mapping);write_csv('review_scores.csv',scores);write_csv('paired_scores.csv',pairrows)
# Check answer content exactly against source, despite masking clinical contexts.
assert all(o['answer']==v['answer'] for source,new in zip(packets,output) for v,o in zip(source['variants'],new['variants']))
manifest={'status':'PRELIMINARY_RESTRICTED_NOT_APPROVED_FOR_EXTERNAL_SHARING','cases':len(output),'paired_answers':len(scores),'outputs_unchanged':True,'direct_id_regex_hits_replaced':{k:counts[k] for k in ('phone','email','identity_number','url')},'adult_age_generalizations':counts['adult_age_generalized'],'calendar_time_redactions':counts['calendar_date'],'counts_note':'counts include repeated contexts between blinded options','indirect_risk':['pediatric age and sex retained for clinical context','diagnosis, exact wording, clinical history and relative time can identify an individual together','source dialogue text is publicly searchable and may be linked back to a source','redaction may remove clinically relevant information; comparison must be interpreted accordingly'],'release_gate':'prohibit external sharing until independent qualified privacy review and source rights confirmation; no sharing performed','human_scores':'PENDING','restricted_packets_sha256':hashlib.sha256((p/'private/restricted_review_packets.jsonl').read_bytes()).hexdigest()}
(p/'PRIVACY_AUDIT.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps(manifest,ensure_ascii=False,indent=2))
