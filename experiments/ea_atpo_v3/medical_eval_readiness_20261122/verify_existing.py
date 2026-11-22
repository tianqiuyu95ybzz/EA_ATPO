#!/usr/bin/env python3
"""Read-only integrity verification; prints aggregate hashes, never medical text."""
import pathlib,json,hashlib,csv,argparse,os
parser=argparse.ArgumentParser();parser.add_argument("--private-digests",type=pathlib.Path);args=parser.parse_args()
p=pathlib.Path(__file__).resolve().parent.parent/'chinese_pilot_20261121';priv=p/'review_privacy_20261122/private';old=p/'quality_audit_20261121/private'
read=lambda x:[json.loads(l) for l in x.open(encoding='utf8')]
original=read(old/'review_packets.jsonl');restricted=read(priv/'restricted_review_packets.jsonl');mapping=read(priv/'private_case_mapping.jsonl');mp={x['case_id']:x['previous_review_id'] for x in mapping};orig={x['review_case']:x for x in original};assert len(orig)==len(restricted)==len(mp)==20 and len(set(mp.values()))==20
pairs=[];individual=[]
for row in restricted:
 o=orig[mp[row['case_id']]];assert len(row['variants'])==len(o['variants'])==2
 for a,b in zip(row['variants'],o['variants']):
  assert a['option']==b['option'] and a['answer']==b['answer'] and a['answer'].strip();digest=hashlib.sha256(a['answer'].encode()).hexdigest();pairs.append(digest);individual.append({'review_id':row['case_id'],'blind_option':a['option'],'sha256':digest})
assert len(pairs)==40
with (priv/'review_scores.csv').open(encoding='utf8',newline='') as f:r=list(csv.DictReader(f))
with (priv/'paired_scores.csv').open(encoding='utf8',newline='') as f:q=list(csv.DictReader(f))
assert len(r)==40 and len(q)==20
filled=sum(bool(x.get(k,'').strip()) for x in r for k in ('medical_correctness_1_5','completeness_1_5','safety_1_5','serious_error_yes_no_unclear'))+sum(bool(x.get(k,'').strip()) for x in q for k in ('added_information_useful_yes_no_unclear','diagnosis_or_advice_changed_yes_no_unclear'))
prev=json.loads((p/'review_privacy_20261122/run_manifest.json').read_text());h=lambda x:hashlib.sha256(x.read_bytes()).hexdigest();integrity={k:h(priv/name)==prev[k] for k,name in [('restricted_packets_sha256','restricted_review_packets.jsonl'),('response_form_sha256','review_scores.csv'),('paired_form_sha256','paired_scores.csv')]}
full=json.loads((p/'run_manifest.json').read_text());pilot_hash_ok=h(p/'private/model_trajectories_retry.jsonl')==full['private_output_sha256']
result={'review_cases':20,'paired_responses':len(pairs),'each_output_equals_original':True,'output_hashes_sha256_of_sorted_multiset':hashlib.sha256(('\n'.join(sorted(pairs))+'\n').encode()).hexdigest(),'review_files_match_prior_hashes':integrity,'pilot_output_file_hash_matches':pilot_hash_ok,'score_fields_filled':filled,'review_status':'PENDING' if filled==0 else 'NONEMPTY_REQUIRES_HUMAN_VALIDATION','defer':'N/A, no qualified matched full cohort','no_raw_text_published':True}
assert all(integrity.values()) and pilot_hash_ok
if args.private_digests:
 assert 'private' in args.private_digests.parts
 fd=os.open(args.private_digests,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w') as f:
  for item in individual:f.write(json.dumps(item)+'\n')
print(json.dumps(result,indent=2))
