#!/usr/bin/env python3
"""Preselect public-source IDs without labels or holdout reads; candidates remain unapproved."""
import ast,csv,hashlib,json,pathlib,re,unicodedata,zipfile
P=pathlib.Path(__file__).resolve().parents[1]; ROOT=P.parents[2];Z=P/'private/medmcqa.zip'; HISTORY=ROOT/'experiments/real_case_pilot_20261103/outputs/full/pilot_public_states.jsonl'
def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(s))).strip().lower()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def optkey(v):return digest(json.dumps(sorted(norm(x) for x in v),ensure_ascii=False))
histq=set();histo=set(); cases=set()
for line in HISTORY.open():
 x=json.loads(line);cases.add(x['case_id'])
 for m in x.get('initial_messages',[]):
  if m.get('role')!='user':continue
  s=m.get('content','');histq.add(digest(norm(s.split('Options:',1)[0])))
  for line2 in s.splitlines():
   if line2.startswith('Options:'):
    try:histo.add(optkey(ast.literal_eval(line2.split(':',1)[1].strip()).values()))
    except (ValueError,SyntaxError,AttributeError):pass
candidates=[];seenq=set();seeno=set();counts={'official_rows':0,'historic_exact_question_or_options_excluded':0,'official_internal_duplicates_excluded':0,'invalid_shape_excluded':0}
with zipfile.ZipFile(Z) as z:
 for line in z.open('dev.json'):
  x=json.loads(line);counts['official_rows']+=1;q=norm(x.get('question',''));choices=[x.get('op'+a) for a in 'abcd'];ident=x.get('id')
  if not q or not ident or any(not isinstance(v,str) or not v.strip() for v in choices):counts['invalid_shape_excluded']+=1;continue
  qh=digest(q);oh=optkey(choices)
  if qh in histq or oh in histo:counts['historic_exact_question_or_options_excluded']+=1;continue
  if qh in seenq or oh in seeno:counts['official_internal_duplicates_excluded']+=1;continue
  seenq.add(qh);seeno.add(oh);candidates.append({'official_dataset':'MedMCQA','official_split':'dev','official_question_id':ident,'question_sha256':qh,'unordered_options_sha256':oh,'selection_rank_hash':digest('ea-atpo-oracle-independent-20261121:'+ident),'status':'CANDIDATE_NOT_EVALUATED'})
candidates.sort(key=lambda x:x['selection_rank_hash']);selected=candidates[:100]
if len(selected)<100:raise ValueError('fewer than 100 candidates')
out=P/'data/independent_candidate_ids.csv'
with out.open('x',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=selected[0].keys(),lineterminator='\n');w.writeheader();w.writerows(selected)
manifest={'status':'CANDIDATE_NOT_CERTIFIED','sample_n':len(selected),'selection':'first 100 IDs sorted by SHA256 of fixed namespace and ID; no answer labels or model outcomes used','source':'MedMCQA author archive dev.json','archive_sha256':hashlib.sha256(Z.read_bytes()).hexdigest(),'history_checked':'291 public states from 100 prior pilot cases; exact question-prefix or unordered 4-option hash only; frozen holdout not opened','history_case_count':len(cases),'history_unique_question_hashes':len(histq),'history_unique_option_hashes':len(histo),'counts':counts,'candidate_csv_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'limitations':['No complete cross-worktree history or paraphrase review','No comparison to frozen holdout','No Router scores, ASK counterfactuals, or Expert predictions','Official dev split is source benchmark dev, not this project router dev','Source author archive dev has answers but they were not used in candidate selection','Does not establish absence from model pretraining']}
(P/'data/independent_candidate_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
