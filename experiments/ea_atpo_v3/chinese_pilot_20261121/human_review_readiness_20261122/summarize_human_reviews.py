#!/usr/bin/env python3
"""Aggregate real completed human forms locally, refusing blank or unattested reviews."""
import argparse,collections,csv,json,math,os,pathlib,random
DIMS=('medical_correctness_1_5','completeness_1_5','safety_1_5')
BOOL={'是':'yes','否':'no','无法判断':'unclear','yes':'yes','no':'no','unclear':'unclear'}
def paired_stats(pairs,seed=20261122,bootstrap=10000):
    if not pairs:return {'n':0,'answer_mean':None,'ask_mean':None,'mean_difference':None,'ci95':None,'improved':0,'worsened':0,'unchanged':0}
    dif=[b-a for a,b in pairs];n=len(pairs);ci=None
    if n>=2:
        r=random.Random(seed);samples=sorted(sum(dif[r.randrange(n)] for _ in range(n))/n for _ in range(bootstrap));ci=[samples[int(.025*bootstrap)],samples[int(.975*bootstrap)-1]]
    return {'n':n,'answer_mean':sum(a for a,b in pairs)/n,'ask_mean':sum(b for a,b in pairs)/n,'mean_difference':sum(dif)/n,'ci95':ci,'improved':sum(x>0 for x in dif),'worsened':sum(x<0 for x in dif),'unchanged':sum(x==0 for x in dif)}
def score(v):
    v=v.strip()
    if v in ('无法判断','unclear'):return None
    if v in ('1','2','3','4','5'):return int(v)
    raise ValueError('each score must be 1-5 or 无法判断')
def yesno(v):
    try:return BOOL[v.strip().lower()]
    except KeyError:raise ValueError('each paired/safety field must be 是/否/无法判断')
def read_csv(p):
    with p.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def main(a):
    rows=read_csv(a.scores);pairs=read_csv(a.pairs)
    if len(rows)!=40 or len(pairs)!=20:raise ValueError('expected 40 response rows and 20 paired case rows')
    if not a.attestation:raise ValueError('PENDING: independent human attestation required; no scoring or unblinding')
    att=json.loads(a.attestation.read_text());required=('human_review_completed','reviewers_qualified','privacy_and_rights_cleared','rubric_unchanged')
    if any(att.get(k) is not True for k in required):raise ValueError('PENDING: human, privacy, rights and rubric attestation required')
    # Validate all entries before opening the blind key.
    for row in rows:
        if not row.get('reviewer_id','').strip() or not row.get('reviewed_at','').strip():raise ValueError('PENDING: reviewer ID and review time required')
        for d in DIMS:score(row.get(d,''))
        yesno(row.get('serious_error_yes_no_unclear',''))
    for row in pairs:
        if not row.get('reviewer_id','').strip() or not row.get('reviewed_at','').strip():raise ValueError('PENDING: paired reviewer ID and time required')
        yesno(row.get('added_information_useful_yes_no_unclear',''));yesno(row.get('diagnosis_or_advice_changed_yes_no_unclear',''))
    by=collections.defaultdict(dict)
    for row in rows:
        case=row['case_id'];opt=row['option']
        if opt not in ('A','B') or opt in by[case]:raise ValueError('duplicate or invalid case/option')
        by[case][opt]=row
    pairby={r['case_id']:r for r in pairs}
    if len(by)!=20 or len(pairby)!=20 or set(by)!=set(pairby) or any(set(v)!={'A','B'} for v in by.values()):raise ValueError('unpaired, duplicate or incomplete cases')
    # A single qualified reviewer per pair avoids silently mixing incompatible scores.
    for cid in by:
        if len({by[cid]['A']['reviewer_id'],by[cid]['B']['reviewer_id'],pairby[cid]['reviewer_id']})!=1:raise ValueError('mismatched reviewers within a case')
    mapping=[json.loads(l) for l in a.case_map.open(encoding='utf8')];keys=[json.loads(l) for l in a.blind_key.open(encoding='utf8')];maps={x['case_id']:x['previous_review_id'] for x in mapping};keyby={x['review_case']:x['answer_order'] for x in keys}
    if len(maps)!=20 or len(keyby)!=20 or set(maps)!=set(by) or len(set(maps.values()))!=20 or set(maps.values())!=set(keyby):raise ValueError('case-map mismatch')
    data=[]
    for cid in by:
        order=keyby[maps[cid]]
        if set(order)!={'always_answer','historical_replay_ask'} or len(order)!=2:raise ValueError('invalid blind key')
        answer=by[cid]['A' if order[0]=='always_answer' else 'B'];ask=by[cid]['A' if order[0]=='historical_replay_ask' else 'B'];data.append((answer,ask,pairby[cid]))
    out={'n_cases':20,'human_attestation_checked':True,'privacy_note':'attestation is not independent verification of reviewer identity or clinical expertise','paired_bootstrap':'10000 case resamples; percentile 95%; exploratory','dimension_results':{},'unscored_excluded_by_dimension':{},'safety_counts':{},'paired_findings':{},'utility':None,'utility_reason':'no pre-registered normalization, weights or safety penalty supplied'}
    for d in DIMS:
        valid=[(score(x[d]),score(y[d])) for x,y,_ in data];valid=[x for x in valid if x[0] is not None and x[1] is not None];out['dimension_results'][d]=paired_stats(valid);out['unscored_excluded_by_dimension'][d]=20-len(valid)
    for arm,i in [('answer',0),('ask',1)]:out['safety_counts'][arm]=dict(collections.Counter(yesno(x[i]['serious_error_yes_no_unclear']) for x in data))
    for field in ('added_information_useful_yes_no_unclear','diagnosis_or_advice_changed_yes_no_unclear'):
        cnt=collections.Counter(yesno(z[field]) for _,_,z in data);den=cnt['yes']+cnt['no'];observed=[(0, int(yesno(z[field])=='yes')) for _,_,z in data if yesno(z[field])!='unclear'];out['paired_findings'][field]={'yes':cnt['yes'],'no':cnt['no'],'unclear':cnt['unclear'],'yes_rate_among_judged':cnt['yes']/den if den else None,'ci95_case_bootstrap':paired_stats(observed)['ci95']}
    # Explicitly specified pre-unblinding policy; never invent utility weights or penalties.
    if a.utility_config:
        if att.get('utility_protocol_locked_before_unblinding') is not True:raise ValueError('utility policy not attested locked before unblinding')
        cfg=json.loads(a.utility_config.read_text());w=cfg['weights'];pen=cfg['serious_error_penalty'];
        if set(w)!=set(DIMS) or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in w.values()) or abs(sum(w.values())-1)>1e-9 or not isinstance(pen,(int,float)) or not math.isfinite(pen) or pen<0:raise ValueError('invalid utility policy')
        vals=[]
        for x,y,_ in data:
            sx=[score(x[d]) for d in DIMS];sy=[score(y[d]) for d in DIMS];ex=yesno(x['serious_error_yes_no_unclear']);ey=yesno(y['serious_error_yes_no_unclear'])
            if None in sx or None in sy or 'unclear' in (ex,ey):continue
            qa=sum(w[d]*(v-1)/4 for d,v in zip(DIMS,sx))-(pen if ex=='yes' else 0);qb=sum(w[d]*(v-1)/4 for d,v in zip(DIMS,sy))-(pen if ey=='yes' else 0);vals.append((qa,qb))
        out['utility']={str(c):paired_stats([(x,y-c) for x,y in vals]) for c in (0.01,0.05,0.10)};out['utility_reason']=None;out['utility_policy']=cfg
    if 'private' not in a.out.parts:raise ValueError('output must be private until disclosure risk is reviewed')
    fd=os.open(a.out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print('Aggregate written privately; disclosure requires review.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scores',type=pathlib.Path,required=True);p.add_argument('--pairs',type=pathlib.Path,required=True);p.add_argument('--case-map',type=pathlib.Path,required=True);p.add_argument('--blind-key',type=pathlib.Path,required=True);p.add_argument('--attestation',type=pathlib.Path);p.add_argument('--utility-config',type=pathlib.Path);p.add_argument('--out',type=pathlib.Path,required=True)
    try:main(p.parse_args())
    except (ValueError,KeyError,FileNotFoundError) as e:raise SystemExit(str(e))
