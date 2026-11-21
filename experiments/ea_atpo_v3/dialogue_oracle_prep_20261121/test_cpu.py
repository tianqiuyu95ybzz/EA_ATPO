#!/usr/bin/env python3
import argparse, collections, hashlib, json, pathlib
from pipeline import load, normalize, oracle, summarize, router_view

def run(ds,path):
    n=0; state_n=0; seen=set(); ids=set(); duplicated_id=0
    for idx,sid,turns in load(ds,path):
        if not all(r in ('patient','doctor') and isinstance(t,str) and t.strip() for r,t in turns): continue
        if {r for r,t in turns}!={'patient','doctor'}: continue
        fingerprint=hashlib.sha256(json.dumps(turns,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        if fingerprint in seen: continue
        seen.add(fingerprint)
        if sid is not None and sid in ids: duplicated_id+=1; continue
        ids.add(sid)
        states=list(normalize(ds,idx,sid,turns))
        for s in states:
            i=s['current_turn']; h=s['dialogue_history']
            assert [(x['speaker'],x['text']) for x in h]==turns[:i]
            assert s['speaker']=='doctor' and s['reference_answer']==turns[i][1]
            assert s['oracle_target'] is None and not s['gold_verified'] and not s['oracle_available']
            assert not ({'patient_feedback','reference_answer','oracle_target','gold_source','gold_verified','oracle_available'} & router_view(s).keys())
            if s['patient_feedback'] is not None:
                assert s['patient_feedback']=={'speaker':'patient','text':turns[i+1][1],'source_turn_index':i+1} and turns[i+1][0]=='patient'
            state_n+=1
        n+=1
    assert n>=50, (ds,n)
    print(f'{ds}: PASS {n} unique structurally valid dialogue records, {state_n} states; chronological prefixes and immediate patient feedback checked; duplicate IDs excluded={duplicated_id}')
    print(ds, json.dumps(summarize(ds,path),sort_keys=True))

def bounds():
    assert oracle('ANSWER','base')=='base'
    assert oracle('DEFER','base','gold',True)== 'gold'
    assert oracle('DEFER','base','gold',True,True,True)=='base'
    assert oracle('DEFER','base','gold',True,True,False)=='gold'
    for args in [('DEFER','base'),('DEFER','base','gold',False),('DEFER','base','gold',True,True,None),('ASK','base')]:
        try: oracle(*args)
        except ValueError: pass
        else: raise AssertionError(args)
    print('Oracle boundary tests: PASS (synthetic fixtures only; correctness is post-hoc)')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--remedi',type=pathlib.Path,required=True);p.add_argument('--meddg',type=pathlib.Path,required=True);a=p.parse_args()
    bounds();run('remedi',a.remedi);run('meddg',a.meddg)
    print('Split check: INDETERMINATE: ReMeDi source split mapping and both datasets patient IDs unavailable; MedDG dev/test not locally present.')
