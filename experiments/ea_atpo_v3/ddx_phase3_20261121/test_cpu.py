import argparse,ast,collections,json,pathlib
from ddx import load_meta,validation_rows,Simulator,format_initial,parse,paired

def main(a):
    onto,labels=load_meta(a.evidences,a.conditions);stats=collections.Counter();ids=set()
    for cid,row in validation_rows(a.zip,50):
        assert cid not in ids;ids.add(cid);assert row['PATHOLOGY'] in labels
        facts=collections.defaultdict(list)
        for raw in ast.literal_eval(row['EVIDENCES']):
            key,sep,v=raw.partition('_@_');facts[key].append(v if sep else True)
        sim=Simulator(row,onto);view=sim.view();assert len(view['observed'])==1
        prompt=format_initial(view,onto,labels);assert all(k not in prompt for k in ('PATHOLOGY','DIFFERENTIAL_DIAGNOSIS'))
        init=row['INITIAL_EVIDENCE'];
        try:sim.ask(init)
        except ValueError:stats['duplicate_rejected']+=1
        else:raise AssertionError('repeated ASK')
        before=sim.view();
        try:sim.ask('INVALID_ID')
        except ValueError:stats['unknown_rejected']+=1
        else:raise AssertionError('unknown evidence accepted')
        assert before==sim.view()
        for key in sorted(onto,key=lambda k:onto[k]['code_question'] != k):
            meta=onto[key];root=meta['code_question']
            if key in sim.view()['observed']:continue
            if root!=key and (root not in sim.view()['observed'] or sim.view()['observed'][root] is False):
                try:sim.ask(key)
                except ValueError:stats['dependency_rejected']+=1;continue
                else:raise AssertionError('dependency accepted')
            old=sim.view();answer=sim.ask(key)
            assert set(sim.view()['observed'])==set(old['observed'])|{key}
            v=facts.get(key,[]);typ=meta['data_type']
            expected=bool(v) if typ=='B' else (v[0] if v else str(meta['default_value'])) if typ=='C' else tuple(v) if v else (str(meta['default_value']),)
            assert answer==expected and sim.view()['trace'][-1]['cost_units']==1
            stats['asked_'+typ]+=1
        gold=row['PATHOLOGY'];wrong=next(x for x in labels if x!=gold)
        assert parse(gold,labels)==(gold,'exact') and parse('not a diagnosis',labels)==(None,'invalid')
        good=paired(gold,gold,labels);bad=paired(wrong,gold,labels)
        assert good['answer']==gold and good['oracle_b_defer']==gold and bad['answer']==wrong and bad['oracle_a_defer']==bad['oracle_b_defer']==gold
        stats['cases']+=1
    assert stats['cases']==50
    print(json.dumps({'result':'PASS','n':50,'counts':dict(stats),'case_id':'archive hash + row locator; not patient identifier','scope':'default/type semantics and independent evidence; dependency policy conservative; no actual Base predictions','cuda_visible_devices':'none'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',type=pathlib.Path,required=True);p.add_argument('--evidences',type=pathlib.Path,required=True);p.add_argument('--conditions',type=pathlib.Path,required=True);main(p.parse_args())
