#!/usr/bin/env python3
"""DDXPlus validation-only, 50 cases; synthetic-classification CPU smoke."""
import argparse, ast, collections, csv, io, json, pathlib, zipfile

class Patient:
    def __init__(self,row,ontology):
        self._ontology=ontology;self._hidden=collections.defaultdict(list)
        for item in ast.literal_eval(row['EVIDENCES']):
            key,sep,value=item.partition('_@_');self._hidden[key].append(value if sep else True)
        initial=row['INITIAL_EVIDENCE']; assert initial in self._hidden
        self._shown={initial:tuple(self._hidden[initial])}; self._trace=[{'event':'initial','evidence':initial,'answer':self._shown[initial],'cost':0}]
    def view(self):
        return {'observed':dict(self._shown),'trace':[dict(x) for x in self._trace]}
    def ask(self,key):
        if key in self._shown: answer=self._shown[key];cost=0;status='repeat'
        elif key not in self._ontology: answer='UNKNOWN';cost=0;status='unknown_code'
        elif key not in self._hidden: answer='UNKNOWN';self._shown[key]=answer;cost=1;status='unobserved'
        else:
            answer=tuple(self._hidden[key]);self._shown[key]=answer;cost=1;status='revealed'
        self._trace.append({'event':'ask','evidence':key,'answer':answer,'cost':cost,'status':status})
        return answer,cost

def oracle(base,gold,valid,protect=False,action='DEFER'):
    if base not in valid or gold not in valid: raise ValueError('label outside official ontology')
    if action == 'ANSWER': return base
    if action != 'DEFER': raise ValueError('invalid oracle action')
    return base if protect and base==gold else gold

def main(a):
    evidence=json.loads(a.evidences.read_text());onto={x['name']:x for x in evidence.values()}; assert len(onto)==len(evidence)
    conditions=json.loads(a.conditions.read_text()); labels=[x['condition_name'] for x in conditions.values()]; valid=set(labels);assert len(valid)==len(labels)
    icds=[x.get('icd10-id') for x in conditions.values()]
    n=0; results=collections.Counter(); prob=[]
    with zipfile.ZipFile(a.zip) as z:
        assert len(z.namelist())==1
        with z.open(z.namelist()[0]) as stream:
            for row in csv.DictReader(io.TextIOWrapper(stream,encoding='utf-8')):
                if n>=50: break
                gold=row['PATHOLOGY']; assert gold in valid
                differential=ast.literal_eval(row['DIFFERENTIAL_DIAGNOSIS']);assert all(label in valid and isinstance(p,(float,int)) and p>=0 for label,p in differential)
                prob.append(sum(p for _,p in differential));patient=Patient(row,onto);view=patient.view()
                assert len(view['observed'])==1 and row['INITIAL_EVIDENCE'] in view['observed']
                assert set(view)=={'observed','trace'} and all(k not in json.dumps(view) for k in ('PATHOLOGY','DIFFERENTIAL_DIAGNOSIS',gold))
                first=row['INITIAL_EVIDENCE'];_,c=patient.ask(first);assert c==0
                before=patient.view(); answer,c=patient.ask('NOT_AN_EVIDENCE');assert (answer,c)==('UNKNOWN',0) and patient.view()['observed']==before['observed']
                candidates=[k for k in patient._hidden if k!=first and k in onto]
                if candidates:
                    key=candidates[0];before=patient.view();ans,c=patient.ask(key); assert c==1 and key not in before['observed'] and key in patient.view()['observed'];assert patient.ask(key)==(ans,0);results['hidden_evidence_tested']+=1
                absent=next((k for k in onto if k not in patient._hidden),None)
                if absent:
                    assert patient.ask(absent)==('UNKNOWN',1) and patient.ask(absent)==('UNKNOWN',0);results['unrecorded_unknown_tested']+=1
                other=next(k for k in labels if k!=gold)
                assert oracle(gold,gold,valid)==gold and oracle(gold,gold,valid,True)==gold
                assert oracle(other,gold,valid)==gold and oracle(other,gold,valid,True)==gold
                assert oracle(other,gold,valid,False,'ANSWER')==other and oracle(other,gold,valid,True,'ANSWER')==other
                # Different surface aliases are not guessed; only ontology labels are accepted.
                try:oracle('not_an_official_label',gold,valid,True)
                except ValueError:pass
                else:raise AssertionError('unknown label accepted')
                n+=1
    assert n==50
    print(json.dumps({'cases':n,'ontology_evidences':len(onto),'ontology_conditions':len(valid),'icd10_present':sum(bool(x) for x in icds),'icd10_unique':len(set(x for x in icds if x)),'hidden_tests':dict(results),'differential_probability_sum_min':min(prob),'differential_probability_sum_max':max(prob),'oracle_A_B_boundaries':'PASS','router_view_isolation':'PASS','ask_repeats_unknown':'PASS','split':'official validate','note':'sample of 50; no estimates of full-dataset coverage'},ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',type=pathlib.Path,required=True);p.add_argument('--evidences',type=pathlib.Path,required=True);p.add_argument('--conditions',type=pathlib.Path,required=True);main(p.parse_args())
