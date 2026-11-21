#!/usr/bin/env python3
"""Re-audit previous 50 private raw outputs; export aggregate only."""
import argparse,collections,csv,hashlib,json,pathlib,sys
root=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'ddx_phase3_20261121'))
from ddx import load_meta,validation_rows,Simulator,format_initial,parse
p=argparse.ArgumentParser();p.add_argument('--base',type=pathlib.Path,required=True);p.add_argument('--ask',type=pathlib.Path,required=True);p.add_argument('--archive',type=pathlib.Path,required=True);p.add_argument('--evidences',type=pathlib.Path,required=True);p.add_argument('--conditions',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
onto,labels=load_meta(a.evidences,a.conditions);base=[json.loads(s) for s in a.base.read_text().splitlines()];ask={x['case_id']:x for x in map(json.loads,a.ask.read_text().splitlines())};raw=list(validation_rows(a.archive,50));assert len(base)==len(ask)==len(raw)==50
truth=collections.Counter();pred=collections.Counter();conf=collections.Counter();invalid=[];private=[]
for (cid,row),b in zip(raw,base):
 assert cid==b['case_id'] and b['gold_pathology']==row['PATHOLOGY'] and ask[cid]['gold_pathology']==row['PATHOLOGY']
 sim=Simulator(row,onto);assert sim.view()['observed']==b['initial_evidence'];askrow=ask[cid];trace=askrow['ask_trace'];assert trace[0]==sim.view()['trace'][0];response=sim.ask(trace[1]['evidence']);assert sim.view()['trace']==trace
 prompt=format_initial({'observed':b['initial_evidence'],'trace':[]},onto,labels);assert 'PATHOLOGY' not in prompt and 'DIFFERENTIAL_DIAGNOSIS' not in prompt
 prediction,status=parse(b['raw_output'],labels);assert status==b['parsing_status'] and prediction==b['base_prediction'] and b['base_correct']==(prediction==row['PATHOLOGY'])
 gold=row['PATHOLOGY'];truth[gold]+=1;pred[prediction if prediction else 'UNPARSEABLE']+=1;conf[gold,prediction if prediction else 'UNPARSEABLE']+=1
 if status!='exact':invalid.append({'case_id':cid,'raw_output':b['raw_output'],'status':status})
 private.append({'case_id':cid,'gold_pathology':gold,'base_prediction':prediction,'raw_output':b['raw_output'],'parsing_status':status,'base_correct':b['base_correct']})
# Verify all canonical labels pass through exactly; this is a unit test, not clinical parsing accuracy.
assert all(parse(v,labels)==(v,'exact') for v in labels);assert parse('not_an_official_label',labels)==(None,'invalid')
columns=labels+['UNPARSEABLE'];a.out.mkdir(exist_ok=True)
with (a.out/'confusion_matrix.csv').open('x',newline='') as f:
 w=csv.writer(f,lineterminator="\n");w.writerow(['official_ground_truth']+columns)
 for label in labels:w.writerow([label]+[conf[label,c] for c in columns])
per={}
for lab in labels:
 tp=conf[lab,lab];fp=pred[lab]-tp;fn=truth[lab]-tp;per[lab]={'support':truth[lab],'predicted':pred[lab],'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn) if tp+fn else 0,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}
summary={'n':50,'strict_accuracy':sum(x['base_correct'] for x in base)/50,'raw_output_nonempty':sum(bool(x['raw_output'].strip()) for x in base),'canonical_exact_parse_count':50-len(invalid),'label_parser_accuracy':'not measurable without independent semantic annotation of output intentions','canonical_parser_unit_test':'49/49 exact labels accepted; invalid token rejected','macro_f1_over_49_official_classes':sum(x['f1'] for x in per.values())/49,'macro_f1_over_supported_classes':sum(x['f1'] for x in per.values() if x['support'])/sum(bool(x['support']) for x in per.values()),'unparseable_count':len(invalid),'patient_simulator_replay':'50/50 traces and response values match stored simulator; not independent validation of official dependency semantics','sample_identity':'50/50 row locators and official gold agree with original validation archive','per_class':per}
(a.out/'classification_report.json').write_text(json.dumps(summary,indent=2)+'\n')
with (a.out/'private'/'baseline_results.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(private[0]));w.writeheader();w.writerows(private)
(a.out/'private'/'invalid_outputs.json').write_text(json.dumps(invalid,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='per_class'},indent=2))
