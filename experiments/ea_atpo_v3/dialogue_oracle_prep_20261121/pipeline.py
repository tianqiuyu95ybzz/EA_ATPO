#!/usr/bin/env python3
"""Read source records locally; output is sensitive and must never be committed."""
import argparse, collections, hashlib, json, pathlib, pickle

class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError('global objects forbidden')

def load(dataset, path):
    if dataset == 'remedi':
        data = json.loads(path.read_text(encoding='utf-8'))
        for index, row in enumerate(data):
            yield index, str(row['dialogue']), [(t.get('role'), t.get('sentence')) for t in row['information']]
    elif dataset == 'meddg':
        with path.open('rb') as f: data = Restricted(f).load()
        for index, row in enumerate(data):
            yield index, None, [({'Patients':'patient','Doctor':'doctor'}.get(t.get('id')), t.get('Sentence')) for t in row]
    else: raise ValueError(dataset)

def normalize(dataset, index, source_id, turns):
    """One state per original doctor turn, history exclusive; no patient ID inferred."""
    split = 'unmapped' if dataset == 'remedi' else 'train'
    locator = str(source_id) if source_id is not None else 'train.pk:index:'+str(index)
    cid = dataset+':'+locator
    history = []
    for i, (role, text) in enumerate(turns):
        if role == 'doctor' and isinstance(text,str) and text.strip() and any(t['speaker']=='patient' for t in history):
            following = turns[i+1] if i+1 < len(turns) else (None,None)
            feedback = {'speaker':'patient','text':following[1],'source_turn_index':i+1} if following[0]=='patient' and isinstance(following[1],str) and following[1].strip() else None
            yield {'dataset':dataset,'case_id':cid,'conversation_id':cid,'source_record_index':index,'source_dialogue_id':source_id,'official_split':split,'dialogue_history':list(history),'speaker':role,'current_turn':i,'patient_feedback':feedback,'reference_answer':text,'reference_semantics':'observed_doctor_utterance_unverified','oracle_target':None,'gold_source':None,'gold_verified':False,'oracle_available':False}
        history.append({'speaker':role,'text':text,'source_turn_index':i})

def router_view(state):
    """Only the pre-decision context; never expose observed future or labels."""
    return {k: state[k] for k in ('dataset','case_id','conversation_id','official_split','dialogue_history','current_turn')}

def oracle(action, base, target=None, gold_verified=False, protecting=False, base_correct=None):
    if action == 'ANSWER': return base
    if action != 'DEFER': raise ValueError('only ANSWER and DEFER supported; ASK requires an observed transition')
    if not gold_verified or target is None: raise ValueError('verified Gold required')
    if protecting:
        if type(base_correct) is not bool: raise ValueError('post-hoc base correctness required')
        if base_correct: return base
    return target

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def summarize(dataset, path):
    count=collections.Counter(); seen={}; ids=collections.Counter(); feedback_cases=set(); multi=0; samples=0
    for index, sid, turns in load(dataset,path):
        count['raw_dialogues']+=1; ids[str(sid)]+=1
        valid=all(r in ('patient','doctor') and isinstance(t,str) and t.strip() for r,t in turns)
        fingerprint=hashlib.sha256(json.dumps(turns,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        if fingerprint in seen: count['exact_duplicate_records']+=1; continue
        seen[fingerprint]=index
        if not valid: count['structurally_invalid_unique']+=1; continue
        roles=collections.Counter(r for r,t in turns)
        if not roles['patient'] or not roles['doctor']: count['missing_role_unique']+=1; continue
        count['usable_unique_dialogues']+=1
        if roles['patient']>=2 and roles['doctor']>=2: multi+=1
        states=list(normalize(dataset,index,sid,turns)); samples+=len(states)
        if any(s['patient_feedback'] for s in states): feedback_cases.add(index)
        count['states_with_patient_feedback']+=sum(s['patient_feedback'] is not None for s in states)
    count['multi_turn_dialogues']=multi; count['doctor_turn_states']=samples; count['dialogues_with_observed_feedback']=len(feedback_cases)
    count['gold_verified_states']=0; count['oracle_available_states']=0
    count['duplicate_source_id_records']=sum(n-1 for n in ids.values() if n>1) if dataset=='remedi' else None
    return dict(count)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--remedi',type=pathlib.Path,required=True); ap.add_argument('--meddg',type=pathlib.Path,required=True); a=ap.parse_args()
    for ds,p in [('remedi',a.remedi),('meddg',a.meddg)]: print(json.dumps({'dataset':ds,'sha256':digest(p),'counts':summarize(ds,p)},ensure_ascii=False))
