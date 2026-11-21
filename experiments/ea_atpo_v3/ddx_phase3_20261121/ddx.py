"""DDXPlus validation-only state, evaluation, and prompt boundaries."""
import ast,collections,csv,hashlib,io,json,zipfile

def load_meta(evidence_path,conditions_path):
    evidence={v['name']:v for v in json.loads(evidence_path.read_text()).values()}
    conditions=json.loads(conditions_path.read_text()); labels=sorted(x['condition_name'] for x in conditions.values())
    assert len(labels)==len(set(labels)) and len(evidence)==223
    return evidence,labels

def validation_rows(path,limit=None):
    # No split switching: this deliberately accepts only the pinned official validation archive.
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!='625a96855becf0efe5da339fd3d440aa99597d48f784957a97dacc3fceaa7e40': raise ValueError('unexpected archive or split')
    with zipfile.ZipFile(path) as z:
        if z.namelist()!=['release_validate_patients']: raise ValueError('unexpected member')
        with z.open(z.namelist()[0]) as f:
            for i,row in enumerate(csv.DictReader(io.TextIOWrapper(f,encoding='utf8'))):
                if limit is not None and i>=limit: break
                # locator, NOT patient ID; salted with source archive hash, not patient contents
                yield f'validate:{digest[:16]}:{i}',row

class Simulator:
    def __init__(self,row,ontology):
        self.ontology=ontology;self._facts=collections.defaultdict(list);self._visible={};self._trace=[]
        for raw in ast.literal_eval(row['EVIDENCES']):
            key,sep,v=raw.partition('_@_')
            if key not in ontology: raise ValueError('evidence outside ontology')
            self._facts[key].append(v if sep else True)
        initial=row['INITIAL_EVIDENCE']
        if initial not in self._facts or ontology[initial]['data_type']!='B': raise ValueError('invalid initial evidence')
        self._visible[initial]=True;self._trace.append({'action':'INITIAL','evidence':initial,'answer':True,'cost_units':0})
    def view(self):
        return {'observed':dict(self._visible),'trace':[dict(x) for x in self._trace]}
    def ask(self,key):
        if key not in self.ontology: raise ValueError('unknown evidence')
        if key in self._visible: raise ValueError('already known')
        meta=self.ontology[key];root=meta['code_question']
        # Conservative gate based on documented dependencies, not claimed exact official environment behavior.
        if root!=key and (root not in self._visible or self._visible[root] is False): raise ValueError('prerequisite not met')
        typ=meta['data_type'];values=self._facts.get(key,[])
        if typ=='B': ans=bool(values)
        elif typ=='C':
            if len(values)>1: raise ValueError('multiple categorical values')
            ans=values[0] if values else str(meta['default_value'])
        elif typ=='M': ans=tuple(values) if values else (str(meta['default_value']),)
        else: raise ValueError('unknown type')
        self._visible[key]=ans;self._trace.append({'action':'ASK','evidence':key,'answer':ans,'cost_units':1});return ans

def format_initial(view,ontology,labels):
    # Only the initial/previously observed facts and fixed ontology can enter the model.
    facts=[]
    for key,val in view['observed'].items():
        meta=ontology[key]; vals=val if isinstance(val,tuple) else (val,)
        rendered=[meta.get('value_meaning',{}).get(str(v),{}).get('en',str(v)) for v in vals]
        facts.append(meta['question_en']+' '+', '.join(rendered))
    return 'Choose exactly one diagnosis from the following list. Return exactly its name, with no explanation.\nDiagnoses: '+ ' | '.join(labels)+'\nObserved evidence: '+ ' ; '.join(facts)

def parse(output,labels):
    # No fuzzy matching or implicit label inference. Ambiguity stays invalid.
    text=output.strip()
    matches=[x for x in labels if x.casefold()==text.casefold()]
    return (matches[0],'exact') if len(matches)==1 else (None,'invalid')

def paired(base,gold,labels):
    if gold not in labels: raise ValueError('unknown official gold')
    # Invalid Base output is retained as invalid, never filled with a guessed diagnosis.
    correct=base is not None and base==gold
    return {'base_correct':correct,'answer':base,'oracle_a_defer':gold,'oracle_b_defer':base if correct else gold}
