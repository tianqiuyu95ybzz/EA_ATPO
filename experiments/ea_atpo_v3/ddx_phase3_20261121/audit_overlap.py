#!/usr/bin/env python3
"""Count exact source-row overlap; do not expose labels or open held-out test."""
import argparse,collections,csv,hashlib,io,json,pathlib,zipfile
EXPECTED={'train':'174ae1d56f36a15b7144838ecd214e1e1748a5369fc90c559529b6dc6ecfb218','validate':'625a96855becf0efe5da339fd3d440aa99597d48f784957a97dacc3fceaa7e40'}
def rows(p,kind):
    if hashlib.sha256(p.read_bytes()).hexdigest()!=EXPECTED[kind]:raise ValueError('unexpected archive')
    with zipfile.ZipFile(p) as z:
        if z.namelist()!=['release_'+kind+'_patients']:raise ValueError('unexpected member')
        with z.open(z.namelist()[0]) as f:
            for r in csv.DictReader(io.TextIOWrapper(f,encoding='utf8')):
                yield hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).digest()
p=argparse.ArgumentParser();p.add_argument('--train',type=pathlib.Path,required=True);p.add_argument('--validate',type=pathlib.Path,required=True);a=p.parse_args()
first50=[]
def validation_stream():
    for h in rows(a.validate,'validate'):
        if len(first50)<50:first50.append(h)
        yield h
v=collections.Counter(validation_stream());t=collections.Counter(rows(a.train,'train'));common=v.keys() & t.keys()
print(json.dumps({'train_rows':sum(t.values()),'validation_rows':sum(v.values()),'train_exact_row_duplicates':sum(n-1 for n in t.values()),'validation_exact_row_duplicates':sum(n-1 for n in v.values()),'first_50_validation_rows_matching_train':sum(h in t for h in first50),'distinct_exact_row_values_shared':len(common),'train_rows_in_cross_split_matches':sum(t[k] for k in common),'validation_rows_in_cross_split_matches':sum(v[k] for k in common),'patient_id_available':False,'semantic_or_patient_level_overlap':'unknown','test_opened':False,'train_archive_sha256':EXPECTED['train'],'validation_archive_sha256':EXPECTED['validate']},indent=2))
