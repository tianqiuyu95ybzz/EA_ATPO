#!/usr/bin/env python3
"""Check score presence only; this does not fabricate or infer ratings."""
import csv,json,pathlib
p=pathlib.Path(__file__).resolve().parent/'private';responses=list(csv.DictReader((p/'review_scores.csv').open(encoding='utf8')));pairs=list(csv.DictReader((p/'paired_scores.csv').open(encoding='utf8')))
assert len(responses)==40 and len(pairs)==20
assert len({r['case_id'] for r in responses})==20 and len({r['case_id'] for r in pairs})==20
assert all({r['option'] for r in responses if r['case_id']==x['case_id']}=={'A','B'} for x in pairs)
fields=('medical_correctness_1_5','completeness_1_5','safety_1_5','serious_error_yes_no_unclear')
complete=sum(all(r.get(f,'').strip() for f in fields) for r in responses)
pair_complete=sum(all(r.get(f,'').strip() for f in ('added_information_useful_yes_no_unclear','diagnosis_or_advice_changed_yes_no_unclear')) for r in pairs)
print(json.dumps({'responses':len(responses),'paired_cases':len(pairs),'fully_scored_responses':complete,'fully_scored_pair_fields':pair_complete,'status':'PENDING_HUMAN_REVIEW' if complete!=40 or pair_complete!=20 else 'SCORES_PRESENT_REQUIRE_VALIDATION','clinical_quality_inferred':False},indent=2))
