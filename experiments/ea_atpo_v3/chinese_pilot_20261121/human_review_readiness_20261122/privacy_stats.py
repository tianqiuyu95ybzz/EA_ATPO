#!/usr/bin/env python3
"""Local-only aggregate privacy risk indicators; never print underlying text."""
import collections,json,pathlib,re
p=pathlib.Path(__file__).resolve().parent;pilot=p.parent
review=pilot/'review_privacy_20261122/private/restricted_review_packets.jsonl'
source=pathlib.Path('/pfs/pfs-spMCeu/image_team/workspace/zjr/EA_ATPO_V1_public_medical_data_20261121/experiments/ea_atpo_v3/public_medical_20261121/private/remedi_base.json')
records=json.loads(source.read_text()); source_strings={t['sentence'].strip() for row in records for t in row['information'] if isinstance(t.get('sentence'),str) and len(t['sentence'].strip())>=12}
items=[json.loads(x) for x in review.open()];counts=collections.Counter()
patterns={'age_or_month_marker':r'\d{1,2}\s*(?:个月|月龄|岁)','relative_duration':r'(?:\d+|[一二三四五六七八九十两几]+)(?:天|月|年|周|个月)','treatment':r'药|治疗|手术|复查|检查','direct_contact':r'(?<!\d)1[3-9]\d{9}(?!\d)|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|https?://\S+'}
for item in items:
 texts=[t['text'] for v in item['variants'] for t in v['visible_context']];answers=[v['answer'] for v in item['variants']];joined='\n'.join(texts+answers)
 for name,regex in patterns.items():counts[name+'_case_count']+=bool(re.search(regex,joined))
 counts['verbatim_source_utterance_at_least_12_chars_case_count']+=any(t.strip() in source_strings for t in texts if len(t.strip())>=12)
print(json.dumps({'cases':len(items),'counts':dict(counts),'method':'simple pattern indicators and exact local-source utterance match; not a de-identification certification or reidentification rate','external_search_or_disclosure':False,'retained_model_responses_unchanged':True},indent=2))
