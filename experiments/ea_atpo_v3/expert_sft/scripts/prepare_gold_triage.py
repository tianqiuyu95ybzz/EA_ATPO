#!/usr/bin/env python3
"""Prioritize historical-label disagreements without deciding medical correctness."""
import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

from audit_data import rows, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--base-answers', type=Path, required=True)
    p.add_argument('--base-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True)
    args = p.parse_args()
    if 'private' not in args.output.resolve().parts or args.output.exists() or args.summary.exists():
        raise ValueError('triage must be new and private; refusing overwrite')
    manifest = json.loads(args.base_manifest.read_text())
    if manifest.get('status') != 'VERIFIED' or manifest.get('raw_answers_sha256') != sha(args.base_answers):
        raise ValueError('triage requires verified complete Base output')
    with args.queue.open(encoding='utf-8', newline='') as f:
        queue = list(csv.DictReader(f))
    if len(queue) != manifest['sample_count'] or any(r['gold_status'] != 'PENDING' for r in queue):
        raise ValueError('triage expects the complete unreviewed queue')
    base_records = {r['sample_id']: r for r in rows(args.base_answers)}
    if len(base_records) != len(queue) or len({r['sample_id'] for r in queue}) != len(queue):
        raise ValueError('queue/Base coverage mismatch')
    triage = []
    for r in queue:
        saved = base_records.get(r['sample_id'])
        if (saved is None or saved.get('status') != 'VERIFIED'
            or (saved['case_id'], saved['split'], saved['raw_output']) != (r['case_id'], r['split'], r['base_answer'])):
            raise ValueError('queue/Base case, split or raw answer mismatch')
        base = re.fullmatch(r'Final Answer: ([A-D])', r['base_answer'])
        historic = re.fullmatch(r'Expert Final Answer: ([A-D])', r['original_gold'])
        if base is None or historic is None:
            raise ValueError('cannot compare historical label with Base output format')
        mismatch = base.group(1) != historic.group(1)
        triage.append({'case_id': r['case_id'], 'sample_id': r['sample_id'], 'split': r['split'],
                       'historical_label_disagreement': str(mismatch).lower(), 'risk_level': 'UNKNOWN',
                       'gold_status': 'PENDING', 'priority': 'DISAGREEMENT_FIRST' if mismatch else 'REVIEW_REQUIRED'})
    triage.sort(key=lambda r: (r['priority'] != 'DISAGREEMENT_FIRST', r['split'], r['case_id'], r['sample_id']))
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(triage[0]))
        writer.writeheader()
        writer.writerows(triage)
    counts = Counter(r['historical_label_disagreement'] for r in triage)
    result = {'samples': len(triage), 'historical_label_disagreement_count': counts['true'],
              'historical_label_agreement_count': counts['false'], 'all_gold_status': 'PENDING',
              'triage_sha256': sha(args.output), 'queue_sha256': sha(args.queue),
              'interpretation': 'disagreement is only a review priority against an unreviewed historical label; it is not Base correctness or accuracy'}
    with args.summary.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
