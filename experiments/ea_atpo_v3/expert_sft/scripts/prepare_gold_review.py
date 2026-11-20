#!/usr/bin/env python3
"""Create a private, unreviewed queue from audited SFT sources; never auto-approve."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from audit_data import FILES, SFT, rows, sha


FIELDS = ['case_id', 'sample_id', 'split', 'conversation', 'base_answer', 'original_gold', 'reviewed_gold', 'base_correct', 'gold_status', 'target_action', 'evidence', 'risk_level', 'reviewer', 'review_timestamp', 'review_note']


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--audit-manifest', type=Path, required=True)
    ap.add_argument('--private-dir', type=Path, required=True)
    ap.add_argument('--summary', type=Path, required=True)
    args = ap.parse_args()
    root = args.source_root.resolve()
    private_dir = args.private_dir.resolve()
    if private_dir.name != 'private':
        raise ValueError('queue must be placed in a private/ directory ignored by Git')
    audit = json.loads(args.audit_manifest.read_text())
    if audit.get('review_status') != 'PENDING_ALL' or audit.get('frozen_holdout_opened') is not False:
        raise ValueError('unexpected audit manifest status')
    private_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(private_dir, 0o700)
    queue = private_dir / 'gold_review_queue.csv'
    if queue.exists() or args.summary.exists():
        raise FileExistsError('refusing to overwrite a queue or summary')
    prepared = []
    counts = Counter()
    for split, file in FILES.items():
        path = root / SFT / file
        if sha(path) != audit['splits'][split]['source_file_sha256']:
            raise ValueError(f'{split} changed since audit')
        for item in rows(path):
            messages = item['messages']
            prepared.append({'case_id': item['case_id'], 'sample_id': item['state_id'], 'split': split,
                             'conversation': messages[1]['content'], 'base_answer': '',
                             'original_gold': messages[2]['content'], 'reviewed_gold': '',
                             'base_correct': '', 'gold_status': 'PENDING', 'target_action': '',
                             'evidence': '', 'risk_level': 'UNKNOWN', 'reviewer': '',
                             'review_timestamp': '', 'review_note': 'Awaiting verified Base output and human review.'})
            counts[split] += 1
    if len(prepared) != sum(v['samples'] for v in audit['splits'].values()):
        raise ValueError('queue row count does not match audit')
    fd = os.open(queue, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(prepared)
    summary = {'schema_version': 1, 'created_queue_sha256': sha(queue), 'queue_path': str(queue),
               'source_audit_sha256': sha(args.audit_manifest), 'samples': len(prepared),
               'samples_by_split': dict(counts), 'status_counts': {'PENDING': len(prepared), 'APPROVED': 0, 'AMBIGUOUS': 0, 'REJECTED': 0},
               'base_answers_verified': 0, 'base_correct_reviewed': 0, 'target_actions_reviewed': 0,
               'ready_for_formal_training': False, 'contains_clinical_certification': False}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open('x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps({'queue': str(queue), 'samples': len(prepared), 'status': 'PENDING_ALL', 'base_answers_verified': 0}))


if __name__ == '__main__':
    main()
