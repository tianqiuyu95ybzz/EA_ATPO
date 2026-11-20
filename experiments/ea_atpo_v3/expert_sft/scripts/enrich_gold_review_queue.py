#!/usr/bin/env python3
"""Copy verified Base outputs into a new private, still-unreviewed queue."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from audit_base_candidates import REVISION
from audit_data import rows, sha
from prepare_gold_review import FIELDS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--base-answers', type=Path, required=True)
    p.add_argument('--base-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    if 'private' not in out.parts or out.exists():
        raise ValueError('output must be a new file in the private directory')
    manifest = json.loads(args.base_manifest.read_text())
    required = ('model_revision', 'tokenizer', 'prompt', 'decoding', 'data', 'case_ids', 'raw_output_completeness')
    if (manifest.get('status') != 'VERIFIED' or manifest.get('base_revision') != REVISION
        or manifest.get('raw_answers_sha256') != sha(args.base_answers)
        or not all(manifest.get('checks', {}).get(k) is True for k in required)):
        raise ValueError('Base manifest is not verified')
    base = {}
    for item in rows(args.base_answers):
        if (item['sample_id'] in base or item.get('status') != 'VERIFIED'
            or hashlib.sha256(item['raw_output'].encode()).hexdigest() != item['raw_output_sha256']):
            raise ValueError('invalid, duplicate or unverified raw Base answer')
        base[item['sample_id']] = item
    with args.queue.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError('unexpected queue columns')
        queue = list(reader)
    if len(queue) != len(base) or len(queue) != manifest.get('sample_count') or len({r['sample_id'] for r in queue}) != len(queue):
        raise ValueError('queue/Base coverage mismatch')
    for row in queue:
        item = base.get(row['sample_id'])
        if (item is None or (item['case_id'], item['split']) != (row['case_id'], row['split'])
            or row['gold_status'] != 'PENDING' or row['base_answer'] or row['reviewed_gold']
            or row['base_correct'] or row['target_action'] or row['reviewer'] or row['review_timestamp']):
            raise ValueError('queue is modified, reviewed or does not match Base identity')
        row['base_answer'] = item['raw_output']
        row['review_note'] = 'Verified Base answer populated; awaiting human review.'
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(queue)
    print(json.dumps({'samples': len(queue), 'status': 'PENDING_ALL', 'output': str(out), 'sha256': sha(out)}))


if __name__ == '__main__':
    main()
