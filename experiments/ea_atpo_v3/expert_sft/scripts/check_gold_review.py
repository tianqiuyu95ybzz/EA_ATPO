#!/usr/bin/env python3
"""Fail closed on incomplete or internally inconsistent human-review records."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path

from audit_data import FILES, SFT, rows, sha
from prepare_gold_review import FIELDS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--audit-manifest', type=Path, required=True)
    p.add_argument('--base-answers', type=Path)
    p.add_argument('--base-manifest', type=Path)
    args = p.parse_args()
    audit = json.loads(args.audit_manifest.read_text())
    expected = {}
    for split, name in FILES.items():
        path = args.source_root / SFT / name
        if sha(path) != audit['splits'][split]['source_file_sha256']:
            raise ValueError('source differs from audit')
        for x in rows(path):
            expected[x['state_id']] = (x['case_id'], split, x['messages'][1]['content'], x['messages'][2]['content'])
    with args.queue.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError('unexpected queue columns')
        review = list(reader)
    if len(review) != len(expected) or len({r['sample_id'] for r in review}) != len(expected):
        raise ValueError('missing or duplicate queue samples')
    base = {}
    if args.base_answers:
        for b in rows(args.base_answers):
            if b['sample_id'] in base:
                raise ValueError('duplicate Base sample')
            import hashlib
            if hashlib.sha256(b['raw_output'].encode()).hexdigest() != b['raw_output_sha256']:
                raise ValueError('Base raw output checksum mismatch')
            if b.get('status') != 'VERIFIED':
                raise ValueError('unverified Base candidate cannot enter review/training')
            base[b['sample_id']] = b['raw_output']
    errors = Counter()
    if not args.base_manifest or not args.base_answers:
        errors['base_pairing_manifest_or_answers_missing'] += 1
    else:
        manifest = json.loads(args.base_manifest.read_text())
        checks = ('model_revision', 'tokenizer', 'prompt', 'decoding', 'data', 'case_ids', 'raw_output_completeness')
        if (manifest.get('status') != 'VERIFIED' or manifest.get('sample_count') != len(expected)
            or manifest.get('raw_answers_sha256') != sha(args.base_answers)
            or manifest.get('base_revision') != 'b968826d9c46dd6066d109eabc6255188de91218'
            or not all(manifest.get('checks', {}).get(k) is True for k in checks)):
            errors['base_pairing_manifest_invalid'] += 1
    statuses = Counter()
    actions = Counter()
    for r in review:
        if expected.get(r['sample_id']) != (r['case_id'], r['split'], r['conversation'], r['original_gold']):
            errors['source_identity_or_text_changed'] += 1
        status = r['gold_status'].strip().upper()
        action = r['target_action'].strip().upper()
        statuses[status] += 1
        if action:
            actions[action] += 1
        if r['sample_id'] not in base or r['base_answer'] != base[r['sample_id']]:
            errors['verified_base_missing_or_changed'] += 1
        if status == 'PENDING':
            errors['pending_review'] += 1
            continue
        if status not in {'APPROVED', 'AMBIGUOUS', 'REJECTED'}:
            errors['invalid_status'] += 1
            continue
        if not r['reviewer'].strip() or not r['evidence'].strip():
            errors['missing_reviewer_or_evidence'] += 1
        try:
            stamp = dt.datetime.fromisoformat(r['review_timestamp'].replace('Z', '+00:00'))
            if stamp.tzinfo is None:
                raise ValueError('timezone required')
        except ValueError:
            errors['invalid_timestamp'] += 1
        if r['risk_level'].strip().upper() not in {'LOW', 'MEDIUM', 'HIGH'}:
            errors['risk_unreviewed'] += 1
        correct = r['base_correct'].strip().lower()
        if status == 'APPROVED':
            if not r['reviewed_gold'].strip() or correct not in {'true', 'false'}:
                errors['approved_missing_gold_or_base_correct'] += 1
            if (correct, action) not in {('true', 'KEEP'), ('false', 'REVISE')}:
                errors['approved_action_inconsistent'] += 1
        elif status == 'AMBIGUOUS':
            if action != 'ABSTAIN' or r['reviewed_gold'].strip():
                errors['ambiguous_must_abstain_without_gold'] += 1
        elif action or r['reviewed_gold'].strip():
            errors['rejected_must_be_excluded'] += 1
    summary = {'ready_for_formal_training': not errors, 'samples': len(review), 'status_counts': dict(statuses),
               'action_counts': dict(actions), 'errors': dict(errors), 'queue_sha256': sha(args.queue),
               'base_answers_sha256': sha(args.base_answers) if args.base_answers else None,
               'note': 'Base provenance metadata/revision/prompt/decoding must also pass the separate pairing gate.'}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
