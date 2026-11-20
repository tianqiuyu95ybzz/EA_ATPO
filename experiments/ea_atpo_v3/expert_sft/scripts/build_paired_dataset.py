#!/usr/bin/env python3
"""Build private paired and model-specific datasets only after review gates pass."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from audit_data import sha

EXP = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--audit-manifest', type=Path, required=True)
    p.add_argument('--base-answers', type=Path, required=True)
    p.add_argument('--base-manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir.resolve()
    if (EXP / 'data/private').resolve() not in out.parents:
        raise ValueError('paired datasets must stay under V3 data/private/')
    check = subprocess.run([sys.executable, str(EXP / 'scripts/check_gold_review.py'),
                            '--queue', str(args.queue), '--source-root', str(args.source_root),
                            '--audit-manifest', str(args.audit_manifest), '--base-answers', str(args.base_answers),
                            '--base-manifest', str(args.base_manifest)], capture_output=True, text=True)
    if check.returncode != 0:
        print(check.stdout, end='')
        print(check.stderr, end='', file=sys.stderr)
        raise RuntimeError('formal paired dataset blocked by review/Base provenance gate')
    verified = json.loads(check.stdout)
    with args.queue.open(encoding='utf-8', newline='') as f:
        queue = list(csv.DictReader(f))
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(out, 0o700)
    splits = ('train', 'validation', 'internal_test')
    counts = Counter()
    paths = {}
    for split in splits:
        selected = [r for r in queue if r['split'] == split and r['gold_status'] in {'APPROVED', 'AMBIGUOUS'}]
        paired, a_data, b_data = [], [], []
        for r in selected:
            correct = {'true': True, 'false': False}.get(r['base_correct'].strip().lower())
            item = {'case_id': r['case_id'], 'sample_id': r['sample_id'], 'split': split,
                    'conversation': r['conversation'], 'base_answer': r['base_answer'],
                    'reviewed_gold': r['reviewed_gold'] if r['gold_status'] == 'APPROVED' else None,
                    'base_correct': correct, 'gold_status': r['gold_status'], 'target_action': r['target_action'],
                    'risk_level': r['risk_level']}
            paired.append(item)
            # Metadata is separate from the model's input object. KEEP/ABSTAIN
            # never receive an answer-loss target and are copied/fallback at runtime.
            a_data.append({'case_id': r['case_id'], 'sample_id': r['sample_id'], 'split': split,
                           'input': {'conversation': r['conversation'], 'base_answer': r['base_answer']},
                           'label': {'action': r['target_action'], 'correction': r['reviewed_gold'] if r['target_action'] == 'REVISE' else None}})
            if r['gold_status'] == 'APPROVED':
                b_data.append({'case_id': r['case_id'], 'sample_id': r['sample_id'], 'split': split,
                               'input': {'conversation': r['conversation']}, 'label': {'answer': r['reviewed_gold']}})
            counts[f'{split}:{r["target_action"]}'] += 1
        for name, examples in [('paired', paired), ('expert_a', a_data), ('expert_b_c', b_data)]:
            file = out / f'{name}_{split}.jsonl'
            fd = os.open(file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + '\n')
            paths[file.name] = {'sha256': sha(file), 'samples': len(examples)}
    manifest = {'status': 'REVIEW_GATED', 'reviewed_queue_sha256': sha(args.queue),
                'review_validation': verified, 'audit_manifest_sha256': sha(args.audit_manifest),
                'base_answers_sha256': sha(args.base_answers), 'base_manifest_sha256': sha(args.base_manifest),
                'case_split_unit': 'case_id', 'files': paths, 'action_counts_by_split': dict(counts),
                'inference_inputs': {'a': ['conversation', 'base_answer'], 'b_c': ['conversation']},
                'rejected_excluded': sum(r['gold_status'] == 'REJECTED' for r in queue),
                'ambiguous_excluded_from_b_c': sum(r['gold_status'] == 'AMBIGUOUS' for r in queue)}
    with (out / 'paired_dataset_manifest.json').open('x', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps({'status': manifest['status'], 'output_dir': str(out), 'files': paths}))


if __name__ == '__main__':
    main()
