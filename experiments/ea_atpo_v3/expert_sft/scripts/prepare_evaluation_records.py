#!/usr/bin/env python3
"""Join frozen raw predictions and reviewed correctness after generation only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from audit_data import rows, sha
from evaluate_experts import aware_datetime, parse_a_output, parse_option, validate_adjudication
from train_experts import EXP, write_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--expert', choices=('A', 'B'), required=True)
    p.add_argument('--raw-predictions', type=Path, required=True)
    p.add_argument('--prediction-run-manifest', type=Path, required=True)
    p.add_argument('--review-queue', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--audit-manifest', type=Path, required=True)
    p.add_argument('--base-answers', type=Path, required=True)
    p.add_argument('--base-manifest', type=Path, required=True)
    p.add_argument('--adjudications', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir.resolve()
    if (EXP / 'data/private').resolve() not in out.parents or out.exists():
        raise ValueError('scored records require a new private V3 directory')
    run = json.loads(args.prediction_run_manifest.read_text())
    if (run.get('status') != 'COMPLETED' or run.get('expert') != args.expert
        or run.get('raw_predictions_sha256') != sha(args.raw_predictions)
        or run.get('reviewed_queue_sha256') != sha(args.review_queue)
        or run.get('base_answers_sha256') != sha(args.base_answers)):
        raise ValueError('prediction run provenance mismatch')
    check = subprocess.run([sys.executable, str(EXP / 'scripts/check_gold_review.py'),
                            '--queue', str(args.review_queue), '--source-root', str(args.source_root),
                            '--audit-manifest', str(args.audit_manifest), '--base-answers', str(args.base_answers),
                            '--base-manifest', str(args.base_manifest)], capture_output=True, text=True)
    if check.returncode:
        raise ValueError('human review/Base provenance gate failed')
    with args.review_queue.open(encoding='utf-8', newline='') as f:
        review = {r['sample_id']: r for r in csv.DictReader(f) if r['split'] == run['split']
                  and (r['gold_status'] in {'APPROVED', 'AMBIGUOUS'} if args.expert == 'A' else r['gold_status'] == 'APPROVED')}
    predictions = list(rows(args.raw_predictions))
    if len(predictions) != len(review) or {r['sample_id'] for r in predictions} != set(review):
        raise ValueError('prediction coverage differs from reviewed split')
    adjudications = {}
    if args.adjudications:
        for r in rows(args.adjudications):
            if r['sample_id'] in adjudications:
                raise ValueError('duplicate adjudication')
            adjudications[r['sample_id']] = r
    scored = []
    generated_at = aware_datetime(run['generated_at_utc'])
    for item in predictions:
        r = review[item['sample_id']]
        if (item['case_id'] != r['case_id'] or item['split'] != r['split'] or item['base_answer'] != r['base_answer']
            or hashlib.sha256(item['expert_output'].encode()).hexdigest() != item['expert_output_sha256']):
            raise ValueError('prediction identity, Base or raw output checksum mismatch')
        record = {**item, 'gold_status': r['gold_status'], 'target_action': r['target_action'],
                  'risk_level': r['risk_level'], 'base_correct': {'true': True, 'false': False}.get(r['base_correct'].strip().lower())}
        candidate = None
        if r['gold_status'] == 'APPROVED':
            if args.expert == 'A':
                try:
                    parsed = parse_a_output(item['expert_output'])
                    if parsed['action'] == 'REVISE':
                        candidate = parsed['answer']
                except ValueError:
                    pass  # invalid A JSON falls back to Base and remains counted
            else:
                candidate = item['expert_output']
            if candidate is not None:
                if args.expert == 'B' and not candidate.strip():
                    record.update({'expert_correct': False, 'correctness_method': 'invalid_output'})
                else:
                    gold, answer = parse_option(r['reviewed_gold']), parse_option(candidate)
                    if gold is not None and answer is not None:
                        record.update({'expert_correct': gold == answer, 'correctness_method': 'approved_gold_exact_option'})
                    else:
                        a = adjudications.get(item['sample_id'])
                        if not a or type(a.get('expert_correct')) is not bool:
                            raise ValueError('non-option output requires independent human correctness adjudication')
                        record.update({'expert_correct': a['expert_correct'], 'correctness_method': 'human_adjudicated',
                                       'correctness_reviewer': a.get('reviewer', ''), 'correctness_evidence': a.get('evidence', ''),
                                       'correctness_timestamp': a.get('timestamp', '')})
                validate_adjudication(record, r, args.expert, generated_at)
        scored.append(record)
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    scored_path = out / 'scored_predictions.jsonl'
    fd = os.open(scored_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        for record in scored:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
    summary = {'status': 'COMPLETED', 'expert': args.expert, 'split': run['split'],
               'model_lock_sha256': run['model_lock_sha256'], 'scored_predictions_sha256': sha(scored_path),
               'raw_predictions_sha256': sha(args.raw_predictions), 'raw_run_manifest_sha256': sha(args.prediction_run_manifest),
               'reviewed_queue_sha256': sha(args.review_queue), 'base_answers_sha256': sha(args.base_answers),
               'score_visibility': run['score_visibility'], 'score_features': run['score_features'],
               'generated_at_utc': run['generated_at_utc'], 'sample_count': len(scored),
               'adjudications_sha256': sha(args.adjudications) if args.adjudications else None}
    write_json(out / 'scored_predictions_manifest.json', summary)
    print(json.dumps({'status': summary['status'], 'samples': len(scored), 'manifest': str(out / 'scored_predictions_manifest.json')}))


if __name__ == '__main__':
    main()
