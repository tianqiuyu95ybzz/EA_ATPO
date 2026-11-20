#!/usr/bin/env python3
"""Choose a fixed decision threshold from reviewed Dev data without Oracle selection."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from audit_data import rows, sha
from evaluate_experts import aware_datetime, parse_a_output, validate_adjudication
from train_experts import EXP, write_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--expert', choices=('A', 'B'), required=True)
    p.add_argument('--scored-predictions', type=Path, required=True)
    p.add_argument('--scored-manifest', type=Path, required=True)
    p.add_argument('--prediction-run-manifest', type=Path, required=True)
    p.add_argument('--raw-predictions', type=Path, required=True)
    p.add_argument('--review-queue', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir.resolve()
    if (EXP / 'outputs').resolve() not in out.parents or out.exists():
        raise ValueError('calibration must use a new V3 outputs/ directory')
    scored_manifest = json.loads(args.scored_manifest.read_text())
    run = json.loads(args.prediction_run_manifest.read_text())
    features = run.get('score_features')
    allowed_features = {'constant', 'random_seed', 'base_entropy', 'base_margin', 'base_max_logprob',
                        'base_option_logprobs', 'base_answer_tokens', 'prompt_tokens', 'dialogue_turns'}
    if args.expert == 'A':
        allowed_features.add('a_action_logprobs')
    if (scored_manifest.get('status') != 'COMPLETED' or scored_manifest.get('split') != 'validation'
        or scored_manifest.get('expert') != args.expert or scored_manifest.get('scored_predictions_sha256') != sha(args.scored_predictions)
        or scored_manifest.get('raw_predictions_sha256') != sha(args.raw_predictions)
        or scored_manifest.get('raw_run_manifest_sha256') != sha(args.prediction_run_manifest)
        or scored_manifest.get('reviewed_queue_sha256') != sha(args.review_queue)
        or run.get('status') != 'COMPLETED' or run.get('expert') != args.expert or run.get('split') != 'validation'
        or run.get('raw_predictions_sha256') != sha(args.raw_predictions) or run.get('reviewed_queue_sha256') != sha(args.review_queue)
        or run.get('score_visibility') != 'predecision' or not isinstance(features, list)
        or not features or not set(features).issubset(allowed_features)):
        raise ValueError('threshold calibration requires matching scored Dev predictions')
    with args.review_queue.open(encoding='utf-8', newline='') as f:
        review = {r['sample_id']: r for r in csv.DictReader(f) if r['split'] == 'validation'
                  and (r['gold_status'] in {'APPROVED', 'AMBIGUOUS'} if args.expert == 'A' else r['gold_status'] == 'APPROVED')}
    records = list(rows(args.scored_predictions))
    raw = list(rows(args.raw_predictions))
    by_id = {r['sample_id']: r for r in raw}
    if len(by_id) != len(raw) or set(by_id) != {r['sample_id'] for r in records}:
        raise ValueError('scored and raw Dev prediction coverage mismatch')
    for record in records:
        original = by_id[record['sample_id']]
        if (any(record.get(k) != original.get(k) for k in ('case_id', 'sample_id', 'split', 'base_answer', 'expert_output', 'decision_score'))
            or hashlib.sha256(original['expert_output'].encode()).hexdigest() != original.get('expert_output_sha256')):
            raise ValueError('scored Dev record differs from raw model prediction')
    if len(records) != len(review) or {r['sample_id'] for r in records} != set(review):
        raise ValueError('scored Dev coverage mismatch')
    scored = []
    generated_at = aware_datetime(run['generated_at_utc'])
    for item in records:
        r = review[item['sample_id']]
        expected = {'true': True, 'false': False}.get(r['base_correct'].strip().lower())
        if (item['case_id'] != r['case_id'] or item['gold_status'] != r['gold_status']
            or item['base_answer'] != r['base_answer'] or item.get('base_correct') != expected
            or item['target_action'] != r['target_action']):
            raise ValueError('scored Dev data do not match the reviewed queue')
        if r['gold_status'] != 'APPROVED':
            continue
        validate_adjudication(item, r, args.expert, generated_at)
        score = item['decision_score']
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('invalid pre-decision score')
        if args.expert == 'A':
            try:
                action = parse_a_output(item['expert_output'])['action']
            except ValueError:
                action = 'INVALID'
        else:
            action = 'REVISE'
        candidate = item['expert_correct'] if action == 'REVISE' else item['base_correct']
        if type(item['base_correct']) is not bool or type(candidate) is not bool:
            raise ValueError('Dev correctness must be independently adjudicated Boolean values')
        scored.append({'base': item['base_correct'], 'candidate': candidate, 'score': score,
                       'action': action, 'case_id': item['case_id']})
    if not scored:
        raise ValueError('no approved Dev examples available')
    results = []
    for threshold in (x / 20 for x in range(21)):
        use = [r['action'] == 'REVISE' and r['score'] >= threshold for r in scored]
        finals = [r['candidate'] if selected else r['base'] for r, selected in zip(scored, use)]
        base_correct = sum(r['base'] for r in scored)
        degradation = sum(r['base'] and not final for r, final in zip(scored, finals))
        correction = sum(not r['base'] and final for r, final in zip(scored, finals))
        results.append({'threshold': threshold, 'accuracy': sum(finals) / len(scored),
                        'degradation_count': degradation, 'base_correct_count': base_correct,
                        'correction_count': correction, 'selected_count': sum(use),
                        'net_gain': correction - degradation})
    chosen = max(results, key=lambda r: (r['accuracy'], -r['degradation_count'], -r['selected_count'], r['threshold']))
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    manifest = {'status': 'COMPLETED', 'expert': args.expert, 'split': 'validation', 'oracle_used': False,
                'selection_rule': 'maximum Dev final accuracy, then lower degradation, fewer Expert selections, higher threshold',
                'threshold': chosen['threshold'], 'grid': results, 'approved_states': len(scored),
                'approved_cases': len({r['case_id'] for r in scored}), 'checkpoint_sha256': run['checkpoint_sha256'],
                'config_sha256': run['config_sha256'], 'scored_predictions_sha256': sha(args.scored_predictions),
                'reviewed_queue_sha256': sha(args.review_queue), 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    write_json(out / 'threshold_calibration_manifest.json', manifest)
    print(json.dumps({'status': 'COMPLETED', 'threshold': chosen['threshold'], 'manifest': str(out / 'threshold_calibration_manifest.json')}))


if __name__ == '__main__':
    main()
