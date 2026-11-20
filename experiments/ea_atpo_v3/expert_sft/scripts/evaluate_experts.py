#!/usr/bin/env python3
"""Offline paired metrics on frozen, adjudicated predictions; no model calls."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from audit_data import rows, sha

ACTIONS = ('KEEP', 'REVISE', 'ABSTAIN')
EXP = Path(__file__).resolve().parents[1]


def parse_a_output(raw: str) -> dict:
    try:
        result = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Expert A output must be JSON') from exc
    if (not isinstance(result, dict) or set(result) != {'action', 'answer'}
        or result['action'] not in ACTIONS or not isinstance(result['answer'], str)):
        raise ValueError('Expert A output must contain only action and answer')
    if result['action'] == 'REVISE' and not result['answer'].strip():
        raise ValueError('REVISE must provide a nonempty correction')
    return result


def finalize_a(base_answer: str, raw: str, selected: bool = True) -> tuple[str, str]:
    result = parse_a_output(raw)
    action = result['action']
    if action == 'REVISE' and not selected:
        action = 'ABSTAIN'
    if action in {'KEEP', 'ABSTAIN'}:
        return action, base_answer
    return action, result['answer']


def ratio(n: int, d: int) -> float | None:
    return n / d if d else None


def contingency(pairs: list[tuple[bool, bool]]) -> dict:
    counts = Counter(('correct' if b else 'wrong', 'correct' if e else 'wrong') for b, e in pairs)
    cc = counts['correct', 'correct']
    cw = counts['correct', 'wrong']
    wc = counts['wrong', 'correct']
    ww = counts['wrong', 'wrong']
    n = len(pairs)
    degradation = ratio(cw, cc + cw)
    return {'base_correct_expert_correct': cc, 'base_correct_expert_wrong': cw,
            'base_wrong_expert_correct': wc, 'base_wrong_expert_wrong': ww,
            'samples': n, 'base_accuracy': ratio(cc + cw, n), 'expert_accuracy': ratio(cc + wc, n),
            'correction_rate': ratio(wc, wc + ww), 'degradation_rate': degradation,
            'preservation_rate': None if degradation is None else 1 - degradation,
            'net_gain_count': wc - cw, 'paired_accuracy_difference': ratio(wc - cw, n)}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    a = sorted(values)
    x = (len(a) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    return a[lo] + (a[hi] - a[lo]) * (x - lo)


def bootstrap(scored: list[dict], key: str, repetitions: int, seed: int) -> dict:
    clusters = defaultdict(list)
    for r in scored:
        clusters[r['case_id']].append(r)
    ids = sorted(clusters)
    if not ids or repetitions < 1:
        raise ValueError('bootstrap requires at least one case and one repetition')
    rng = random.Random(seed)
    values = {k: [] for k in ('paired_accuracy_difference', 'correction_rate', 'degradation_rate')}
    for _ in range(repetitions):
        sampled = [r for case in rng.choices(ids, k=len(ids)) for r in clusters[case]]
        result = contingency([(r['base_correct'], r[key]) for r in sampled])
        for metric in ('paired_accuracy_difference', 'correction_rate', 'degradation_rate'):
            if result[metric] is not None:
                values[metric].append(result[metric])
    return {k: {'low': percentile(v, 0.025), 'high': percentile(v, 0.975), 'valid_replicates': len(v)}
            for k, v in values.items()}


def action_metrics(observed: list[tuple[str, str]]) -> dict:
    result = {}
    for action in ACTIONS:
        tp = sum(a == action and b == action for a, b in observed)
        fp = sum(a != action and b == action for a, b in observed)
        fn = sum(a == action and b != action for a, b in observed)
        precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
        f1 = ratio(2 * tp, 2 * tp + fp + fn)
        result[action] = {'precision': precision, 'recall': recall, 'f1': f1,
                          'true_count': tp + fn, 'predicted_count': tp + fp}
    return result


def evaluate(records: list[dict], expert: str, threshold: float, repetitions: int, seed: int) -> tuple[dict, list[dict], list[dict]]:
    if expert not in {'A', 'B', 'C'} or not 0 <= threshold <= 1:
        raise ValueError('invalid expert or threshold')
    if not records or len({r['sample_id'] for r in records}) != len(records):
        raise ValueError('empty or duplicate prediction records')
    scored, observed_actions = [], []
    status_counts, action_counts, predicted_counts = Counter(), Counter(), Counter()
    invalid_outputs = 0
    for r in records:
        status = r.get('gold_status')
        if status not in {'APPROVED', 'AMBIGUOUS', 'REJECTED'}:
            raise ValueError('PENDING or unknown review status cannot be evaluated')
        status_counts[status] += 1
        if status == 'REJECTED':
            continue
        if not isinstance(r.get('base_answer'), str) or not r['base_answer']:
            raise ValueError('missing raw Base answer')
        score = r.get('decision_score')
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('each included row requires a frozen 0..1 decision score')
        if expert == 'A':
            try:
                predicted_action = parse_a_output(r['expert_output'])['action']
                action, final_answer = finalize_a(r['base_answer'], r['expert_output'], score >= threshold)
            except ValueError:
                predicted_action, action, final_answer = 'INVALID', 'ABSTAIN', r['base_answer']
                invalid_outputs += 1
            expected = r.get('target_action')
            if expected not in ACTIONS:
                raise ValueError('missing reviewed target action')
            observed_actions.append((expected, predicted_action))
            predicted_counts[predicted_action] += 1
            action_counts[action] += 1
        else:
            if not isinstance(r.get('expert_output'), str):
                raise ValueError('independent Expert requires a raw output, including empty invalid outputs')
            predicted_action = 'REVISE'
            if not r['expert_output'].strip():
                invalid_outputs += 1
            action = 'REVISE' if score >= threshold else 'KEEP'
            final_answer = r['expert_output'] if action == 'REVISE' else r['base_answer']
            action_counts[action] += 1
        if action in {'KEEP', 'ABSTAIN'} and final_answer != r['base_answer']:
            raise AssertionError('Base preservation failed')
        if status == 'AMBIGUOUS':
            continue
        if type(r.get('base_correct')) is not bool:
            raise ValueError('approved sample requires reviewed Base correctness')
        if str(r.get('risk_level', '')).upper() not in {'LOW', 'MEDIUM', 'HIGH'}:
            raise ValueError('approved sample requires a reviewed risk level')
        if predicted_action == 'REVISE' and type(r.get('expert_correct')) is not bool:
            raise ValueError('missing adjudicated Expert correctness')
        if predicted_action == 'REVISE' and r.get('correctness_method') not in {'human_adjudicated', 'approved_gold_exact_option', 'invalid_output'}:
            raise ValueError('Expert correctness requires recorded independent adjudication method')
        if predicted_action == 'REVISE' and r.get('correctness_method') == 'invalid_output' and r['expert_correct']:
            raise ValueError('invalid output cannot be marked correct')
        candidate_correct = r['expert_correct'] if predicted_action == 'REVISE' else r['base_correct']
        final_correct = candidate_correct if action == 'REVISE' else r['base_correct']
        scored.append({'case_id': r['case_id'], 'sample_id': r['sample_id'], 'base_correct': r['base_correct'],
                       'candidate_correct': candidate_correct, 'final_correct': final_correct,
                       'predicted_action': predicted_action, 'actual_action': action,
                       'decision_score': score, 'risk_level': r.get('risk_level', 'UNKNOWN')})
    if not scored:
        raise ValueError('no approved cases with adjudicated correctness')
    comparison_key = 'final_correct' if expert == 'A' else 'candidate_correct'
    paired = contingency([(r['base_correct'], r[comparison_key]) for r in scored])
    final = contingency([(r['base_correct'], r['final_correct']) for r in scored])
    by_case = defaultdict(list)
    for r in scored:
        by_case[r['case_id']].append(r)
    case_results = [{'case_id': case, 'contingency': contingency([(r['base_correct'], r[comparison_key]) for r in items]),
                     'high_risk_errors': sum(r['risk_level'].upper() == 'HIGH' and not r['final_correct'] for r in items)}
                    for case, items in sorted(by_case.items())]
    curve = []
    for t in [x / 20 for x in range(21)]:
        selected = [r['predicted_action'] == 'REVISE' and r['decision_score'] >= t for r in scored]
        final_values = [r['candidate_correct'] if use else r['base_correct'] for r, use in zip(scored, selected)]
        selected_values = [r['candidate_correct'] for r, use in zip(scored, selected) if use]
        curve.append({'threshold': t, 'expert_selection_rate': sum(selected) / len(scored),
                      'base_coverage': 1 - sum(selected) / len(scored),
                      'final_error_rate': 1 - sum(final_values) / len(scored),
                      'selected_expert_error_rate': None if not selected_values else 1 - sum(selected_values) / len(selected_values)})
    oracle_selector = sum(r['base_correct'] or r['candidate_correct'] for r in scored) / len(scored)
    summary = {'expert': expert, 'approved_samples': len(scored), 'cases': len(by_case),
               'status_counts': dict(status_counts), 'comparison': paired, 'final_policy': final,
               'paired_bootstrap_95ci': bootstrap(scored, comparison_key, repetitions, seed),
               'final_policy_bootstrap_95ci': bootstrap(scored, 'final_correct', repetitions, seed),
               'bootstrap_unit': 'case_id', 'bootstrap_repetitions': repetitions, 'bootstrap_seed': seed,
               'action_counts_including_ambiguous': dict(action_counts),
               'model_action_counts_including_ambiguous': dict(predicted_counts) if expert == 'A' else None,
               'invalid_output_count_including_ambiguous': invalid_outputs,
               'action_rates_including_ambiguous': {k: ratio(action_counts[k], sum(action_counts.values())) for k in ACTIONS},
               'action_classification': action_metrics(observed_actions) if expert == 'A' else None,
               'high_risk_final_errors': sum(r['risk_level'].upper() == 'HIGH' and not r['final_correct'] for r in scored),
               'oracle_expert_accuracy_theoretical': 1.0, 'oracle_selector_accuracy_theoretical': oracle_selector,
               'oracle_usage': 'offline theoretical reporting only; never a training feature or threshold objective',
               'risk_coverage_definition': 'base_coverage=1-expert_selection_rate; final_error_rate is over approved samples',
               'decision_threshold': threshold}
    return summary, case_results, curve


def parse_option(value: str) -> str | None:
    match = re.fullmatch(r'\s*(?:(?:Expert )?Final Answer:\s*)?([A-D])\s*', value or '', re.I)
    return match.group(1).upper() if match else None


def aware_datetime(value: str) -> dt.datetime:
    result = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('timezone-aware timestamps are required')
    return result


def validate_adjudication(row: dict, review: dict, expert: str, generated_at: dt.datetime) -> None:
    if row['gold_status'] != 'APPROVED':
        return
    if expert == 'A':
        try:
            candidate = parse_a_output(row['expert_output'])
        except ValueError:
            return
        if candidate['action'] != 'REVISE':
            return
        answer = candidate['answer']
    else:
        answer = row['expert_output']
    method = row.get('correctness_method')
    if method == 'human_adjudicated':
        if (not str(row.get('correctness_reviewer', '')).strip()
            or not str(row.get('correctness_evidence', '')).strip()
            or aware_datetime(row.get('correctness_timestamp', '')) < generated_at):
            raise ValueError('human correctness adjudication lacks reviewer, evidence, or valid timestamp')
    elif method == 'approved_gold_exact_option':
        gold, pred = parse_option(review['reviewed_gold']), parse_option(answer)
        if gold is None or pred is None or row.get('expert_correct') is not (gold == pred):
            raise ValueError('exact-option adjudication cannot be independently verified')
    elif method == 'invalid_output':
        if expert == 'A' or answer.strip() or row.get('expert_correct') is not False:
            raise ValueError('invalid-output adjudication inconsistent with raw prediction')
    else:
        raise ValueError('unsupported correctness adjudication method')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', type=Path, required=True, help='private, post-generation adjudicated records')
    p.add_argument('--expert', choices=('A', 'B', 'C'), required=True)
    p.add_argument('--split', choices=('train', 'validation', 'internal_test'), required=True)
    p.add_argument('--model-lock', type=Path, required=True)
    p.add_argument('--prediction-manifest', type=Path, required=True)
    p.add_argument('--review-queue', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--audit-manifest', type=Path, required=True)
    p.add_argument('--base-answers', type=Path, required=True)
    p.add_argument('--base-manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=10000)
    p.add_argument('--seed', type=int, default=20261120)
    args = p.parse_args()
    out = args.output_dir.resolve()
    if (EXP / 'outputs').resolve() not in out.parents or out.exists():
        raise ValueError('evaluation output must be a new V3 outputs/ directory')
    lock = json.loads(args.model_lock.read_text())
    prediction = json.loads(args.prediction_manifest.read_text())
    calibration_path = Path(lock['threshold_calibration_manifest'])
    calibration = json.loads(calibration_path.read_text())
    frozen_at = aware_datetime(lock.get('frozen_at_utc', ''))
    generated_at = aware_datetime(prediction.get('generated_at_utc', ''))
    features = prediction.get('score_features')
    allowed_features = {'constant', 'random_seed', 'base_entropy', 'base_margin', 'base_max_logprob',
                        'base_option_logprobs', 'base_answer_tokens', 'prompt_tokens', 'dialogue_turns',
                        'a_action_logprobs'}
    if args.expert != 'A':
        allowed_features.remove('a_action_logprobs')
    if (lock.get('status') != 'FROZEN' or lock.get('expert') != args.expert
        or lock.get('threshold_chosen_on') != 'validation' or lock.get('oracle_used_for_threshold') is not False
        or lock.get('checkpoint_sha256') != sha(Path(lock['checkpoint_file']))
        or lock.get('config_sha256') != sha(Path(lock['config_file']))
        or lock.get('threshold_calibration_sha256') != sha(calibration_path)
        or calibration.get('split') != 'validation' or calibration.get('oracle_used') is not False
        or calibration.get('threshold') != lock.get('threshold')
        or aware_datetime(calibration.get('completed_at_utc', '')) > frozen_at
        or prediction.get('status') != 'COMPLETED' or prediction.get('split') != args.split
        or prediction.get('model_lock_sha256') != sha(args.model_lock)
        or prediction.get('scored_predictions_sha256') != sha(args.predictions)
        or prediction.get('reviewed_queue_sha256') != sha(args.review_queue)
        or prediction.get('base_answers_sha256') != sha(args.base_answers)
        or prediction.get('score_visibility') != 'predecision' or generated_at < frozen_at
        or not isinstance(features, list) or not features or not set(features).issubset(allowed_features)):
        raise ValueError('frozen model, split, review, prediction, score, or threshold provenance is inconsistent')
    review_check = subprocess.run([sys.executable, str(EXP / 'scripts/check_gold_review.py'),
                                   '--queue', str(args.review_queue), '--source-root', str(args.source_root),
                                   '--audit-manifest', str(args.audit_manifest), '--base-answers', str(args.base_answers),
                                   '--base-manifest', str(args.base_manifest)], capture_output=True, text=True)
    if review_check.returncode:
        raise ValueError('human Gold review or Base provenance gate failed')
    source = list(rows(args.predictions))
    if any(r.get('split') != args.split for r in source):
        raise ValueError('mixed or wrong evaluation split')
    with args.review_queue.open(encoding='utf-8', newline='') as f:
        review = {r['sample_id']: r for r in csv.DictReader(f)
                  if r['split'] == args.split and r['gold_status'] != 'REJECTED'}
    if set(review) != {r['sample_id'] for r in source} or len(review) != len(source):
        raise ValueError('prediction coverage differs from reviewed split')
    for record in source:
        r = review[record['sample_id']]
        correct = {'true': True, 'false': False}.get(r['base_correct'].strip().lower())
        if (record['case_id'] != r['case_id'] or record['gold_status'] != r['gold_status']
            or record['base_answer'] != r['base_answer'] or record.get('base_correct') != correct
            or record.get('target_action') != r['target_action'] or record.get('risk_level') != r['risk_level']):
            raise ValueError('prediction record differs from the frozen human review queue')
        validate_adjudication(record, r, args.expert, generated_at)
    if args.split == 'internal_test':
        marker = EXP / 'outputs/evaluation' / f'internal_test_{args.expert}_consumed.json'
        marker.parent.mkdir(parents=True, exist_ok=True)
        with marker.open('x', encoding='utf-8') as f:
            json.dump({'expert': args.expert, 'model_lock_sha256': sha(args.model_lock), 'predictions_sha256': sha(args.predictions)}, f)
    summary, by_case, curve = evaluate(source, args.expert, float(lock['threshold']), args.bootstrap, args.seed)
    out.mkdir(parents=True, mode=0o700)
    for name, value in [('aggregate_metrics.json', summary), ('by_case.json', by_case)]:
        with (out / name).open('x', encoding='utf-8') as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write('\n')
    with (out / 'risk_coverage.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    coords = ' '.join(f'{50 + 480 * r["base_coverage"]:.1f},{270 - 220 * r["final_error_rate"]:.1f}' for r in curve)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="600" height="320" viewBox="0 0 600 320">'
           '<rect width="600" height="320" fill="white"/><path d="M50 50V270H530" fill="none" stroke="black"/>'
           f'<polyline points="{coords}" fill="none" stroke="#2767b2" stroke-width="3"/>'
           '<text x="245" y="305" font-family="sans-serif" font-size="14">Base coverage (0–1)</text>'
           '<text x="12" y="30" font-family="sans-serif" font-size="14">Final error rate (0–1)</text>'
           '<text x="45" y="288" font-family="sans-serif" font-size="12">0</text>'
           '<text x="526" y="288" font-family="sans-serif" font-size="12">1</text>'
           '<text x="32" y="55" font-family="sans-serif" font-size="12">1</text></svg>')
    with (out / 'risk_coverage.svg').open('x', encoding='utf-8') as f:
        f.write(svg + '\n')
    print(json.dumps({'status': 'COMPLETED', 'summary': str(out / 'aggregate_metrics.json')}))


if __name__ == '__main__':
    main()
