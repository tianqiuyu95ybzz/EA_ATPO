#!/usr/bin/env python3
"""Fail-closed Router admission using independently frozen policy criteria."""
import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--metrics', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    metrics = json.loads(args.metrics.read_text())
    registration = json.loads(args.registration.read_text())
    issues = []
    if registration.get('status') != 'FROZEN_BEFORE_TEST':
        issues.append('no frozen numerical admission criteria')
    correction = registration.get('min_correction_rate')
    degradation = registration.get('max_degradation_rate')
    if not isinstance(correction, (int, float)) or not 0 < correction <= 1:
        issues.append('minimum meaningful correction rate is not defined')
    if not isinstance(degradation, (int, float)) or not 0 <= degradation <= 1:
        issues.append('maximum acceptable degradation rate is not defined')
    final = metrics.get('final_policy', {})
    ci = metrics.get('final_policy_bootstrap_95ci', {}).get('paired_accuracy_difference', {})
    if final.get('correction_rate') is None or final.get('degradation_rate') is None:
        issues.append('paired correction/degradation not estimable')
    elif not issues:
        if final['correction_rate'] < correction:
            issues.append('correction rate below frozen criterion')
        if final['degradation_rate'] > degradation:
            issues.append('degradation rate above frozen criterion')
        if final.get('net_gain_count', 0) <= 0:
            issues.append('no positive net gain')
        if ci.get('low') is None or ci['low'] <= 0:
            issues.append('paired 95% CI does not establish improvement')
    if registration.get('score_features_predecision_only') is not True:
        issues.append('pre-decision feature constraint not verified')
    allowed = {'constant', 'random_seed', 'base_entropy', 'base_margin', 'base_max_logprob',
               'base_option_logprobs', 'base_answer_tokens', 'prompt_tokens', 'dialogue_turns'}
    features = metrics.get('score_features')
    if (metrics.get('score_visibility') != 'predecision' or not isinstance(features, list)
        or not features or not set(features).issubset(allowed)):
        issues.append('evaluated policy is not established as a pre-Expert decision')
    if registration.get('oracle_used_for_selection') is not False:
        issues.append('independence from Oracle selection not verified')
    result = {'decision': 'BLOCKED' if issues else 'ELIGIBLE_FOR_ROUTER', 'reasons': issues,
              'scope': 'accuracy-improvement route only; separately validated cost-sensitive evidence may require independent review',
              'no_router_or_rl_started': True}
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
