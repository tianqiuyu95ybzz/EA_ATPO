#!/usr/bin/env python3
"""Extract the two theoretical Oracle upper bounds from offline aggregate metrics."""
import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--aggregate-metrics', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    metrics = json.loads(args.aggregate_metrics.read_text())
    expert = metrics['oracle_expert_accuracy_theoretical']
    selector = metrics['oracle_selector_accuracy_theoretical']
    result = {'expert': metrics['expert'], 'approved_samples': metrics['approved_samples'],
              'oracle_expert_accuracy': expert, 'oracle_selector_accuracy': selector,
              'oracle_expert_definition': 'Base correct -> KEEP; Base wrong -> correct with Gold',
              'oracle_selector_definition': 'Offline choose correct Base or fixed Expert answer; both wrong remains wrong',
              'usage': 'theoretical offline upper bounds only; never training inputs or threshold objectives'}
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
