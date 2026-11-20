#!/usr/bin/env python3
"""Freeze a trained A/B checkpoint only after non-Oracle Dev calibration."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from audit_data import sha
from evaluate_experts import aware_datetime
from train_experts import EXP, write_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--expert', choices=('A', 'B'), required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--training-manifest', type=Path, required=True)
    p.add_argument('--calibration-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    checkpoint = args.checkpoint.resolve()
    expected_parent = (EXP / 'outputs' / f'expert_{args.expert.lower()}').resolve()
    if expected_parent not in checkpoint.parents or expected_parent not in args.output.resolve().parents or args.output.exists():
        raise ValueError('lock and checkpoint require the matching Expert directory and a new output')
    train = json.loads(args.training_manifest.read_text())
    cal = json.loads(args.calibration_manifest.read_text())
    adapter = checkpoint / 'adapter_model.safetensors'
    adapter_sha, config_sha = sha(adapter), sha(args.config)
    known = {r['adapter_sha256'] for r in train.get('epochs', [])}
    now = datetime.now(timezone.utc)
    if (train.get('status') != 'TRAINED_DEV_SELECTION_PENDING' or train.get('expert') != args.expert
        or adapter_sha not in known or train.get('config_sha256') != config_sha
        or cal.get('status') != 'COMPLETED' or cal.get('expert') != args.expert
        or cal.get('split') != 'validation' or cal.get('oracle_used') is not False
        or cal.get('checkpoint_sha256') != adapter_sha or cal.get('config_sha256') != config_sha
        or cal.get('reviewed_queue_sha256') != train.get('reviewed_queue_sha256')
        or aware_datetime(cal.get('completed_at_utc', '')) > now
        or isinstance(cal.get('threshold'), bool) or not isinstance(cal.get('threshold'), (float, int))
        or not 0 <= cal['threshold'] <= 1):
        raise ValueError('checkpoint, training and Dev calibration provenance are inconsistent')
    lock = {'status': 'FROZEN', 'expert': args.expert, 'frozen_at_utc': now.isoformat(),
            'checkpoint_file': str(adapter), 'checkpoint_sha256': adapter_sha,
            'config_file': str(args.config.resolve()), 'config_sha256': config_sha,
            'training_manifest_sha256': sha(args.training_manifest), 'reviewed_queue_sha256': train['reviewed_queue_sha256'],
            'threshold': cal['threshold'], 'threshold_chosen_on': 'validation', 'oracle_used_for_threshold': False,
            'threshold_calibration_manifest': str(args.calibration_manifest.resolve()),
            'threshold_calibration_sha256': sha(args.calibration_manifest)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, lock)
    print(json.dumps({'status': 'FROZEN', 'expert': args.expert, 'model_lock': str(args.output)}))


if __name__ == '__main__':
    main()
