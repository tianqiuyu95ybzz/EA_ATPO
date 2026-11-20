#!/usr/bin/env python3
"""Bind already-scored Dev predictions to their subsequently frozen checkpoint."""
import argparse
import json
from pathlib import Path

from audit_data import sha
from train_experts import write_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--scored-manifest', type=Path, required=True)
    p.add_argument('--scored-predictions', type=Path, required=True)
    p.add_argument('--model-lock', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    scored = json.loads(args.scored_manifest.read_text())
    lock = json.loads(args.model_lock.read_text())
    calibration_path = Path(lock['threshold_calibration_manifest'])
    calibration = json.loads(calibration_path.read_text())
    if (lock.get('status') != 'FROZEN' or scored.get('status') != 'COMPLETED'
        or scored.get('split') != 'validation' or scored.get('expert') != lock.get('expert')
        or scored.get('scored_predictions_sha256') != sha(args.scored_predictions)
        or lock.get('threshold_calibration_sha256') != sha(calibration_path)
        or calibration.get('scored_predictions_sha256') != sha(args.scored_predictions)
        or scored.get('reviewed_queue_sha256') != lock.get('reviewed_queue_sha256')):
        raise ValueError('Dev results do not match the frozen checkpoint/calibration')
    scored['model_lock_sha256'] = sha(args.model_lock)
    write_json(args.output, scored)
    print(json.dumps({'status': 'BOUND', 'manifest': str(args.output)}))


if __name__ == '__main__':
    main()
