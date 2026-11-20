#!/usr/bin/env python3
"""Freeze 291 public-only V2 ANSWER requests without loading a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from audit_base_candidates import REVISION, TOKENIZER_HASHES
from audit_data import PILOT, SPLIT, rows, sha

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT / 'experiments/ea_atpo_v2/src'))
from v2_protocol import Action, build_base_messages, public_state_from_row  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--audit-manifest', type=Path, required=True)
    p.add_argument('--base-model-dir', type=Path, required=True)
    p.add_argument('--private-dir', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True)
    args = p.parse_args()
    root, private = args.source_root.resolve(), args.private_dir.resolve()
    if private.name != 'private':
        raise ValueError('request plan must stay in the private/ directory')
    private.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(private, 0o700)
    dest = private / 'base_request_plan.jsonl'
    if dest.exists() or args.summary.exists():
        raise FileExistsError('refusing to overwrite request plan')
    audit = json.loads(args.audit_manifest.read_text())
    states_path, split_path = root / PILOT / 'pilot_public_states.jsonl', root / SPLIT
    if sha(states_path) != audit['public_states']['sha256'] or sha(split_path) != audit['split_manifest_sha256']:
        raise ValueError('source states or split differ from audit')
    for name, expected in TOKENIZER_HASHES.items():
        if sha(args.base_model_dir / name) != expected:
            raise ValueError(f'tokenizer mismatch: {name}')
    split = json.loads(split_path.read_text())
    case_to_split = {}
    for name in ('train', 'validation', 'internal_test'):
        for case_id in split[name]:
            if case_id in case_to_split:
                raise ValueError('case appears in multiple splits')
            case_to_split[case_id] = name
    params = {'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 24, 'seed': 20261120,
              'structured_outputs_regex': '^Final Answer: [A-D]$', 'enable_thinking': False}
    requests = []
    for state in sorted(rows(states_path), key=lambda r: (r['case_id'], r['state_index'], r['state_id'])):
        if state['case_id'] not in case_to_split:
            raise ValueError('state outside split')
        public = public_state_from_row(state, max_asks=3)
        messages = build_base_messages(public, Action.ANSWER)
        request = {'messages': messages, 'params': params, 'model_revision': REVISION, 'tokenizer_hashes': TOKENIZER_HASHES}
        canonical = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
        requests.append({'case_id': state['case_id'], 'sample_id': state['state_id'],
                         'split': case_to_split[state['case_id']], 'request': request,
                         'request_sha256': hashlib.sha256(canonical).hexdigest()})
    if len(requests) != audit['public_states']['states'] or len({r['sample_id'] for r in requests}) != len(requests):
        raise ValueError('incomplete or duplicate request plan')
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        for item in requests:
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + '\n')
    summary = {'status': 'REQUESTS_PREPARED_NO_MODEL_RUN', 'sample_count': len(requests),
               'request_plan_sha256': sha(dest), 'private_plan': str(dest), 'base_revision': REVISION,
               'tokenizer_hashes': TOKENIZER_HASHES, 'decoding': params,
               'public_states_sha256': sha(states_path), 'case_split_sha256': sha(split_path),
               'v2_protocol_code_sha256': sha(PROJECT / 'experiments/ea_atpo_v2/src/v2_protocol.py'),
               'request_fields_exclude_gold_and_review_labels': True}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open('x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps({'status': summary['status'], 'sample_count': len(requests), 'request_plan_sha256': summary['request_plan_sha256']}))


if __name__ == '__main__':
    main()
