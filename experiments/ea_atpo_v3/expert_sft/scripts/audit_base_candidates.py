#!/usr/bin/env python3
"""Archive historical V2 raw Base outputs as *unverified* reuse candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from audit_data import FILES, PILOT, SFT, V2, rows, sha

REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
TOKENIZER_HASHES = {'tokenizer.json': 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
                    'tokenizer_config.json': 'd5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--audit-manifest', type=Path, required=True)
    ap.add_argument('--base-model-dir', type=Path, required=True)
    ap.add_argument('--private-dir', type=Path, required=True)
    ap.add_argument('--summary', type=Path, required=True)
    args = ap.parse_args()
    root, private = args.source_root.resolve(), args.private_dir.resolve()
    if private.name != 'private':
        raise ValueError('candidate raw answers must stay in the private/ directory')
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(private, 0o700)
    dest = private / 'base_v2_reuse_candidates.jsonl'
    if dest.exists() or args.summary.exists():
        raise FileExistsError('refusing to overwrite prior candidate outputs')
    audit = json.loads(args.audit_manifest.read_text())
    for name, expected in TOKENIZER_HASHES.items():
        if sha(args.base_model_dir / name) != expected:
            raise ValueError(f'Base tokenizer hash mismatch: {name}')
    run_root = root / V2
    merged = json.loads((run_root / 'merged/merged_manifest.json').read_text())
    trajectory = run_root / 'merged/pilot_trajectories.jsonl'
    if sha(trajectory) != merged['trajectory_sha256'] or sha(trajectory) != audit['v2_pilot']['trajectory_sha256']:
        raise ValueError('V2 trajectory hash mismatch')
    cfg_path = run_root / 'configs/phase1_patient_review_v2.json'
    config = json.loads(cfg_path.read_text())
    if (sha(cfg_path) != merged['config_sha256'] or config['roles']['base']['model_revision'] != REVISION
        or config['roles']['base']['enable_thinking'] is not False
        or config['inference']['temperature'] != 0.0 or config['inference']['top_p'] != 1.0
        or config['inference']['answer_max_tokens'] != 24):
        raise ValueError('V2 Base config differs from recorded contract')
    for i in range(1, 6):
        m = json.loads((run_root / f'batches/batch_{i:02}/run_manifest.json').read_text())
        if (m['config_sha256'] != sha(cfg_path) or m['models']['base']['model_revision'] != REVISION
            or m['models']['base']['enable_thinking'] is not False
            or m['data_sha256']['public_states'] != audit['public_states']['sha256']
            or m['data_sha256']['case_split'] != audit['split_manifest_sha256']
            or m['data_sha256']['source_jsonl'] != audit['official_source']['sha256']):
            raise ValueError('V2 batch provenance mismatch')
    states = {r['state_id']: r for r in rows(root / PILOT / 'pilot_public_states.jsonl')}
    train = {r['state_id']: r for r in rows(root / SFT / FILES['train'])}
    selected = []
    for r in rows(trajectory):
        if r['action'] != 'ANSWER':
            continue
        s = states[r['state_id']]
        visible = [m for m in s['initial_messages'] + s['dialogue'] if m['role'] != 'system']
        visible_hash = hashlib.sha256(json.dumps(visible, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        trace = r['trace']
        if (r['state_id'] not in train or train[r['state_id']]['case_id'] != r['case_id']
            or s['case_id'] != r['case_id'] or r['public_state'] != visible
            or r['public_state_digest'] != visible_hash or not r['valid'] or len(trace) != 1
            or trace[0]['role'] != 'base' or trace[0]['stage'] != 'initial_answer'
            or not re.fullmatch(r'\s*Final Answer:\s*[A-D]\s*', trace[0].get('output', ''), re.I)):
            raise ValueError('V2 candidate failed case/state/raw-output checks')
        raw = trace[0]['output']
        selected.append({'case_id': r['case_id'], 'sample_id': r['state_id'], 'split': 'train',
                         'raw_output': raw, 'raw_output_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                         'public_state_digest': visible_hash, 'rollout_seed': r['rollout_seed'],
                         'source_trajectory_sha256': sha(trajectory), 'status': 'UNVERIFIED_PROMPT_AT_RUNTIME'})
    if len(selected) != 146 or len({r['sample_id'] for r in selected}) != 146:
        raise ValueError('unexpected V2 Base candidate count')
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')
    summary = {'status': 'CANDIDATES_ONLY_NOT_APPROVED_FOR_REUSE', 'candidates': len(selected),
               'case_split': 'train', 'base_revision': REVISION, 'tokenizer_hashes': TOKENIZER_HASHES,
               'recorded_decoding': {'temperature': 0.0, 'top_p': 1.0, 'max_tokens': 24, 'enable_thinking': False, 'structured_output_regex': '^Final Answer: [A-D]$'},
               'trajectory_sha256': sha(trajectory), 'config_sha256': sha(cfg_path),
               'candidate_file_sha256': sha(dest), 'raw_candidate_file_private': str(dest),
               'verified_for_v3_reuse': 0,
               'blocker': 'The historical records do not preserve the exact per-request prompt or its hash and runtime code was not locked to the recorded commit. Reconstructing the prompt is not proof of what was sent. Re-generate using one frozen V3 prompt and capture request hashes.'}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open('x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps({'candidates': len(selected), 'verified_for_v3_reuse': 0, 'status': summary['status']}))


if __name__ == '__main__':
    main()
