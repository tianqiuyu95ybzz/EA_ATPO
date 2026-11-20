"""Public-only Expert prompts and reviewed-target boundary validation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from audit_data import rows, sha

ACTIONS = ('KEEP', 'REVISE', 'ABSTAIN')
SYSTEMS = {
    'a_action': ('You are a medical diagnostic assistant reviewing a Base answer. '
                 'Using only the visible case and Base answer, choose KEEP if the answer should be preserved, '
                 'REVISE if a reliable correction is needed, or ABSTAIN if you cannot safely decide.'),
    'a_correction': ('You are a medical diagnostic assistant. Using only the visible case and Base answer, '
                     'provide a complete corrected answer. Do not invent unavailable patient facts.'),
    'b_answer': ('You are a medical diagnostic assistant. Using only the visible case and dialogue, '
                 'provide a complete answer. Do not invent unavailable patient facts.'),
}


def public_messages(expert: str, mode: str, sample_input: dict) -> list[dict]:
    expected = {'conversation', 'base_answer'} if expert == 'A' else {'conversation'}
    if expert not in {'A', 'B'} or set(sample_input) != expected:
        raise ValueError('model input contains unexpected, missing or privileged fields')
    if any(not isinstance(v, str) or not v for v in sample_input.values()):
        raise ValueError('public input fields must be nonempty strings')
    key = {'A': {'action': 'a_action', 'answer': 'a_correction'}, 'B': {'answer': 'b_answer'}}[expert].get(mode)
    if key is None:
        raise ValueError('invalid Expert prompt mode')
    public = {name: sample_input[name] for name in sorted(expected)}
    return [{'role': 'system', 'content': SYSTEMS[key]},
            {'role': 'user', 'content': json.dumps(public, ensure_ascii=False, sort_keys=True)}]


def validate_dataset(paired_dir: Path, review_queue: Path, expert: str,
                     splits: tuple[str, ...] = ('train', 'validation')) -> tuple[dict, dict]:
    """Verify the explicitly selected splits match the reviewed queue."""
    manifest = json.loads((paired_dir / 'paired_dataset_manifest.json').read_text())
    if (manifest.get('status') != 'REVIEW_GATED' or not manifest.get('review_validation', {}).get('ready_for_formal_training')
        or manifest.get('reviewed_queue_sha256') != sha(review_queue)):
        raise ValueError('paired dataset is not bound to a fully reviewed queue')
    with review_queue.open(encoding='utf-8', newline='') as f:
        review = list(csv.DictReader(f))
    datasets = {}
    prefix = 'expert_a' if expert == 'A' else 'expert_b_c'
    if not splits or any(s not in {'train', 'validation', 'internal_test'} for s in splits):
        raise ValueError('unexpected dataset split')
    for split in splits:
        filename = f'{prefix}_{split}.jsonl'
        path = paired_dir / filename
        if manifest.get('files', {}).get(filename, {}).get('sha256') != sha(path):
            raise ValueError('dataset checksum differs from paired manifest')
        data = list(rows(path))
        relevant = {r['sample_id']: r for r in review if r['split'] == split
                    and (r['gold_status'] in {'APPROVED', 'AMBIGUOUS'} if expert == 'A' else r['gold_status'] == 'APPROVED')}
        if len(relevant) != len(data) or set(relevant) != {r['sample_id'] for r in data}:
            raise ValueError('training dataset coverage differs from reviewed queue')
        for item in data:
            r = relevant[item['sample_id']]
            if (set(item) != {'case_id', 'sample_id', 'split', 'input', 'label'}
                or item['case_id'] != r['case_id'] or item['split'] != split):
                raise ValueError('dataset sample identity/schema mismatch')
            expected_input = {'conversation': r['conversation']}
            if expert == 'A':
                expected_input['base_answer'] = r['base_answer']
                expected_label = {'action': r['target_action'],
                                  'correction': r['reviewed_gold'] if r['target_action'] == 'REVISE' else None}
                if expected_label['action'] not in ACTIONS:
                    raise ValueError('invalid reviewed action')
            else:
                expected_label = {'answer': r['reviewed_gold']}
            if item['input'] != expected_input or item['label'] != expected_label:
                raise ValueError('dataset public input or target differs from reviewed queue')
            public_messages(expert, 'action' if expert == 'A' else 'answer', item['input'])
        if not data:
            raise ValueError(f'{split} split has no usable reviewed samples')
        datasets[split] = data
    return manifest, datasets
