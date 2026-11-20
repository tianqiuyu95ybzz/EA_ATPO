#!/usr/bin/env python3
"""Audit local source files without opening the frozen holdout or emitting text/Gold."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


SFT = Path('experiments/expert_sft/multiturn_dialogue/data')
PILOT = Path('experiments/real_case_pilot_20261103/outputs/full')
SOURCE = Path('experiments/expert_sft/data/source/dataset/merged_train_dataset.jsonl')
SPLIT = Path('experiments/ea_atpo_value_router_20261105/artifacts/case_split.json')
V2 = Path('experiments/ea_atpo_v2/outputs/ask_unknown_repair_50case_20261120')
FILES = {'train': 'train_sft.jsonl', 'validation': 'validation_sft.jsonl', 'internal_test': 'internal_test_sft.jsonl'}


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path: Path):
    with path.open(encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{number}: invalid JSON') from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def reconstruct(state: dict) -> tuple[str, int]:
    initial, dialogue, trace = (state[k] for k in ('initial_messages', 'dialogue', 'state_construction_trace'))
    users = [m['content'] for m in initial if m['role'] == 'user']
    require(len(users) == 1 and bool(users[0].strip()), 'invalid initial public case')
    require(len(dialogue) == len(trace) and len(dialogue) % 2 == 0, 'invalid dialogue/trace length')
    lines = ['Visible case and answer options:\n' + users[0].strip()]
    for i, (message, event) in enumerate(zip(dialogue, trace)):
        actor, role = (('base', 'assistant'), ('patient', 'user'))[i % 2]
        require(message.get('role') == role and event.get('actor') == actor, 'invalid dialogue actor')
        require(bool(message.get('content', '').strip()) and message['content'] == event.get('response'), 'invalid dialogue event')
        label = 'Clinician question' if actor == 'base' else 'Patient reply'
        lines.append(f'{label}: {message["content"].strip()}')
    return '\n\n'.join(lines), len(dialogue) // 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-root', required=True, type=Path)
    ap.add_argument('--output-dir', required=True, type=Path)
    args = ap.parse_args()
    root = args.source_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    outputs = [out / 'data_manifest.json', out / 'split_manifest.json', out / 'sample_manifest.json']
    require(not any(p.exists() for p in outputs), 'refusing to overwrite audit manifest')

    previous = json.loads((root / SFT / 'dataset_manifest.json').read_text())
    split = json.loads((root / SPLIT).read_text())
    split_ids = {name: set(split[name]) for name in FILES}
    for a, b in [('train', 'validation'), ('train', 'internal_test'), ('validation', 'internal_test')]:
        require(not split_ids[a] & split_ids[b], f'case overlap between {a} and {b}')
    states = list(rows(root / PILOT / 'pilot_public_states.jsonl'))
    by_state = {s['state_id']: s for s in states}
    require(len(by_state) == len(states), 'duplicate source state_id')
    source = list(rows(root / SOURCE))
    by_index = {r['index']: r for r in source}
    require(len(by_index) == len(source), 'duplicate official source index')
    source_shapes = Counter(','.join(m['role'] for m in r['prompt']) for r in source)
    require(set(source_shapes) == {'system,user'}, 'official ATPO source has unexpected message structure')

    sample_manifest = []
    datasets = {}
    all_seen = set()
    observed_case_ids = {}
    for name, filename in FILES.items():
        path = root / SFT / filename
        file_hash = sha(path)
        require(file_hash == previous['files'][name]['sha256'], f'{name} file hash differs from earlier manifest')
        entries = list(rows(path))
        observed_case_ids[name] = {r['case_id'] for r in entries}
        require(observed_case_ids[name] == split_ids[name], f'{name} membership differs from frozen split')
        turns = Counter()
        for r in entries:
            case, state_id, source_index = r['case_id'], r['state_id'], r['source_row_index']
            require(r['split'] == name and state_id not in all_seen and state_id in by_state, 'invalid/duplicate split or state')
            all_seen.add(state_id)
            state = by_state[state_id]
            require(state['case_id'] == case and state['source_row_index'] == source_index, 'state source identity mismatch')
            require(r['state_index'] == state['state_index'], 'state index mismatch')
            public_text, count = reconstruct(state)
            turns[count] += 1
            msgs = r['messages']
            require([m['role'] for m in msgs] == ['system', 'user', 'assistant'], 'unexpected SFT message roles')
            require(msgs[1]['content'] == public_text and r['dialogue_turns'] == count, 'public dialogue reconstruction mismatch')
            require(r['prompt_sha256'] == digest_bytes(public_text.encode()), 'public prompt hash mismatch')
            src = by_index[source_index]
            require(next(m['content'] for m in src['prompt'] if m['role'] == 'user') == next(m['content'] for m in state['initial_messages'] if m['role'] == 'user'), 'source prompt mismatch')
            target = re.fullmatch(r'Expert Final Answer: ([A-D])', msgs[2]['content'])
            gold = re.match(r'^\s*([A-D])(?:\b|[.():])', str(src['ground_truth']['answer']), re.I)
            require(target is not None and gold is not None and target.group(1) == gold.group(1).upper(), 'unmatched historical answer label')
            # Identifiers and hashes provide provenance without copying case text or answer labels.
            sample_manifest.append({'case_id_sha256': digest_bytes(case.encode()), 'sample_id_sha256': digest_bytes(state_id.encode()), 'split': name, 'source_file': str(SFT / filename), 'source_file_sha256': file_hash, 'public_conversation_sha256': r['prompt_sha256']})
        datasets[name] = {'samples': len(entries), 'cases': len(observed_case_ids[name]), 'dialogue_turns': dict(sorted(turns.items())), 'source_file_sha256': file_hash}
    require(all_seen == set(by_state), 'SFT states and public-state source do not match exactly')

    v2_ids = []
    for path in sorted((root / V2 / 'case_batches').glob('batch_*_case_ids.json')):
        v2_ids.extend(json.loads(path.read_text()))
    require(len(v2_ids) == len(set(v2_ids)), 'V2 duplicate case selection')
    v2_rows = list(rows(root / V2 / 'merged/pilot_trajectories.jsonl'))
    v2_states = {r['state_id'] for r in v2_rows}
    require(all(r['case_id'] in set(v2_ids) and r['state_id'] in by_state and by_state[r['state_id']]['case_id'] == r['case_id'] for r in v2_rows), 'V2 state/case mapping failure')
    overlaps = {k: len(v & set(v2_ids)) for k, v in observed_case_ids.items()}
    manifest = {
        'schema_version': 1, 'source_kind': 'model_simulated_pilot_states_not_human_patient_transcripts',
        'official_source': {'path': str(SOURCE), 'sha256': sha(root / SOURCE), 'rows': len(source), 'prompt_role_shapes': dict(source_shapes)},
        'public_states': {'path': str(PILOT / 'pilot_public_states.jsonl'), 'sha256': sha(root / PILOT / 'pilot_public_states.jsonl'), 'states': len(states)},
        'historical_trajectories': {'path': str(PILOT / 'pilot_trajectories.jsonl'), 'sha256': sha(root / PILOT / 'pilot_trajectories.jsonl')},
        'prior_manifest_sha256': sha(root / SFT / 'dataset_manifest.json'),
        'split_manifest_sha256': sha(root / SPLIT), 'splits': datasets,
        'v2_pilot': {'cases': len(v2_ids), 'states': len(v2_states), 'trajectories': len(v2_rows), 'actions': dict(Counter(r['action'] for r in v2_rows)), 'case_overlap_by_split': overlaps, 'state_overlap_with_sft': len(v2_states & all_seen), 'trajectory_sha256': sha(root / V2 / 'merged/pilot_trajectories.jsonl'), 'included_as_extra_sft_rows': False},
        'case_overlap_between_splits': 0, 'review_status': 'PENDING_ALL',
        'frozen_holdout_opened': False, 'source_root_read_only': str(root),
    }
    split_manifest = {'unit': 'case_id', 'source_file': str(SPLIT), 'source_sha256': sha(root / SPLIT), 'seed': split.get('seed'), 'splits': {k: {'cases': len(v), 'case_ids_sha256': digest_bytes('\n'.join(sorted(v)).encode()), 'samples': datasets[k]['samples']} for k, v in split_ids.items()}, 'pairwise_case_overlap': 0, 'holdout_opened': False}
    for path, obj in zip(outputs, (manifest, split_manifest, {'schema_version': 1, 'samples': sample_manifest})):
        with path.open('x', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write('\n')
    print(json.dumps({'status': 'PASS', 'splits': datasets, 'v2_pilot': manifest['v2_pilot'], 'official_rows': len(source)}, indent=2))


if __name__ == '__main__':
    main()
