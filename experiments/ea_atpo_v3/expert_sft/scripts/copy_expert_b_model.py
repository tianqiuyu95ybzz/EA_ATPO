#!/usr/bin/env python3
"""Physically copy the pinned AWQ initialization for Expert B, without overwrites."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from audit_data import sha
from verify_awq_lora import LOCK, LOCK_SHA, REVISION

EXP = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True)
    args = p.parse_args()
    source, dest = args.source.resolve(), args.destination.resolve()
    if (EXP / 'models').resolve() not in dest.parents or dest == source or source in dest.parents:
        raise ValueError('Expert B copy must be a new independent V3 models/ directory')
    if dest.exists() or args.summary.exists():
        raise FileExistsError('refusing to overwrite an existing copy or manifest')
    if sha(LOCK) != LOCK_SHA:
        raise ValueError('AWQ source lock mismatch')
    lock = json.loads(LOCK.read_text())
    if lock['huggingface_revision'] != REVISION:
        raise ValueError('incorrect model revision')
    extras = [{'name': name, 'size': (source / name).stat().st_size, 'sha256': sha(source / name)}
              for name in ('LICENSE', 'README.md', '.gitattributes', 'model_source_lock.json') if (source / name).is_file()]
    entries = lock['weight_files'] + lock['auxiliary_files'] + extras
    if any((source / x['name']).stat().st_size != x['size'] for x in entries):
        raise ValueError('source file missing or size mismatch')
    dest.mkdir(parents=True, exist_ok=False, mode=0o700)
    records = []
    for entry in entries:
        src, target = source / entry['name'], dest / entry['name']
        h = hashlib.sha256()
        with src.open('rb') as inp, target.open('xb') as out:
            for block in iter(lambda: inp.read(8 * 1024 * 1024), b''):
                out.write(block)
                h.update(block)
            out.flush()
            os.fsync(out.fileno())
        if h.hexdigest() != entry['sha256'] or sha(target) != entry['sha256']:
            raise ValueError(f'copy fingerprint mismatch: {entry["name"]}; partial copy retained')
        a, b = src.stat(), target.stat()
        if (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino):
            raise ValueError('copy unexpectedly shares the source inode')
        records.append({'name': entry['name'], 'size': entry['size'], 'sha256': entry['sha256'], 'independent_inode': True})
        print(f'Copied and verified {entry["name"]}', flush=True)
    result = {'status': 'COPIED_AND_VERIFIED', 'repo_id': 'Qwen/Qwen3-32B-AWQ', 'revision': REVISION,
              'tokenizer_revision': REVISION, 'source_lock_sha256': LOCK_SHA, 'source_directory': str(source),
              'independent_copy_directory': str(dest), 'files': records,
              'weight_copy_is_not_adapter_or_checkpoint': True, 'training_started': False}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'files': len(records), 'summary': str(args.summary)}))


if __name__ == '__main__':
    main()
