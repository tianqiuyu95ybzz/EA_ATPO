#!/usr/bin/env python3
"""Hash this V3 experiment's ignored PFS artifacts without exporting their contents."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from audit_data import sha

EXP = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[4]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('refusing to overwrite an existing inventory')
    files, links = [], []
    for root in (EXP / 'data/private', EXP / 'outputs', EXP / 'models',
                 EXP / 'scripts/__pycache__', EXP / 'tests/__pycache__'):
        if not root.exists():
            continue
        for directory, subdirs, names in os.walk(root, followlinks=False):
            current = Path(directory)
            for name in sorted(subdirs + names):
                path = current / name
                if path.is_symlink():
                    links.append({'path': str(path.relative_to(EXP)), 'target': os.readlink(path)})
                elif path.is_file():
                    files.append({'path': str(path.relative_to(EXP)), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    for name in ('.v3b_3831174', '.v3b_3843078', '.v3b_3849589'):
        path = PROJECT.parent / name
        if path.is_symlink():
            links.append({'path': str(path), 'target': os.readlink(path)})
    files.sort(key=lambda r: r['path'])
    result = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'experiment_root': str(EXP),
              'file_count': len(files), 'total_bytes': sum(x['bytes'] for x in files), 'files': files,
              'symlinks': links, 'contents_exported': False,
              'shared_cache_note': 'Attempt 02 tried a FlashInfer JIT build in /pfs/pfs-spMCeu/image_team/persistent-cache-agnes-intern-image/cache/flashinfer/0.7.2.post1/100a/cached_ops/sampling; pre-existing state is unknown and no cleanup was performed.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps({'file_count': result['file_count'], 'total_bytes': result['total_bytes'], 'symlink_count': len(links), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
