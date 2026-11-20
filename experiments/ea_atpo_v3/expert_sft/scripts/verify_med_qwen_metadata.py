#!/usr/bin/env python3
"""Read-only med_qwen metadata/local inventory. This is not an inference test."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def request(url: str) -> tuple[int | None, dict, str | None, str | None]:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'ea-atpo-v3-expert-audit/1'})
        with urllib.request.urlopen(req, timeout=20) as response:
            raw, status = response.read(), response.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read(), exc.code
    except Exception as exc:
        return None, {}, None, f'{type(exc).__name__}: {exc}'
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
    return status, parsed, hashlib.sha256(raw).hexdigest(), None


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('refusing to overwrite verification report')
    meta_url = 'https://modelscope.cn/api/v1/models/zky001/med_qwen'
    files_url = 'https://modelscope.cn/api/v1/models/zky001/med_qwen/repo/files?Revision=master&Recursive=True'
    ms, meta, meta_sha, meta_error = request(meta_url)
    fs, tree, tree_sha, tree_error = request(files_url)
    data = meta.get('Data') if isinstance(meta.get('Data'), dict) else {}
    local = args.model_dir.resolve()
    filenames = sorted(f.name for f in local.iterdir() if f.is_file()) if local.is_dir() else []
    index_path = local / 'model.safetensors.index.json'
    shards = []
    index_error = None
    if index_path.is_file():
        try:
            idx = json.loads(index_path.read_text())
            shards = sorted(set(idx['weight_map'].values()))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            index_error = type(exc).__name__
    missing_shards = [name for name in shards if not (local / name).is_file()]
    tokenizer_present = (local / 'tokenizer.json').is_file() and (local / 'tokenizer_config.json').is_file()
    config_present = (local / 'config.json').is_file()
    revision = data.get('Revision')
    license_name = data.get('LicenseName') or data.get('License') or ''
    immutable_revision = isinstance(revision, str) and len(revision) >= 40 and all(x in '0123456789abcdefABCDEF' for x in revision)
    prerequisites = {'weight_index': index_path.is_file() and index_error is None,
                     'nonempty_weight_shard_set': bool(shards), 'all_indexed_shards_present': bool(shards) and not missing_shards,
                     'config': config_present, 'tokenizer': tokenizer_present,
                     'immutable_revision': immutable_revision, 'license_identified': bool(license_name)}
    result = {'status': 'BLOCKED' if not all(prerequisites.values()) else 'PENDING_GPU_INFERENCE',
              'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'model_id': 'zky001/med_qwen',
              'metadata_url': meta_url, 'metadata_http_status': ms, 'metadata_response_sha256': meta_sha,
              'metadata_error': meta_error, 'architecture_metadata': data.get('Architectures'),
              'revision_metadata': revision, 'license_metadata': license_name,
              'file_tree_url': files_url, 'file_tree_http_status': fs, 'file_tree_api_code': tree.get('Code'),
              'file_tree_response_sha256': tree_sha, 'file_tree_error': tree_error,
              'local_model_dir': str(local), 'local_filenames': filenames, 'indexed_shards': len(shards),
              'missing_indexed_shards': len(missing_shards), 'index_parse_error': index_error,
              'prerequisites': prerequisites, 'runtime_versions': {x: version(x) for x in ('torch', 'transformers', 'peft', 'safetensors')},
              'full_shard_hash_verification': 'NOT_RUN', 'actual_gpu_inference': 'NOT_RUN',
              'output_format_verified': False, 'training_started': False,
              'note': 'Metadata/local inventory alone cannot approve this model. Verify shard hashes, architecture, dependencies, license and immutable revision, then perform real GPU inference and output parsing before training.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'metadata_http_status': ms, 'file_tree_http_status': fs,
                      'revision': revision, 'license': license_name, 'local_file_count': len(filenames),
                      'actual_gpu_inference': result['actual_gpu_inference']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
