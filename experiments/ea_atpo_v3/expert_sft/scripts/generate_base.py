#!/usr/bin/env python3
"""Generate frozen Base responses, only after an immediate no-sharing GPU gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from audit_base_candidates import REVISION, TOKENIZER_HASHES
from audit_data import rows, sha

PROJECT = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parents[1]
FINGERPRINT = {
    'config.json': 'f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30',
    'generation_config.json': '2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2',
    'model.safetensors.index.json': 'f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc',
    'merges.txt': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
    'model-00001-of-00005.safetensors': '31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f',
    'model-00002-of-00005.safetensors': '5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282',
    'model-00003-of-00005.safetensors': 'c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836',
    'model-00004-of-00005.safetensors': 'b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a',
    'model-00005-of-00005.safetensors': '20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff',
    **TOKENIZER_HASHES,
}


def write_json(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


def gate(gpu: int, output: Path) -> None:
    script = PROJECT / 'experiments/expert_sft/multiturn_dialogue/scripts/gpu_safety_gate.py'
    result = subprocess.run([sys.executable, str(script), '--gpu', str(gpu)], capture_output=True, text=True)
    with output.open('x', encoding='utf-8') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write('\n' + result.stderr)
    if result.returncode:
        raise RuntimeError(f'GPU {gpu} safety gate blocked; see {output}')


def gpu_used(gpu: int) -> int:
    value = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    return int(value.strip())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.gpu not in {0, 2, 3, 4, 5, 6} or os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        raise ValueError('set CUDA_VISIBLE_DEVICES to exactly one allowed physical GPU; 1 and 7 are prohibited')
    private_root = (EXP / 'data/private').resolve()
    out = args.output_dir.resolve()
    if private_root not in out.parents:
        raise ValueError('raw Base outputs must stay under the V3 private data directory')
    out.mkdir(mode=0o700, parents=False, exist_ok=False)
    os.chmod(out, 0o700)
    gate(args.gpu, out / 'gpu_before_fingerprinting.json')
    started = time.monotonic()
    plan_meta = json.loads(args.plan_manifest.read_text())
    if (plan_meta['status'] != 'REQUESTS_PREPARED_NO_MODEL_RUN' or plan_meta['base_revision'] != REVISION
        or sha(args.plan) != plan_meta['request_plan_sha256']
        or sha(PROJECT / 'experiments/ea_atpo_v2/src/v2_protocol.py') != plan_meta['v2_protocol_code_sha256']):
        raise ValueError('request plan or protocol code does not match its frozen manifest')
    requests = list(rows(args.plan))
    if len(requests) != plan_meta['sample_count'] or len({r['sample_id'] for r in requests}) != len(requests):
        raise ValueError('incomplete or duplicate request plan')
    for r in requests:
        req = r['request']
        calculated = hashlib.sha256(json.dumps(req, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if (calculated != r['request_sha256'] or req['model_revision'] != REVISION
            or req['tokenizer_hashes'] != TOKENIZER_HASHES or req['params'] != plan_meta['decoding']):
            raise ValueError('per-request hash or frozen generation contract mismatch')
    model_dir = args.model_dir.resolve()
    fingerprints = {name: sha(model_dir / name) for name in FINGERPRINT}
    if fingerprints != FINGERPRINT:
        raise ValueError('Base checkpoint/tokenizer does not match pinned source fingerprint')
    runtime = out / 'runtime'
    for subdir in ('hf', 'xdg', 'tmp', 'triton', 'torch'):
        (runtime / subdir).mkdir(parents=True, mode=0o700)
    os.environ.update({'HF_HOME': str(runtime / 'hf'), 'XDG_CACHE_HOME': str(runtime / 'xdg'),
                       'TMPDIR': str(runtime / 'tmp'), 'TRITON_CACHE_DIR': str(runtime / 'triton'),
                       'TORCH_HOME': str(runtime / 'torch'), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    import vllm
    import torch
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    if vllm.__version__ != '0.33.0':
        raise RuntimeError('use the pinned V2 vLLM 0.33.0 runtime')
    gate(args.gpu, out / 'gpu_immediately_before_model_load.json')
    engine = LLM(model=str(model_dir), tokenizer=str(model_dir), dtype='bfloat16',
                 tensor_parallel_size=1, trust_remote_code=False, gpu_memory_utilization=0.12,
                 max_model_len=4096, max_num_seqs=1)
    decoding = plan_meta['decoding']
    sampling = SamplingParams(temperature=decoding['temperature'], top_p=decoding['top_p'],
                              max_tokens=decoding['max_tokens'], seed=decoding['seed'],
                              structured_outputs=StructuredOutputsParams(regex=decoding['structured_outputs_regex']))
    peak = gpu_used(args.gpu)
    bad = 0
    raw_path = out / 'base_answers.jsonl'
    fd = os.open(raw_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        for r in requests:
            result = engine.chat(r['request']['messages'], sampling_params=sampling, use_tqdm=False,
                                 chat_template_kwargs={'enable_thinking': decoding['enable_thinking']})
            raw = result[0].outputs[0].text
            valid = bool(raw and re.fullmatch(decoding['structured_outputs_regex'], raw))
            bad += not valid
            record = {'case_id': r['case_id'], 'sample_id': r['sample_id'], 'split': r['split'],
                      'request_sha256': r['request_sha256'], 'raw_output': raw,
                      'raw_output_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                      'status': 'VERIFIED' if valid else 'INVALID_FORMAT'}
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
            f.flush()
            peak = max(peak, gpu_used(args.gpu))
    if bad:
        write_json(out / 'generation_incomplete.json', {'status': 'BLOCKED_INVALID_OUTPUTS', 'invalid_outputs': bad, 'samples': len(requests), 'raw_answers_sha256': sha(raw_path)})
        raise RuntimeError('one or more Base outputs failed the fixed answer format')
    result = {'status': 'VERIFIED', 'sample_count': len(requests), 'base_revision': REVISION,
              'raw_answers_sha256': sha(raw_path), 'request_plan_sha256': sha(args.plan),
              'plan_manifest_sha256': sha(args.plan_manifest), 'model_fingerprints': fingerprints,
              'gpu': args.gpu, 'gpu_peak_used_mib': peak, 'elapsed_seconds': round(time.monotonic() - started, 3),
              'vllm': vllm.__version__, 'torch': torch.__version__, 'cuda_build': torch.version.cuda,
              'decoding': decoding, 'checks': {k: True for k in ('model_revision', 'tokenizer', 'prompt', 'decoding', 'data', 'case_ids', 'raw_output_completeness')}}
    write_json(out / 'base_generation_manifest.json', result)
    print(json.dumps({'status': result['status'], 'samples': result['sample_count'], 'manifest': str(out / 'base_generation_manifest.json')}))


if __name__ == '__main__':
    main()
