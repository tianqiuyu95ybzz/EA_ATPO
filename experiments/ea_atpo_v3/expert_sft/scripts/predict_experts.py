#!/usr/bin/env python3
"""Generate private A/B predictions without exposing any review labels to models."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from audit_data import rows, sha
from expert_training_data import ACTIONS, validate_dataset
from train_experts import EXP, PROJECT, REVISION, encode_prompt, prompt_text, validate_model, validate_smoke, write_json
from evaluate_experts import aware_datetime
from verify_awq_lora import assert_post_load_safe, gate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--expert', choices=('A', 'B'), required=True)
    p.add_argument('--split', choices=('validation', 'internal_test'), required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--expert-b-copy-manifest', type=Path)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--model-lock', type=Path)
    p.add_argument('--paired-dir', type=Path, required=True)
    p.add_argument('--review-queue', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--audit-manifest', type=Path, required=True)
    p.add_argument('--base-answers', type=Path, required=True)
    p.add_argument('--base-manifest', type=Path, required=True)
    p.add_argument('--smoke-report', type=Path, required=True)
    p.add_argument('--smoke-adapter', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.gpu not in {0, 2, 3, 4, 5, 6} or os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        raise ValueError('exactly one allowed physical GPU is required; 1 and 7 are prohibited')
    out = args.output_dir.resolve()
    if (EXP / 'data/private').resolve() not in out.parents or out.exists():
        raise ValueError('raw predictions require a new private V3 directory')
    checkpoint = args.checkpoint.resolve()
    if (EXP / 'outputs' / f'expert_{args.expert.lower()}').resolve() not in checkpoint.parents:
        raise ValueError('checkpoint must belong to the corresponding independent Expert')
    adapter = checkpoint / 'adapter_model.safetensors'
    checkpoint_manifest = json.loads((checkpoint / 'epoch_manifest.json').read_text())
    if checkpoint_manifest['adapter_sha256'] != sha(adapter):
        raise ValueError('checkpoint adapter hash mismatch')
    config = json.loads(args.config.read_text())
    if config.get('revision') != REVISION or config.get('tokenizer_revision') != REVISION:
        raise ValueError('wrong model/tokenizer revision')
    if args.split == 'internal_test':
        if args.model_lock is None:
            raise ValueError('Internal Test requires a pre-existing Dev-frozen model lock')
        lock = json.loads(args.model_lock.read_text())
        calibration = Path(lock['threshold_calibration_manifest'])
        cal = json.loads(calibration.read_text())
        if (lock.get('status') != 'FROZEN' or lock.get('expert') != args.expert
            or lock.get('checkpoint_sha256') != sha(adapter) or lock.get('config_sha256') != sha(args.config)
            or lock.get('threshold_chosen_on') != 'validation' or lock.get('oracle_used_for_threshold') is not False
            or lock.get('threshold_calibration_sha256') != sha(calibration)
            or cal.get('split') != 'validation' or cal.get('oracle_used') is not False
            or cal.get('threshold') != lock.get('threshold')
            or aware_datetime(cal.get('completed_at_utc', '')) > aware_datetime(lock.get('frozen_at_utc', ''))):
            raise ValueError('Internal Test model/threshold lock is invalid')
    check = subprocess.run([sys.executable, str(EXP / 'scripts/check_gold_review.py'),
                            '--queue', str(args.review_queue), '--source-root', str(args.source_root),
                            '--audit-manifest', str(args.audit_manifest), '--base-answers', str(args.base_answers),
                            '--base-manifest', str(args.base_manifest)], capture_output=True, text=True)
    if check.returncode:
        raise ValueError('human Gold review/Base provenance gate failed')
    validate_smoke(args.smoke_report, args.smoke_adapter)
    paired, data = validate_dataset(args.paired_dir, args.review_queue, args.expert, splits=(args.split,))
    if paired['base_answers_sha256'] != sha(args.base_answers) or paired['base_manifest_sha256'] != sha(args.base_manifest):
        raise ValueError('paired data do not match Base artifact')
    model_path = args.model_dir.resolve()
    if args.expert == 'A' and model_path != (PROJECT.parent / 'EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ').resolve():
        raise ValueError('Expert A must use the original pinned AWQ initialization')
    fingerprints = validate_model(args.expert, model_path, args.expert_b_copy_manifest)
    base_answers = {r['sample_id']: r for r in rows(args.base_answers)}
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    started = time.monotonic()
    report = {'status': 'FAILED', 'expert': args.expert, 'split': args.split, 'stage': 'resource_gate'}
    try:
        gate(args.gpu, out / 'gpu_before_load.json')
        for subdir in ('hf', 'xdg', 'tmp', 'triton', 'torch', 'cuda', 'flashinfer'):
            (out / 'runtime' / subdir).mkdir(parents=True, mode=0o700)
        os.environ.update({'HF_HOME': str(out / 'runtime/hf'), 'XDG_CACHE_HOME': str(out / 'runtime/xdg'),
                           'TMPDIR': str(out / 'runtime/tmp'), 'TRITON_CACHE_DIR': str(out / 'runtime/triton'),
                           'TORCH_HOME': str(out / 'runtime/torch'), 'CUDA_CACHE_PATH': str(out / 'runtime/cuda'),
                           'FLASHINFER_WORKSPACE_BASE': str(out / 'runtime/flashinfer'),
                           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
        import torch
        import gptqmodel  # noqa: F401
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        gate(args.gpu, out / 'gpu_immediately_before_model_load.json')
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model_config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        quant = getattr(model_config, 'quantization_config', None) or {}
        if model_config.model_type != 'qwen3' or str(quant.get('quant_method', '')).lower() != 'awq':
            raise ValueError('not the pinned Qwen3 AWQ initialization')
        quant['backend'] = 'torch_awq'
        model_config.quantization_config = quant
        base = AutoModelForCausalLM.from_pretrained(model_path, config=model_config, device_map={'': 0},
                                                    local_files_only=True, trust_remote_code=False, attn_implementation='sdpa')
        model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False)
        model.eval()
        assert_post_load_safe(args.gpu)
        action_tokens = [tokenizer.encode(label, add_special_tokens=False)[0] for label in ACTIONS]
        if len(set(action_tokens)) != len(ACTIONS):
            raise ValueError('action token prefixes are not distinct')
        if args.split == 'internal_test':
            marker = EXP / 'outputs/evaluation' / f'internal_test_{args.expert}_generation_started.json'
            marker.parent.mkdir(parents=True, exist_ok=True)
            write_json(marker, {'expert': args.expert, 'model_lock_sha256': sha(args.model_lock),
                                'checkpoint_sha256': sha(adapter), 'started_at_utc': datetime.now(timezone.utc).isoformat()})
        generated_at = datetime.now(timezone.utc).isoformat()
        report['stage'] = 'generation'
        prediction_path = out / 'raw_predictions.jsonl'
        fd = os.open(prediction_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        count = 0
        with os.fdopen(fd, 'w', encoding='utf-8') as f, torch.inference_mode():
            for item in data[args.split]:
                base_item = base_answers[item['sample_id']]
                if (base_item['case_id'], base_item['split'], base_item['status']) != (item['case_id'], args.split, 'VERIFIED'):
                    raise ValueError('Base sample identity mismatch')
                if args.expert == 'A':
                    ids = encode_prompt(tokenizer, prompt_text(tokenizer, 'A', 'action', item['input']), 3712, torch, torch.device('cuda:0'))
                    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1, action_tokens].float()
                    probabilities = torch.softmax(logits, dim=-1)
                    action = ACTIONS[int(probabilities.argmax().item())]
                    score = float(probabilities[ACTIONS.index('REVISE')].item())
                else:
                    action, score = 'REVISE', 0.5  # fixed pre-decision constant; no Expert output feature
                answer = ''
                if action == 'REVISE':
                    ids = encode_prompt(tokenizer, prompt_text(tokenizer, args.expert, 'answer', item['input']), 3712, torch, torch.device('cuda:0'))
                    generation = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=384,
                                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
                    answer = tokenizer.decode(generation[0, ids.shape[1]:], skip_special_tokens=True)
                raw = json.dumps({'action': action, 'answer': answer}, ensure_ascii=False) if args.expert == 'A' else answer
                record = {'case_id': item['case_id'], 'sample_id': item['sample_id'], 'split': args.split,
                          'base_answer': base_item['raw_output'], 'expert_output': raw,
                          'expert_output_sha256': hashlib.sha256(raw.encode()).hexdigest(), 'decision_score': score}
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
                f.flush()
                count += 1
                assert_post_load_safe(args.gpu)
        report.update({'status': 'COMPLETED', 'stage': 'complete', 'sample_count': count,
                       'raw_predictions_sha256': sha(prediction_path), 'base_answers_sha256': sha(args.base_answers),
                       'paired_manifest_sha256': sha(args.paired_dir / 'paired_dataset_manifest.json'),
                       'reviewed_queue_sha256': sha(args.review_queue), 'checkpoint_sha256': sha(adapter),
                       'config_sha256': sha(args.config), 'model_lock_sha256': sha(args.model_lock) if args.model_lock else None,
                       'model_fingerprints': fingerprints, 'generated_at_utc': generated_at,
                       'score_visibility': 'predecision', 'score_features': ['a_action_logprobs'] if args.expert == 'A' else ['constant'],
                       'generation': {'do_sample': False, 'top_p': 1.0, 'max_new_tokens': 384, 'enable_thinking': False},
                       'inputs_sent_to_model': ['conversation', 'base_answer'] if args.expert == 'A' else ['conversation'],
                       'gpu': args.gpu, 'gpu_peak_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1),
                       'gpu_peak_reserved_mib': round(torch.cuda.max_memory_reserved() / 2**20, 1)})
    except Exception as exc:
        report.update({'error_class': type(exc).__name__, 'error': str(exc)[:1000]})
        raise
    finally:
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        write_json(out / 'prediction_run_manifest.json', report)
    print(json.dumps({'status': report['status'], 'samples': report['sample_count'], 'manifest': str(out / 'prediction_run_manifest.json')}))


if __name__ == '__main__':
    main()
