#!/usr/bin/env python3
"""Exercise the A/B loss implementation on AWQ with synthetic non-medical text."""
import argparse
import importlib.metadata
import json
import math
import os
import signal
import time
from pathlib import Path

from audit_data import sha
from expert_training_data import ACTIONS
from train_experts import EXP, compute_sample_loss, validate_smoke, write_json
from verify_awq_lora import LOCK, LOCK_SHA, REVISION, assert_post_load_safe, gate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--smoke-report', type=Path, required=True)
    p.add_argument('--smoke-adapter', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.gpu not in {0, 2, 3, 4, 5, 6} or os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        raise ValueError('one allowed physical GPU is required')
    out = args.output_dir.resolve()
    if (EXP / 'outputs').resolve() not in out.parents or out.exists():
        raise ValueError('use a new V3 outputs/ directory')
    validate_smoke(args.smoke_report, args.smoke_adapter)
    if sha(LOCK) != LOCK_SHA:
        raise ValueError('source lock changed')
    lock = json.loads(LOCK.read_text())
    model_path = args.model_dir.resolve()
    if lock['huggingface_revision'] != REVISION:
        raise ValueError('wrong revision')
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    started = time.monotonic()
    report = {'status': 'FAILED', 'stage': 'resource_gate', 'formal_training': False,
              'synthetic_nonmedical_only': True, 'model_revision': REVISION, 'gpu': args.gpu, 'steps': []}
    def interrupted(signum, frame):
        raise InterruptedError(f'process received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        gate(args.gpu, out / 'gpu_before_fingerprinting.json')
        for x in lock['weight_files'] + lock['auxiliary_files']:
            path = model_path / x['name']
            if path.stat().st_size != x['size'] or sha(path) != x['sha256']:
                raise ValueError('AWQ model fingerprint mismatch')
        for d in ('hf', 'xdg', 'tmp', 'triton', 'torch', 'cuda', 'flashinfer'):
            (out / 'runtime' / d).mkdir(parents=True, mode=0o700)
        os.environ.update({'HF_HOME': str(out / 'runtime/hf'), 'XDG_CACHE_HOME': str(out / 'runtime/xdg'),
                           'TMPDIR': str(out / 'runtime/tmp'), 'TRITON_CACHE_DIR': str(out / 'runtime/triton'),
                           'TORCH_HOME': str(out / 'runtime/torch'), 'CUDA_CACHE_PATH': str(out / 'runtime/cuda'),
                           'FLASHINFER_WORKSPACE_BASE': str(out / 'runtime/flashinfer'),
                           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
        import torch
        import torch.nn.functional as F
        import gptqmodel  # noqa: F401
        from peft import LoraConfig, get_peft_model
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        gate(args.gpu, out / 'gpu_immediately_before_model_load.json')
        torch.cuda.set_device(0)
        torch.manual_seed(20261120)
        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        qconfig = config.quantization_config
        if config.model_type != 'qwen3' or str(qconfig.get('quant_method', '')).lower() != 'awq':
            raise ValueError('wrong AWQ architecture')
        qconfig['backend'] = 'torch_awq'
        config.quantization_config = qconfig
        base = AutoModelForCausalLM.from_pretrained(model_path, config=config, device_map={'': 0},
                                                    local_files_only=True, trust_remote_code=False, attn_implementation='sdpa')
        assert_post_load_safe(args.gpu)
        base.config.use_cache = False
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
                                                task_type='CAUSAL_LM', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
        del base
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        params = [v for n, v in model.named_parameters() if v.requires_grad and 'lora_' in n]
        if not params:
            raise ValueError('no LoRA parameters')
        optimizer = torch.optim.AdamW(params, lr=1e-4)
        tokens = [tokenizer.encode(a, add_special_tokens=False)[0] for a in ACTIONS]
        if len(set(tokens)) != 3:
            raise ValueError('action tokens are not distinct')
        device = torch.device('cuda:0')
        model.train()
        examples = [('A', 'KEEP'), ('A', 'REVISE'), ('A', 'ABSTAIN'), ('B', 'ANSWER')]
        step_data = []
        report['stage'] = 'synthetic_losses'
        for expert, action in examples:
            inputs = {'conversation': 'Synthetic non-medical case: a visitor asks for a color.'}
            if expert == 'A':
                inputs['base_answer'] = 'The color is blue.'
                label = {'action': action, 'correction': 'The color is green.' if action == 'REVISE' else None}
            else:
                label = {'answer': 'The color is green.'}
            optimizer.zero_grad(set_to_none=True)
            loss, action_loss, correction_loss, _ = compute_sample_loss(model, tokenizer, expert,
                {'input': inputs, 'label': label}, device, torch, F, tokens, {a: 1.0 for a in ACTIONS})
            if not torch.isfinite(loss) or ((correction_loss is not None) != (action in {'REVISE', 'ANSWER'})):
                raise RuntimeError('loss or correction mask is invalid')
            loss.backward()
            norms = [p.grad.detach().float().norm() for p in params if p.grad is not None]
            norm = float(torch.stack(norms).norm().item()) if norms else 0.0
            if not math.isfinite(norm) or norm <= 0:
                raise RuntimeError('zero or non-finite LoRA gradient')
            chosen = next(p for p in params if p.grad is not None and torch.count_nonzero(p.grad).item())
            before = chosen.detach().clone()
            optimizer.step()
            if torch.equal(before, chosen.detach()):
                raise RuntimeError('optimizer did not update LoRA')
            assert_post_load_safe(args.gpu)
            step_data.append({'expert': expert, 'action': action, 'loss': float(loss.detach().item()),
                              'action_loss_present': action_loss is not None, 'correction_loss_present': correction_loss is not None,
                              'gradient_norm': norm, 'adapter_parameter_changed': True})
            report['steps'] = list(step_data)
            with (out / 'synthetic_step_progress.jsonl').open('a', encoding='utf-8') as progress:
                progress.write(json.dumps(step_data[-1], sort_keys=True) + '\n')
        report.update({'status': 'PASSED', 'stage': 'complete', 'steps': step_data,
                       'action_first_token_ids': dict(zip(ACTIONS, tokens)), 'torch': torch.__version__,
                       'cuda_build': torch.version.cuda, 'gpu_peak_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1),
                       'gpu_peak_reserved_mib': round(torch.cuda.max_memory_reserved() / 2**20, 1),
                       'trainer_script_sha256': sha(EXP / 'scripts/train_experts.py'),
                       'prompt_script_sha256': sha(EXP / 'scripts/expert_training_data.py'),
                       'package_versions': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'peft', 'gptqmodel')}})
    except Exception as exc:
        if isinstance(exc, InterruptedError):
            report['status'] = 'INTERRUPTED'
        report.update({'error_class': type(exc).__name__, 'error': str(exc)[:1000]})
        raise
    finally:
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        write_json(out / 'trainer_loss_smoke_report.json', report)
    print(json.dumps({'status': report['status'], 'report': str(out / 'trainer_loss_smoke_report.json')}))


if __name__ == '__main__':
    main()
