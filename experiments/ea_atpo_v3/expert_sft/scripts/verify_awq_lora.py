#!/usr/bin/env python3
"""Real AWQ/LoRA compatibility smoke, using only synthetic non-medical text."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from audit_data import sha

PROJECT = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parents[1]
LOCK = PROJECT / 'experiments/expert_sft/multiturn_dialogue/data/qwen3_32b_awq_source_lock.json'
LOCK_SHA = 'c3016466299afca01670361bc40dc6868ff1ffe4e3d118197b164167354a3bbe'
REVISION = '0499c3ac83fdef8810b907a23894ba91e95eddd8'


def gate(gpu: int, dest: Path) -> None:
    gate_script = PROJECT / 'experiments/expert_sft/multiturn_dialogue/scripts/gpu_safety_gate.py'
    r = subprocess.run([sys.executable, str(gate_script), '--gpu', str(gpu)], capture_output=True, text=True)
    with dest.open('x', encoding='utf-8') as f:
        f.write(r.stdout + r.stderr)
    if r.returncode:
        raise RuntimeError(f'GPU safety gate failed: {dest}')


def assert_post_load_safe(gpu: int) -> dict:
    line = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    free, used = (int(x.strip()) for x in line.split(','))
    raw = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    other = [x.strip() for x in raw.splitlines() if x.strip() and x.strip().upper() != 'N/A' and x.strip() != str(os.getpid())]
    if free < 32768 or other:
        raise RuntimeError(f'post-load resource gate failed: free={free} MiB, other process count={len(other)}')
    return {'free_mib': free, 'used_mib': used, 'other_compute_process_count': len(other)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.gpu not in {0, 2, 3, 4, 5, 6} or os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        raise ValueError('CUDA_VISIBLE_DEVICES must name exactly one allowed physical GPU; 1 and 7 are prohibited')
    out = args.output_dir.resolve()
    if (EXP / 'outputs').resolve() not in out.parents:
        raise ValueError('smoke outputs must remain under V3 outputs/')
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(out, 0o700)
    started = time.monotonic()
    report = {'status': 'FAILED', 'stage': 'resource_gate', 'gpu': args.gpu,
              'model_revision': REVISION, 'synthetic_nonmedical_only': True,
              'tokenizer_revision': REVISION, 'seed': 20261120, 'source_lock_sha256': LOCK_SHA,
              'script_sha256': sha(Path(__file__)), 'formal_training': False, 'optimizer_steps': 0}
    try:
        gate(args.gpu, out / 'gpu_before_fingerprinting.json')
        report['stage'] = 'fingerprint'
        if sha(LOCK) != LOCK_SHA:
            raise ValueError('AWQ source lock differs from pinned hash')
        lock = json.loads(LOCK.read_text())
        if lock['huggingface_revision'] != REVISION:
            raise ValueError('AWQ model revision mismatch')
        model_path = args.model_dir.resolve()
        fingerprints = {}
        for item in lock['weight_files'] + lock['auxiliary_files']:
            path = model_path / item['name']
            if path.stat().st_size != item['size'] or sha(path) != item['sha256']:
                raise ValueError(f'AWQ fingerprint mismatch: {item["name"]}')
            fingerprints[item['name']] = item['sha256']
        for subdir in ('hf', 'xdg', 'tmp', 'triton', 'torch', 'cuda', 'flashinfer'):
            (out / 'runtime' / subdir).mkdir(parents=True, mode=0o700)
        os.environ.update({'HF_HOME': str(out / 'runtime/hf'), 'XDG_CACHE_HOME': str(out / 'runtime/xdg'),
                           'TMPDIR': str(out / 'runtime/tmp'), 'TRITON_CACHE_DIR': str(out / 'runtime/triton'),
                           'TORCH_HOME': str(out / 'runtime/torch'), 'CUDA_CACHE_PATH': str(out / 'runtime/cuda'),
                           'FLASHINFER_WORKSPACE_BASE': str(out / 'runtime/flashinfer'),
                           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
        import torch
        import gptqmodel  # noqa: F401 -- installs the trainable torch_awq bridge
        from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        report['package_versions'] = {name: importlib.metadata.version(name)
                                      for name in ('torch', 'transformers', 'peft', 'gptqmodel', 'safetensors')}
        report['stage'] = 'immediate_resource_gate'
        gate(args.gpu, out / 'gpu_immediately_before_model_load.json')
        report['gpu_model'] = json.loads((out / 'gpu_immediately_before_model_load.json').read_text())['name']
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        torch.cuda.set_device(0)
        torch.manual_seed(20261120)
        torch.cuda.manual_seed_all(20261120)
        torch.cuda.reset_peak_memory_stats()
        report['stage'] = 'load_awq'
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        qconfig = getattr(config, 'quantization_config', None) or {}
        if config.model_type != 'qwen3' or str(qconfig.get('quant_method', '')).lower() != 'awq':
            raise RuntimeError('checkpoint is not the pinned Qwen3 AWQ model')
        qconfig['backend'] = 'torch_awq'
        config.quantization_config = qconfig
        base = AutoModelForCausalLM.from_pretrained(model_path, config=config, device_map={'': 0},
                                                    local_files_only=True, trust_remote_code=False, attn_implementation='sdpa')
        report['resource_after_load'] = assert_post_load_safe(args.gpu)
        base.config.use_cache = False
        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
                          target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
        model = get_peft_model(base, lora)
        del base
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        params = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_' in n]
        bad_trainable = [n for n, p in model.named_parameters() if p.requires_grad and 'lora_' not in n]
        if not params or bad_trainable:
            raise RuntimeError('unexpected trainable parameter set')
        prompt = [{'role': 'user', 'content': 'Reply with the word OK.'}]
        full = prompt + [{'role': 'assistant', 'content': 'OK'}]
        prefix = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        text = tokenizer.apply_chat_template(full, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
        ids = tokenizer(text, add_special_tokens=False)['input_ids']
        if ids[:len(prefix_ids)] != prefix_ids or len(ids) <= len(prefix_ids):
            raise RuntimeError('synthetic chat template loss boundary invalid')
        device = torch.device('cuda:0')
        inputs = torch.tensor([ids], dtype=torch.long, device=device)
        labels = torch.tensor([[-100] * len(prefix_ids) + ids[len(prefix_ids):]], dtype=torch.long, device=device)
        attention = torch.ones_like(inputs)
        optimizer = torch.optim.AdamW(params, lr=1e-4)
        model.train()
        records = []
        report['stage'] = 'forward_backward_optimizer'
        for step in range(1, 11):
            optimizer.zero_grad(set_to_none=True)
            output = model(input_ids=inputs, attention_mask=attention, labels=labels, use_cache=False)
            loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at step {step}')
            loss.backward()
            grads = [p.grad.detach().float().norm() for p in params if p.grad is not None]
            grad_norm = float(torch.stack(grads).norm().item()) if grads else 0.0
            if not math.isfinite(grad_norm) or grad_norm <= 0:
                raise RuntimeError(f'zero/non-finite LoRA gradient at step {step}')
            chosen = next(p for p in params if p.grad is not None and torch.count_nonzero(p.grad).item() > 0)
            before = chosen.detach().clone()
            optimizer.step()
            if torch.equal(before, chosen.detach()):
                raise RuntimeError(f'LoRA weight did not change at step {step}')
            records.append({'step': step, 'loss': float(loss.detach().item()), 'lora_gradient_norm': grad_norm,
                            'lora_parameter_changed': True})
            report['optimizer_steps'] = step
            report['resource_after_step'] = assert_post_load_safe(args.gpu)
        report['stage'] = 'checkpoint_save_and_reload'
        adapter = out / 'adapter'
        model.save_pretrained(adapter, safe_serialization=True)
        saved = load_file(str(adapter / 'adapter_model.safetensors'), device='cpu')
        saved_hash = sha(adapter / 'adapter_model.safetensors')
        del optimizer, params, chosen, before, output, loss, grads, model
        torch.cuda.empty_cache()
        reloaded_base = AutoModelForCausalLM.from_pretrained(model_path, config=config, device_map={'': 0},
                                                             local_files_only=True, trust_remote_code=False, attn_implementation='sdpa')
        reloaded = PeftModel.from_pretrained(reloaded_base, adapter, is_trainable=False)
        actual = get_peft_model_state_dict(reloaded)
        if set(actual) != set(saved) or not all(torch.equal(actual[k].detach().cpu(), saved[k]) for k in saved):
            raise RuntimeError('reloaded adapter differs from saved tensors')
        report['stage'] = 'reload_inference'
        reloaded.eval()
        with torch.inference_mode():
            input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
            generated = reloaded.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                          max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(generated[0, len(prefix_ids):], skip_special_tokens=True)
        if not generated_text.strip():
            raise RuntimeError('reloaded adapter inference produced no visible output')
        report.update({'status': 'PASSED', 'stage': 'complete', 'steps': records,
                       'trainable_lora_parameters': sum(v.numel() for v in saved.values()),
                       'adapter_sha256': saved_hash, 'adapter_reload_exact': True,
                       'reload_inference_output_sha256': hashlib.sha256(generated_text.encode()).hexdigest(),
                       'reload_inference_output_chars': len(generated_text),
                       'gpu_peak_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1),
                       'gpu_peak_reserved_mib': round(torch.cuda.max_memory_reserved() / 2**20, 1),
                       'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                       'model_fingerprints': fingerprints, 'lora': {'rank': 16, 'alpha': 32, 'dropout': 0.05},
                       'optimizer': 'AdamW', 'learning_rate': 1e-4})
    except Exception as exc:
        if report['stage'] in {'resource_gate', 'immediate_resource_gate'}:
            report['status'] = 'BLOCKED_RESOURCE'
        report.update({'error_class': type(exc).__name__, 'error': str(exc)[:1000]})
        raise
    finally:
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        with (out / 'smoke_report.json').open('x', encoding='utf-8') as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write('\n')
    print(json.dumps({'status': report['status'], 'steps': report['optimizer_steps'], 'report': str(out / 'smoke_report.json')}))


if __name__ == '__main__':
    main()
