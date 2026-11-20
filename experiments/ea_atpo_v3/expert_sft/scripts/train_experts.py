#!/usr/bin/env python3
"""Gated, independent AWQ-LoRA A/B SFT. No PENDING data or fallback model."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from audit_data import sha
from expert_training_data import ACTIONS, public_messages, validate_dataset
from verify_awq_lora import LOCK, LOCK_SHA, REVISION, assert_post_load_safe, gate

EXP = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[4]
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def write_json(path: Path, obj: dict) -> None:
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write('\n')


def validate_smoke(path: Path, adapter: Path) -> None:
    smoke = json.loads(path.read_text())
    if (smoke.get('status') != 'PASSED' or smoke.get('model_revision') != REVISION
        or smoke.get('optimizer_steps') != 10 or len(smoke.get('steps', [])) != 10
        or not all(s.get('lora_parameter_changed') is True
                   and isinstance(s.get('lora_gradient_norm'), (float, int))
                   and math.isfinite(s['lora_gradient_norm']) and s['lora_gradient_norm'] > 0
                   and isinstance(s.get('loss'), (float, int)) and math.isfinite(s['loss']) for s in smoke['steps'])
        or smoke.get('adapter_reload_exact') is not True or smoke.get('reload_inference_output_chars', 0) < 1
        or smoke.get('adapter_sha256') != sha(adapter)
        or smoke.get('script_sha256') != sha(EXP / 'scripts/verify_awq_lora.py')):
        raise ValueError('the real 10-step AWQ LoRA smoke gate is not satisfied')


def validate_loss_smoke(path: Path) -> None:
    smoke = json.loads(path.read_text())
    steps = smoke.get('steps', [])
    expected = [('A', 'KEEP'), ('A', 'REVISE'), ('A', 'ABSTAIN'), ('B', 'ANSWER')]
    if (smoke.get('status') != 'PASSED' or smoke.get('model_revision') != REVISION
        or smoke.get('formal_training') is not False or smoke.get('synthetic_nonmedical_only') is not True
        or smoke.get('trainer_script_sha256') != sha(Path(__file__))
        or smoke.get('prompt_script_sha256') != sha(EXP / 'scripts/expert_training_data.py')
        or [(s.get('expert'), s.get('action')) for s in steps] != expected
        or not all(isinstance(s.get('gradient_norm'), (float, int)) and math.isfinite(s['gradient_norm'])
                   and s['gradient_norm'] > 0 and s.get('adapter_parameter_changed') is True
                   and isinstance(s.get('loss'), (float, int)) and math.isfinite(s['loss']) for s in steps)
        or [s.get('action_loss_present') for s in steps] != [True, True, True, False]
        or [s.get('correction_loss_present') for s in steps] != [False, True, False, True]):
        raise ValueError('the synthetic A/B AWQ loss integration gate is not satisfied')


def validate_model(expert: str, model: Path, copy_manifest: Path | None) -> dict:
    if sha(LOCK) != LOCK_SHA:
        raise ValueError('AWQ source lock changed')
    lock = json.loads(LOCK.read_text())
    if lock['huggingface_revision'] != REVISION:
        raise ValueError('wrong AWQ revision')
    if expert == 'B':
        if copy_manifest is None:
            raise ValueError('Expert B requires an independent weight-copy manifest')
        copied = json.loads(copy_manifest.read_text())
        source = Path(copied['source_directory']).resolve()
        if (copied.get('status') != 'COPIED_AND_VERIFIED' or copied.get('revision') != REVISION
            or Path(copied['independent_copy_directory']).resolve() != model or model == source):
            raise ValueError('Expert B model copy not independently verified')
    fingerprints = {}
    for entry in lock['weight_files'] + lock['auxiliary_files']:
        path = model / entry['name']
        if path.stat().st_size != entry['size'] or sha(path) != entry['sha256']:
            raise ValueError(f'model fingerprint mismatch: {entry["name"]}')
        if expert == 'B':
            a, b = (source / entry['name']).stat(), path.stat()
            if (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino):
                raise ValueError('Expert B must not share a source weight inode')
        fingerprints[entry['name']] = entry['sha256']
    return fingerprints


def prompt_text(tokenizer, expert: str, mode: str, inputs: dict) -> str:
    return tokenizer.apply_chat_template(public_messages(expert, mode, inputs), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def encode_prompt(tokenizer, text: str, maximum: int, torch, device):
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    if not ids or len(ids) > maximum:
        raise ValueError('public prompt empty or exceeds max sequence length; no truncation allowed')
    return torch.tensor([ids], dtype=torch.long, device=device)


def supervised_ids(tokenizer, expert: str, inputs: dict, answer: str, maximum: int):
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('missing reviewed answer target')
    messages = public_messages(expert, 'answer', inputs)
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': answer}], tokenize=False,
                                         add_generation_prompt=False, enable_thinking=False)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
    ids = tokenizer(full, add_special_tokens=False)['input_ids']
    if not prefix_ids or ids[:len(prefix_ids)] != prefix_ids or len(ids) <= len(prefix_ids) or len(ids) > maximum:
        raise ValueError('invalid supervised target boundary or sequence too long; no truncation allowed')
    return ids, [-100] * len(prefix_ids) + ids[len(prefix_ids):]


def encode_supervised(tokenizer, expert: str, inputs: dict, answer: str, maximum: int, torch, device):
    ids, labels = supervised_ids(tokenizer, expert, inputs, answer, maximum)
    return (torch.tensor([ids], dtype=torch.long, device=device),
            torch.tensor([labels], dtype=torch.long, device=device))


def classification_stats(pairs: list[tuple[str, str]]) -> dict:
    report = {}
    for cls in ACTIONS:
        tp = sum(a == cls and b == cls for a, b in pairs)
        fp = sum(a != cls and b == cls for a, b in pairs)
        fn = sum(a == cls and b != cls for a, b in pairs)
        report[cls] = {'support': tp + fn, 'predicted': tp + fp,
                       'precision': tp / (tp + fp) if tp + fp else None,
                       'recall': tp / (tp + fn) if tp + fn else None,
                       'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
    return report


def compute_sample_loss(model, tokenizer, expert: str, item: dict, device, torch, F,
                        first_tokens: list[int], class_weights: dict, max_len: int = 4096):
    """Exactly the requested A action/correction or B assistant-token losses."""
    action_loss, correction_loss, predicted = None, None, None
    if expert == 'A':
        action = item['label']['action']
        if action not in ACTIONS or (action != 'REVISE' and item['label'].get('correction') is not None):
            raise ValueError('KEEP/ABSTAIN may not carry a correction target')
        if action == 'REVISE' and not item['label'].get('correction'):
            raise ValueError('REVISE requires a reviewed correction target')
        ids = encode_prompt(tokenizer, prompt_text(tokenizer, 'A', 'action', item['input']), max_len, torch, device)
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1, first_tokens].float()
        target = ACTIONS.index(action)
        action_loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target], device=device)) * class_weights[action]
        predicted = ACTIONS[int(logits.argmax().item())]
        if action == 'REVISE':
            full, labels = encode_supervised(tokenizer, 'A', item['input'], item['label']['correction'], max_len, torch, device)
            correction_loss = model(input_ids=full, attention_mask=torch.ones_like(full), labels=labels, use_cache=False).loss.float()
        loss = action_loss + (correction_loss if correction_loss is not None else 0)
    elif expert == 'B':
        full, labels = encode_supervised(tokenizer, 'B', item['input'], item['label']['answer'], max_len, torch, device)
        correction_loss = model(input_ids=full, attention_mask=torch.ones_like(full), labels=labels, use_cache=False).loss.float()
        loss = correction_loss
    else:
        raise ValueError('unsupported Expert')
    return loss, action_loss, correction_loss, predicted


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--expert', choices=('A', 'B'), required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--expert-b-copy-manifest', type=Path)
    p.add_argument('--paired-dir', type=Path, required=True)
    p.add_argument('--review-queue', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--audit-manifest', type=Path, required=True)
    p.add_argument('--base-answers', type=Path, required=True)
    p.add_argument('--base-manifest', type=Path, required=True)
    p.add_argument('--smoke-report', type=Path, required=True)
    p.add_argument('--smoke-adapter', type=Path, required=True)
    p.add_argument('--loss-smoke-report', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=10)
    args = p.parse_args()
    if args.gpu not in {0, 2, 3, 4, 5, 6} or os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        raise ValueError('exactly one allowed physical GPU must be specified; 1 and 7 are prohibited')
    if not 1 <= args.epochs <= 10:
        raise ValueError('formal epochs must be in 1..10')
    out = args.output_dir.resolve()
    expected_parent = (EXP / 'outputs' / f'expert_{args.expert.lower()}').resolve()
    if expected_parent not in out.parents or out.exists():
        raise ValueError('use a new Expert-specific output directory')
    config = json.loads(args.config.read_text())
    if (config.get('model_id') != 'Qwen/Qwen3-32B-AWQ' or config.get('revision') != REVISION
        or config.get('tokenizer_revision') != REVISION or config.get('max_epochs') != 10):
        raise ValueError('formal Expert configuration does not match the locked experiment')
    expected_training = {'learning_rate': 0.0001, 'micro_batch': 1, 'gradient_accumulation': 8,
                         'max_sequence_length': 4096, 'gradient_checkpointing': True,
                         'optimizer': 'AdamW', 'weight_decay': 0.01}
    if config.get('training') != expected_training or config.get('lora') != {'rank': 16, 'alpha': 32, 'dropout': 0.05}:
        raise ValueError('formal training or adapter configuration differs from the implemented settings')
    if (args.expert == 'A' and (config.get('action_loss_weight') != 1.0
        or config.get('correction_loss_weight') != 1.0 or config.get('correction_loss_only_for') != 'REVISE')):
        raise ValueError('Expert A loss weights or correction mask differ from the fixed experiment')
    check = subprocess.run([sys.executable, str(EXP / 'scripts/check_gold_review.py'),
                            '--queue', str(args.review_queue), '--source-root', str(args.source_root),
                            '--audit-manifest', str(args.audit_manifest), '--base-answers', str(args.base_answers),
                            '--base-manifest', str(args.base_manifest)], capture_output=True, text=True)
    if check.returncode:
        print(check.stdout, end='')
        raise ValueError('human Gold review/Base provenance gate failed')
    validate_smoke(args.smoke_report, args.smoke_adapter)
    validate_loss_smoke(args.loss_smoke_report)
    paired, data = validate_dataset(args.paired_dir, args.review_queue, args.expert)
    if (paired['base_answers_sha256'] != sha(args.base_answers) or paired['base_manifest_sha256'] != sha(args.base_manifest)
        or paired['audit_manifest_sha256'] != sha(args.audit_manifest)):
        raise ValueError('paired dataset provenance differs from verified inputs')
    model_path = args.model_dir.resolve()
    if args.expert == 'A' and model_path != (PROJECT.parent / 'EA_ATPO/experiments/real_case_pilot_20261103/assets/models/Qwen3-32B-AWQ').resolve():
        raise ValueError('Expert A must use the original pinned AWQ initialization')
    fingerprints = validate_model(args.expert, model_path, args.expert_b_copy_manifest)
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(out, 0o700)
    report = {'status': 'FAILED', 'stage': 'resource_gate', 'expert': args.expert, 'epochs_completed': 0,
              'model_revision': REVISION, 'tokenizer_revision': REVISION, 'gpu': args.gpu,
              'config_sha256': sha(args.config), 'paired_manifest_sha256': sha(args.paired_dir / 'paired_dataset_manifest.json'),
              'reviewed_queue_sha256': sha(args.review_queue), 'smoke_report_sha256': sha(args.smoke_report),
              'loss_smoke_report_sha256': sha(args.loss_smoke_report),
              'source_lock_sha256': LOCK_SHA, 'model_fingerprints': fingerprints}
    started = time.monotonic()
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
        import torch.nn.functional as F
        import gptqmodel  # noqa: F401 -- install torch_awq bridge before Transformers
        from peft import LoraConfig, get_peft_model
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        gate(args.gpu, out / 'gpu_immediately_before_model_load.json')
        torch.cuda.set_device(0)
        torch.manual_seed(20261120)
        torch.cuda.manual_seed_all(20261120)
        torch.cuda.reset_peak_memory_stats()
        report['stage'] = 'load_awq'
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        model_config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        qconfig = getattr(model_config, 'quantization_config', None) or {}
        if model_config.model_type != 'qwen3' or str(qconfig.get('quant_method', '')).lower() != 'awq':
            raise ValueError('only pinned Qwen3 AWQ is allowed')
        qconfig['backend'] = 'torch_awq'
        model_config.quantization_config = qconfig
        base = AutoModelForCausalLM.from_pretrained(model_path, config=model_config, device_map={'': 0},
                                                    local_files_only=True, trust_remote_code=False, attn_implementation='sdpa')
        assert_post_load_safe(args.gpu)
        base.config.use_cache = False
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
                                                task_type='CAUSAL_LM', target_modules=TARGETS))
        del base
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        params = [v for n, v in model.named_parameters() if v.requires_grad and 'lora_' in n]
        if not params or any(v.requires_grad and 'lora_' not in n for n, v in model.named_parameters()):
            raise ValueError('unexpected trainable parameter set')
        tokens = {label: tokenizer.encode(label, add_special_tokens=False) for label in ACTIONS}
        first_tokens = [tokens[label][0] for label in ACTIONS]
        if len(set(first_tokens)) != len(ACTIONS):
            raise ValueError('action classes do not have distinct first tokens in pinned tokenizer')
        counts = Counter(r['label']['action'] for r in data['train']) if args.expert == 'A' else Counter()
        present = [n for n in counts.values() if n]
        imbalance = bool(present and max(present) / min(present) > 3)
        class_weights = {a: (sum(present) / (len(present) * counts[a]) if imbalance and counts[a] else 1.0) for a in ACTIONS}
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
        device, max_len, accumulation = torch.device('cuda:0'), 4096, 8

        def sample_loss(item):
            return compute_sample_loss(model, tokenizer, args.expert, item, device, torch, F,
                                       first_tokens, class_weights, max_len)

        report.update({'seed': 20261120, 'training_config': {'epochs': args.epochs, 'micro_batch': 1,
                       'gradient_accumulation': accumulation, 'max_sequence_length': max_len,
                       'gradient_checkpointing': True, 'learning_rate': 1e-4, 'optimizer': 'AdamW',
                       'lora_rank': 16, 'lora_alpha': 32, 'lora_dropout': 0.05, 'target_modules': TARGETS,
                       'lambda_action': 1.0, 'lambda_answer': 1.0, 'class_weighted': imbalance,
                       'class_weights': class_weights, 'action_token_ids': tokens,
                       'action_classifier': 'cross-entropy over the first distinct tokenizer token for each action; JSON is assembled by the inference wrapper'},
                       'train_samples': len(data['train']), 'dev_samples': len(data['validation']),
                       'train_cases': len({r['case_id'] for r in data['train']}),
                       'dev_cases': len({r['case_id'] for r in data['validation']}), 'train_action_counts': dict(counts),
                       'gpu_model': json.loads((out / 'gpu_immediately_before_model_load.json').read_text())['name'],
                       'package_versions': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'peft', 'gptqmodel', 'safetensors')},
                       'cuda_build': torch.version.cuda, 'model_parameter_dtype': str(next(model.parameters()).dtype),
                       'lora_parameter_dtypes': sorted({str(p.dtype) for p in params}),
                       'trainable_lora_parameters': sum(p.numel() for p in params),
                       'checkpoint_selection': 'NOT_SELECTED_REQUIRES_DEV_EVALUATION',
                       'inference_inputs': ['conversation', 'base_answer'] if args.expert == 'A' else ['conversation']})
        epoch_records = []
        report['stage'] = 'training'
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = list(range(len(data['train'])))
            random.Random(20261120 + epoch).shuffle(order)
            per_class, decisions = defaultdict(lambda: {'count': 0, 'action_loss': 0.0, 'answer_loss': 0.0, 'answer_count': 0}), []
            all_loss = []
            for start in range(0, len(order), accumulation):
                group = order[start:start + accumulation]
                optimizer.zero_grad(set_to_none=True)
                for index in group:
                    item = data['train'][index]
                    loss, action_loss, answer_loss, pred = sample_loss(item)
                    if not torch.isfinite(loss):
                        raise RuntimeError('non-finite loss')
                    (loss / len(group)).backward()
                    all_loss.append(float(loss.detach().item()))
                    key = item['label']['action'] if args.expert == 'A' else 'ANSWER'
                    per_class[key]['count'] += 1
                    if action_loss is not None:
                        per_class[key]['action_loss'] += float(action_loss.detach().item())
                        decisions.append((key, pred))
                    if answer_loss is not None:
                        per_class[key]['answer_loss'] += float(answer_loss.detach().item())
                        per_class[key]['answer_count'] += 1
                grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if not torch.isfinite(grad) or float(grad) == 0:
                    raise RuntimeError('zero or non-finite LoRA gradient')
                optimizer.step()
                assert_post_load_safe(args.gpu)
            model.eval()
            val_loss, val_actions = [], []
            val_classes = defaultdict(lambda: {'count': 0, 'action_loss': 0.0, 'answer_loss': 0.0, 'answer_count': 0})
            with torch.inference_mode():
                for item in data['validation']:
                    loss, a_loss, ans_loss, pred = sample_loss(item)
                    if not torch.isfinite(loss):
                        raise RuntimeError('non-finite Dev loss')
                    val_loss.append(float(loss.item()))
                    cls = item['label']['action'] if args.expert == 'A' else 'ANSWER'
                    val_classes[cls]['count'] += 1
                    if a_loss is not None:
                        val_classes[cls]['action_loss'] += float(a_loss.item())
                    if ans_loss is not None:
                        val_classes[cls]['answer_loss'] += float(ans_loss.item())
                        val_classes[cls]['answer_count'] += 1
                    if args.expert == 'A':
                        val_actions.append((item['label']['action'], pred))
            checkpoint = out / f'epoch-{epoch:02d}'
            checkpoint.mkdir()
            model.save_pretrained(checkpoint, safe_serialization=True)
            optimizer_path = checkpoint / 'optimizer.pt'
            torch.save({'epoch': epoch, 'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state()}, optimizer_path)
            record = {'epoch': epoch, 'train_loss': sum(all_loss) / len(all_loss),
                      'dev_loss': sum(val_loss) / len(val_loss), 'train_class_sums': dict(per_class),
                      'dev_class_sums': dict(val_classes),
                      'train_action_classification': classification_stats(decisions) if decisions else None,
                      'dev_action_classification': classification_stats(val_actions) if val_actions else None,
                      'adapter_sha256': sha(checkpoint / 'adapter_model.safetensors'),
                      'optimizer_sha256': sha(optimizer_path), 'checkpoint': str(checkpoint)}
            epoch_records.append(record)
            write_json(checkpoint / 'epoch_manifest.json', record)
            report['epochs_completed'] = epoch
        report.update({'status': 'TRAINED_DEV_SELECTION_PENDING', 'stage': 'complete', 'epochs': epoch_records,
                       'gpu_peak_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1),
                       'gpu_peak_reserved_mib': round(torch.cuda.max_memory_reserved() / 2**20, 1)})
    except Exception as exc:
        report.update({'error_class': type(exc).__name__, 'error': str(exc)[:1000]})
        raise
    finally:
        report['elapsed_seconds'] = round(time.monotonic() - started, 3)
        write_json(out / 'training_manifest.json', report)
    print(json.dumps({'status': report['status'], 'expert': args.expert, 'epochs': report['epochs_completed']}))


if __name__ == '__main__':
    main()
