"""Synthetic end-to-end coverage of review/provenance and test-safe evaluation."""
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / 'scripts'))
from audit_data import FILES, SFT
from prepare_gold_review import FIELDS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')


class EvaluationPipelineTest(unittest.TestCase):
    def test_validation_pipeline_and_privileged_score_rejection(self):
        outputs = EXP / 'outputs'
        outputs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='synthetic_eval_', dir=outputs) as directory:
            root = Path(directory)
            source = root / 'source'
            data = source / SFT
            data.mkdir(parents=True)
            expected = {}
            queue, answers = [], []
            for split, filename in FILES.items():
                sample, case = f'{split}-sample', f'{split}-case'
                row = {'case_id': case, 'state_id': sample, 'messages': [
                    {'role': 'system', 'content': 'synthetic'},
                    {'role': 'user', 'content': f'synthetic {split} conversation'},
                    {'role': 'assistant', 'content': 'Expert Final Answer: A'}]}
                path = data / filename
                path.write_text(json.dumps(row) + '\n')
                expected[split] = {'source_file_sha256': digest(path)}
                raw = 'Final Answer: B'
                answers.append({'sample_id': sample, 'case_id': case, 'split': split, 'raw_output': raw,
                                'raw_output_sha256': hashlib.sha256(raw.encode()).hexdigest(), 'status': 'VERIFIED'})
                queue.append({'case_id': case, 'sample_id': sample, 'split': split,
                              'conversation': row['messages'][1]['content'], 'base_answer': raw,
                              'original_gold': 'Expert Final Answer: A', 'reviewed_gold': 'A', 'base_correct': 'false',
                              'gold_status': 'APPROVED', 'target_action': 'REVISE', 'evidence': 'synthetic evidence',
                              'risk_level': 'LOW', 'reviewer': 'synthetic reviewer',
                              'review_timestamp': '2026-01-01T00:00:00+00:00', 'review_note': ''})
            audit, base, base_manifest, review = [root / name for name in ('audit.json', 'base.jsonl', 'base_manifest.json', 'review.csv')]
            dump(audit, {'splits': expected})
            base.write_text(''.join(json.dumps(r) + '\n' for r in answers))
            checks = {k: True for k in ('model_revision', 'tokenizer', 'prompt', 'decoding', 'data', 'case_ids', 'raw_output_completeness')}
            dump(base_manifest, {'status': 'VERIFIED', 'sample_count': 3, 'raw_answers_sha256': digest(base),
                                 'base_revision': 'b968826d9c46dd6066d109eabc6255188de91218', 'checks': checks})
            with review.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(queue)
            calibration, checkpoint, config, lock = [root / name for name in ('calibration.json', 'adapter.bin', 'config.json', 'lock.json')]
            dump(calibration, {'split': 'validation', 'oracle_used': False, 'threshold': 0.5, 'completed_at_utc': '2026-01-01T01:00:00+00:00'})
            checkpoint.write_text('synthetic adapter')
            config.write_text('synthetic config')
            dump(lock, {'status': 'FROZEN', 'expert': 'A', 'threshold': 0.5, 'threshold_chosen_on': 'validation',
                        'oracle_used_for_threshold': False, 'frozen_at_utc': '2026-01-01T02:00:00+00:00',
                        'checkpoint_file': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                        'config_file': str(config), 'config_sha256': digest(config),
                        'threshold_calibration_manifest': str(calibration), 'threshold_calibration_sha256': digest(calibration)})
            pred = root / 'scored.jsonl'
            row = {'sample_id': 'validation-sample', 'case_id': 'validation-case', 'split': 'validation',
                   'gold_status': 'APPROVED', 'base_answer': 'Final Answer: B', 'base_correct': False,
                   'target_action': 'REVISE', 'risk_level': 'LOW', 'decision_score': 0.9,
                   'expert_output': json.dumps({'action': 'REVISE', 'answer': 'A'}),
                   'expert_correct': True, 'correctness_method': 'approved_gold_exact_option'}
            pred.write_text(json.dumps(row) + '\n')
            pred_manifest = root / 'pred_manifest.json'
            manifest = {'status': 'COMPLETED', 'split': 'validation', 'model_lock_sha256': digest(lock),
                        'scored_predictions_sha256': digest(pred), 'reviewed_queue_sha256': digest(review),
                        'base_answers_sha256': digest(base), 'score_visibility': 'predecision',
                        'score_features': ['base_entropy'], 'generated_at_utc': '2026-01-01T03:00:00+00:00'}
            dump(pred_manifest, manifest)
            out = root / 'eval'
            command = [sys.executable, str(EXP / 'scripts/evaluate_experts.py'), '--predictions', str(pred),
                       '--expert', 'A', '--split', 'validation', '--model-lock', str(lock),
                       '--prediction-manifest', str(pred_manifest), '--review-queue', str(review),
                       '--source-root', str(source), '--audit-manifest', str(audit), '--base-answers', str(base),
                       '--base-manifest', str(base_manifest), '--output-dir', str(out), '--bootstrap', '20']
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            metrics = json.loads((out / 'aggregate_metrics.json').read_text())
            self.assertEqual(metrics['final_policy']['expert_accuracy'], 1.0)
            self.assertTrue((out / 'risk_coverage.svg').exists())
            manifest['score_features'] = ['reviewed_gold']
            dump(pred_manifest, manifest)
            command[command.index('--output-dir') + 1] = str(root / 'leaky_eval')
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertFalse((root / 'leaky_eval').exists())


if __name__ == '__main__':
    unittest.main()
