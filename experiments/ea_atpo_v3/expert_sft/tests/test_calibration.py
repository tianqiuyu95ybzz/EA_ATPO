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
from audit_data import sha
from prepare_gold_review import FIELDS


class CalibrationTest(unittest.TestCase):
    def test_dev_threshold_uses_predecision_scores_and_no_oracle(self):
        root_dir = EXP / 'outputs'
        root_dir.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='synthetic_cal_', dir=root_dir) as tmp:
            root = Path(tmp)
            review, scored, raw, run_path, scored_meta = [root / x for x in ('review.csv', 'scored.jsonl', 'raw.jsonl', 'run.json', 'scored_manifest.json')]
            review_rows, predictions = [], []
            for n, base_correct, score, answer in [(1, True, 0.1, 'B'), (2, False, 0.9, 'A')]:
                case, sample = f'case-{n}', f'sample-{n}'
                base = 'Final Answer: A' if base_correct else 'Final Answer: B'
                action = 'KEEP' if base_correct else 'REVISE'
                review_rows.append({'case_id': case, 'sample_id': sample, 'split': 'validation',
                                    'conversation': 'synthetic', 'base_answer': base, 'original_gold': 'Expert Final Answer: A',
                                    'reviewed_gold': 'A', 'base_correct': str(base_correct).lower(), 'gold_status': 'APPROVED',
                                    'target_action': action, 'evidence': 'synthetic', 'risk_level': 'LOW',
                                    'reviewer': 'test', 'review_timestamp': '2026-01-01T00:00:00+00:00', 'review_note': ''})
                predictions.append({'case_id': case, 'sample_id': sample, 'split': 'validation',
                                    'base_answer': base, 'base_correct': base_correct, 'gold_status': 'APPROVED',
                                    'target_action': action, 'decision_score': score,
                                    'expert_output': json.dumps({'action': 'REVISE', 'answer': answer}),
                                    'expert_correct': answer == 'A', 'correctness_method': 'approved_gold_exact_option'})
            with review.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(review_rows)
            scored.write_text(''.join(json.dumps(r) + '\n' for r in predictions))
            raw_rows = [{**{k: r[k] for k in ('case_id', 'sample_id', 'split', 'base_answer', 'expert_output', 'decision_score')},
                         'expert_output_sha256': hashlib.sha256(r['expert_output'].encode()).hexdigest()} for r in predictions]
            raw.write_text(''.join(json.dumps(r) + '\n' for r in raw_rows))
            run = {'status': 'COMPLETED', 'expert': 'A', 'split': 'validation', 'score_visibility': 'predecision',
                   'score_features': ['a_action_logprobs'], 'generated_at_utc': '2026-01-01T01:00:00+00:00',
                   'raw_predictions_sha256': sha(raw), 'reviewed_queue_sha256': sha(review),
                   'checkpoint_sha256': 'a' * 64, 'config_sha256': 'b' * 64}
            run_path.write_text(json.dumps(run))
            scored_meta.write_text(json.dumps({'status': 'COMPLETED', 'split': 'validation', 'expert': 'A',
                                              'scored_predictions_sha256': sha(scored), 'raw_run_manifest_sha256': sha(run_path),
                                              'raw_predictions_sha256': sha(raw),
                                              'reviewed_queue_sha256': sha(review)}))
            out = root / 'calibration'
            command = [sys.executable, str(EXP / 'scripts/calibrate_threshold.py'), '--expert', 'A',
                       '--scored-predictions', str(scored), '--scored-manifest', str(scored_meta),
                       '--prediction-run-manifest', str(run_path), '--raw-predictions', str(raw),
                       '--review-queue', str(review), '--output-dir', str(out)]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out / 'threshold_calibration_manifest.json').read_text())
            self.assertEqual(manifest['threshold'], 0.9)
            self.assertFalse(manifest['oracle_used'])
            run['score_features'] = ['expert_output']
            run_path.write_text(json.dumps(run))
            meta = json.loads(scored_meta.read_text())
            meta['raw_run_manifest_sha256'] = sha(run_path)
            scored_meta.write_text(json.dumps(meta))
            command[command.index('--output-dir') + 1] = str(root / 'leaky')
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
