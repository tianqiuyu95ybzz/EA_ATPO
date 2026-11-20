import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from expert_training_data import public_messages
from train_experts import classification_stats, supervised_ids, validate_loss_smoke


class TinyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        text = ''.join(f'<{m["role"]}>{m["content"]}' for m in messages)
        return text + ('<assistant>' if add_generation_prompt else '<end>')

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': list(text.encode())}


class TrainingBoundaryTest(unittest.TestCase):
    def test_a_inputs_contain_only_conversation_and_base(self):
        value = {'conversation': 'synthetic public dialogue', 'base_answer': 'synthetic base'}
        for mode in ('action', 'answer'):
            messages = public_messages('A', mode, value)
            self.assertEqual(json.loads(messages[1]['content']), value)
            self.assertNotIn('reviewed_gold', messages[1]['content'])
            self.assertNotIn('base_correct', messages[1]['content'])

    def test_b_never_accepts_base_or_review_labels(self):
        value = {'conversation': 'synthetic public dialogue'}
        self.assertEqual(json.loads(public_messages('B', 'answer', value)[1]['content']), value)
        for forbidden in ('base_answer', 'reviewed_gold', 'base_correct', 'target_action'):
            with self.assertRaises(ValueError):
                public_messages('B', 'answer', {**value, forbidden: 'synthetic'})

    def test_a_refuses_privileged_features(self):
        value = {'conversation': 'synthetic dialogue', 'base_answer': 'synthetic base'}
        for forbidden in ('reviewed_gold', 'base_correct', 'target_action', 'reviewer'):
            with self.assertRaises(ValueError):
                public_messages('A', 'action', {**value, forbidden: 'synthetic'})

    def test_assistant_target_mask_and_class_f1(self):
        values = {'conversation': 'synthetic dialogue', 'base_answer': 'synthetic base'}
        ids, labels = supervised_ids(TinyTokenizer(), 'A', values, 'synthetic correction', 4096)
        first = next(i for i, label in enumerate(labels) if label != -100)
        self.assertTrue(all(label == -100 for label in labels[:first]))
        self.assertEqual(labels[first:], ids[first:])
        metrics = classification_stats([('REVISE', 'KEEP'), ('KEEP', 'KEEP')])
        self.assertEqual(metrics['REVISE']['recall'], 0.0)
        self.assertEqual(metrics['REVISE']['f1'], 0.0)

    def test_interrupted_loss_smoke_cannot_unlock_training(self):
        root = Path(__file__).resolve().parents[1] / 'outputs'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='synthetic_gate_', dir=root) as tmp:
            report = Path(tmp) / 'report.json'
            report.write_text(json.dumps({'status': 'INTERRUPTED', 'steps': []}))
            with self.assertRaises(ValueError):
                validate_loss_smoke(report)


if __name__ == '__main__':
    unittest.main()
