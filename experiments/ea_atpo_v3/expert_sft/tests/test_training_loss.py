import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from test_training_boundary import TinyTokenizer
from train_experts import ACTIONS, compute_sample_loss

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is not installed in this interpreter')
class TrainingLossTest(unittest.TestCase):
    def test_action_and_correction_masks_with_synthetic_model(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(256, 8)
                self.output = torch.nn.Linear(8, 256)

            def forward(self, input_ids, attention_mask, use_cache=False, labels=None):
                logits = self.output(self.embedding(input_ids))
                loss = None
                if labels is not None:
                    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), labels[:, 1:].reshape(-1), ignore_index=-100)
                return SimpleNamespace(logits=logits, loss=loss)

        tokenizer, model = TinyTokenizer(), TinyModel()
        weights = {action: 1.0 for action in ACTIONS}
        common = {'conversation': 'synthetic public dialogue', 'base_answer': 'synthetic base'}
        for action in ACTIONS:
            model.zero_grad(set_to_none=True)
            item = {'input': common, 'label': {'action': action, 'correction': 'synthetic correction' if action == 'REVISE' else None}}
            total, action_loss, answer_loss, _ = compute_sample_loss(model, tokenizer, 'A', item, torch.device('cpu'), torch, F, [1, 2, 3], weights)
            self.assertIsNotNone(action_loss)
            self.assertEqual(answer_loss is not None, action == 'REVISE')
            total.backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None), 0)
        model.zero_grad(set_to_none=True)
        item = {'input': {'conversation': 'synthetic public dialogue'}, 'label': {'answer': 'synthetic answer'}}
        total, action_loss, answer_loss, _ = compute_sample_loss(model, tokenizer, 'B', item, torch.device('cpu'), torch, F, [1, 2, 3], weights)
        self.assertIsNone(action_loss)
        self.assertIsNotNone(answer_loss)
        total.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None), 0)


if __name__ == '__main__':
    unittest.main()
