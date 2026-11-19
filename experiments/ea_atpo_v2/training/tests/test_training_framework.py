from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.ea_atpo_v2.training import ACTION_NAMES
from experiments.ea_atpo_v2.training.calibration import ActionValueCalibrator
from experiments.ea_atpo_v2.training.data import (
    QExample, load_public_embedding_cache, public_state_hash, q_example_from_dict,
    q_sampling_summary, require_single_continuation_policy, split_cases_by_id,
    validate_official_train_cases,
)
from experiments.ea_atpo_v2.training.extract_state_embeddings import (
    build_embedding_config, extract_batches, pool_hidden_states, unique_public_states,
)
from experiments.ea_atpo_v2.training.evaluate_policies import case_level_metrics, main as evaluate_main, q_diagnostics
from experiments.ea_atpo_v2.training.features import ExistingPublicStateEncoder
from experiments.ea_atpo_v2.training.losses import masked_distillation_loss, masked_regression_loss
from experiments.ea_atpo_v2.training.metrics import aggregate_rollouts, utility_from_raw
from experiments.ea_atpo_v2.training.models import ConstantMeanQRegressor, PolicyHead, QRegressor, RidgeQRegressor
from experiments.ea_atpo_v2.training.trainer_utils import atomic_torch_save, load_checkpoint, optimize
from v2_protocol import ACTION_ORDER as PROTOCOL_ACTION_ORDER


def qrow(case_id: str, state_id: str, q=(0.2, 0.5, 0.4), mask=(True, True, True)) -> dict:
    return {
        "case_id": case_id,
        "state_id": state_id,
        "Q_ASK": q[0], "Q_ANSWER": q[1], "Q_DEFER": q[2],
        "valid_mask": dict(zip(ACTION_NAMES, mask)),
        "return_std": {"Q_ASK": None, "Q_ANSWER": 0.1, "Q_DEFER": 0.2},
        "valid_sample_count": {"ASK": 1, "ANSWER": 2, "DEFER": 1},
        "attempted_sample_count": {"ASK": 1, "ANSWER": 3, "DEFER": 1},
        "action_cost": {"ASK": 0.05, "ANSWER": 0.0, "DEFER": 0.60},
        "metadata": {"case_id": case_id, "state_id": state_id, "public_state": [{"role": "user", "content": "Public symptom question?"}], "costs_included_in_targets": True, "continuation_policy_id": "fixed-test-policy"},
    }


def make_example(i: int) -> QExample:
    x = float(i) / 10.0
    return QExample(
        case_id=f"case-{i}", state_id=f"state-{i}",
        public_state=({"role": "user", "content": f"Public state {i}?"},),
        q_target=np.asarray([x, 1.0 - x, 0.5 * x], dtype=np.float64),
        valid_mask=np.asarray([True, True, True]),
        action_cost=np.asarray([0.05, 0.0, 0.60]),
        sample_count=np.asarray([1.0, 1.0, 1.0]),
        attempted_sample_count=np.asarray([1.0, 1.0, 1.0]),
        return_std=np.asarray([np.nan, np.nan, np.nan]),
        costs_included_in_targets=True,
        raw_metadata={"costs_included_in_targets": True, "continuation_policy_id": "fixed-test-policy"},
    )


def outcome(case_id: str, action: str, correctness: int) -> dict:
    ask_count = int(action == "ASK")
    defer_count = int(action == "DEFER")
    return {
        "record_type": "v2_counterfactual_rollout", "synthetic_data": True, "case_id": case_id,
        "state_id": f"{case_id}:s0", "public_state_digest": "same-public-digest",
        "action": action, "valid": True, "failure_reason": None,
        "final_accuracy": float(correctness), "ask_count": ask_count, "defer_count": defer_count,
        "cost_components": {"ask_cost": 0.05 * ask_count, "defer_cost": 0.60 * defer_count,
                             "expert_calls": defer_count, "expert_tokens": 10 * defer_count},
        "return": float(correctness) - 0.05 * ask_count - 0.60 * defer_count,
    }


def predictions_for(case_id: str, correct_answer: int, correct_defer: int):
    gross = {"ASK": 1.0, "ANSWER": float(correct_answer), "DEFER": float(correct_defer)}
    return {
        "synthetic_data": True, "case_id": case_id, "state_id": f"{case_id}:s0", "q_gross_values": gross,
        "q_values_reference_net": {"ASK": 0.95, "ANSWER": float(correct_answer), "DEFER": float(correct_defer) - 0.60},
        "policy_logits": {"ASK": -1.0, "ANSWER": float(correct_answer), "DEFER": float(correct_defer)},
        "policy_temperature": 1.0, "policy_reference_costs": {"ASK": 0.05, "ANSWER": 0.0, "DEFER": 0.60},
        "uncertainty_score": 0.8 if not correct_answer else 0.2,
        "sft_action": "ANSWER",
    }


class TrainingFrameworkTests(unittest.TestCase):
    def test_action_order_is_the_v2_protocol_order(self):
        self.assertEqual(ACTION_NAMES, ("ASK", "ANSWER", "DEFER"))
        self.assertEqual(ACTION_NAMES, tuple(action.value for action in PROTOCOL_ACTION_ORDER))

    def test_q_dataset_loading_preserves_missing_target_and_metadata(self):
        row = qrow("c1", "s1", q=(None, 0.5, 0.2), mask=(False, True, True))
        example = q_example_from_dict(row)
        self.assertFalse(example.valid_mask[0])
        self.assertTrue(np.isnan(example.q_target[0]))
        self.assertEqual(example.sample_count.tolist(), [1.0, 2.0, 1.0])
        self.assertEqual(example.attempted_sample_count.tolist(), [1.0, 3.0, 1.0])
        self.assertTrue(np.isnan(example.return_std[0]))
        self.assertTrue(example.costs_included_in_targets)
        self.assertEqual(example.public_state[0]["role"], "user")

    def test_existing_public_text_encoder_fit_transform_and_restore(self):
        train_rows = [make_example(0), make_example(1), make_example(2)]
        encoder = ExistingPublicStateEncoder(max_features=32).fit(train_rows)
        first = encoder.transform(train_rows)
        restored = ExistingPublicStateEncoder.from_json(encoder.to_json())
        second = restored.transform(train_rows)
        self.assertEqual(first.shape[0], 3)
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.allclose(first, second))

    def test_masked_regression_ignores_missing_actions(self):
        pred = torch.tensor([[100.0, 1.0, 2.0]], requires_grad=True)
        target = torch.tensor([[float("nan"), 0.0, 1.0]])
        mask = torch.tensor([[False, True, True]])
        count = torch.tensor([[0.0, 1.0, 1.0]])
        loss = masked_regression_loss(pred, target, mask, loss="mse", sample_count=count)
        self.assertAlmostEqual(float(loss.detach()), 1.0)
        loss.backward()
        self.assertEqual(float(pred.grad[0, 0]), 0.0)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_q_regression_uses_two_layer_head_and_early_stopping(self):
        torch.manual_seed(3)
        train_rows = [make_example(i) for i in range(8)]
        val_rows = [make_example(i + 20) for i in range(2)]
        train_x = np.asarray([[i / 10.0, 1.0, -0.5] for i in range(8)], dtype=np.float32)
        val_x = np.asarray([[2.0, 1.0, -0.5], [2.1, 1.0, -0.5]], dtype=np.float32)
        model = QRegressor(3, 8)
        self.assertEqual(sum(isinstance(m, torch.nn.Linear) for m in model.modules()), 2)

        def loss_fn(pred, target, mask, counts):
            return masked_regression_loss(pred, target, mask, sample_count=counts)

        result = optimize(model, train_x, train_rows, val_x, val_rows, epochs=30, patience=6,
                          batch_size=4, learning_rate=0.01, weight_decay=0.0, loss_fn=loss_fn, seed=3)
        self.assertGreaterEqual(result["best_epoch"], 1)
        self.assertTrue(np.isfinite(result["best_validation_loss"]))
        self.assertEqual(model(torch.from_numpy(val_x)).shape, (2, 3))

    def test_distillation_masks_missing_actions_and_skips_unpaired_state(self):
        logits = torch.tensor([[50.0, 0.2, -0.1], [0.0, 0.0, 0.0]], requires_grad=True)
        targets = torch.tensor([[float("nan"), 1.0, 0.0], [float("nan"), 1.0, float("nan")]])
        mask = torch.tensor([[False, True, True], [False, True, False]])
        loss, used = masked_distillation_loss(logits, targets, mask, temperature=0.5, allow_empty=True)
        self.assertEqual(used, 1)
        expected_target = torch.softmax(torch.tensor([2.0, 0.0]), dim=0)
        expected_loss = -(expected_target * torch.log_softmax(logits[0, 1:], dim=0)).sum()
        self.assertAlmostEqual(float(loss.detach()), float(expected_loss.detach()), places=6)
        loss.backward()
        self.assertEqual(float(logits.grad[0, 0]), 0.0)
        empty, used_empty = masked_distillation_loss(logits[1:], targets[1:], mask[1:], allow_empty=True)
        self.assertIsNone(empty)
        self.assertEqual(used_empty, 0)

    def test_case_id_split_and_official_train_gate(self):
        examples = [make_example(i // 2) for i in range(12)]
        train_ids, val_ids = split_cases_by_id(examples, validation_fraction=0.25, seed=11)
        self.assertFalse(train_ids & val_ids)
        for case_id in {row.case_id for row in examples}:
            assigned = [case_id in train_ids, case_id in val_ids]
            self.assertEqual(sum(assigned), 1)
        with tempfile.TemporaryDirectory() as temp:
            split_path = Path(temp) / "split.json"
            split_path.write_text(json.dumps({"unit": "case_id", "train": sorted({x.case_id for x in examples}), "validation": ["val"], "internal_test": ["test"]}))
            audit = validate_official_train_cases(examples, split_path)
            self.assertFalse(audit["heldout_cases_opened"])
            self.assertTrue(audit["official_train_pool_complete"])
            split_path.write_text(json.dumps({"unit": "case_id", "train": ["case-0"], "validation": ["case-1"], "internal_test": []}))
            with self.assertRaises(ValueError):
                validate_official_train_cases(examples, split_path)

    def test_continuation_policy_and_k1_sampling_audit(self):
        rows = [make_example(0), make_example(1)]
        self.assertEqual(require_single_continuation_policy(rows), "fixed-test-policy")
        summary = q_sampling_summary(rows)
        self.assertTrue(summary["all_valid_targets_are_K1"])
        self.assertEqual(summary["rollouts_per_state_action_K_distribution"], {"1": 6})
        mixed = [rows[0], replace(rows[1], raw_metadata={"continuation_policy_id": "other-policy"})]
        with self.assertRaisesRegex(ValueError, "continuation_policy_id"):
            require_single_continuation_policy(mixed)

    def test_constant_mean_and_ridge_q_baselines(self):
        rows = [make_example(i) for i in range(8)]
        x = np.asarray([[float(i), float(i % 3)] for i in range(8)], dtype=np.float32)
        constant = ConstantMeanQRegressor(2)
        constant.fit_closed_form(rows)
        constant_predictions = constant(torch.from_numpy(x))
        expected = np.mean(np.stack([row.q_target for row in rows]), axis=0)
        self.assertTrue(np.allclose(constant_predictions[0].detach().numpy(), expected))
        ridge = RidgeQRegressor(2)
        ridge.fit_closed_form(x, rows, alpha=0.5)
        ridge_predictions = ridge(torch.from_numpy(x))
        self.assertEqual(tuple(ridge_predictions.shape), (8, 3))
        self.assertTrue(torch.isfinite(ridge_predictions).all())

    def test_public_embedding_cache_checks_manifest_and_state_hash(self):
        example = make_example(0)
        config = build_embedding_config(
            model_id="frozen-base-test", revision="a" * 40,
            tokenizer_id="frozen-base-test", tokenizer_revision="a" * 40,
            prompt_template_sha256="b" * 64,
            pooling="mean_last_hidden_state", max_length=128,
            truncation_side="left", normalize_l2=False, dtype="float32",
            device_name="cpu", batch_size=1, torch_version="test-torch",
            transformers_version="test-transformers",
        )
        with tempfile.TemporaryDirectory() as temp:
            cache_path = Path(temp) / "features.jsonl"
            row = {
                "case_id": example.case_id, "state_id": example.state_id,
                "feature_boundary": "public_state_before_action_only",
                "state_hash": public_state_hash(example.public_state),
                "rendered_prompt_hash": "c" * 64,
                "embedding": [0.25, -0.5],
                "model_id": config["model_id"], "model_revision": config["model_revision"],
                "pooling": config["pooling"], "max_length": config["max_length"],
                "prompt_template_hash": config["prompt_template_hash"],
                "embedding_config_hash": config["embedding_config_hash"],
            }
            cache_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest_path = cache_path.with_suffix(cache_path.suffix + ".manifest.json")
            manifest_path.write_text(json.dumps({
                "format": "ea_atpo_v2_public_embedding_cache_v1",
                "embedding_config_hash": config["embedding_config_hash"],
                "state_count": 1,
                "output_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            }), encoding="utf-8")
            cache = load_public_embedding_cache(cache_path, expected_examples=[example])
            self.assertTrue(np.allclose(cache.vectors[(example.case_id, example.state_id)], [0.25, -0.5]))
            self.assertEqual(cache.metadata["embedding_config_hash"], config["embedding_config_hash"])
            changed = replace(example, public_state=({"role": "user", "content": "different public state"},))
            with self.assertRaisesRegex(ValueError, "state hash mismatch"):
                load_public_embedding_cache(cache_path, expected_examples=[changed])

    def test_embedding_mean_pooling_ignores_padding(self):
        hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
        mask = torch.tensor([[1, 1, 0]])
        pooled = pool_hidden_states(hidden, mask)
        self.assertTrue(torch.allclose(pooled, torch.tensor([[2.0, 4.0]])))

    def test_embedding_extractor_uses_only_public_state_and_passes_template_kwargs(self):
        class FakeTokenizer:
            chat_template = "fake-template-v1"
            eos_token = "<eos>"
            eos_token_id = 0
            pad_token_id = 0

            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
                self.kwargs = kwargs
                return "|".join(f"{m['role']}:{m['content']}" for m in messages)

            def __call__(self, texts, **kwargs):
                ids = [[len(text) % 10 + 1, 2] for text in texts]
                return {"input_ids": torch.tensor(ids), "attention_mask": torch.ones(len(ids), 2, dtype=torch.long)}

        class FakeModel:
            def __call__(self, *, input_ids, attention_mask):
                hidden = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 2)
                return type("Result", (), {"last_hidden_state": hidden})()

        tokenizer = FakeTokenizer()
        rows = [make_example(0)]
        vectors, prompt_hashes = extract_batches(
            FakeModel(), tokenizer, rows, device="cpu", max_length=32,
            batch_size=1, normalize_l2=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        self.assertEqual(tokenizer.kwargs, {"enable_thinking": False})
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 2)
        self.assertEqual(len(prompt_hashes[0]), 64)

    def test_embedding_state_reader_deduplicates_public_states_and_enforces_split(self):
        with tempfile.TemporaryDirectory() as temp:
            states_path = Path(temp) / "states.jsonl"
            split_path = Path(temp) / "split.json"
            state = {"case_id": "case-0", "state_id": "state-0", "public_state": [{"role": "user", "content": "public"}], "final_accuracy": 1, "gold": "must_not_be_used"}
            states_path.write_text(json.dumps(state) + "\n" + json.dumps({**state, "action": "ASK"}) + "\n", encoding="utf-8")
            split_path.write_text(json.dumps({"unit": "case_id", "train": ["case-0"], "validation": ["case-1"], "internal_test": ["case-2"]}), encoding="utf-8")
            states, audit = unique_public_states(states_path, split_path, "train")
            self.assertEqual(len(states), 1)
            self.assertTrue(audit["no_gold_or_outcome_fields_used"])
            with self.assertRaisesRegex(ValueError, "outside official validation"):
                unique_public_states(states_path, split_path, "validation")

    def test_cost_recomputation_uses_raw_cost_components_once(self):
        self.assertAlmostEqual(utility_from_raw(1.0, 1, 1, 0.60, ask_cost=0.05), 0.35)
        self.assertAlmostEqual(utility_from_raw(1.0, 1, 1, 0.00, ask_cost=0.05), 0.95)

    def test_affine_action_calibration(self):
        pred = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
        target = 0.8 * pred + 0.1 - np.asarray([0.05, 0.0, 0.60])[None, :]
        mask = np.ones_like(pred, dtype=bool)
        costs = np.asarray([0.05, 0.0, 0.60])
        calibrator = ActionValueCalibrator().fit(pred - costs[None, :], target, mask, costs)
        gross = calibrator.calibrate_gross(pred - costs[None, :], costs)
        self.assertTrue(np.allclose(gross, 0.8 * pred + 0.1, atol=1e-6))

    def test_checkpoint_save_restore_roundtrip(self):
        torch.manual_seed(7)
        model = QRegressor(4, 5)
        features = torch.randn(3, 4)
        before = model(features).detach()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "q.pt"
            atomic_torch_save({"format": "ea_atpo_v2_training_checkpoint_v1", "model_type": "q_regressor",
                               "action_order": list(ACTION_NAMES), "input_dim": 4, "hidden_dim": 5,
                               "model_state": model.state_dict()}, path)
            checkpoint = load_checkpoint(path)
            restored = QRegressor(checkpoint["input_dim"], checkpoint["hidden_dim"])
            restored.load_state_dict(checkpoint["model_state"])
            self.assertTrue(torch.equal(before, restored(features).detach()))

    def test_policy_head_and_distillation_checkpoint(self):
        model = PolicyHead(4, 6)
        self.assertEqual(model(torch.ones(2, 4)).shape, (2, 3))
        targets = torch.tensor([[0.3, 0.7, 0.4], [0.1, 0.2, 0.9]])
        mask = torch.ones_like(targets, dtype=torch.bool)
        logits = model(torch.randn(2, 4))
        loss, rows = masked_distillation_loss(logits, targets, mask, temperature=0.7)
        self.assertEqual(rows, 2)
        self.assertTrue(torch.isfinite(loss))

    def test_evaluation_metrics_accuracy_cost_correction_spoilage_and_q_diagnostics(self):
        trajectories = []
        outcome_map = {"c1": (0, 1), "c2": (1, 0), "c3": (1, 1)}
        preds = {}
        for case_id, (answer, defer) in outcome_map.items():
            trajectories.extend([outcome(case_id, "ANSWER", answer), outcome(case_id, "DEFER", defer), outcome(case_id, "ASK", 1)])
            preds[(case_id, f"{case_id}:s0")] = predictions_for(case_id, answer, defer)
        states = aggregate_rollouts(trajectories)
        policies = ["always_answer", "always_defer", "static_l2d", "uncertainty_threshold", "sft_only", "q_guided_distillation"]
        summaries, paired, curves, normalized = case_level_metrics(states, preds, policies, [0.0, 0.6], 0.5, 0.05, 99)
        always = next(row for row in summaries if row["policy"] == "always_answer" and row["lambda_defer"] == 0.6)
        self.assertAlmostEqual(always["accuracy"], 2 / 3)
        self.assertEqual(len(paired), len(policies) * 2)
        self.assertEqual(len(curves), len(policies) * 2)
        self.assertEqual(len(normalized), len(policies) * 2 * 3)
        d = next(row for row in summaries if row["policy"] == "always_defer" and row["lambda_defer"] == 0.6)
        self.assertEqual(d["expert_correction_count"], 1)
        self.assertEqual(d["expert_spoilage_count"], 1)
        diagnostics = q_diagnostics(states, preds, np.asarray([0.05, 0.0, 0.60]))
        self.assertAlmostEqual(diagnostics["q_mae"], 0.0)
        self.assertAlmostEqual(diagnostics["advantage_auroc"], 1.0)

    def test_evaluate_cli_selects_uncertainty_on_validation_and_writes_curve(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rollout_path = root / "rollouts.jsonl"
            pred_path = root / "predictions.jsonl"
            split_path = root / "split.json"
            output_dir = root / "eval"
            records, predictions = [], []
            for case_id, (answer, defer) in {"c1": (0, 1), "c2": (1, 0), "c3": (1, 1)}.items():
                records.extend([outcome(case_id, action, correct) for action, correct in (("ANSWER", answer), ("DEFER", defer), ("ASK", 1))])
                predictions.append(predictions_for(case_id, answer, defer))
            rollout_path.write_text("".join(json.dumps(row) + "\n" for row in records))
            pred_path.write_text("".join(json.dumps(row) + "\n" for row in predictions))
            split_path.write_text(json.dumps({"unit": "case_id", "train": [], "validation": ["c1", "c2", "c3"], "internal_test": []}))
            argv = ["evaluate_policies", "--trajectories", str(rollout_path), "--predictions", str(pred_path),
                    "--case-split", str(split_path), "--split", "validation", "--output-dir", str(output_dir),
                    "--select-uncertainty-threshold", "--cost-grid", "0", "0.6"]
            with patch("sys.argv", argv):
                evaluate_main()
            report = json.loads((output_dir / "evaluation.json").read_text())
            self.assertTrue(report["synthetic_data"])
            self.assertTrue(report["uncertainty_threshold_selected_on_validation"])
            self.assertTrue((output_dir / "uncertainty_threshold_lock.json").exists())
            self.assertEqual(len((output_dir / "accuracy_cost_curve.csv").read_text().splitlines()), 1 + 6 * 2)


if __name__ == "__main__":
    unittest.main()
