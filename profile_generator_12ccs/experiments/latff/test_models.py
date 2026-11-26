"""Logic-only tests for LATFF A/B/C models.

The inputs below are artificial random tensors. They are never used as dataset
samples or experiment results.
"""

from __future__ import annotations

import math
from pathlib import Path
import unittest

import torch

from checkpoint import load_original_classifier_checkpoint
from models import (
    LengthAwareAdaptiveFusionClassifier,
    MaskedPoolingClassifier,
    OriginalPoolingClassifier,
    build_padding_mask,
    masked_pool,
)


torch.manual_seed(2026)


def make_model(model_class):
    # Small Encoder depth keeps the logic tests fast; dimensions still match the
    # real 12-CCA model's 16/128/176/12 interface.
    model = model_class(num_layers=1, dropout_rate=0.0)
    model.eval()
    return model


def artificial_batch(lengths, max_length=8):
    batch_size = len(lengths)
    trace = torch.randn(batch_size, max_length, 2)
    env = torch.randn(batch_size, 2)
    return trace, torch.tensor(lengths, dtype=torch.long), env, max_length


class PoolingLogicTests(unittest.TestCase):
    def test_same_length_outputs_have_expected_dimensions(self):
        trace, lengths, env, max_length = artificial_batch([8, 8, 8])
        for model_class in (
            OriginalPoolingClassifier,
            MaskedPoolingClassifier,
            LengthAwareAdaptiveFusionClassifier,
        ):
            model = make_model(model_class)
            logits = model(trace, lengths, env, max_length)
            feature, details = model.extract_features(trace, lengths, env, max_length)
            self.assertEqual(tuple(logits.shape), (3, 12))
            self.assertEqual(tuple(details["encoded"].shape), (3, 8, 16))
            self.assertEqual(tuple(details["sequence_feature"].shape), (3, 48))
            self.assertEqual(tuple(feature.shape), (3, 176))
            self.assertEqual(model.fc[0].in_features, 176)
            self.assertTrue(torch.isfinite(logits).all())

    def test_mixed_lengths_and_length_one(self):
        trace, lengths, env, max_length = artificial_batch([1, 3, 7, 8])
        model = make_model(MaskedPoolingClassifier)
        logits = model(trace, lengths, env, max_length)
        _, details = model.extract_features(trace, lengths, env, max_length)
        self.assertEqual(tuple(logits.shape), (4, 12))
        self.assertTrue(details["padding_mask"][0, 1:].all())
        self.assertFalse(details["padding_mask"][0, 0].item())
        self.assertFalse(details["padding_mask"][3].any().item())
        self.assertTrue(torch.isfinite(logits).all())

    def test_masked_pooling_ignores_extreme_padding_values(self):
        model = make_model(MaskedPoolingClassifier)
        trace, lengths, env, max_length = artificial_batch([3, 6])
        padded_trace = trace.clone()
        padded_trace[0, 3:] = 1.0e20
        padded_trace[1, 6:] = -1.0e20

        feature_a, detail_a = model.extract_features(trace, lengths, env, max_length)
        feature_b, detail_b = model.extract_features(padded_trace, lengths, env, max_length)
        logits_b = model(padded_trace, lengths, env, max_length)
        self.assertTrue(torch.allclose(feature_a, feature_b, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(detail_a["max"], detail_b["max"], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.isfinite(feature_b).all())
        self.assertTrue(torch.isfinite(logits_b).all())

    def test_gate_is_initially_identity_scaled(self):
        model = make_model(LengthAwareAdaptiveFusionClassifier)
        trace, lengths, env, max_length = artificial_batch([1, 4, 8])
        _, details = model.extract_features(trace, lengths, env, max_length)
        weights = details["gate_weights"]
        self.assertTrue(torch.allclose(weights, torch.ones_like(weights), atol=1e-7))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.full((3,), 3.0), atol=1e-7))

    def test_invalid_lengths_and_mask_have_clear_errors(self):
        encoded = torch.randn(2, 4, 16)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            masked_pool(encoded, torch.tensor([0, 2], dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            masked_pool(encoded, torch.tensor([5, 2], dtype=torch.long))
        bad_mask = torch.zeros(2, 4, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "does not match lengths"):
            masked_pool(encoded, torch.tensor([2, 4], dtype=torch.long), bad_mask)
        with self.assertRaisesRegex(ValueError, "max_length must equal"):
            model = make_model(MaskedPoolingClassifier)
            trace, lengths, env, _ = artificial_batch([2, 4])
            model(trace, lengths, env, 7)


class CheckpointCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = (
            Path(__file__).resolve().parents[2]
            / "models"
            / "best_model_accuracy.pth.tar"
        )

    def test_original_checkpoint_loads_for_a_b_and_c(self):
        self.assertTrue(self.checkpoint.is_file(), f"Missing test checkpoint: {self.checkpoint}")

        report_a = load_original_classifier_checkpoint(
            OriginalPoolingClassifier(), self.checkpoint
        )
        report_b = load_original_classifier_checkpoint(
            MaskedPoolingClassifier(), self.checkpoint
        )
        report_c = load_original_classifier_checkpoint(
            LengthAwareAdaptiveFusionClassifier(), self.checkpoint
        )

        self.assertTrue(report_a["strict"])
        self.assertTrue(report_b["strict"])
        self.assertFalse(report_c["strict"])
        self.assertEqual(report_a["missing_keys"], [])
        self.assertEqual(report_b["missing_keys"], [])
        self.assertEqual(report_c["missing_keys"], ["gate.bias", "gate.weight"])
        self.assertEqual(report_c["unexpected_keys"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

