"""
Unit Tests for EfficientNetB3 Classifier, Losses, and Threshold Tuning.
"""

import unittest
import numpy as np
import torch

from src.classification.efficientnet_model import build_fracture_classifier, BoneFractureEfficientNet
from src.classification.losses import FocalLoss, get_loss_function
from src.classification.threshold_tuning import tune_classification_threshold


class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.model = build_fracture_classifier(pretrained=False, device=self.device)

    def test_model_forward_pass_shape(self):
        # EfficientNetB3 takes [B, 3, 300, 300]
        x = torch.randn(2, 3, 300, 300, device=self.device)
        logits, anatomy_logits = self.model(x)
        self.assertEqual(logits.shape, (2,))
        self.assertIsNone(anatomy_logits)

        # Test predict_proba
        probs = self.model.predict_proba(x)
        self.assertEqual(probs.shape, (2,))
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

    def test_freeze_and_unfreeze_backbone(self):
        # Freeze backbone
        self.model.freeze_backbone()
        counts_frozen = self.model.count_trainable_parameters()
        # Head has (1536 * 1 + 1) = 1537 parameters
        self.assertEqual(counts_frozen["trainable"], 1537)

        # Unfreeze top layers
        self.model.unfreeze_upper_layers(num_blocks=3)
        counts_unfrozen = self.model.count_trainable_parameters()
        self.assertGreater(counts_unfrozen["trainable"], 1537)

    def test_focal_loss_computation(self):
        focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
        logits = torch.tensor([1.5, -2.0, 0.5])
        targets = torch.tensor([1.0, 0.0, 1.0])
        loss = focal_loss(logits, targets)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_threshold_tuning(self):
        # Synthetic ground truth and predictions
        y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 1])
        y_probs = np.array([0.1, 0.2, 0.3, 0.65, 0.8, 0.4, 0.75, 0.15, 0.85, 0.9])
        
        best_thresh, metrics = tune_classification_threshold(
            y_true=y_true,
            y_probs=y_probs,
            target_metric="f1",
            save_path="scratch/test_threshold.json",
            plot_path="scratch/test_threshold_curve.png"
        )
        self.assertGreater(best_thresh, 0.0)
        self.assertLess(best_thresh, 1.0)
        self.assertIn("fracture_f1", metrics)


if __name__ == '__main__':
    unittest.main()
