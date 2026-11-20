"""
Unit Tests for End-to-End Inference Pipeline and Grad-CAM Generation.
"""

import unittest
from PIL import Image
import torch

from src.inference.pipeline import FractureInferencePipeline


class TestInferencePipeline(unittest.TestCase):
    def setUp(self):
        # Initialize pipeline on CPU for quick deterministic unit testing
        self.pipeline = FractureInferencePipeline(
            classifier_checkpoint="models/checkpoints/non_existent.pth",  # Should safely run with initialized weights
            detector_path="yolov8n.pt",
            device=torch.device('cpu')
        )
        self.sample_img = Image.new('RGB', (500, 700), color=(120, 120, 120))

    def test_pipeline_output_contract(self):
        result = self.pipeline.predict(self.sample_img)

        # Verify all contract keys are present
        required_keys = [
            "prediction_label",
            "is_fractured",
            "fracture_probability",
            "non_fracture_probability",
            "decision_threshold",
            "model_confidence",
            "anatomy_name",
            "detector_confidence",
            "is_fallback",
            "original_image",
            "annotated_detection_image",
            "cropped_image",
            "gradcam_heatmap",
            "gradcam_overlay",
            "focus_box",
            "focused_image",
            "peak_focus_intensity"
        ]
        for k in required_keys:
            self.assertIn(k, result, f"Missing output key: {k}")

        self.assertIn(result["prediction_label"], ["FRACTURED", "NON-FRACTURED"])
        self.assertIsInstance(result["is_fractured"], bool)
        self.assertTrue(0.0 <= result["fracture_probability"] <= 1.0)
        self.assertTrue(0.0 <= result["non_fracture_probability"] <= 1.0)
        self.assertEqual(result["cropped_image"].size, (300, 300))
        self.assertEqual(result["gradcam_overlay"].size, (300, 300))
        self.assertEqual(result["focused_image"].size, (300, 300))
        self.assertIsInstance(result["focus_box"], tuple)
        self.assertEqual(len(result["focus_box"]), 4)

        # Verify that Step 5 focus box is strictly inside Step 2 bounding box
        fb = result["focus_box"]
        bb = result["bounding_box"]
        self.assertGreaterEqual(fb[0], bb[0], f"Focus box x1 ({fb[0]}) must be >= Step 2 box x1 ({bb[0]})")
        self.assertGreaterEqual(fb[1], bb[1], f"Focus box y1 ({fb[1]}) must be >= Step 2 box y1 ({bb[1]})")
        self.assertLessEqual(fb[2], bb[2], f"Focus box x2 ({fb[2]}) must be <= Step 2 box x2 ({bb[2]})")
        self.assertLessEqual(fb[3], bb[3], f"Focus box y2 ({fb[3]}) must be <= Step 2 box y2 ({bb[3]})")

        # Verify crop space constraint
        cfb = result["crop_focus_box"]
        s2b = result["step2_crop_box"]
        self.assertGreaterEqual(cfb[0], s2b[0], f"Crop focus x1 ({cfb[0]}) must be >= Step 2 crop x1 ({s2b[0]})")
        self.assertGreaterEqual(cfb[1], s2b[1], f"Crop focus y1 ({cfb[1]}) must be >= Step 2 crop y1 ({s2b[1]})")
        self.assertLessEqual(cfb[2], s2b[2], f"Crop focus x2 ({cfb[2]}) must be <= Step 2 crop x2 ({s2b[2]})")
        self.assertLessEqual(cfb[3], s2b[3], f"Crop focus y2 ({cfb[3]}) must be <= Step 2 crop y2 ({s2b[3]})")


if __name__ == '__main__':
    unittest.main()
