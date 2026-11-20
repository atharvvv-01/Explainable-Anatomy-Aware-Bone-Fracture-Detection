"""
Unit Tests for Anatomical Object Detector and Padded Cropping Pipeline.
"""

import unittest
from PIL import Image

from src.localization.crop_generator import generate_anatomical_crop
from src.localization.detector import AnatomicalDetector


class TestDetectorAndCropping(unittest.TestCase):
    def setUp(self):
        # Create a test synthetic image
        self.test_img = Image.new('RGB', (800, 600), color=(120, 120, 120))

    def test_anatomical_crop_valid_box(self):
        # Bounding box (x1, y1, x2, y2)
        box = (200, 150, 400, 450)
        padded_crop, eff_box, is_fallback, warning = generate_anatomical_crop(
            image=self.test_img,
            bounding_box=box,
            padding_ratio=0.12,
            target_size=(300, 300)
        )
        self.assertFalse(is_fallback)
        self.assertEqual(warning, "")
        self.assertEqual(padded_crop.size, (300, 300))
        # Effective box should be padded:
        # width = 200, pad_x = 24 -> x1 = 176, x2 = 424
        # height = 300, pad_y = 36 -> y1 = 114, y2 = 486
        self.assertEqual(eff_box, (176, 114, 424, 486))

    def test_anatomical_crop_fallback_mechanism(self):
        # Test None box triggers fallback
        padded_crop, eff_box, is_fallback, warning = generate_anatomical_crop(
            image=self.test_img,
            bounding_box=None,
            target_size=(300, 300)
        )
        self.assertTrue(is_fallback)
        self.assertIn("fallback", warning.lower())
        self.assertEqual(padded_crop.size, (300, 300))
        self.assertEqual(eff_box, (0, 0, 800, 600))

    def test_anatomical_crop_degenerate_box(self):
        # Test degenerate small box triggers fallback
        box = (50, 50, 52, 53)
        padded_crop, eff_box, is_fallback, warning = generate_anatomical_crop(
            image=self.test_img,
            bounding_box=box,
            target_size=(300, 300)
        )
        self.assertTrue(is_fallback)
        self.assertIn("fallback", warning.lower())

    def test_detector_inference_interface(self):
        detector = AnatomicalDetector(conf_threshold=0.35, device="cpu")
        res = detector.detect(self.test_img)
        self.assertIn("has_detection", res)
        self.assertIn("is_fallback", res)
        self.assertIn("annotated_image", res)
        self.assertIn("confidence", res)
        self.assertEqual(res["annotated_image"].size, self.test_img.size)


if __name__ == '__main__':
    unittest.main()
