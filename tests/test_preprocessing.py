"""
Unit Tests for Image Preprocessing and Aspect-Ratio Padding.
"""

import unittest
import numpy as np
from PIL import Image
import torch

from src.preprocessing.image_preprocessing import (
    load_and_validate_image,
    resize_with_aspect_ratio_padding,
    apply_clahe,
    preprocess_for_inference
)
from src.preprocessing.augmentation import (
    get_training_transforms,
    get_validation_transforms
)


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Create synthetic images: RGB, RGBA, and Grayscale
        self.rgb_img = Image.new('RGB', (400, 600), color=(100, 100, 100))
        self.rgba_img = Image.new('RGBA', (500, 300), color=(150, 150, 150, 200))
        self.gray_img = Image.new('L', (350, 350), color=80)

    def test_load_and_validate_image_modes(self):
        # Test RGB
        res_rgb = load_and_validate_image(self.rgb_img)
        self.assertEqual(res_rgb.mode, 'RGB')
        self.assertEqual(res_rgb.size, (400, 600))

        # Test RGBA conversion to 3-channel RGB
        res_rgba = load_and_validate_image(self.rgba_img)
        self.assertEqual(res_rgba.mode, 'RGB')
        self.assertEqual(res_rgba.size, (500, 300))

        # Test Grayscale conversion to 3-channel RGB
        res_gray = load_and_validate_image(self.gray_img)
        self.assertEqual(res_gray.mode, 'RGB')
        self.assertEqual(res_gray.size, (350, 350))

    def test_resize_with_aspect_ratio_padding(self):
        target_size = (300, 300)
        padded_img, pad_coords, scale = resize_with_aspect_ratio_padding(self.rgb_img, target_size=target_size)
        
        # Dimensions must match target
        self.assertEqual(padded_img.size, target_size)
        pad_left, pad_top, new_w, new_h = pad_coords
        
        # Image was 400x600 (height > width). Height should scale to 300, width should be 200.
        self.assertEqual(new_h, 300)
        self.assertEqual(new_w, 200)
        self.assertEqual(pad_top, 0)
        self.assertEqual(pad_left, 50)
        self.assertEqual(pad_left * 2 + new_w, 300)

    def test_clahe_enhancement(self):
        enhanced = apply_clahe(self.rgb_img, clip_limit=2.0)
        self.assertEqual(enhanced.size, self.rgb_img.size)
        self.assertEqual(enhanced.mode, 'RGB')

    def test_preprocess_for_inference(self):
        tensor, orig, coords = preprocess_for_inference(self.rgb_img, target_size=(300, 300))
        self.assertEqual(tensor.shape, (1, 3, 300, 300))
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(orig.size, (400, 600))


if __name__ == '__main__':
    unittest.main()
