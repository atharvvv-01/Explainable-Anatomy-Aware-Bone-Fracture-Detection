"""
Reusable Image Preprocessing Pipeline for Bone Fracture Radiographs.
Maintains anatomical aspect ratios with symmetric padding, supports
grayscale/RGB/RGBA formats, optional CLAHE, and ImageNet normalization.
"""

import os
from typing import Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_and_validate_image(image_input: Union[str, np.ndarray, Image.Image]) -> Image.Image:
    """
    Safely loads and validates an image from filepath, numpy array, or PIL Image.
    Converts RGBA or Grayscale (L) to 3-channel RGB.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at path: {image_input}")
        with Image.open(image_input) as img:
            # Handle alpha channel or single-channel grayscale
            if img.mode == 'RGBA':
                # Composite over black background to preserve bone radiodensity
                background = Image.new('RGB', img.size, (0, 0, 0))
                background.paste(img, mask=img.split()[3])
                rgb_img = background
            elif img.mode != 'RGB':
                rgb_img = img.convert('RGB')
            else:
                rgb_img = img.copy()
            return rgb_img
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            rgb_arr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            rgb_arr = cv2.cvtColor(image_input, cv2.COLOR_BGRA2RGB)
        elif image_input.shape[2] == 3:
            rgb_arr = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported numpy image shape: {image_input.shape}")
        return Image.fromarray(rgb_arr)
    elif isinstance(image_input, Image.Image):
        if image_input.mode == 'RGBA':
            bg = Image.new('RGB', image_input.size, (0, 0, 0))
            bg.paste(image_input, mask=image_input.split()[3])
            return bg
        elif image_input.mode != 'RGB':
            return image_input.convert('RGB')
        return image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def resize_with_aspect_ratio_padding(
    image: Image.Image,
    target_size: Tuple[int, int] = (300, 300),
    fill_color: Tuple[int, int, int] = (0, 0, 0)
) -> Tuple[Image.Image, Tuple[int, int, int, int], float]:
    """
    Resizes image maintaining original aspect ratio and applies symmetric padding.
    Returns:
        padded_image: The resized and padded PIL Image (target_size)
        pad_coords: (pad_left, pad_top, new_width, new_height)
        scale_factor: Scaling multiplier applied to original dimensions
    """
    orig_w, orig_h = image.size
    target_w, target_h = target_size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Calculate symmetric padding
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2

    padded = Image.new('RGB', (target_w, target_h), fill_color)
    padded.paste(resized, (pad_left, pad_top))

    pad_coords = (pad_left, pad_top, new_w, new_h)
    return padded, pad_coords, scale


def apply_clahe(image: Image.Image, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> Image.Image:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to enhance subtle trabecular bone contours without over-amplifying noise.
    """
    img_np = np.array(image)
    # Convert to LAB color space
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_rgb)


def preprocess_for_inference(
    image_input: Union[str, np.ndarray, Image.Image],
    target_size: Tuple[int, int] = (300, 300),
    use_clahe: bool = False
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int, int, int]]:
    """
    Complete inference preprocessing pipeline:
    1. Loads and validates image to 3-channel RGB.
    2. Retains original image for Streamlit rendering.
    3. Resizes with aspect ratio preservation and symmetric padding to target_size.
    4. Optionally applies subtle CLAHE.
    5. Converts to PyTorch Tensor [C, H, W] in [0, 1].
    6. Normalizes using ImageNet statistics.
    Returns:
        tensor: [1, 3, target_h, target_w] normalized tensor for EfficientNetB3
        rgb_orig: Original uncropped PIL image
        pad_coords: Padding coordinates (pad_left, pad_top, new_w, new_h)
    """
    rgb_orig = load_and_validate_image(image_input)
    
    # Process for classifier
    proc_img = rgb_orig
    if use_clahe:
        proc_img = apply_clahe(proc_img)

    padded_img, pad_coords, _ = resize_with_aspect_ratio_padding(proc_img, target_size=target_size)

    # Tensor conversion & normalization
    tensor = TF.to_tensor(padded_img)  # Shape [3, H, W], range [0.0, 1.0]
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    tensor = tensor.unsqueeze(0)  # Shape [1, 3, H, W]

    return tensor, rgb_orig, pad_coords
