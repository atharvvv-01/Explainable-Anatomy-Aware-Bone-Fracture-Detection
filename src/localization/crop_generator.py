"""
Anatomical Crop Generator with Symmetric Padding and Full-Image Fallback.
Safely extracts localized skeletal regions, expands boundary margin by ~12%,
and preserves trabecular bone context for Stage 2 classification.
"""

from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image

from src.preprocessing.image_preprocessing import (
    load_and_validate_image,
    resize_with_aspect_ratio_padding
)


def generate_anatomical_crop(
    image: Union[str, Image.Image, np.ndarray],
    bounding_box: Optional[Tuple[int, int, int, int]] = None,
    padding_ratio: float = 0.12,
    target_size: Tuple[int, int] = (300, 300)
) -> Tuple[Image.Image, Tuple[int, int, int, int], bool, str]:
    """
    Extracts anatomical region with symmetric padding and aspect-ratio preservation.
    Falls back to full image if bounding box is absent, invalid, or degenerated.

    Args:
        image: Input image (path, PIL Image, or numpy array).
        bounding_box: (x1, y1, x2, y2) in absolute pixel coordinates.
        padding_ratio: Fractional padding to expand around detected box (default: 0.12 = 12%).
        target_size: Target resolution for EfficientNetB3 classifier (300, 300).

    Returns:
        crop_image: Resized & padded PIL Image (target_size)
        effective_box: (x1, y1, x2, y2) pixel coordinates used for crop
        is_fallback: True if full image was used due to detection failure
        warning_msg: Informative warning if fallback was triggered, else empty string
    """
    pil_img = load_and_validate_image(image)
    orig_w, orig_h = pil_img.size

    # Fallback Condition 1: Bounding box is None
    if bounding_box is None:
        padded_full, _, _ = resize_with_aspect_ratio_padding(pil_img, target_size=target_size)
        return padded_full, (0, 0, orig_w, orig_h), True, "No anatomical bounding box detected. Full-image fallback engaged."

    x1, y1, x2, y2 = bounding_box
    bw = x2 - x1
    bh = y2 - y1

    # Fallback Condition 2: Degenerate or negative box
    if bw <= 10 or bh <= 10 or x1 >= orig_w or y1 >= orig_h:
        padded_full, _, _ = resize_with_aspect_ratio_padding(pil_img, target_size=target_size)
        return padded_full, (0, 0, orig_w, orig_h), True, f"Degenerate bounding box ({x1},{y1},{x2},{y2}). Full-image fallback engaged."

    # Compute symmetric padding
    pad_x = int(round(bw * padding_ratio))
    pad_y = int(round(bh * padding_ratio))

    # Expand box while clamping to image boundary
    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(orig_w, x2 + pad_x)
    crop_y2 = min(orig_h, y2 + pad_y)

    cropped_region = pil_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    effective_box = (crop_x1, crop_y1, crop_x2, crop_y2)

    # Resize cropped region with aspect-ratio preserving padding to 300x300
    padded_crop, _, _ = resize_with_aspect_ratio_padding(cropped_region, target_size=target_size)

    return padded_crop, effective_box, False, ""
