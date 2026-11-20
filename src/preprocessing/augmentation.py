"""
Medically-Realistic Radiograph Data Augmentation Pipeline.
Preserves subtle cortical bone fracture lines while introducing
controlled rotational, contrast, and affine diversity.
"""

from typing import Tuple
import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_training_transforms(target_size: Tuple[int, int] = (300, 300)) -> T.Compose:
    """
    Returns enhanced training transform pipeline with medically realistic augmentations.
    Preserves subtle bone features while exposing the model to position, angle, and illumination variance.
    """
    return T.Compose([
        # Random rotation (+/- 15 degrees)
        T.RandomRotation(degrees=15, fill=0),
        # Horizontal flip (bilaterally valid for limb radiographs)
        T.RandomHorizontalFlip(p=0.5),
        # Affine translations and scale variation
        T.RandomAffine(
            degrees=0,
            translate=(0.06, 0.06),
            scale=(0.92, 1.08),
            fill=0
        ),
        # Subtle perspective adjustment simulating slight angle tilt
        T.RandomPerspective(distortion_scale=0.10, p=0.25, fill=0),
        # Controlled brightness and contrast adjustment
        T.ColorJitter(brightness=0.12, contrast=0.15),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_validation_transforms(target_size: Tuple[int, int] = (300, 300)) -> T.Compose:
    """
    Returns deterministic validation/testing transform pipeline without augmentations.
    """
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
