"""
PyTorch Dataset and DataLoader Pipeline for Bone Fracture Detection.
Supports aspect-ratio preserving padding, online data augmentation,
and class-balanced WeightedRandomSampler.
"""

import os
from typing import Optional, Callable, Tuple, Union, Dict, Any
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.preprocessing.image_preprocessing import (
    load_and_validate_image,
    resize_with_aspect_ratio_padding,
    apply_clahe
)
from src.preprocessing.augmentation import (
    get_training_transforms,
    get_validation_transforms
)


class BoneFractureDataset(Dataset):
    """
    Custom PyTorch Dataset for Bone Fracture Radiographs.
    """
    def __init__(
        self,
        data: Union[str, pd.DataFrame],
        target_size: Tuple[int, int] = (300, 300),
        transform: Optional[Callable] = None,
        use_clahe: bool = False,
        crop_boxes: Optional[Dict[str, Tuple[int, int, int, int]]] = None
    ):
        """
        Args:
            data: Split name ('train', 'val', 'test'), path to split CSV, or DataFrame.
            target_size: Desired model resolution (300, 300).
            transform: Transformation / augmentation pipeline.
            use_clahe: Whether to apply CLAHE contrast enhancement.
            crop_boxes: Optional dict mapping filepath to (x1, y1, x2, y2) bounding box crops.
        """
        if isinstance(data, str):
            if data in ['train', 'val', 'test']:
                csv_path = os.path.join("data", "splits", f"{data}.csv")
            else:
                csv_path = data
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Split CSV not found: {csv_path}")
            self.df = pd.read_csv(csv_path)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise TypeError("data must be a split name, csv path, or pandas DataFrame")

        self.target_size = target_size
        self.transform = transform
        self.use_clahe = use_clahe
        self.crop_boxes = crop_boxes or {}

        # Precompute class labels and weights
        self.labels = self.df["fracture_id"].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        label = row["fracture_id"]

        # 1. Load and validate image
        pil_img = load_and_validate_image(img_path)

        # 2. If crop box exists for this image, extract the anatomical region
        if img_path in self.crop_boxes:
            box = self.crop_boxes[img_path]
            # box is (x1, y1, x2, y2)
            pil_img = pil_img.crop(box)

        # 3. Optional CLAHE
        if self.use_clahe:
            pil_img = apply_clahe(pil_img)

        # 4. Aspect-ratio preserving padding to 300x300
        padded_img, _, _ = resize_with_aspect_ratio_padding(pil_img, target_size=self.target_size)

        # 5. Apply transforms / tensor conversion / normalization
        if self.transform is not None:
            tensor = self.transform(padded_img)
        else:
            default_t = get_validation_transforms(self.target_size)
            tensor = default_t(padded_img)

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return tensor, label_tensor, img_path


def get_data_loaders(
    train_data: Union[str, pd.DataFrame] = "train",
    val_data: Union[str, pd.DataFrame] = "val",
    test_data: Optional[Union[str, pd.DataFrame]] = "test",
    batch_size: int = 16,
    num_workers: int = 0,
    use_weighted_sampler: bool = False,
    target_size: Tuple[int, int] = (300, 300),
    use_clahe: bool = False,
    crop_boxes: Optional[Dict[str, Tuple[int, int, int, int]]] = None
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Constructs PyTorch DataLoaders for train, val, and test splits.
    Supports WeightedRandomSampler for class imbalance handling.
    """
    train_transform = get_training_transforms(target_size)
    eval_transform = get_validation_transforms(target_size)

    train_dataset = BoneFractureDataset(
        data=train_data,
        target_size=target_size,
        transform=train_transform,
        use_clahe=use_clahe,
        crop_boxes=crop_boxes
    )

    val_dataset = BoneFractureDataset(
        data=val_data,
        target_size=target_size,
        transform=eval_transform,
        use_clahe=use_clahe,
        crop_boxes=crop_boxes
    )

    # Imbalance handling: WeightedRandomSampler
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        labels = train_dataset.labels
        class_counts = np.bincount(labels.astype(int))
        class_weights = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights[labels.astype(int)]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False  # Sampler and shuffle are mutually exclusive

    # On Windows, num_workers=0 avoids multiprocessing spawn overhead
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = None
    if test_data is not None:
        test_dataset = BoneFractureDataset(
            data=test_data,
            target_size=target_size,
            transform=eval_transform,
            use_clahe=use_clahe,
            crop_boxes=crop_boxes
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    return train_loader, val_loader, test_loader
