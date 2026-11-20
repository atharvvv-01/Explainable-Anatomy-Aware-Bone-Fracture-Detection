"""
Reproducible Dataset Splitting Module.
Implements stratified splitting (70% Train, 15% Val, 15% Test)
with strict leakage prevention (isolating duplicates into training set).
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def create_reproducible_splits(
    df: pd.DataFrame,
    splits_dir: str = "data/splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Creates stratified splits with leak prevention for duplicate images.
    All duplicate pairs are confined strictly to the training partition.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    os.makedirs(splits_dir, exist_ok=True)

    # 1. Separate exact duplicate images to prevent them from landing in test/val
    dup_mask = df["is_exact_duplicate"]
    duplicates_df = df[dup_mask].copy()
    unique_df = df[~dup_mask].copy().reset_index(drop=True)

    # Stratified split on unique images
    strat_col = "fracture_label"
    
    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=random_state)
    
    train_idx, temp_idx = next(sss1.split(unique_df, unique_df[strat_col]))
    train_unique = unique_df.iloc[train_idx].copy()
    temp_unique = unique_df.iloc[temp_idx].copy().reset_index(drop=True)

    # Second split: Val vs Test (50/50 of temp)
    val_prop_of_temp = val_ratio / temp_ratio
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_prop_of_temp), random_state=random_state)
    val_idx, test_idx = next(sss2.split(temp_unique, temp_unique[strat_col]))
    
    val_df = temp_unique.iloc[val_idx].copy().reset_index(drop=True)
    test_df = temp_unique.iloc[test_idx].copy().reset_index(drop=True)

    # Add duplicate images strictly into train_df to preserve them without polluting val or test
    train_df = pd.concat([train_unique, duplicates_df], ignore_index=True)

    # Assign split labels
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Save to CSV files
    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "val.csv")
    test_path = os.path.join(splits_dir, "test.csv")
    catalog_path = os.path.join(splits_dir, "dataset_catalog.csv")
    summary_path = os.path.join(splits_dir, "split_summary.json")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    combined_df.to_csv(catalog_path, index=False)

    summary = {
        "random_state": random_state,
        "total_records": len(combined_df),
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "train_distribution": train_df["fracture_label"].value_counts().to_dict(),
        "val_distribution": val_df["fracture_label"].value_counts().to_dict(),
        "test_distribution": test_df["fracture_label"].value_counts().to_dict(),
        "duplicate_records_in_train": int(len(duplicates_df)),
        "duplicate_records_in_val": int(val_df["is_exact_duplicate"].sum()),
        "duplicate_records_in_test": int(test_df["is_exact_duplicate"].sum()),
        "files": {
            "train_csv": train_path,
            "val_csv": val_path,
            "test_csv": test_path,
            "catalog_csv": catalog_path
        }
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print("\n--- Reproducible Splits Generated ---")
    print(f"Train Set: {len(train_df)} ({train_df['fracture_label'].value_counts().to_dict()})")
    print(f"Val Set:   {len(val_df)} ({val_df['fracture_label'].value_counts().to_dict()})")
    print(f"Test Set:  {len(test_df)} ({test_df['fracture_label'].value_counts().to_dict()})")
    print(f"Catalog saved to {catalog_path}")

    return train_df, val_df, test_df, summary


if __name__ == "__main__":
    from src.dataset.data_cleaning import scan_and_clean_dataset
    df, _ = scan_and_clean_dataset()
    create_reproducible_splits(df)
