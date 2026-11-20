"""
Trainer and Evaluator for Stage 1 YOLO Anatomical Localization Detector.
Prepares representative annotated dataset, trains lightweight YOLOv8n,
evaluates IoU, Precision, Recall, mAP@50, mAP@50-95, and saves best model.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import torch
from ultralytics import YOLO

from src.localization.annotation_utils import (
    generate_assistive_anatomical_annotations,
    validate_yolo_dataset,
    create_yolo_dataset_yaml
)


def train_anatomical_detector(
    dataset_yaml: str = "data/annotations/yolo/dataset.yaml",
    epochs: int = 15,
    imgsz: int = 320,
    batch: int = 16,
    model_name: str = "yolov8n.pt",
    save_dir: str = "models/detector",
    results_dir: str = "results"
) -> Dict[str, Any]:
    """
    Executes training and evaluation of YOLO anatomical detector.
    """
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"Training YOLO Anatomical Detector on device: {device}")
    print(f"Epochs: {epochs} | Batch: {batch} | ImgSz: {imgsz}")
    print(f"=======================================================\n")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)

    # Check if dataset yaml exists; if not, generate bootstrap annotations
    if not os.path.exists(dataset_yaml):
        print("YOLO dataset not found. Generating representative annotated subset (350 images)...")
        catalog_path = "data/splits/dataset_catalog.csv"
        if not os.path.exists(catalog_path):
            from src.dataset.data_cleaning import scan_and_clean_dataset
            from src.dataset.split_dataset import create_reproducible_splits
            df, _ = scan_and_clean_dataset()
            create_reproducible_splits(df)
        df_catalog = pd.read_csv(catalog_path)
        gen_summary = generate_assistive_anatomical_annotations(df_catalog, num_samples=350)
        print(f"Generated {gen_summary['total_annotated']} annotations across {gen_summary['class_distribution']}")

    # Initialize YOLO model
    model = YOLO(model_name)

    # Train
    train_results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,
        project="experiments/detector_run",
        name="anatomy_yolo",
        exist_ok=True,
        verbose=True
    )

    # Evaluate on validation split
    print("\nEvaluating YOLO Detector on Validation Set...")
    metrics = model.val(data=dataset_yaml, device=device, verbose=False)

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    detector_metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "map_50": round(map50, 4),
        "map_50_95": round(map50_95, 4),
        "epochs": epochs,
        "batch_size": batch,
        "input_resolution": imgsz
    }

    # Save metrics JSON
    metrics_path = os.path.join(results_dir, "metrics", "detector_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(detector_metrics, f, indent=2)

    # Save best model checkpoint
    best_pt = os.path.join("experiments", "detector_run", "anatomy_yolo", "weights", "best.pt")
    target_pt = os.path.join(save_dir, "yolov8n_anatomy.pt")
    if os.path.exists(best_pt):
        shutil.copy2(best_pt, target_pt)
        print(f"Best detector weights copied to: {target_pt}")
    else:
        # Fallback to last.pt or current weights
        model.save(target_pt)
        print(f"Detector weights saved to: {target_pt}")

    print("\n--- YOLO Detector Evaluation Metrics ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"mAP@50:    {map50:.4f}")
    print(f"mAP@50-95: {map50_95:.4f}")

    return detector_metrics


if __name__ == "__main__":
    train_anatomical_detector(epochs=15, batch=16, imgsz=320)
