"""
CLAHE-Enhanced Contrast-Boosted Fracture Classifier Training.
Applies Contrast Limited Adaptive Histogram Equalization to enhance
bone cortex and hairline trabecular fracture lines before feeding to
EfficientNetB3.
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.classification.train_classifier import train_classifier
from src.evaluation.evaluate import evaluate_checkpoint_on_test_set


def run_clahe_experiments():
    print("=================================================================")
    print(" TRAINING CLAHE-ENHANCED CONTRAST-BOOSTED FRACTURE CLASSIFIERS")
    print(" Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("=================================================================\n")

    # 1. Train CLAHE Full-Image Model
    print("\n[PHASE 1/2] Training CLAHE Full-Image Classifier (12 Epochs)...")
    _ = train_classifier(
        experiment_name="clahe_full_image",
        loss_type="class_weighted",
        use_sampler=False,
        use_crops=False,
        use_clahe=True,
        epochs_p1=4,
        epochs_p2=8,
        batch_size=32,
        lr_p1=1e-3,
        lr_p2=1.5e-4,
        weight_decay=1e-3,
        dropout=0.35,
        patience=5
    )

    test_metrics_full = evaluate_checkpoint_on_test_set(
        checkpoint_path="models/checkpoints/clahe_full_image_best.pth",
        test_csv="data/splits/test.csv",
        threshold_path="config/clahe_full_image_threshold.json",
        experiment_name="clahe_full_image",
        batch_size=32,
        use_crops=False,
        use_clahe=True
    )
    print("\n>>> CLAHE Full-Image Test Results:")
    print(f"Accuracy:  {test_metrics_full['accuracy']*100:.2f}%")
    print(f"Recall:    {test_metrics_full['fracture_recall_sensitivity']*100:.2f}%")
    print(f"Precision: {test_metrics_full['fracture_precision']*100:.2f}%")
    print(f"F1 Score:  {test_metrics_full['fracture_f1']:.4f}")
    print(f"ROC-AUC:   {test_metrics_full['roc_auc']:.4f}")

    # 2. Train CLAHE Anatomy-Guided Model
    print("\n[PHASE 2/2] Training CLAHE Anatomy-Guided Classifier (12 Epochs)...")
    _ = train_classifier(
        experiment_name="clahe_anatomy_guided",
        loss_type="class_weighted",
        use_sampler=False,
        use_crops=True,
        use_clahe=True,
        epochs_p1=4,
        epochs_p2=8,
        batch_size=32,
        lr_p1=1e-3,
        lr_p2=1.5e-4,
        weight_decay=1e-3,
        dropout=0.35,
        patience=5
    )

    test_metrics_anat = evaluate_checkpoint_on_test_set(
        checkpoint_path="models/checkpoints/clahe_anatomy_guided_best.pth",
        test_csv="data/splits/test.csv",
        threshold_path="config/clahe_anatomy_guided_threshold.json",
        experiment_name="clahe_anatomy_guided",
        batch_size=32,
        use_crops=True,
        use_clahe=True
    )
    print("\n>>> CLAHE Anatomy-Guided Test Results:")
    print(f"Accuracy:  {test_metrics_anat['accuracy']*100:.2f}%")
    print(f"Recall:    {test_metrics_anat['fracture_recall_sensitivity']*100:.2f}%")
    print(f"Precision: {test_metrics_anat['fracture_precision']*100:.2f}%")
    print(f"F1 Score:  {test_metrics_anat['fracture_f1']:.4f}")
    print(f"ROC-AUC:   {test_metrics_anat['roc_auc']:.4f}")


if __name__ == "__main__":
    run_clahe_experiments()
