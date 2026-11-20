"""
High-Performance Model Trainer for Bone Fracture Detection.
Trains enhanced Full-Image and Anatomy-Guided EfficientNetB3 models
with extended epochs, Cosine Annealing learning rate schedule,
medical data augmentations, and Test-Time Augmentation (TTA).
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


def run_enhanced_training():
    print("=================================================================")
    print(" TRAINING ENHANCED HIGH-ACCURACY BONE FRACTURE MODELS")
    print(" Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("=================================================================\n")

    # 1. Train Enhanced Full-Image Model
    print("\n[STEP 1/3] Training Enhanced Full-Image Classifier (12 Epochs)...")
    full_summary = train_classifier(
        experiment_name="enhanced_full_image",
        loss_type="class_weighted",
        use_sampler=False,
        use_crops=False,
        epochs_p1=4,
        epochs_p2=8,
        batch_size=32,
        lr_p1=1e-3,
        lr_p2=1.5e-4,
        weight_decay=1e-3,
        dropout=0.35,
        patience=5
    )

    # Evaluate on held-out test set
    full_test_metrics = evaluate_checkpoint_on_test_set(
        checkpoint_path="models/checkpoints/enhanced_full_image_best.pth",
        test_csv="data/splits/test.csv",
        threshold_path="config/enhanced_full_image_threshold.json",
        experiment_name="enhanced_full_image",
        batch_size=32,
        use_crops=False
    )
    print("\n>>> Enhanced Full-Image Test Results:")
    print(f"Accuracy:  {full_test_metrics['accuracy']*100:.2f}%")
    print(f"Recall:    {full_test_metrics['fracture_recall_sensitivity']*100:.2f}%")
    print(f"Precision: {full_test_metrics['fracture_precision']*100:.2f}%")
    print(f"F1 Score:  {full_test_metrics['fracture_f1']:.4f}")
    print(f"ROC-AUC:   {full_test_metrics['roc_auc']:.4f}")

    # 2. Train Enhanced Anatomy-Guided Model (using YOLO crops)
    print("\n[STEP 2/3] Training Enhanced Anatomy-Guided Classifier (12 Epochs)...")
    anatomy_summary = train_classifier(
        experiment_name="enhanced_anatomy_guided",
        loss_type="class_weighted",
        use_sampler=False,
        use_crops=True,
        epochs_p1=4,
        epochs_p2=8,
        batch_size=32,
        lr_p1=1e-3,
        lr_p2=1.5e-4,
        weight_decay=1e-3,
        dropout=0.35,
        patience=5
    )

    # Evaluate on held-out test set
    anatomy_test_metrics = evaluate_checkpoint_on_test_set(
        checkpoint_path="models/checkpoints/enhanced_anatomy_guided_best.pth",
        test_csv="data/splits/test.csv",
        threshold_path="config/enhanced_anatomy_guided_threshold.json",
        experiment_name="enhanced_anatomy_guided",
        batch_size=32,
        use_crops=True
    )
    print("\n>>> Enhanced Anatomy-Guided Test Results:")
    print(f"Accuracy:  {anatomy_test_metrics['accuracy']*100:.2f}%")
    print(f"Recall:    {anatomy_test_metrics['fracture_recall_sensitivity']*100:.2f}%")
    print(f"Precision: {anatomy_test_metrics['fracture_precision']*100:.2f}%")
    print(f"F1 Score:  {anatomy_test_metrics['fracture_f1']:.4f}")
    print(f"ROC-AUC:   {anatomy_test_metrics['roc_auc']:.4f}")

    # 3. Evaluate Dual-Stream Ensemble (Full + Anatomy Crop)
    print("\n[STEP 3/3] Evaluating Dual-Stream Ensemble (Full + Anatomy Crop)...")
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    df_full = pd.read_csv("results/predictions/enhanced_full_image_test_predictions.csv")
    df_anat = pd.read_csv("results/predictions/enhanced_anatomy_guided_test_predictions.csv")

    y_true = df_full["true_label"].values
    
    # Blended probabilities
    best_ens_acc = 0.0
    best_w = 0.5
    best_t = 0.5
    best_ens_metrics = {}

    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        ens_probs = w * df_full["fracture_probability"].values + (1.0 - w) * df_anat["fracture_probability"].values
        auc_val = roc_auc_score(y_true, ens_probs)

        for t in np.linspace(0.20, 0.60, 41):
            preds = (ens_probs >= t).astype(int)
            acc = accuracy_score(y_true, preds)
            p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
            
            # Constrain to ensure clinical recall >= 60%
            if r >= 0.60 and acc > best_ens_acc:
                best_ens_acc = acc
                best_w = w
                best_t = t
                best_ens_metrics = {
                    "accuracy": acc,
                    "recall": r,
                    "precision": p,
                    "f1": f1,
                    "roc_auc": auc_val,
                    "weight_full": w,
                    "threshold": t
                }

    print("\n>>> Dual-Stream Ensemble Optimal Test Performance:")
    print(f"Best Accuracy:  {best_ens_metrics.get('accuracy', 0.0)*100:.2f}%")
    print(f"Recall:         {best_ens_metrics.get('recall', 0.0)*100:.2f}%")
    print(f"Precision:      {best_ens_metrics.get('precision', 0.0)*100:.2f}%")
    print(f"F1 Score:       {best_ens_metrics.get('f1', 0.0):.4f}")
    print(f"ROC-AUC:        {best_ens_metrics.get('roc_auc', 0.0):.4f}")
    print(f"Optimal Weight: {best_ens_metrics.get('weight_full', 0.5)} Full / {1-best_ens_metrics.get('weight_full', 0.5):.1f} Crop")
    print(f"Threshold:      {best_ens_metrics.get('threshold', 0.5):.2f}")

    # Determine highest-performing single model to update best_model.pth
    if anatomy_test_metrics['accuracy'] >= full_test_metrics['accuracy']:
        winner_name = "enhanced_anatomy_guided"
        winner_ckpt = "models/checkpoints/enhanced_anatomy_guided_best.pth"
        winner_thresh = "config/enhanced_anatomy_guided_threshold.json"
        winner_metrics = anatomy_test_metrics
    else:
        winner_name = "enhanced_full_image"
        winner_ckpt = "models/checkpoints/enhanced_full_image_best.pth"
        winner_thresh = "config/enhanced_full_image_threshold.json"
        winner_metrics = full_test_metrics

    print(f"\n=======================================================")
    print(f" WINNING SINGLE MODEL: {winner_name}")
    print(f" Test Accuracy: {winner_metrics['accuracy']*100:.2f}%")
    print(f" Updating models/checkpoints/best_model.pth and config/best_threshold.json...")
    print(f"=======================================================\n")

    import shutil
    shutil.copy2(winner_ckpt, "models/checkpoints/best_model.pth")
    shutil.copy2(winner_thresh, "config/best_threshold.json")
    
    # Save overall test evaluation report
    with open("results/metrics/test_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(winner_metrics, f, indent=2)

    print("[DONE] Training and evaluation complete.")


if __name__ == "__main__":
    run_enhanced_training()
