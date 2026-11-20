"""
Two-Phase Transfer Learning Trainer for EfficientNetB3 Fracture Classifier.
Implements Phase 1 (frozen backbone) and Phase 2 (fine-tuning upper layers)
with early stopping, learning rate scheduling, class-weighted/focal loss,
and automated validation threshold optimization.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_recall_fscore_support,
    accuracy_score
)
from tqdm import tqdm

from src.classification.efficientnet_model import build_fracture_classifier
from src.classification.losses import get_loss_function
from src.classification.threshold_tuning import tune_classification_threshold
from src.dataset.dataset_loader import get_data_loaders


def evaluate_model(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluates model on validation set.
    Returns:
        (loss, accuracy, recall, precision, f1, roc_auc, all_probs, all_targets)
    """
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits, _ = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * images.size(0)

            # Test-Time Augmentation (Original + Horizontal Flip)
            images_flip = torch.flip(images, dims=[3])
            logits_flip, _ = model(images_flip)
            probs = ((torch.sigmoid(logits) + torch.sigmoid(logits_flip)) / 2.0).cpu().numpy()

            all_probs.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    num_samples = len(all_targets)
    avg_loss = total_loss / max(num_samples, 1)

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # Standard 0.5 threshold metrics for monitoring
    preds_05 = (all_probs >= 0.50).astype(int)
    acc = accuracy_score(all_targets, preds_05)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, preds_05, pos_label=1, average='binary', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        roc_auc = 0.5

    return avg_loss, acc, r, p, f1, roc_auc, all_probs, all_targets


def train_classifier(
    experiment_name: str = "baseline_full_image",
    loss_type: str = "class_weighted",
    use_sampler: bool = False,
    use_crops: bool = False,
    use_clahe: bool = False,
    crop_boxes: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
    epochs_p1: int = 5,
    epochs_p2: int = 10,
    lr_p1: float = 1e-3,
    lr_p2: float = 1e-4,
    batch_size: int = 16,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    patience: int = 5,
    save_dir: str = "models/checkpoints",
    results_dir: str = "results"
) -> Dict[str, Any]:
    """
    Executes complete two-phase training workflow.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=======================================================")
    print(f"Starting Training: {experiment_name}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Loss: {loss_type} | Sampler: {use_sampler} | Crop-Guided: {use_crops} | CLAHE: {use_clahe}")
    print(f"=======================================================\n")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)

    # Load crop boxes if requested
    if use_crops and crop_boxes is None:
        det_json = "data/processed/yolo_detections.json"
        if os.path.exists(det_json):
            print(f"Loading precomputed YOLO bounding boxes from {det_json}...")
            with open(det_json, 'r', encoding='utf-8') as f:
                raw_dets = json.load(f)
                crop_boxes = {k: v["box"] for k, v in raw_dets.items() if v.get("box") is not None}
                print(f"Loaded {len(crop_boxes)} anatomical crop boxes.")

    # Load data
    train_loader, val_loader, _ = get_data_loaders(
        train_data="train",
        val_data="val",
        batch_size=batch_size,
        num_workers=0,
        use_weighted_sampler=use_sampler,
        crop_boxes=crop_boxes if use_crops else None,
        use_clahe=use_clahe
    )

    # Calculate class weight from train split (moderated to balance accuracy and recall)
    train_df = pd.read_csv("data/splits/train.csv")
    num_neg = (train_df["fracture_id"] == 0).sum()
    num_pos = (train_df["fracture_id"] == 1).sum()
    raw_pos_weight = float(num_neg / max(num_pos, 1))
    pos_weight = min(raw_pos_weight, 2.0)

    # Initialize loss
    criterion = get_loss_function(
        method=loss_type,
        pos_weight=pos_weight,
        focal_gamma=2.0,
        focal_alpha=0.75,
        device=device
    )

    # Build model
    model = build_fracture_classifier(pretrained=True, dropout=dropout, device=device)

    history: List[Dict[str, Any]] = []
    best_val_score = -1.0
    best_checkpoint_path = os.path.join(save_dir, f"{experiment_name}_best.pth")
    epochs_no_improve = 0

    # -------------------------------------------------------------
    # PHASE 1: Train Classification Head (Backbone Frozen)
    # -------------------------------------------------------------
    print(f"\n>>> PHASE 1: Training Classification Head ({epochs_p1} Epochs, lr={lr_p1})")
    model.freeze_backbone()
    optimizer_p1 = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_p1,
        weight_decay=weight_decay
    )

    for epoch in range(1, epochs_p1 + 1):
        model.train()
        train_loss = 0.0

        for images, targets, _ in tqdm(train_loader, desc=f"P1 Epoch {epoch}/{epochs_p1}"):
            images = images.to(device)
            targets = targets.to(device)

            optimizer_p1.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer_p1.step()

            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss, val_acc, val_r, val_p, val_f1, val_auc, _, _ = evaluate_model(model, val_loader, criterion, device)

        print(f"P1 Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val Recall: {val_r:.4f} | AUC: {val_auc:.4f}")

        history.append({
            "epoch": epoch,
            "phase": 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "val_recall": val_r,
            "val_precision": val_p,
            "val_roc_auc": val_auc,
            "lr": lr_p1
        })

        val_score = 0.4 * val_acc + 0.3 * val_f1 + 0.3 * val_auc
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save({
                "epoch": epoch,
                "phase": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_p1.state_dict(),
                "val_score": val_score,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_auc": val_auc,
                "experiment_name": experiment_name,
                "loss_type": loss_type
            }, best_checkpoint_path)

    # -------------------------------------------------------------
    # PHASE 2: Partial Fine-Tuning (Upper Backbone Layers Unfrozen)
    # -------------------------------------------------------------
    print(f"\n>>> PHASE 2: Partial Fine-Tuning ({epochs_p2} Epochs, lr={lr_p2})")
    model.unfreeze_upper_layers(num_blocks=4, freeze_bn=True)
    optimizer_p2 = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_p2,
        weight_decay=weight_decay
    )
    scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=epochs_p2, eta_min=1e-6)

    for epoch in range(1, epochs_p2 + 1):
        global_epoch = epochs_p1 + epoch
        model.train()
        train_loss = 0.0

        for images, targets, _ in tqdm(train_loader, desc=f"P2 Epoch {epoch}/{epochs_p2}"):
            images = images.to(device)
            targets = targets.to(device)

            optimizer_p2.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer_p2.step()

            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_loss, val_acc, val_r, val_p, val_f1, val_auc, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler_p2.step()

        current_lr = optimizer_p2.param_groups[0]['lr']
        print(f"P2 Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val Recall: {val_r:.4f} | AUC: {val_auc:.4f} | lr: {current_lr:.1e}")

        history.append({
            "epoch": global_epoch,
            "phase": 2,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "val_recall": val_r,
            "val_precision": val_p,
            "val_roc_auc": val_auc,
            "lr": current_lr
        })

        val_score = 0.4 * val_acc + 0.3 * val_f1 + 0.3 * val_auc
        if val_score > best_val_score:
            best_val_score = val_score
            epochs_no_improve = 0
            torch.save({
                "epoch": global_epoch,
                "phase": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_p2.state_dict(),
                "val_score": val_score,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_auc": val_auc,
                "experiment_name": experiment_name,
                "loss_type": loss_type
            }, best_checkpoint_path)
            print(f"--> Saved New Best Checkpoint (Score: {best_val_score:.4f} | Acc: {val_acc*100:.1f}% | AUC: {val_auc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at P2 epoch {epoch} (no improvement for {patience} epochs).")
                break

    # Save training history CSV
    history_df = pd.DataFrame(history)
    history_csv = os.path.join(results_dir, "metrics", f"{experiment_name}_history.csv")
    history_df.to_csv(history_csv, index=False)

    # Plot Training Loss and Metrics
    plot_training_history(history_df, experiment_name, os.path.join(results_dir, "plots", f"{experiment_name}_history.png"))

    # Load best checkpoint for validation threshold optimization
    print(f"\nLoading best checkpoint for threshold tuning: {best_checkpoint_path}")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, _, _, _, _, _, val_probs, val_targets = evaluate_model(model, val_loader, criterion, device)
    
    thresh_plot_path = os.path.join(results_dir, "plots", f"{experiment_name}_threshold_tuning.png")
    thresh_json_path = os.path.join("config", f"{experiment_name}_threshold.json")
    best_thresh, thresh_metrics = tune_classification_threshold(
        y_true=val_targets,
        y_probs=val_probs,
        target_metric="accuracy_constrained",
        min_recall=0.65,
        save_path=thresh_json_path,
        plot_path=thresh_plot_path
    )

    summary = {
        "experiment_name": experiment_name,
        "best_checkpoint": best_checkpoint_path,
        "best_val_score": round(float(best_val_score), 4),
        "optimal_threshold": round(float(best_thresh), 4),
        "threshold_metrics": thresh_metrics,
        "history_file": history_csv
    }

    return summary


def plot_training_history(history_df: pd.DataFrame, experiment_name: str, save_path: str) -> None:
    """Plots multi-panel training loss, F1, and learning rate curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Loss Curves
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", color='#1f77b4', linewidth=2)
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", color='#d62728', linewidth=2)
    axes[0].set_title(f"Training & Validation Loss ({experiment_name})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 2. Validation F1 and Recall
    axes[1].plot(history_df["epoch"], history_df["val_f1"], label="Val F1", color='#2ca02c', linewidth=2)
    axes[1].plot(history_df["epoch"], history_df["val_recall"], label="Val Recall", color='#ff7f0e', linewidth=2, linestyle='--')
    axes[1].plot(history_df["epoch"], history_df["val_roc_auc"], label="Val ROC-AUC", color='#9467bd', linewidth=1.5, linestyle=':')
    axes[1].set_title("Validation Diagnostic Metrics", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # 3. Learning Rate Schedule
    axes[2].plot(history_df["epoch"], history_df["lr"], color='#8c564b', linewidth=2)
    axes[2].set_title("Learning Rate Schedule", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_yscale('log')
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training history visualization saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNetB3 Fracture Classifier")
    parser.add_argument("--experiment_name", type=str, default="baseline_full_image")
    parser.add_argument("--loss_type", type=str, default="class_weighted", choices=["class_weighted", "focal_loss", "standard_bce"])
    parser.add_argument("--use_sampler", action="store_true")
    parser.add_argument("--use_crops", action="store_true")
    parser.add_argument("--epochs_p1", type=int, default=4)
    parser.add_argument("--epochs_p2", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    train_classifier(
        experiment_name=args.experiment_name,
        loss_type=args.loss_type,
        use_sampler=args.use_sampler,
        use_crops=args.use_crops,
        epochs_p1=args.epochs_p1,
        epochs_p2=args.epochs_p2,
        batch_size=args.batch_size
    )
