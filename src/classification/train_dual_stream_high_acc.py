"""
High-Accuracy Dual-Stream Anatomy-Aware EfficientNetB3 Classifier.
Integrates Global Radiograph context with Localized Anatomical Bounding-Box Crops.
Tracks Train Accuracy, Validation Accuracy, and Test Accuracy across epochs.
Pushes diagnostic accuracy towards 95% on held-out partitions.
"""

import os
import sys
import json
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
from src.preprocessing.image_preprocessing import (
    load_and_validate_image,
    resize_with_aspect_ratio_padding,
    IMAGENET_MEAN,
    IMAGENET_STD
)
from src.localization.crop_generator import generate_anatomical_crop
from src.evaluation.metrics import compute_comprehensive_metrics, plot_roc_and_pr_curves
from src.evaluation.confusion_matrix import plot_and_save_confusion_matrix


# ---------------------------------------------------------------------
# 1. Dual-Stream In-Memory Dataset
# ---------------------------------------------------------------------
class CachedDualStreamDataset(Dataset):
    """
    High-speed in-memory cached dataset providing synchronized Full and Crop pairs.
    """
    def __init__(self, split_csv: str, det_json: str, is_train: bool = False, target_size=(300, 300), existing_data=None):
        self.df = pd.read_csv(split_csv)
        self.is_train = is_train
        self.target_size = target_size
        self.labels = self.df["fracture_id"].values.astype(np.float32)
        self.filepaths = self.df["filepath"].tolist()

        if existing_data is not None:
            self.full_images = existing_data["full_images"]
            self.crop_images = existing_data["crop_images"]
        else:
            with open(det_json, "r", encoding="utf-8") as f:
                detections = json.load(f)

            print(f"Pre-caching {len(self.df)} dual images for {'Train' if is_train else 'Eval'} split...")
            self.full_images = []
            self.crop_images = []

            for p in tqdm(self.filepaths, desc=f"Caching {os.path.basename(split_csv)}"):
                pil_img = load_and_validate_image(p)
                det = detections.get(p, {})
                box = det.get("box", None)

                # 1. Crop image
                c_img, _, _, _ = generate_anatomical_crop(
                    image=pil_img,
                    bounding_box=box,
                    padding_ratio=0.12,
                    target_size=target_size
                )
                # 2. Full image
                f_img, _, _ = resize_with_aspect_ratio_padding(pil_img, target_size=target_size)

                self.full_images.append(f_img)
                self.crop_images.append(c_img)

        # Augmentation transforms
        if is_train:
            self.spatial_jitter = T.RandomAffine(degrees=10, translate=(0.04, 0.04), scale=(0.96, 1.04), fill=0)
            self.color_jitter = T.ColorJitter(brightness=0.10, contrast=0.10)
        else:
            self.spatial_jitter = None
            self.color_jitter = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        f_img = self.full_images[idx]
        c_img = self.crop_images[idx]
        label = self.labels[idx]
        path = self.filepaths[idx]

        if self.is_train:
            # Synchronized horizontal flip
            if torch.rand(1).item() > 0.5:
                f_img = TF.hflip(f_img)
                c_img = TF.hflip(c_img)

            f_img = self.spatial_jitter(f_img)
            f_img = self.color_jitter(f_img)
            c_img = self.spatial_jitter(c_img)
            c_img = self.color_jitter(c_img)

        t_full = TF.normalize(TF.to_tensor(f_img), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        t_crop = TF.normalize(TF.to_tensor(c_img), mean=IMAGENET_MEAN, std=IMAGENET_STD)

        return t_full, t_crop, torch.tensor(label, dtype=torch.float32), path


# ---------------------------------------------------------------------
# 2. Dual-Stream Architecture
# ---------------------------------------------------------------------
class DualStreamEfficientNetB3(nn.Module):
    """
    Two-Stream Feature Fusion Model for Bone Fracture Detection.
    Combines global radiograph context with high-resolution anatomical crop.
    Loads medical backbone weights directly from local checkpoint.
    """
    def __init__(self, pretrained_backbone_path: str = "models/checkpoints/enhanced_full_image_best.pth", dropout: float = 0.3):
        super().__init__()
        base = efficientnet_b3(weights=None)

        if os.path.exists(pretrained_backbone_path):
            print(f"Loading local medical backbone weights from {pretrained_backbone_path}...")
            ckpt = torch.load(pretrained_backbone_path, map_location="cpu")
            sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            feat_sd = {k.replace("features.", ""): v for k, v in sd.items() if k.startswith("features.")}
            base.features.load_state_dict(feat_sd, strict=True)
            print("Loaded 100% of EfficientNetB3 backbone feature weights cleanly.")
        else:
            print(f"Warning: Local checkpoint {pretrained_backbone_path} not found. Using initialized weights.")

        self.features = base.features
        self.avgpool = base.avgpool
        feat_dim = 1536

        # Deep Fusion Classifier Head
        self.fusion_head = nn.Sequential(
            nn.Linear(feat_dim * 2, 768),
            nn.BatchNorm1d(768),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(768, 192),
            nn.BatchNorm1d(192),
            nn.SiLU(),
            nn.Dropout(p=dropout * 0.7),
            nn.Linear(192, 1)
        )

        # Auxiliary single-stream heads for multi-task regularization
        self.aux_full = nn.Linear(feat_dim, 1)
        self.aux_crop = nn.Linear(feat_dim, 1)

    def forward(self, x_full: torch.Tensor, x_crop: torch.Tensor):
        # Extract features
        feats_full = torch.flatten(self.avgpool(self.features(x_full)), 1)
        feats_crop = torch.flatten(self.avgpool(self.features(x_crop)), 1)

        # Concatenate joint features
        joint_feats = torch.cat([feats_full, feats_crop], dim=1)
        logit_fusion = self.fusion_head(joint_feats).squeeze(-1)

        logit_full = self.aux_full(feats_full).squeeze(-1)
        logit_crop = self.aux_crop(feats_crop).squeeze(-1)

        return logit_fusion, logit_full, logit_crop

    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False
        print("Backbone frozen for Phase 1 Head Training.")

    def unfreeze_upper_layers(self, num_blocks: int = 5):
        total_layers = len(self.features)
        unfreeze_start = max(0, total_layers - num_blocks)
        for i, layer in enumerate(self.features):
            requires_grad = (i >= unfreeze_start)
            for param in layer.parameters():
                param.requires_grad = requires_grad
        print(f"Unfrozen top {num_blocks} blocks of EfficientNetB3 backbone for Phase 2 fine-tuning.")


# ---------------------------------------------------------------------
# 3. Label-Smoothing BCE Loss
# ---------------------------------------------------------------------
class SmoothBCELoss(nn.Module):
    def __init__(self, smoothing: float = 0.04, pos_weight: float = 1.0):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = torch.tensor(pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        smooth_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, smooth_targets,
            pos_weight=self.pos_weight.to(logits.device)
        )
        return bce


# ---------------------------------------------------------------------
# 4. Evaluation Function (With TTA)
# ---------------------------------------------------------------------
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5):
    model.eval()
    all_probs = []
    all_targets = []
    all_paths = []

    with torch.no_grad():
        for x_full, x_crop, targets, paths in loader:
            x_full = x_full.to(device)
            x_crop = x_crop.to(device)

            # View 1: Original
            l_f1, l_full1, l_crop1 = model(x_full, x_crop)
            p1 = torch.sigmoid(l_f1)

            # View 2: Horizontal Flip TTA
            x_full_flip = torch.flip(x_full, dims=[3])
            x_crop_flip = torch.flip(x_crop, dims=[3])
            l_f2, _, _ = model(x_full_flip, x_crop_flip)
            p2 = torch.sigmoid(l_f2)

            probs = ((p1 + p2) / 2.0).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(targets.numpy())
            all_paths.extend(paths)

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    preds = (all_probs >= threshold).astype(int)

    acc = accuracy_score(all_targets, preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, preds, pos_label=1, average='binary', zero_division=0)
    try:
        auc_score = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc_score = 0.5

    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "roc_auc": auc_score,
        "probabilities": all_probs,
        "targets": all_targets,
        "filepaths": all_paths
    }


# ---------------------------------------------------------------------
# 5. Master Training Routine
# ---------------------------------------------------------------------
def train_high_accuracy_dual_stream(
    epochs_p1: int = 4,
    epochs_p2: int = 14,
    batch_size: int = 24,
    lr_p1: float = 8e-4,
    lr_p2_bb: float = 3e-5,
    lr_p2_head: float = 2e-4,
    save_dir: str = "models/checkpoints",
    results_dir: str = "results"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" TRAINING HIGH-ACCURACY DUAL-STREAM EFFICIENTNET-B3")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Target: Elevating Accuracy from 91% towards 95%")
    print(f"=======================================================\n")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "predictions"), exist_ok=True)

    # 1. Load Data Splits
    det_json = "data/processed/yolo_detections.json"
    train_ds = CachedDualStreamDataset("data/splits/train.csv", det_json, is_train=True)
    val_ds = CachedDualStreamDataset("data/splits/val.csv", det_json, is_train=False)
    test_ds = CachedDualStreamDataset("data/splits/test.csv", det_json, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    # Eval loaders without shuffling
    train_eval_ds = CachedDualStreamDataset(
        "data/splits/train.csv", det_json, is_train=False,
        existing_data={"full_images": train_ds.full_images, "crop_images": train_ds.crop_images}
    )
    train_eval_loader = DataLoader(train_eval_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. Build Model & Criterion
    model = DualStreamEfficientNetB3(
        pretrained_backbone_path="models/checkpoints/enhanced_full_image_best.pth",
        dropout=0.30
    ).to(device)
    criterion = SmoothBCELoss(smoothing=0.03, pos_weight=1.0)

    best_val_acc = 0.0
    best_checkpoint_path = os.path.join(save_dir, "dual_stream_high_acc_best.pth")
    history = []

    # -------------------------------------------------------------
    # PHASE 1: Train Classification Fusion Head
    # -------------------------------------------------------------
    print(f"\n>>> PHASE 1: Training Fusion Head ({epochs_p1} Epochs, lr={lr_p1})")
    model.freeze_backbone()
    optimizer_p1 = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_p1,
        weight_decay=1e-3
    )

    for epoch in range(1, epochs_p1 + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x_full, x_crop, targets, _ in tqdm(train_loader, desc=f"P1 Epoch {epoch}/{epochs_p1}"):
            x_full, x_crop, targets = x_full.to(device), x_crop.to(device), targets.to(device)
            optimizer_p1.zero_grad()

            l_fusion, l_full, l_crop = model(x_full, x_crop)
            loss_fus = criterion(l_fusion, targets)
            loss_ful = criterion(l_full, targets)
            loss_crp = criterion(l_crop, targets)
            loss = loss_fus + 0.20 * loss_ful + 0.20 * loss_crp

            loss.backward()
            optimizer_p1.step()

            train_loss += loss.item() * targets.size(0)
            preds = (torch.sigmoid(l_fusion) >= 0.50).float()
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        train_loss = train_loss / train_total
        train_epoch_acc = train_correct / train_total

        # Evaluate Train & Validation Accuracy
        val_eval = evaluate_loader(model, val_loader, device)
        val_acc = val_eval["accuracy"]
        val_f1 = val_eval["f1"]
        val_auc = val_eval["roc_auc"]

        print(f"P1 Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_epoch_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")

        history.append({
            "epoch": epoch,
            "phase": 1,
            "train_loss": train_loss,
            "train_accuracy": train_epoch_acc,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "val_auc": val_auc
        })

    # -------------------------------------------------------------
    # PHASE 2: Deep Backbone Fine-Tuning
    # -------------------------------------------------------------
    print(f"\n>>> PHASE 2: Fine-Tuning EfficientNetB3 ({epochs_p2} Epochs)")
    model.unfreeze_upper_layers(num_blocks=5)

    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = [p for p in model.fusion_head.parameters()] + [p for p in model.aux_full.parameters()] + [p for p in model.aux_crop.parameters()]

    optimizer_p2 = AdamW([
        {"params": backbone_params, "lr": lr_p2_bb, "weight_decay": 1e-4},
        {"params": head_params, "lr": lr_p2_head, "weight_decay": 1e-3}
    ])
    scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=epochs_p2, eta_min=1e-6)

    for epoch in range(1, epochs_p2 + 1):
        global_epoch = epochs_p1 + epoch
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x_full, x_crop, targets, _ in tqdm(train_loader, desc=f"P2 Epoch {epoch}/{epochs_p2}"):
            x_full, x_crop, targets = x_full.to(device), x_crop.to(device), targets.to(device)
            optimizer_p2.zero_grad()

            l_fusion, l_full, l_crop = model(x_full, x_crop)
            loss_fus = criterion(l_fusion, targets)
            loss_ful = criterion(l_full, targets)
            loss_crp = criterion(l_crop, targets)
            loss = loss_fus + 0.20 * loss_ful + 0.20 * loss_crp

            loss.backward()
            optimizer_p2.step()

            train_loss += loss.item() * targets.size(0)
            preds = (torch.sigmoid(l_fusion) >= 0.50).float()
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        train_loss = train_loss / train_total
        train_epoch_acc = train_correct / train_total
        scheduler_p2.step()

        # Evaluate on Validation
        val_eval = evaluate_loader(model, val_loader, device)
        val_acc = val_eval["accuracy"]
        val_f1 = val_eval["f1"]
        val_auc = val_eval["roc_auc"]

        print(f"P2 Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_epoch_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")

        history.append({
            "epoch": global_epoch,
            "phase": 2,
            "train_loss": train_loss,
            "train_accuracy": train_epoch_acc,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "val_auc": val_auc
        })

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_f1 > 0.70):
            best_val_acc = val_acc
            torch.save({
                "epoch": global_epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "val_auc": val_auc
            }, best_checkpoint_path)
            print(f"--> [SAVED BEST MODEL] Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")

    # Save training history
    history_df = pd.DataFrame(history)
    history_csv = os.path.join(results_dir, "metrics", "dual_stream_high_acc_history.csv")
    history_df.to_csv(history_csv, index=False)

    # Plot training loss & accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", color="#3b82f6", linewidth=2)
    axes[0].set_title("Training Loss across Epochs", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(history_df["epoch"], history_df["train_accuracy"]*100, label="Train Accuracy", color="#10b981", linewidth=2)
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"]*100, label="Validation Accuracy", color="#8b5cf6", linewidth=2)
    axes[1].set_title("Train & Validation Accuracy (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    hist_plot_path = os.path.join(results_dir, "plots", "dual_stream_high_acc_history.png")
    plt.tight_layout()
    plt.savefig(hist_plot_path, dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # 6. Final Comprehensive Evaluation: Train, Val, and Test Accuracy
    # -------------------------------------------------------------
    print(f"\n=======================================================")
    print(f" EVALUATING BEST DUAL-STREAM MODEL ON ALL 3 PARTITIONS")
    print(f" Loading checkpoint: {best_checkpoint_path}")
    print(f"=======================================================")

    ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Step A: Tune threshold on Validation Set to maximize accuracy
    val_results = evaluate_loader(model, val_loader, device, threshold=0.50)
    best_t = 0.50
    best_t_acc = 0.0

    for t in np.linspace(0.20, 0.80, 61):
        acc = accuracy_score(val_results["targets"], (val_results["probabilities"] >= t).astype(int))
        if acc > best_t_acc:
            best_t_acc = acc
            best_t = t

    print(f"Optimal Validation Threshold: {best_t:.2f} (Val Acc: {best_t_acc*100:.2f}%)")

    # Save threshold config
    with open("config/dual_stream_high_acc_threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": round(float(best_t), 4), "val_accuracy": round(float(best_t_acc), 4)}, f, indent=2)

    # Step B: Evaluate on Train Set
    train_eval = evaluate_loader(model, train_eval_loader, device, threshold=best_t)
    train_acc = train_eval["accuracy"]

    # Step C: Evaluate on Validation Set with optimal threshold
    val_eval = evaluate_loader(model, val_loader, device, threshold=best_t)
    val_acc = val_eval["accuracy"]

    # Step D: Evaluate on Held-Out Test Set with optimal threshold
    test_eval = evaluate_loader(model, test_loader, device, threshold=best_t)
    test_acc = test_eval["accuracy"]

    # Compute comprehensive test metrics
    test_metrics = compute_comprehensive_metrics(test_eval["targets"], test_eval["probabilities"], threshold=best_t)

    # Save prediction CSVs for all 3 splits (Train, Val, Test)
    train_preds_df = pd.DataFrame({
        "filepath": train_eval["filepaths"],
        "true_label": train_eval["targets"].astype(int),
        "fracture_probability": np.round(train_eval["probabilities"], 4),
        "predicted_label": (train_eval["probabilities"] >= best_t).astype(int)
    })
    train_preds_df.to_csv(os.path.join(results_dir, "predictions", "dual_stream_high_acc_train_predictions.csv"), index=False)

    val_preds_df = pd.DataFrame({
        "filepath": val_eval["filepaths"],
        "true_label": val_eval["targets"].astype(int),
        "fracture_probability": np.round(val_eval["probabilities"], 4),
        "predicted_label": (val_eval["probabilities"] >= best_t).astype(int)
    })
    val_preds_df.to_csv(os.path.join(results_dir, "predictions", "dual_stream_high_acc_val_predictions.csv"), index=False)

    test_preds_df = pd.DataFrame({
        "filepath": test_eval["filepaths"],
        "true_label": test_eval["targets"].astype(int),
        "fracture_probability": np.round(test_eval["probabilities"], 4),
        "predicted_label": (test_eval["probabilities"] >= best_t).astype(int)
    })
    preds_csv = os.path.join(results_dir, "predictions", "dual_stream_high_acc_test_predictions.csv")
    test_preds_df.to_csv(preds_csv, index=False)

    # Save metrics JSON
    metrics_json = os.path.join(results_dir, "metrics", "dual_stream_high_acc_metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # Generate Confusion Matrix, ROC, and PR Plots
    cm_path = os.path.join(results_dir, "plots", "dual_stream_high_acc_confusion_matrix.png")
    plot_and_save_confusion_matrix(
        test_eval["targets"],
        (test_eval["probabilities"] >= best_t).astype(int),
        save_path=cm_path,
        title="Dual-Stream EfficientNetB3 Test Confusion Matrix"
    )

    roc_path = os.path.join(results_dir, "plots", "dual_stream_high_acc_roc_curve.png")
    pr_path = os.path.join(results_dir, "plots", "dual_stream_high_acc_pr_curve.png")
    plot_roc_and_pr_curves(
        test_eval["targets"],
        test_eval["probabilities"],
        roc_save_path=roc_path,
        pr_save_path=pr_path
    )

    print("\n-------------------------------------------------------------")
    print(f"             FINAL MODEL ACCURACY METRICS                    ")
    print("-------------------------------------------------------------")
    print(f"  Train Accuracy:        {train_acc * 100:.2f}%")
    print(f"  Validation Accuracy:   {val_acc * 100:.2f}%")
    print(f"  Test Accuracy:         {test_acc * 100:.2f}%")
    print("-------------------------------------------------------------")
    print(f"  Test Fracture Recall:  {test_metrics['fracture_recall_sensitivity'] * 100:.2f}%")
    print(f"  Test Precision:        {test_metrics['fracture_precision'] * 100:.2f}%")
    print(f"  Test Fracture F1:      {test_metrics['fracture_f1']:.4f}")
    print(f"  Test Macro F1:         {test_metrics['macro_f1']:.4f}")
    print(f"  Test ROC-AUC:          {test_metrics['roc_auc']:.4f}")
    print(f"  Test PR-AUC:           {test_metrics['pr_auc']:.4f}")
    print(f"  Optimal Threshold:     {best_t:.2f}")
    print("-------------------------------------------------------------\n")

    # Update active production checkpoint and threshold if test accuracy exceeds baseline
    if test_acc >= 0.88:
        shutil.copy2(best_checkpoint_path, "models/checkpoints/best_model.pth")
        shutil.copy2("config/dual_stream_high_acc_threshold.json", "config/best_threshold.json")
        with open("results/metrics/test_evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump({
                "experiment": "Dual-Stream EfficientNet-B3 (High Accuracy)",
                "train_accuracy": round(float(train_acc), 4),
                "val_accuracy": round(float(val_acc), 4),
                "test_accuracy": round(float(test_acc), 4),
                "threshold": round(float(best_t), 4),
                **test_metrics
            }, f, indent=2)
        print("Updated models/checkpoints/best_model.pth and config/best_threshold.json with latest model.")

    return {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "test_metrics": test_metrics,
        "checkpoint": best_checkpoint_path
    }


if __name__ == "__main__":
    train_high_accuracy_dual_stream()
