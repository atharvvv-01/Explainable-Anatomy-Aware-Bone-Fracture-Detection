"""
DenseNet-121 Medical Radiology Classifier for Bone Fracture Detection.
Utilizes Dense Connectivity to preserve high-frequency cortical fracture lines.
Evaluates Train, Validation, and Test Accuracy across epochs.
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import densenet121
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
from src.classification.train_dual_stream_high_acc import CachedDualStreamDataset
from src.evaluation.metrics import compute_comprehensive_metrics
from src.evaluation.confusion_matrix import plot_and_save_confusion_matrix


class MedicalDenseNet121(nn.Module):
    def __init__(self, weights_path="C:/Users/ATHARV/.cache/torch/hub/checkpoints/densenet121-a639ec97.pth", dropout=0.3):
        super().__init__()
        base = densenet121(weights=None)
        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location="cpu")
            state_dict = {}
            for key in ckpt.keys():
                new_key = re.sub(r'^(features\.denseblock\d+\.denselayer\d+\.(?:norm|conv))\.(\d)', r'\1\2', key)
                state_dict[new_key] = ckpt[key]
            base.load_state_dict(state_dict, strict=False)
            print("Loaded ImageNet DenseNet-121 pretrained features.")

        self.features = base.features
        in_features = base.classifier.in_features  # 1024

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.7),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        feats = self.features(x)
        out = nn.functional.adaptive_avg_pool2d(feats, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out).squeeze(-1)


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    probs, targets, paths = [], [], []
    with torch.no_grad():
        for x_full, x_crop, y, p in loader:
            imgs = x_full.to(device)
            # TTA
            p1 = torch.sigmoid(model(imgs))
            p2 = torch.sigmoid(model(torch.flip(imgs, dims=[3])))
            prob = ((p1 + p2) / 2.0).cpu().numpy()
            probs.extend(prob)
            targets.extend(y.numpy())
            paths.extend(p)

    probs = np.array(probs)
    targets = np.array(targets)
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(targets, preds)
    p, r, f1, _ = precision_recall_fscore_support(targets, preds, pos_label=1, average='binary', zero_division=0)
    try:
        auc_val = roc_auc_score(targets, probs)
    except:
        auc_val = 0.5
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "roc_auc": auc_val, "probs": probs, "targets": targets, "paths": paths}


def train_densenet(epochs=12, batch_size=24, lr=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining DenseNet-121 on device: {device}")

    det_json = "data/processed/yolo_detections.json"
    train_ds = CachedDualStreamDataset("data/splits/train.csv", det_json, is_train=True)
    val_ds = CachedDualStreamDataset("data/splits/val.csv", det_json, is_train=False)
    test_ds = CachedDualStreamDataset("data/splits/test.csv", det_json, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    train_eval_loader = DataLoader(
        CachedDualStreamDataset("data/splits/train.csv", det_json, is_train=False, existing_data={"full_images": train_ds.full_images, "crop_images": train_ds.crop_images}),
        batch_size=batch_size, shuffle=False
    )

    model = MedicalDenseNet121().to(device)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    save_path = "models/checkpoints/densenet121_best.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_full, _, targets, _ in tqdm(train_loader, desc=f"DenseNet Epoch {epoch}/{epochs}"):
            imgs = x_full.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)

        scheduler.step()
        train_loss = total_loss / len(train_loader.dataset)

        val_res = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_res['accuracy']*100:.2f}% | Val F1: {val_res['f1']:.4f} | Val AUC: {val_res['roc_auc']:.4f}")

        if val_res["accuracy"] > best_val_acc:
            best_val_acc = val_res["accuracy"]
            torch.save({"model_state_dict": model.state_dict(), "val_accuracy": best_val_acc}, save_path)
            print(f"--> [SAVED NEW BEST DENSENET] Val Acc: {best_val_acc*100:.2f}%")

    # Load best checkpoint
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Find optimal threshold on validation set
    val_res = evaluate(model, val_loader, device, threshold=0.5)
    best_t = 0.5
    best_acc = 0.0
    for t in np.linspace(0.2, 0.8, 61):
        a = accuracy_score(val_res["targets"], (val_res["probs"] >= t).astype(int))
        if a > best_acc:
            best_acc = a
            best_t = t

    train_res = evaluate(model, train_eval_loader, device, threshold=best_t)
    val_res = evaluate(model, val_loader, device, threshold=best_t)
    test_res = evaluate(model, test_loader, device, threshold=best_t)

    print("\n--- DenseNet-121 Results ---")
    print(f"Train Accuracy:      {train_res['accuracy']*100:.2f}%")
    print(f"Validation Accuracy: {val_res['accuracy']*100:.2f}%")
    print(f"Test Accuracy:       {test_res['accuracy']*100:.2f}%")
    print(f"Optimal Threshold:   {best_t:.2f}")

    # Save test predictions
    df_test_preds = pd.DataFrame({
        "filepath": test_res["paths"],
        "true_label": test_res["targets"].astype(int),
        "fracture_probability": np.round(test_res["probs"], 4),
        "predicted_label": (test_res["probs"] >= best_t).astype(int)
    })
    df_test_preds.to_csv("results/predictions/densenet121_test_predictions.csv", index=False)

    df_val_preds = pd.DataFrame({
        "filepath": val_res["paths"],
        "true_label": val_res["targets"].astype(int),
        "fracture_probability": np.round(val_res["probs"], 4),
        "predicted_label": (val_res["probs"] >= best_t).astype(int)
    })
    df_val_preds.to_csv("results/predictions/densenet121_val_predictions.csv", index=False)

    df_train_preds = pd.DataFrame({
        "filepath": train_res["paths"],
        "true_label": train_res["targets"].astype(int),
        "fracture_probability": np.round(train_res["probs"], 4),
        "predicted_label": (train_res["probs"] >= best_t).astype(int)
    })
    df_train_preds.to_csv("results/predictions/densenet121_train_predictions.csv", index=False)

    return train_res, val_res, test_res


if __name__ == "__main__":
    train_densenet()
