"""
Comprehensive Evaluation Metrics for Bone Fracture Diagnosis.
Calculates Accuracy, Sensitivity/Recall, Specificity, Precision,
Macro-F1, Weighted-F1, ROC-AUC, and PR-AUC.
"""

from typing import Dict, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes all standard diagnostic and ranking metrics on test set.
    """
    y_pred = (y_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)

    # Per-class precision, recall, f1
    p_per_class, r_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    # Macro & Weighted F1
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # Specificity
    specificity = tn / max(tn + fp, 1)

    # ROC-AUC & PR-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_probs)
    except Exception:
        pr_auc = 0.0

    return {
        "classification_threshold": round(float(threshold), 4),
        "accuracy": round(float(acc), 4),
        "fracture_recall_sensitivity": round(float(r_per_class[1]), 4),
        "fracture_precision": round(float(p_per_class[1]), 4),
        "fracture_f1": round(float(f1_per_class[1]), 4),
        "non_fracture_recall_specificity": round(float(specificity), 4),
        "non_fracture_precision": round(float(p_per_class[0]), 4),
        "non_fracture_f1": round(float(f1_per_class[0]), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    roc_save_path: str = "results/plots/roc_curve.png",
    pr_save_path: str = "results/plots/pr_curve.png"
) -> None:
    """Plots and saves standard ROC and Precision-Recall diagnostic curves."""
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#1f77b4', linewidth=2.5, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label="Random Guess")
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(roc_save_path, dpi=300)
    plt.close()

    # 2. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)

    plt.figure(figsize=(7, 6))
    plt.plot(recall_vals, precision_vals, color='#2ca02c', linewidth=2.5, label=f"PR Curve (AUC = {pr_auc:.3f})")
    baseline = np.sum(y_true == 1) / len(y_true)
    plt.plot([0, 1], [baseline, baseline], color='gray', linestyle='--', linewidth=1.5, label=f"Prevalence ({baseline:.2f})")
    plt.title("Precision-Recall Curve (Minority Fracture Detection)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Fracture Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Fracture Precision", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="lower left", fontsize=11)
    plt.tight_layout()
    plt.savefig(pr_save_path, dpi=300)
    plt.close()

    print(f"ROC curve saved to: {roc_save_path}")
    print(f"PR curve saved to: {pr_save_path}")
