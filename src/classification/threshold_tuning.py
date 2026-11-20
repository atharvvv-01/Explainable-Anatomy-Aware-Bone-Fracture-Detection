"""
Optimal Classification Threshold Tuning on Validation Set.
Prevents false negatives on medical X-rays by systematically sweeping
thresholds and maximizing Fracture F1 / Clinical Recall.
"""

import os
import json
from typing import Dict, Any, Tuple, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score


def tune_classification_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    target_metric: str = "f1",
    min_recall: float = 0.80,
    save_path: str = "config/best_threshold.json",
    plot_path: str = "results/plots/threshold_tuning_curve.png"
) -> Tuple[float, Dict[str, Any]]:
    """
    Sweeps thresholds on validation probabilities to select the clinically optimal threshold.
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_probs: Predicted probabilities of fracture [0.0, 1.0].
        target_metric: Metric to optimize ('f1', 'balanced_accuracy', 'recall_constrained').
        min_recall: Minimum required clinical recall when using 'recall_constrained'.
        save_path: Path to save threshold configuration JSON.
        plot_path: Path to save diagnostic tuning plot.
    """
    thresholds = np.linspace(0.10, 0.90, 81)
    precisions: List[float] = []
    recalls: List[float] = []
    f1_scores: List[float] = []
    balanced_accs: List[float] = []
    fnrs: List[float] = []  # False negative rates

    accs: List[float] = []

    best_threshold = 0.50
    best_score = -1.0
    best_metrics = {}

    for thresh in thresholds:
        preds = (y_probs >= thresh).astype(int)
        
        # Binary classification metrics for positive (Fracture = 1) class
        p, r, f1, _ = precision_recall_fscore_support(y_true, preds, pos_label=1, average='binary', zero_division=0)
        bacc = balanced_accuracy_score(y_true, preds)
        acc = accuracy_score(y_true, preds)
        
        # Calculate false negative rate
        pos_indices = (y_true == 1)
        if pos_indices.sum() > 0:
            fnr = (preds[pos_indices] == 0).sum() / pos_indices.sum()
        else:
            fnr = 0.0

        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)
        balanced_accs.append(bacc)
        accs.append(acc)
        fnrs.append(fnr)

        # Optimization criterion
        if target_metric == "f1":
            score = f1
        elif target_metric == "balanced_accuracy":
            score = bacc
        elif target_metric == "accuracy_constrained":
            # Maximize accuracy subject to minimum acceptable fracture recall (e.g. >= 0.60)
            score = (acc + 0.3 * f1) if r >= min_recall else (acc * 0.7)
        elif target_metric == "youden_j":
            # Youden's J Index = Sensitivity + Specificity - 1 = 2 * BalancedAcc - 1
            score = 2.0 * bacc - 1.0
        elif target_metric == "recall_constrained":
            score = f1 if r >= min_recall else -1.0
        else:
            score = f1

        if score > best_score:
            best_score = score
            best_threshold = thresh
            best_metrics = {
                "threshold": round(float(thresh), 4),
                "accuracy": round(float(acc), 4),
                "fracture_f1": round(float(f1), 4),
                "fracture_recall": round(float(r), 4),
                "fracture_precision": round(float(p), 4),
                "balanced_accuracy": round(float(bacc), 4),
                "false_negative_rate": round(float(fnr), 4)
            }

    # Plot Threshold Tuning Curves
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(thresholds, f1_scores, label="Fracture F1-Score", color='#2ca02c', linewidth=2)
    plt.plot(thresholds, recalls, label="Fracture Recall (Sensitivity)", color='#d62728', linewidth=2, linestyle='--')
    plt.plot(thresholds, precisions, label="Fracture Precision", color='#1f77b4', linewidth=2, linestyle=':')
    plt.plot(thresholds, fnrs, label="False Negative Rate (FNR)", color='#9467bd', linewidth=1.5, linestyle='-.')

    plt.axvline(x=best_threshold, color='black', linestyle='-', alpha=0.8,
                label=f"Selected Threshold ({best_threshold:.2f})")
    
    plt.title("Validation Set Threshold Optimization", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Classification Decision Threshold", fontsize=11)
    plt.ylabel("Metric Score", fontsize=11)
    plt.ylim(-0.02, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Save to config JSON
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(best_metrics, f, indent=2)

    print(f"Optimal threshold found: {best_threshold:.4f} (F1: {best_metrics['fracture_f1']}, Recall: {best_metrics['fracture_recall']})")
    print(f"Threshold metadata saved to: {save_path}")
    print(f"Tuning plot saved to: {plot_path}")

    return best_threshold, best_metrics
