"""
Confusion Matrix Calculation and Heatmap Visualization.
Plots normalized and absolute confusion matrices for fracture classification.
"""

import os
from typing import Tuple, List, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ["Non-fractured", "Fractured"],
    save_path: str = "results/plots/confusion_matrix.png",
    title: str = "Bone Fracture Confusion Matrix"
) -> np.ndarray:
    """
    Computes and plots formatted confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=axes[0],
                annot_kws={"size": 13, "weight": "bold"})
    axes[0].set_title(f"{title} (Counts)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("True Diagnosis", fontsize=11)
    axes[0].set_xlabel("Predicted Diagnosis", fontsize=11)

    # Normalized proportions
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=axes[1],
                annot_kws={"size": 13, "weight": "bold"})
    axes[1].set_title(f"{title} (Normalized)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("True Diagnosis", fontsize=11)
    axes[1].set_xlabel("Predicted Diagnosis", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Confusion matrix plot saved to: {save_path}")
    return cm
