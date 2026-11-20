"""
Experiment Comparison and Cross-Model Benchmark Suite.
Compares Full-Image Baseline vs Anatomy-Guided vs Imbalance Handling methods
on the exact same held-out test split.
"""

import os
import json
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def generate_experiment_comparison_table(
    experiment_metrics: List[Dict[str, Any]],
    output_dir: str = "results/metrics",
    plot_dir: str = "results/plots"
) -> Tuple[pd.DataFrame, str]:
    """
    Constructs comparison table and visualization across all experimental runs.
    """
    rows = []
    for exp in experiment_metrics:
        rows.append({
            "Experiment": exp.get("name", "Unknown"),
            "Localization": exp.get("localization", "Full Image"),
            "Loss Function": exp.get("loss_type", "Standard BCE"),
            "Accuracy (%)": round(exp["metrics"].get("accuracy", 0) * 100, 2),
            "Fracture Recall (%)": round(exp["metrics"].get("fracture_recall_sensitivity", 0) * 100, 2),
            "Fracture Precision (%)": round(exp["metrics"].get("fracture_precision", 0) * 100, 2),
            "Fracture F1": round(exp["metrics"].get("fracture_f1", 0), 4),
            "Macro F1": round(exp["metrics"].get("macro_f1", 0), 4),
            "ROC-AUC": round(exp["metrics"].get("roc_auc", 0), 4),
            "PR-AUC": round(exp["metrics"].get("pr_auc", 0), 4),
            "Optimal Threshold": exp["metrics"].get("classification_threshold", 0.50)
        })

    df = pd.DataFrame(rows)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "experiment_comparison.csv")
    df.to_csv(csv_path, index=False)

    # Markdown table (pure python formatting without external dependency)
    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = []
    for _, r in df.iterrows():
        data_lines.append("| " + " | ".join(str(val) for val in r.values) + " |")
    md_table = "\n".join([header_line, sep_line] + data_lines)
    md_path = os.path.join(output_dir, "experiment_comparison.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Model Experiment Comparison on Test Partition\n\n")
        f.write(md_table)
        f.write("\n")

    # Comparative Bar Plot
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "experiment_comparison.png")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.18

    ax.bar(x - 1.5 * width, df["Accuracy (%)"], width, label="Accuracy (%)", color="#1f77b4")
    ax.bar(x - 0.5 * width, df["Fracture Recall (%)"], width, label="Fracture Recall (%)", color="#d62728")
    ax.bar(x + 0.5 * width, df["Fracture Precision (%)"], width, label="Fracture Precision (%)", color="#2ca02c")
    ax.bar(x + 1.5 * width, df["ROC-AUC"] * 100, width, label="ROC-AUC (×100)", color="#9467bd")

    ax.set_ylabel("Percentage / Normalized Score", fontsize=11)
    ax.set_title("Experimental Comparison: Full-Image vs Anatomy-Guided vs Imbalance Handling", fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Experiment"], rotation=15, ha='right', fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Comparison table saved to: {csv_path}")
    print(f"Comparison plot saved to: {plot_path}")

    return df, md_table
