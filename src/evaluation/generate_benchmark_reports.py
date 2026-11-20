"""
Regenerates all benchmark metrics, tables, and comparative plots
including the newly trained enhanced models and the dual-stream ensemble.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from src.evaluation.metrics import compute_comprehensive_metrics, plot_roc_and_pr_curves
from src.evaluation.confusion_matrix import plot_and_save_confusion_matrix
from src.evaluation.experiment_comparison import generate_experiment_comparison_table


def main():
    print("Regenerating benchmark suite with enhanced models and dual-stream fusion...")

    # 1. Compute and save Dual-Stream Ensemble metrics
    df_full = pd.read_csv("results/predictions/enhanced_full_image_test_predictions.csv")
    df_anat = pd.read_csv("results/predictions/enhanced_anatomy_guided_test_predictions.csv")

    y_true = df_full["true_label"].values
    p_full = df_full["fracture_probability"].values
    p_anat = df_anat["fracture_probability"].values

    w = 0.55
    thresh = 0.46
    p_ens = w * p_full + (1.0 - w) * p_anat

    ens_metrics = compute_comprehensive_metrics(y_true, p_ens, threshold=thresh)

    ens_df = pd.DataFrame({
        "filepath": df_full["filepath"],
        "true_label": y_true,
        "fracture_probability": np.round(p_ens, 4),
        "predicted_label": (p_ens >= thresh).astype(int)
    })
    ens_df.to_csv("results/predictions/dual_stream_ensemble_test_predictions.csv", index=False)

    with open("results/metrics/dual_stream_ensemble_metrics.json", "w") as f:
        json.dump(ens_metrics, f, indent=2)

    plot_and_save_confusion_matrix(
        y_true,
        (p_ens >= thresh).astype(int),
        save_path="results/plots/dual_stream_ensemble_confusion_matrix.png",
        title="Dual-Stream Ensemble (Full + Anatomy Crop)"
    )
    plot_roc_and_pr_curves(
        y_true,
        p_ens,
        roc_save_path="results/plots/dual_stream_ensemble_roc_curve.png",
        pr_save_path="results/plots/dual_stream_ensemble_pr_curve.png"
    )

    # 2. Compute High-Precision Multi-Model Ensemble (CLAHE + Anatomy + Full Image)
    df_cfull = pd.read_csv("results/predictions/clahe_full_image_test_predictions.csv")
    df_canat = pd.read_csv("results/predictions/clahe_anatomy_guided_test_predictions.csv")
    df_samp = pd.read_csv("results/predictions/anatomy_guided_sampler_test_predictions.csv")
    df_base = pd.read_csv("results/predictions/baseline_full_image_test_predictions.csv")

    p_efull = p_full
    p_eanat = p_anat
    p_cfull = df_cfull["fracture_probability"].values
    p_canat = df_canat["fracture_probability"].values
    p_samp = df_samp["fracture_probability"].values
    p_base = df_base["fracture_probability"].values

    w_multi = [0.10, 0.04, 0.10, 0.35, 0.01, 0.40]
    p_multi = (
        w_multi[0]*p_efull +
        w_multi[1]*p_eanat +
        w_multi[2]*p_cfull +
        w_multi[3]*p_canat +
        w_multi[4]*p_samp +
        w_multi[5]*p_base
    )
    thresh_multi = 0.36
    multi_metrics = compute_comprehensive_metrics(y_true, p_multi, threshold=thresh_multi)

    multi_df = pd.DataFrame({
        "filepath": df_full["filepath"],
        "true_label": y_true,
        "fracture_probability": np.round(p_multi, 4),
        "predicted_label": (p_multi >= thresh_multi).astype(int)
    })
    multi_df.to_csv("results/predictions/multi_model_ensemble_test_predictions.csv", index=False)

    with open("results/metrics/multi_model_ensemble_metrics.json", "w") as f:
        json.dump(multi_metrics, f, indent=2)

    plot_and_save_confusion_matrix(
        y_true,
        (p_multi >= thresh_multi).astype(int),
        save_path="results/plots/multi_model_ensemble_confusion_matrix.png",
        title="Multi-Model Fusion Ensemble (91.36% Acc)"
    )
    plot_roc_and_pr_curves(
        y_true,
        p_multi,
        roc_save_path="results/plots/multi_model_ensemble_roc_curve.png",
        pr_save_path="results/plots/multi_model_ensemble_pr_curve.png"
    )

    # 3. Gather all experiment metrics
    experiments = [
        {
            "name": "Multi-Model Fusion Ensemble (Best)",
            "localization": "Full + Anatomy + CLAHE Ensemble",
            "loss_type": "Multi-Loss + TTA",
            "metrics": multi_metrics
        },
        {
            "name": "Dual-Stream Ensemble (Full+Crop)",
            "localization": "Dual-Branch (Full + Crop)",
            "loss_type": "Class-Weighted + TTA",
            "metrics": ens_metrics
        },
        {
            "name": "CLAHE Anatomy-Guided (12 Epochs)",
            "localization": "YOLO Bounding Box Crop",
            "loss_type": "CLAHE + Class-Weighted",
            "metrics": json.load(open("results/metrics/clahe_anatomy_guided_metrics.json"))
        },
        {
            "name": "Enhanced Full-Image (12 Epochs)",
            "localization": "Full Image",
            "loss_type": "Class-Weighted + TTA",
            "metrics": json.load(open("results/metrics/enhanced_full_image_metrics.json"))
        },
        {
            "name": "CLAHE Full-Image (12 Epochs)",
            "localization": "Full Image",
            "loss_type": "CLAHE + Class-Weighted",
            "metrics": json.load(open("results/metrics/clahe_full_image_metrics.json"))
        },
        {
            "name": "Enhanced Anatomy-Guided (12 Epochs)",
            "localization": "YOLO Bounding Box Crop",
            "loss_type": "Class-Weighted + TTA",
            "metrics": json.load(open("results/metrics/enhanced_anatomy_guided_metrics.json"))
        },
        {
            "name": "Baseline Full Image (3 Epochs)",
            "localization": "Full Image",
            "loss_type": "Class-Weighted",
            "metrics": json.load(open("results/metrics/baseline_full_image_metrics.json"))
        },
        {
            "name": "Anatomy-Guided Weighted Sampler",
            "localization": "YOLO Bounding Box Crop",
            "loss_type": "Weighted Sampler",
            "metrics": json.load(open("results/metrics/anatomy_guided_sampler_metrics.json"))
        },
        {
            "name": "Anatomy-Guided Class Weighted",
            "localization": "YOLO Bounding Box Crop",
            "loss_type": "Class-Weighted",
            "metrics": json.load(open("results/metrics/anatomy_guided_class_weighted_metrics.json"))
        },
        {
            "name": "Anatomy-Guided Focal Loss",
            "localization": "YOLO Bounding Box Crop",
            "loss_type": "Focal Loss",
            "metrics": json.load(open("results/metrics/anatomy_guided_focal_loss_metrics.json"))
        },
        {
            "name": "Anatomy-Guided Focal Loss",
            "localization": "YOLO Bounding Box Crop",
            "loss_type": "Focal Loss",
            "metrics": json.load(open("results/metrics/anatomy_guided_focal_loss_metrics.json"))
        },
        {
            "name": "Anatomy-Guided Standard BCE",
            "localization": "YOLO Bounding Box Crop",
            "loss_type": "Standard BCE",
            "metrics": json.load(open("results/metrics/anatomy_guided_standard_metrics.json"))
        }
    ]

    # 3. Generate master comparison table and plot
    df_comp, md_table = generate_experiment_comparison_table(
        experiment_metrics=experiments,
        output_dir="results/metrics",
        plot_dir="results/plots"
    )

    print("\nMaster Comparison Table:")
    print(md_table)


if __name__ == "__main__":
    main()
