import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import shutil
import pandas as pd
from src.evaluation.experiment_comparison import generate_experiment_comparison_table

def compile_artifacts():
    experiments = [
        {
            "name": "Full-Image EfficientNetB3 Baseline",
            "localization": "Full Image",
            "loss_type": "Standard BCE",
            "metrics_file": "results/metrics/baseline_full_image_metrics.json",
            "checkpoint": "models/checkpoints/baseline_full_image_best.pth",
            "threshold_file": "config/baseline_full_image_threshold.json"
        },
        {
            "name": "Anatomy-Guided Crop (Standard BCE)",
            "localization": "Anatomy Crop (12% Pad)",
            "loss_type": "Standard BCE",
            "metrics_file": "results/metrics/anatomy_guided_standard_metrics.json",
            "checkpoint": "models/checkpoints/anatomy_guided_standard_best.pth",
            "threshold_file": "config/anatomy_guided_standard_threshold.json"
        },
        {
            "name": "Anatomy-Guided Crop + Class-Weighted Loss",
            "localization": "Anatomy Crop (12% Pad)",
            "loss_type": "Class-Weighted BCE",
            "metrics_file": "results/metrics/anatomy_guided_class_weighted_metrics.json",
            "checkpoint": "models/checkpoints/anatomy_guided_class_weighted_best.pth",
            "threshold_file": "config/anatomy_guided_class_weighted_threshold.json"
        },
        {
            "name": "Anatomy-Guided Crop + Focal Loss (gamma=2)",
            "localization": "Anatomy Crop (12% Pad)",
            "loss_type": "Focal Loss",
            "metrics_file": "results/metrics/anatomy_guided_focal_loss_metrics.json",
            "checkpoint": "models/checkpoints/anatomy_guided_focal_loss_best.pth",
            "threshold_file": "config/anatomy_guided_focal_loss_threshold.json"
        },
        {
            "name": "Anatomy-Guided Crop + Weighted Sampler",
            "localization": "Anatomy Crop (12% Pad)",
            "loss_type": "Standard BCE + Sampler",
            "metrics_file": "results/metrics/anatomy_guided_sampler_metrics.json",
            "checkpoint": "models/checkpoints/anatomy_guided_sampler_best.pth",
            "threshold_file": "config/anatomy_guided_sampler_threshold.json"
        }
    ]

    records = []
    best_f1 = -1.0
    best_exp = None

    for exp in experiments:
        if os.path.exists(exp["metrics_file"]):
            with open(exp["metrics_file"]) as f:
                m = json.load(f)
            records.append({
                "name": exp["name"],
                "localization": exp["localization"],
                "loss_type": exp["loss_type"],
                "metrics": m
            })
            if m.get("fracture_f1", 0) > best_f1:
                best_f1 = m["fracture_f1"]
                best_exp = exp

    # Generate comparison table & plots
    df, md = generate_experiment_comparison_table(records)
    print("--- BENCHMARK COMPARISON TABLE ---")
    print(md)

    # Set up best model checkpoint
    if best_exp:
        best_ckpt = best_exp["checkpoint"]
        target_ckpt = "models/checkpoints/best_model.pth"
        shutil.copy2(best_ckpt, target_ckpt)
        print(f"\nBest Model Checkpoint copied to: {target_ckpt} ({best_exp['name']}, F1={best_f1})")

        best_thresh = best_exp["threshold_file"]
        target_thresh = "config/best_threshold.json"
        shutil.copy2(best_thresh, target_thresh)
        print(f"Optimal Threshold copied to: {target_thresh}")

        # Also save test evaluation report for app
        shutil.copy2(best_exp["metrics_file"], "results/metrics/test_evaluation_report.json")
        print("Saved test_evaluation_report.json for Web Dashboard")

if __name__ == "__main__":
    compile_artifacts()
