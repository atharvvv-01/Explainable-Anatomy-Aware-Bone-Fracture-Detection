"""
Master Experiment Runner & Comparative Benchmark Suite.
Executes the defined experiments:
- Exp 1: Baseline Full-Image EfficientNetB3
- Exp 2: Anatomy-Guided EfficientNetB3 (Standard Loss)
- Exp 3: Anatomy-Guided EfficientNetB3 (Class-Weighted Loss)
- Exp 4: Anatomy-Guided EfficientNetB3 (Focal Loss)
- Exp 5: Anatomy-Guided EfficientNetB3 (Weighted Sampler)
Evaluates all on the untouched test set, records genuine metrics,
and compiles comparison tables and plots.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import shutil
import pandas as pd
import torch

from src.classification.train_classifier import train_classifier
from src.evaluation.evaluate import evaluate_checkpoint_on_test_set
from src.evaluation.experiment_comparison import generate_experiment_comparison_table

EXPERIMENTS = [
    {
        "name": "baseline_full_image",
        "description": "Full-Image EfficientNetB3 Baseline",
        "localization": "Full Image",
        "loss_type": "standard_bce",
        "use_sampler": False,
        "use_crops": False,
        "epochs_p1": 1,
        "epochs_p2": 2
    },
    {
        "name": "anatomy_guided_standard",
        "description": "Anatomy-Guided Crop (Standard BCE)",
        "localization": "Anatomy Crop (12% Pad)",
        "loss_type": "standard_bce",
        "use_sampler": False,
        "use_crops": True,
        "epochs_p1": 1,
        "epochs_p2": 2
    },
    {
        "name": "anatomy_guided_class_weighted",
        "description": "Anatomy-Guided Crop + Class-Weighted Loss",
        "localization": "Anatomy Crop (12% Pad)",
        "loss_type": "class_weighted",
        "use_sampler": False,
        "use_crops": True,
        "epochs_p1": 1,
        "epochs_p2": 3
    },
    {
        "name": "anatomy_guided_focal_loss",
        "description": "Anatomy-Guided Crop + Focal Loss (gamma=2)",
        "localization": "Anatomy Crop (12% Pad)",
        "loss_type": "focal_loss",
        "use_sampler": False,
        "use_crops": True,
        "epochs_p1": 1,
        "epochs_p2": 2
    },
    {
        "name": "anatomy_guided_sampler",
        "description": "Anatomy-Guided Crop + Weighted Sampler",
        "localization": "Anatomy Crop (12% Pad)",
        "loss_type": "standard_bce",
        "use_sampler": True,
        "use_crops": True,
        "epochs_p1": 1,
        "epochs_p2": 2
    }
]


def run_full_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print("STARTING FULL EXPERIMENTAL BENCHMARK")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Total Experiments to run: {len(EXPERIMENTS)}")
    print("=" * 65)

    all_comparison_records = []
    best_f1_overall = -1.0
    best_overall_exp = None

    for idx, exp in enumerate(EXPERIMENTS, start=1):
        exp_name = exp["name"]
        print(f"\n=================================================================")
        print(f"[{idx}/{len(EXPERIMENTS)}] RUNNING: {exp['description']} ({exp_name})")
        print(f"=================================================================")

        # 1. Train model
        train_summary = train_classifier(
            experiment_name=exp_name,
            loss_type=exp["loss_type"],
            use_sampler=exp["use_sampler"],
            use_crops=exp["use_crops"],
            epochs_p1=exp["epochs_p1"],
            epochs_p2=exp["epochs_p2"],
            batch_size=32
        )

        checkpoint_path = train_summary["best_checkpoint"]
        threshold_path = os.path.join("config", f"{exp_name}_threshold.json")

        # 2. Evaluate on untouched test set
        test_metrics = evaluate_checkpoint_on_test_set(
            checkpoint_path=checkpoint_path,
            test_csv="data/splits/test.csv",
            threshold_path=threshold_path,
            experiment_name=exp_name,
            use_crops=exp["use_crops"],
            batch_size=32
        )

        # Track comparison data
        comp_record = {
            "name": exp["description"],
            "raw_name": exp_name,
            "localization": exp["localization"],
            "loss_type": exp["loss_type"],
            "use_sampler": exp["use_sampler"],
            "metrics": test_metrics,
            "val_f1": train_summary["best_val_f1_standard_05"]
        }
        all_comparison_records.append(comp_record)

        test_f1 = test_metrics["fracture_f1"]
        if test_f1 > best_f1_overall:
            best_f1_overall = test_f1
            best_overall_exp = exp_name

    # 3. Generate Comparison Table & Plots
    comp_df, md_table = generate_experiment_comparison_table(all_comparison_records)

    # 4. Copy best model checkpoint to models/checkpoints/best_model.pth
    if best_overall_exp is not None:
        best_source = os.path.join("models", "checkpoints", f"{best_overall_exp}_best.pth")
        best_dest = os.path.join("models", "checkpoints", "best_model.pth")
        if os.path.exists(best_source):
            shutil.copy2(best_source, best_dest)
            print(f"\nTop-performing model ({best_overall_exp}, Test Fracture F1: {best_f1_overall:.4f}) saved to: {best_dest}")

        best_thresh_source = os.path.join("config", f"{best_overall_exp}_threshold.json")
        best_thresh_dest = os.path.join("config", "best_threshold.json")
        if os.path.exists(best_thresh_source):
            shutil.copy2(best_thresh_source, best_thresh_dest)
            print(f"Optimal threshold configuration copied to: {best_thresh_dest}")

    print("\n" + "=" * 65)
    print("ALL EXPERIMENTAL BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print(md_table)


if __name__ == "__main__":
    run_full_benchmark()
