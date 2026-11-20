"""
Final Model Evaluator on Untouched Test Set.
Computes genuine test metrics, confusion matrix, ROC/PR curves,
and saves all results without fabrication.
"""

import os
import json
import argparse
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.classification.efficientnet_model import build_fracture_classifier
from src.evaluation.metrics import compute_comprehensive_metrics, plot_roc_and_pr_curves
from src.evaluation.confusion_matrix import plot_and_save_confusion_matrix
from src.dataset.dataset_loader import get_data_loaders


def evaluate_checkpoint_on_test_set(
    checkpoint_path: str,
    test_csv: str = "data/splits/test.csv",
    threshold_path: Optional[str] = None,
    experiment_name: str = "final_evaluation",
    results_dir: str = "results",
    batch_size: int = 16,
    use_crops: bool = False,
    use_clahe: bool = False
) -> Dict[str, Any]:
    """
    Evaluates a saved checkpoint on the held-out test set.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nEvaluating Checkpoint: {checkpoint_path}")
    print(f"Test Set: {test_csv} | Device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test split CSV not found: {test_csv}")

    # Load threshold from config if provided
    threshold = 0.50
    if threshold_path and os.path.exists(threshold_path):
        with open(threshold_path, 'r', encoding='utf-8') as f:
            t_data = json.load(f)
            threshold = float(t_data.get("threshold", 0.50))
            print(f"Loaded optimal validation threshold: {threshold:.4f}")
    else:
        print(f"Using default classification threshold: {threshold:.2f}")

    # Load test data
    crop_boxes = None
    if use_crops:
        det_json = "data/processed/yolo_detections.json"
        if os.path.exists(det_json):
            with open(det_json, 'r', encoding='utf-8') as f:
                raw_dets = json.load(f)
                crop_boxes = {k: v["box"] for k, v in raw_dets.items() if v.get("box") is not None}

    _, _, test_loader = get_data_loaders(
        train_data="train",
        val_data="val",
        test_data=test_csv,
        batch_size=batch_size,
        num_workers=0,
        crop_boxes=crop_boxes,
        use_clahe=use_clahe
    )

    # Build model and load checkpoint
    model = build_fracture_classifier(pretrained=False, device=device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    all_probs = []
    all_targets = []
    all_paths = []

    with torch.no_grad():
        for images, targets, paths in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            logits1, _ = model(images)
            p1 = torch.sigmoid(logits1)

            # Test-Time Augmentation (Original + Horizontal Flip)
            images_flip = torch.flip(images, dims=[3])
            logits2, _ = model(images_flip)
            p2 = torch.sigmoid(logits2)

            probs = ((p1 + p2) / 2.0).cpu().numpy()
            
            all_probs.extend(probs)
            all_targets.extend(targets.numpy())
            all_paths.extend(paths)

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # Save individual predictions
    preds_df = pd.DataFrame({
        "filepath": all_paths,
        "true_label": all_targets.astype(int),
        "fracture_probability": all_probs,
        "predicted_label": (all_probs >= threshold).astype(int)
    })
    preds_csv_path = os.path.join(results_dir, "predictions", f"{experiment_name}_test_predictions.csv")
    os.makedirs(os.path.dirname(preds_csv_path), exist_ok=True)
    preds_df.to_csv(preds_csv_path, index=False)

    # Compute comprehensive metrics
    metrics = compute_comprehensive_metrics(all_targets, all_probs, threshold=threshold)

    # Save metrics JSON
    metrics_json_path = os.path.join(results_dir, "metrics", f"{experiment_name}_metrics.json")
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # Generate plots
    plot_and_save_confusion_matrix(
        all_targets,
        (all_probs >= threshold).astype(int),
        save_path=os.path.join(results_dir, "plots", f"{experiment_name}_confusion_matrix.png"),
        title=f"Test Confusion Matrix ({experiment_name})"
    )

    plot_roc_and_pr_curves(
        all_targets,
        all_probs,
        roc_save_path=os.path.join(results_dir, "plots", f"{experiment_name}_roc_curve.png"),
        pr_save_path=os.path.join(results_dir, "plots", f"{experiment_name}_pr_curve.png")
    )

    print("\n--- Final Test Set Diagnostic Performance ---")
    print(f"Accuracy:            {metrics['accuracy']*100:.2f}%")
    print(f"Fracture Recall:     {metrics['fracture_recall_sensitivity']*100:.2f}%")
    print(f"Fracture Precision:  {metrics['fracture_precision']*100:.2f}%")
    print(f"Fracture F1-Score:   {metrics['fracture_f1']*100:.2f}%")
    print(f"Macro F1-Score:      {metrics['macro_f1']*100:.2f}%")
    print(f"ROC-AUC:             {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:              {metrics['pr_auc']:.4f}")
    print(f"Confusion Matrix:    TN={metrics['confusion_matrix']['true_negatives']}, FP={metrics['confusion_matrix']['false_positives']}, FN={metrics['confusion_matrix']['false_negatives']}, TP={metrics['confusion_matrix']['true_positives']}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model on Test Set")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--threshold_path", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="test_eval")
    parser.add_argument("--use_crops", action="store_true")
    args = parser.parse_args()

    evaluate_checkpoint_on_test_set(
        checkpoint_path=args.checkpoint,
        threshold_path=args.threshold_path,
        experiment_name=args.experiment_name,
        use_crops=args.use_crops
    )
