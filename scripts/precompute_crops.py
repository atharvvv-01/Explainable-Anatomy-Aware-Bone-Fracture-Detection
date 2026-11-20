"""
Precomputes Stage 1 YOLO anatomical detections for all images in the dataset.
Saves detection boxes and class predictions to data/processed/yolo_detections.json.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch
from PIL import Image

from src.localization.detector import AnatomicalDetector

def precompute_all_detections(
    catalog_csv: str = "data/splits/dataset_catalog.csv",
    output_json: str = "data/processed/yolo_detections.json",
    conf_threshold: float = 0.30
):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    df = pd.read_csv(catalog_csv)
    detector = AnatomicalDetector(
        model_path="models/detector/yolov8n_anatomy.pt",
        conf_threshold=conf_threshold,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    detections = {}
    fallback_count = 0

    print(f"Precomputing detections for {len(df)} images...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="YOLO Detecting"):
        filepath = row["filepath"]
        try:
            res = detector.detect(filepath)
            detections[filepath] = {
                "has_detection": res["has_detection"],
                "is_fallback": res["is_fallback"],
                "box": res["best_box"],
                "anatomy_name": res["anatomy_name"],
                "confidence": res["confidence"]
            }
            if res["is_fallback"]:
                fallback_count += 1
        except Exception as e:
            detections[filepath] = {
                "has_detection": False,
                "is_fallback": True,
                "box": None,
                "anatomy_name": "Full-Image Fallback",
                "confidence": 0.0
            }
            fallback_count += 1

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(detections, f, indent=2)

    print(f"\nPrecomputation finished. Total: {len(detections)}, Fallbacks: {fallback_count} ({fallback_count/len(detections)*100:.1f}%)")
    print(f"Saved to: {output_json}")

if __name__ == "__main__":
    precompute_all_detections()
