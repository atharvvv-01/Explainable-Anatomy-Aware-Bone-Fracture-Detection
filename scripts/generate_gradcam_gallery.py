"""
Generates and saves representative Grad-CAM explainability visualizations
for fractured and non-fractured cases in results/gradcam/.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from PIL import Image

from src.inference.pipeline import FractureInferencePipeline

def generate_sample_gradcams(num_samples: int = 4):
    os.makedirs("results/gradcam", exist_ok=True)
    pipeline = FractureInferencePipeline(
        classifier_checkpoint="models/checkpoints/best_model.pth",
        detector_path="models/detector/yolov8n_anatomy.pt",
        threshold_path="config/best_threshold.json"
    )

    cat_df = pd.read_csv("data/splits/test.csv")
    
    frac_samples = cat_df[cat_df["fracture_label"] == "Fractured"].head(num_samples // 2)
    norm_samples = cat_df[cat_df["fracture_label"] == "Non-fractured"].head(num_samples // 2)
    selected = pd.concat([frac_samples, norm_samples], ignore_index=True)

    for idx, row in selected.iterrows():
        filepath = row["filepath"]
        label = row["fracture_label"]
        fname = row["filename"]
        base_name = os.path.splitext(fname)[0]

        res = pipeline.predict(filepath)
        
        # Save composite image: [Original, Cropped, Heatmap, Overlay]
        w, h = 300, 300
        canvas = Image.new("RGB", (w * 4, h))
        canvas.paste(res["original_image"].resize((w, h)), (0, 0))
        canvas.paste(res["cropped_image"].resize((w, h)), (w, 0))
        canvas.paste(res["gradcam_heatmap"].resize((w, h)), (w * 2, 0))
        canvas.paste(res["gradcam_overlay"].resize((w, h)), (w * 3, 0))

        out_path = f"results/gradcam/sample_{idx+1}_{label}_{base_name[:15]}.png"
        canvas.save(out_path)
        print(f"Saved: {out_path} (True: {label}, Pred: {res['prediction_label']}, Prob: {res['fracture_probability']:.3f})")

if __name__ == "__main__":
    generate_sample_gradcams()
