"""
Interactive Radiograph Diagnostic Tool.
Allows the user to choose an X-ray image from their PC via a native Windows file dialog
or command-line argument, then displays the complete two-stage anatomy-aware diagnosis:
- Stage 1: YOLO Anatomical Region Localization (Arm, Leg, Hand, Shoulder, Spine)
- Anatomical Padded Crop (300x300, 12% padding)
- Stage 2: EfficientNetB3 Fracture Probability & Prediction
- Grad-CAM Explainability Overlay
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference.pipeline import FractureInferencePipeline


def choose_image_file_dialog() -> str:
    """Opens a native OS file dialog for the user to select an image from their PC."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()  # Hide root Tk window
        root.attributes('-topmost', True)  # Bring file dialog to front

        print("\n[INFO] Opening file dialog... Please select an X-ray image from your PC.")
        file_path = filedialog.askopenfilename(
            title="Select an X-ray Radiograph from your PC",
            initialdir=os.path.abspath("Dataset/Dataset"),
            filetypes=[
                ("X-ray Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"),
                ("All Files", "*.*")
            ]
        )
        root.destroy()
        return file_path
    except Exception as e:
        print(f"[WARNING] Could not open GUI file dialog: {e}")
        return ""


def plot_and_save_diagnosis(result: dict, image_source: str, save_path: str = "results/predictions/latest_diagnosis.png", show: bool = True) -> str:
    """Renders and saves a 5-stage visual diagnosis highlighting localization, cropping, explainability, and pathology focus."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    is_fx = result.get("is_fractured", False)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5.5), facecolor="#0b111e")
    
    # 1. Original Radiograph
    axes[0].imshow(result["original_image"])
    axes[0].set_title("1. Original X-Ray", color="white", fontsize=12, fontweight="bold", pad=10)
    axes[0].axis("off")

    # 2. Stage 1: Bone ROI Localization (Generic Bone Structure, no anatomical body-part classifier)
    axes[1].imshow(result["annotated_detection_image"])
    status_str = f"Bone ROI ({result['detector_confidence']*100:.1f}%)"
    if result["is_fallback"]:
        status_str = "Full-Image Fallback"
    axes[1].set_title(f"2. Stage 1: Bone ROI\n{status_str}", color="#38bdf8", fontsize=11, fontweight="bold", pad=8)
    axes[1].axis("off")

    # 3. Stage 2: Padded Bone Crop
    axes[2].imshow(result["cropped_image"])
    axes[2].set_title("3. Padded Bone Crop\n(12% Margin, 300x300)", color="#a78bfa", fontsize=11, fontweight="bold", pad=8)
    axes[2].axis("off")

    # 4. Stage 3/4: Grad-CAM Explainability Overlay
    axes[3].imshow(result["gradcam_overlay"])
    axes[3].set_title("4. Grad-CAM Heatmap\n(Cortical Saliency)", color="#fb923c", fontsize=11, fontweight="bold", pad=8)
    axes[3].axis("off")

    # 5. Stage 5: Pathology Focus ROI (Brightest Grad-CAM Region Box strictly inside Step 2 Box)
    axes[4].imshow(result["focused_image"])
    crop_box_str = str(result.get("crop_focus_box", result.get("focus_box", "(N/A)")))
    peak_act = result.get("peak_focus_intensity", 0.0) * 100
    axes[4].set_title(f"5. Pathology Focus ROI\nInside Step 2 Box: {crop_box_str} ({peak_act:.0f}%)", color="#ff1744" if is_fx else "#22c55e", fontsize=10, fontweight="bold", pad=8)
    axes[4].axis("off")

    # Main Diagnostic Title Banner
    is_fx = result["is_fractured"]
    pred_label = result["prediction_label"]
    fx_prob = result["fracture_probability"] * 100
    model_conf = result["model_confidence"] * 100
    thresh = result["decision_threshold"]

    banner_color = "#ef4444" if is_fx else "#22c55e"
    banner_icon = "[!]" if is_fx else "[OK]"

    main_title = (
        f"{banner_icon} DIAGNOSIS: {pred_label}  |  "
        f"Fracture Probability: {fx_prob:.1f}%  |  "
        f"Confidence: {model_conf:.1f}%  |  "
        f"Threshold: {thresh:.2f}"
    )

    if result["is_fallback"]:
        main_title += "  [DETECTOR FALLBACK ACTIVE]"

    fig.suptitle(main_title, fontsize=15, fontweight="bold", color=banner_color, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()
    return os.path.abspath(save_path)


def main():
    parser = argparse.ArgumentParser(description="Anatomy-Guided Bone Fracture Diagnostic Tool")
    parser.add_argument("--image", type=str, default="", help="Path to X-ray image (if omitted, file dialog opens)")
    parser.add_argument("--no-show", action="store_true", help="Do not display interactive matplotlib window")
    args = parser.parse_args()

    image_path = args.image

    # If no image path supplied via command line, open Windows file dialog
    if not image_path:
        image_path = choose_image_file_dialog()

    # If user cancelled file dialog, fallback to asking or picking sample
    if not image_path or not os.path.exists(image_path):
        print("\nNo image was selected from the file picker.")
        print("Using a default sample from Dataset/Dataset/Fracture/...")
        sample_dir = "Dataset/Dataset/Fracture"
        if os.path.exists(sample_dir):
            files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                image_path = files[0]
                print(f"Selected sample: {image_path}")
            else:
                print("No sample images found. Exiting.")
                return
        else:
            print("Dataset directory not found. Exiting.")
            return

    print(f"\n========================================================")
    print(f" PROCESSING X-RAY RADIOGRAPH")
    print(f" Image: {image_path}")
    print(f"========================================================")

    pipeline = FractureInferencePipeline()
    result = pipeline.predict(image_path)

    print("\n------------------- DIAGNOSTIC RESULTS -------------------")
    print(f"Prediction:            {result['prediction_label']}")
    print(f"Fracture Probability:  {result['fracture_probability']*100:.2f}%")
    print(f"Decision Threshold:    {result['decision_threshold']:.2f}")
    print(f"Model Confidence:      {result['model_confidence']*100:.2f}%")
    print(f"Localized ROI:         {result.get('region_name', 'Bone Structure')} (Detector Conf: {result['detector_confidence']*100:.2f}%)")
    print(f"Step 2 Bone Box:       {result['bounding_box']}")
    print(f"Step 5 Focus Box:      {result['focus_box']} (strictly inside Step 2 Box, Peak Focus: {result['peak_focus_intensity']*100:.1f}%)")
    print(f"Step 5 Crop Box:       {result.get('crop_focus_box')} (inside Step 2 crop: {result.get('step2_crop_box')})")
    print(f"Fallback Active:       {result['is_fallback']}")
    if result["uncertainty_warning"]:
        print(f"Warning:               {result['uncertainty_warning']}")
    print("----------------------------------------------------------")

    # Generate visual diagnosis figure
    saved_img_path = plot_and_save_diagnosis(result, image_path, show=not args.no_show)
    print(f"\n[SUCCESS] 5-Stage anatomy-aware diagnosis image saved to:")
    print(f"file:///{saved_img_path.replace(os.sep, '/')}")
    print("You can Ctrl+Click or open this file directly in VS Code to see all 5 diagnostic panels!")

    # Generate clinical-grade PDF diagnostic report
    try:
        from src.evaluation.pdf_report_generator import generate_diagnostic_pdf_report
        pdf_path = generate_diagnostic_pdf_report(
            result=result,
            image_source=image_path,
            output_path="results/reports/latest_diagnostic_report.pdf"
        )
        print(f"\n[SUCCESS] AI Diagnostic PDF Report generated and saved to:")
        print(f"file:///{pdf_path.replace(os.sep, '/')}")
        print("You can Ctrl+Click or open this PDF report directly in your viewer!")
    except Exception as pdf_err:
        print(f"[WARNING] Could not generate PDF report: {pdf_err}")


if __name__ == "__main__":
    main()
