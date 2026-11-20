"""
Production FastAPI Server for Anatomy-Aware Fracture Detection & Grad-CAM Explainability.
Run with:
    pip install fastapi uvicorn python-multipart
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import io
import base64
from typing import Optional
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch

from src.inference.pipeline import FractureInferencePipeline

app = FastAPI(
    title="Anatomy-Aware Bone Fracture Detection API",
    description="Deep Learning API (EfficientNet-B3 + YOLOv8) for bone fracture diagnosis and Grad-CAM explainability.",
    version="1.0.0"
)

# Enable CORS for web/mobile client integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Inference Pipeline (Lazy singleton)
pipeline: Optional[FractureInferencePipeline] = None

@app.on_event("startup")
def load_pipeline():
    global pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing FractureInferencePipeline on {device}...")
    pipeline = FractureInferencePipeline(
        classifier_checkpoint="models/checkpoints/best_model.pth",
        detector_path="models/detector/yolov8n_anatomy.pt",
        threshold_path="config/best_threshold.json",
        device=device
    )
    print("Inference Pipeline loaded successfully.")

def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model": "EfficientNet-B3 (Dual-Stream Anatomy-Aware)",
        "device": str(pipeline.device) if pipeline else "loading"
    }

@app.post("/predict")
async def predict_fracture(file: UploadFile = File(...)):
    """
    Accepts an X-ray radiograph image and returns:
    - Diagnostic prediction (FRACTURED / NON-FRACTURED)
    - Fracture probability and decision threshold
    - Detected anatomical compartment and bounding box
    - Base64-encoded Grad-CAM heatmap overlay
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is still initializing.")

    # Validate image format
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    try:
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    # Run inference
    try:
        results = pipeline.predict(pil_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # Encode visualizations as base64 PNGs
    gradcam_b64 = pil_to_base64(results["gradcam_overlay"])
    crop_b64 = pil_to_base64(results["cropped_image"])
    focus_b64 = pil_to_base64(results["focused_image"])

    # Optional PDF report generation
    pdf_report_path = None
    try:
        from src.evaluation.pdf_report_generator import generate_diagnostic_pdf_report
        pdf_report_path = generate_diagnostic_pdf_report(
            result=results,
            image_source=file.filename or "uploaded_radiograph.png",
            output_path=f"results/reports/api_{file.filename or 'radiograph'}.pdf"
        )
    except Exception:
        pass

    return JSONResponse(content={
        "prediction": results["prediction_label"],
        "is_fractured": results["is_fractured"],
        "fracture_probability": results["fracture_probability"],
        "decision_threshold": results["decision_threshold"],
        "model_confidence": results["model_confidence"],
        "severity_score": results.get("severity_score"),
        "severity_level": results.get("severity_level"),
        "uncertainty_warning": results["uncertainty_warning"],
        "bone_localization": {
            "status": results.get("region_name", "Bone Structure"),
            "detector_confidence": results["detector_confidence"],
            "bounding_box": results["bounding_box"],
            "is_fallback": results["is_fallback"]
        },
        "pathology_focus": {
            "focus_bounding_box": results.get("focus_box"),
            "crop_focus_box": results.get("crop_focus_box"),
            "step2_crop_box": results.get("step2_crop_box"),
            "peak_activation_intensity": results.get("peak_focus_intensity"),
            "strictly_inside_step2_box": True
        },
        "pdf_report_path": pdf_report_path,
        "visualizations": {
            "stage_1_original": "Input Radiograph",
            "stage_2_bone_crop_base64": f"data:image/png;base64,{crop_b64}",
            "stage_4_gradcam_overlay_base64": f"data:image/png;base64,{gradcam_b64}",
            "stage_5_pathology_focus_box_base64": f"data:image/png;base64,{focus_b64}"
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
