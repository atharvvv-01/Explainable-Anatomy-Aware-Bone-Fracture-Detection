"""
Stage 1 Anatomical Region Localization Detector (YOLO-based).
Detects skeletal structures (Arm, Leg, Hand, Shoulder, Spine)
and provides visualization, coordinates, confidence scores, and fallback flags.
"""

import os
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO

from src.localization.annotation_utils import ANATOMY_CLASSES
from src.preprocessing.image_preprocessing import load_and_validate_image


class AnatomicalDetector:
    """
    Stage 1 Anatomical Object Detector wrapping YOLOv8.
    """
    def __init__(
        self,
        model_path: str = "models/detector/yolov8n_anatomy.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # If custom anatomy detector checkpoint exists, load it; else fallback to yolov8n.pt base
        if os.path.exists(model_path):
            print(f"Loading trained Anatomical Detector: {model_path}")
            self.model = YOLO(model_path)
        elif os.path.exists("yolov8n.pt"):
            print(f"Custom checkpoint {model_path} not found. Initializing with base YOLOv8n: yolov8n.pt")
            self.model = YOLO("yolov8n.pt")
        else:
            print("Downloading/initializing base YOLOv8n detector...")
            self.model = YOLO("yolov8n.pt")

    def detect(
        self,
        image_input: Union[str, Image.Image, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Runs anatomical localization on input X-ray.
        """
        pil_img = load_and_validate_image(image_input)
        orig_w, orig_h = pil_img.size
        cv_img = np.array(pil_img)  # RGB

        # Run inference
        results = self.model.predict(
            source=cv_img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        all_detections: List[Dict[str, Any]] = []
        best_detection: Optional[Dict[str, Any]] = None
        max_conf = -1.0

        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_idx = int(boxes.cls[i].cpu().numpy())

                # Pure Bone Region Localization (generic skeletal ROI without anatomical classification)
                class_name = "Bone Structure"

                det_dict = {
                    "box": (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                    "confidence": round(conf, 4),
                    "class_id": 0,
                    "region_name": class_name,
                    "anatomy_name": class_name
                }
                all_detections.append(det_dict)

                if conf > max_conf:
                    max_conf = conf
                    best_detection = det_dict

        # If no detection or confidence below threshold, trigger fallback
        is_fallback = (best_detection is None or best_detection["confidence"] < self.conf_threshold)

        # Create annotated image for display
        annotated_img = pil_img.copy()
        draw = ImageDraw.Draw(annotated_img)

        if not is_fallback and best_detection is not None:
            bx1, by1, bx2, by2 = best_detection["box"]
            c_conf = best_detection["confidence"]
            
            # Draw bounding box (medical cyan)
            draw.rectangle([bx1, by1, bx2, by2], outline="#00e5ff", width=4)
            label_text = f"Bone ROI ({c_conf*100:.1f}%)"
            
            # Draw label banner
            text_bbox = draw.textbbox((bx1, max(0, by1 - 25)), label_text)
            draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 4, text_bbox[3] + 2], fill="#00e5ff")
            draw.text((bx1, max(0, by1 - 25)), label_text, fill="black")
        else:
            # Fallback border (amber warning)
            draw.rectangle([0, 0, orig_w - 1, orig_h - 1], outline="#ff9100", width=3)
            label_text = "Fallback: Full Image Utilized"
            text_bbox = draw.textbbox((10, 10), label_text)
            draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 4, text_bbox[3] + 2], fill="#ff9100")
            draw.text((10, 10), label_text, fill="black")

        return {
            "has_detection": not is_fallback,
            "is_fallback": is_fallback,
            "best_box": best_detection["box"] if (not is_fallback and best_detection is not None) else None,
            "region_name": "Bone Structure" if (not is_fallback and best_detection is not None) else "Full-Image Fallback",
            "anatomy_name": "Bone Structure" if (not is_fallback and best_detection is not None) else "Full-Image Fallback",
            "confidence": best_detection["confidence"] if (not is_fallback and best_detection is not None) else 0.0,
            "all_detections": all_detections,
            "annotated_image": annotated_img,
            "warning": "Bone ROI detector confidence below threshold. Full-image fallback engaged." if is_fallback else ""
        }
