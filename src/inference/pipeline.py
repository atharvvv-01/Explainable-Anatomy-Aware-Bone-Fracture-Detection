"""
Unified Two-Stage Inference Pipeline for Bone Fracture Detection.
Orchestrates:
Input Image -> Preprocessing -> Stage 1 YOLO Localization -> Padded Cropping
-> Fallback Mechanism -> Stage 2 EfficientNetB3 Classifier -> Thresholding
-> Grad-CAM Explainability -> Final Diagnostic Output.
"""

import os
import json
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from src.preprocessing.image_preprocessing import (
    load_and_validate_image,
    resize_with_aspect_ratio_padding,
    apply_clahe,
    IMAGENET_MEAN,
    IMAGENET_STD
)
from src.localization.detector import AnatomicalDetector
from src.localization.crop_generator import generate_anatomical_crop
from src.classification.efficientnet_model import build_fracture_classifier
from src.explainability.gradcam import GradCAM


class FractureInferencePipeline:
    """
    Two-stage medical inference pipeline integrating localization,
    classification, fallback safeguards, and explainability.
    """
    def __init__(
        self,
        classifier_checkpoint: str = "models/checkpoints/best_model.pth",
        detector_path: str = "models/detector/yolov8n_anatomy.pt",
        threshold_path: str = "config/best_threshold.json",
        conf_threshold: float = 0.35,
        padding_ratio: float = 0.12,
        use_clahe: bool = False,
        device: Optional[torch.device] = None
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        print(f"Initializing Inference Pipeline on device: {self.device}")

        # 1. Load Stage 1 YOLO Detector
        self.detector = AnatomicalDetector(
            model_path=detector_path,
            conf_threshold=conf_threshold,
            device=str(self.device)
        )

        # 2. Load Stage 2 EfficientNetB3 Classifiers (Dual-Stream: Full-Image + Anatomy-Guided)
        self.classifier = build_fracture_classifier(pretrained=False, device=self.device)
        if os.path.exists(classifier_checkpoint):
            print(f"Loading classifier checkpoint: {classifier_checkpoint}")
            ckpt = torch.load(classifier_checkpoint, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.classifier.load_state_dict(state_dict)
        else:
            print(f"Warning: Checkpoint {classifier_checkpoint} not found. Running with initialized weights.")
        self.classifier.eval()

        self.crop_classifier = None
        crop_ckpt = "models/checkpoints/enhanced_anatomy_guided_best.pth"
        if os.path.exists(crop_ckpt):
            try:
                self.crop_classifier = build_fracture_classifier(pretrained=False, device=self.device)
                c_ckpt = torch.load(crop_ckpt, map_location=self.device)
                c_sd = c_ckpt["model_state_dict"] if "model_state_dict" in c_ckpt else c_ckpt
                self.crop_classifier.load_state_dict(c_sd)
                self.crop_classifier.eval()
                print("Loaded Dual-Stream Anatomy-Guided Crop Classifier.")
            except Exception as e:
                print(f"Could not load crop classifier: {e}")
                self.crop_classifier = None

        # 3. Load Grad-CAM (explains localized anatomical features)
        gradcam_model = self.crop_classifier if self.crop_classifier is not None else self.classifier
        self.gradcam = GradCAM(gradcam_model)

        # 4. Load Classification Decision Threshold
        self.threshold = 0.50
        if os.path.exists(threshold_path):
            try:
                with open(threshold_path, 'r', encoding='utf-8') as f:
                    t_info = json.load(f)
                    self.threshold = float(t_info.get("threshold", 0.50))
                    print(f"Loaded tuned classification threshold: {self.threshold:.4f}")
            except Exception as e:
                print(f"Failed to parse threshold file {threshold_path}: {e}")
        else:
            print(f"Threshold file not found. Defaulting to 0.50.")

        self.padding_ratio = padding_ratio
        self.use_clahe = use_clahe

    def predict(
        self,
        image_input: Union[str, Image.Image, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Processes a single radiograph through the entire two-stage pipeline.
        """
        # Step 1: Validate and load image
        original_img = load_and_validate_image(image_input)

        # Step 2: Stage 1 Anatomical Localization
        det_result = self.detector.detect(original_img)
        has_detection = det_result["has_detection"]
        is_fallback = det_result["is_fallback"]
        best_box = det_result["best_box"]
        anatomy_name = det_result["anatomy_name"]
        det_confidence = det_result["confidence"]
        annotated_orig = det_result["annotated_image"]

        # Step 3: Padded Anatomical Cropping or Fallback
        cropped_img, effective_box, crop_fallback, fallback_warning = generate_anatomical_crop(
            image=original_img,
            bounding_box=best_box,
            padding_ratio=self.padding_ratio,
            target_size=(300, 300)
        )

        overall_fallback = is_fallback or crop_fallback

        # Step 4: Classifier Preprocessing (Crop & Global Branches)
        proc_img = cropped_img
        if self.use_clahe:
            proc_img = apply_clahe(proc_img)

        # Tensor conversion & ImageNet normalization for crop
        crop_tensor = TF.to_tensor(proc_img).unsqueeze(0).to(self.device)
        crop_tensor = TF.normalize(crop_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Global Full-Image Tensor
        full_padded, _, _ = resize_with_aspect_ratio_padding(original_img, target_size=(300, 300))
        full_tensor = TF.to_tensor(full_padded).unsqueeze(0).to(self.device)
        full_tensor = TF.normalize(full_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Step 5: Dual-Stream Fracture Classification with TTA
        with torch.no_grad():
            # 1. Global Full-Image Branch
            logit_full, _ = self.classifier(full_tensor)
            p1_full = torch.sigmoid(logit_full).squeeze()
            logit_full_flip, _ = self.classifier(torch.flip(full_tensor, dims=[3]))
            p2_full = torch.sigmoid(logit_full_flip).squeeze()
            p_full = float(((p1_full + p2_full) / 2.0).cpu().numpy())

            # 2. Localized Anatomy Crop Branch
            if self.crop_classifier is not None and not overall_fallback:
                logit_crop, _ = self.crop_classifier(crop_tensor)
                p1_crop = torch.sigmoid(logit_crop).squeeze()
                logit_crop_flip, _ = self.crop_classifier(torch.flip(crop_tensor, dims=[3]))
                p2_crop = torch.sigmoid(logit_crop_flip).squeeze()
                p_crop = float(((p1_crop + p2_crop) / 2.0).cpu().numpy())

                # Optimal dual-stream fusion: 55% global context + 45% localized bone micro-texture
                fracture_prob = float(0.55 * p_full + 0.45 * p_crop)
            else:
                fracture_prob = p_full

            non_fracture_prob = 1.0 - fracture_prob

        # Step 6: Threshold Decision
        is_fractured = bool(fracture_prob >= self.threshold)
        prediction_label = "FRACTURED" if is_fractured else "NON-FRACTURED"

        # Diagnostic confidence score (calibrated distance from threshold)
        if is_fractured:
            model_confidence = float(min(1.0, fracture_prob / max(self.threshold, 1e-4)))
        else:
            model_confidence = float(min(1.0, non_fracture_prob / max(1.0 - self.threshold, 1e-4)))

        # Uncertainty warning if close to threshold
        margin = abs(fracture_prob - self.threshold)
        uncertainty_warning = ""
        if margin < 0.10:
            uncertainty_warning = f"High Diagnostic Uncertainty: Prediction is within {margin*100:.1f}% margin of the decision threshold ({self.threshold:.2f}). Expert review advised."

        # Step 7: Grad-CAM Explainability (Stage 4)
        grad_input = crop_tensor if (self.crop_classifier is not None and not overall_fallback) else full_tensor
        heatmap = self.gradcam.generate_heatmap(grad_input)
        colored_heatmap, heatmap_overlay = self.gradcam.overlay_heatmap(
            cropped_img,
            heatmap,
            alpha=0.45
        )

        # Step 8: Stage 5 - Pathology Focus ROI (Brightest Grad-CAM Region Box)
        # Handle fallback when no detection occurred
        safe_step2_box = best_box if (best_box is not None and not overall_fallback) else effective_box

        # Compute exact pixel boundary of the Step 2 box inside the 300x300 crop canvas
        crop_w = max(1, effective_box[2] - effective_box[0])
        crop_h = max(1, effective_box[3] - effective_box[1])
        target_w, target_h = (300, 300)
        scale = min(target_w / crop_w, target_h / crop_h)
        new_w = max(1, int(round(crop_w * scale)))
        new_h = max(1, int(round(crop_h * scale)))
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2

        if not overall_fallback and best_box is not None:
            # Step 2 box relative to effective cropped region
            reg_x1 = max(0, safe_step2_box[0] - effective_box[0])
            reg_y1 = max(0, safe_step2_box[1] - effective_box[1])
            reg_x2 = min(crop_w, safe_step2_box[2] - effective_box[0])
            reg_y2 = min(crop_h, safe_step2_box[3] - effective_box[1])

            # Step 2 box in 300x300 crop coordinates
            s2_cx1 = max(0, min(target_w - 1, int(round(pad_left + reg_x1 * scale))))
            s2_cy1 = max(0, min(target_h - 1, int(round(pad_top + reg_y1 * scale))))
            s2_cx2 = max(s2_cx1 + 1, min(target_w, int(round(pad_left + reg_x2 * scale))))
            s2_cy2 = max(s2_cy1 + 1, min(target_h, int(round(pad_top + reg_y2 * scale))))
            step2_crop_box = (s2_cx1, s2_cy1, s2_cx2, s2_cy2)
        else:
            s2_cx1, s2_cy1, s2_cx2, s2_cy2 = 0, 0, target_w, target_h
            step2_crop_box = (0, 0, target_w, target_h)

        # Extract small box strictly constrained within the box made in Step 2
        crop_focus_box, focused_image, peak_focus_intensity = self.gradcam.extract_brightest_region_box(
            base_image=cropped_img,
            heatmap=heatmap,
            constraint_box=step2_crop_box,
            is_fractured=is_fractured,
            draw_constraint_box=not overall_fallback,
            constraint_box_label="Bone ROI (Step 2)"
        )

        # Map the small focus box back to original radiograph coordinates (strictly inside Step 2 box)
        sc_w = max(1, s2_cx2 - s2_cx1)
        sc_h = max(1, s2_cy2 - s2_cy1)
        orig_bw = max(1, safe_step2_box[2] - safe_step2_box[0])
        orig_bh = max(1, safe_step2_box[3] - safe_step2_box[1])

        orig_fx1 = int(round(safe_step2_box[0] + (crop_focus_box[0] - s2_cx1) / sc_w * orig_bw))
        orig_fy1 = int(round(safe_step2_box[1] + (crop_focus_box[1] - s2_cy1) / sc_h * orig_bh))
        orig_fx2 = int(round(safe_step2_box[0] + (crop_focus_box[2] - s2_cx1) / sc_w * orig_bw))
        orig_fy2 = int(round(safe_step2_box[1] + (crop_focus_box[3] - s2_cy1) / sc_h * orig_bh))

        # Strict clamp inside the box made in Step 2 on original image
        orig_fx1 = max(safe_step2_box[0], min(safe_step2_box[2] - 1, orig_fx1))
        orig_fy1 = max(safe_step2_box[1], min(safe_step2_box[3] - 1, orig_fy1))
        orig_fx2 = max(orig_fx1 + 1, min(safe_step2_box[2], orig_fx2))
        orig_fy2 = max(orig_fy1 + 1, min(safe_step2_box[3], orig_fy2))
        orig_focus_box = (orig_fx1, orig_fy1, orig_fx2, orig_fy2)

        # Generate full-field radiograph with both Step 2 box and Step 5 small box
        from PIL import ImageDraw
        focused_full_image = original_img.copy().convert("RGB")
        full_draw = ImageDraw.Draw(focused_full_image)
        s2_color = "#00e5ff"
        full_draw.rectangle(list(safe_step2_box), outline=s2_color, width=3)
        s2_badge = full_draw.textbbox((safe_step2_box[0], max(0, safe_step2_box[1] - 18)), "Bone ROI (Step 2)")
        full_draw.rectangle([s2_badge[0] - 2, s2_badge[1] - 1, s2_badge[2] + 2, s2_badge[3] + 1], fill=s2_color)
        full_draw.text((safe_step2_box[0], max(0, safe_step2_box[1] - 18)), "Bone ROI (Step 2)", fill="#000000")

        focus_color = "#ff1744" if is_fractured else "#22c55e"
        full_draw.rectangle(list(orig_focus_box), outline=focus_color, width=4)
        tag_text = f"Pathology Focus ({peak_focus_intensity*100:.0f}%)" if is_fractured else f"Intact Focus ({peak_focus_intensity*100:.0f}%)"
        tag_y = max(safe_step2_box[1], orig_focus_box[1] - 18)
        tag_bbox = full_draw.textbbox((orig_focus_box[0], tag_y), tag_text)
        full_draw.rectangle([tag_bbox[0] - 2, tag_bbox[1] - 1, tag_bbox[2] + 2, tag_bbox[3] + 1], fill=focus_color)
        full_draw.text((orig_focus_box[0], tag_y), tag_text, fill="#ffffff")

        # Grad-CAM Quantitative Spatial Attention Metrics (strictly evaluated inside bone ROI)
        search_mask = np.zeros_like(heatmap)
        search_mask[s2_cy1:s2_cy2, s2_cx1:s2_cx2] = heatmap[s2_cy1:s2_cy2, s2_cx1:s2_cx2]
        bone_roi_pixels = search_mask[s2_cy1:s2_cy2, s2_cx1:s2_cx2]
        mean_act = float(np.mean(bone_roi_pixels)) if bone_roi_pixels.size > 0 else 0.0
        high_act_ratio = float(np.mean(bone_roi_pixels >= 0.50)) if bone_roi_pixels.size > 0 else 0.0

        # Multi-factor Severity & Diagnostic Score (0 - 100 scale)
        if is_fractured:
            score_conf = float(model_confidence * 60.0)
            score_spread = float(min(25.0, high_act_ratio * 100.0 * 0.5))
            score_peak = float(peak_focus_intensity * 15.0)
            severity_score = round(min(100.0, max(1.0, score_conf + score_spread + score_peak)), 1)
            if severity_score >= 75.0:
                severity_level = "SEVERE"
            elif severity_score >= 45.0:
                severity_level = "MODERATE"
            else:
                severity_level = "MILD"
        else:
            score_conf = float(model_confidence * 60.0)
            score_spread = float(max(0.0, (1.0 - high_act_ratio) * 25.0))
            score_peak = float(max(0.0, (1.0 - peak_focus_intensity) * 15.0))
            severity_score = round(max(0.0, fracture_prob * 100.0), 1)
            severity_level = "NORMAL"

        bone_region_label = "Bone Structure" if has_detection else "Full-Image Fallback"

        return {
            "prediction_label": prediction_label,
            "is_fractured": is_fractured,
            "fracture_probability": round(fracture_prob, 4),
            "non_fracture_probability": round(non_fracture_prob, 4),
            "decision_threshold": round(self.threshold, 4),
            "model_confidence": round(model_confidence, 4),
            "uncertainty_warning": uncertainty_warning,
            
            # Severity & Quantitative Grad-CAM Metrics
            "severity_score": severity_score,
            "severity_level": severity_level,
            "mean_activation": round(mean_act * 100.0, 2),
            "high_activation_area": round(high_act_ratio * 100.0, 2),
            "score_breakdown": {
                "model_confidence": round(score_conf, 1),
                "activation_spread": round(score_spread, 1),
                "peak_intensity": round(score_peak, 1),
                "total": severity_score
            },
            
            # Stage 1 Localization (Generic Bone ROI, no body-part classification)
            "region_name": bone_region_label,
            "anatomy_name": bone_region_label,
            "detector_confidence": round(det_confidence, 4),
            "bounding_box": effective_box,
            "is_fallback": overall_fallback,
            "fallback_warning": fallback_warning if overall_fallback else "",
            
            # Stage 5 Pathology Focus Details (Strictly constrained within Step 2 box)
            "focus_box": orig_focus_box,
            "crop_focus_box": crop_focus_box,
            "step2_crop_box": step2_crop_box,
            "peak_focus_intensity": round(peak_focus_intensity, 4),

            # Visual Artifacts (5 Stages)
            "original_image": original_img,                   # Stage 1: Original X-Ray
            "annotated_detection_image": annotated_orig,      # Stage 2: Bone ROI Localization
            "cropped_image": cropped_img,                     # Stage 3: Padded Bone Crop
            "gradcam_heatmap": colored_heatmap,               # Stage 4: Grad-CAM Heatmap
            "gradcam_overlay": heatmap_overlay,               # Stage 4: Grad-CAM Overlay
            "focused_image": focused_image,                   # Stage 5: Pathology Focus ROI Box (Crop)
            "focused_full_image": focused_full_image          # Stage 5: Pathology Focus ROI on Full Radiograph
        }
