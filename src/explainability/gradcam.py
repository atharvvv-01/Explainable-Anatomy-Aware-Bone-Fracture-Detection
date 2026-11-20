"""
High-Precision Grad-CAM and Layer-CAM Module for EfficientNetB3 Bone Fracture Detection.
Visualizes class activation heatmaps to highlight anatomical cortical bone disruptions
and fracture lines, eliminating diffuse background haze and random artifact highlights.
"""

from typing import Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.preprocessing.image_preprocessing import (
    load_and_validate_image,
    resize_with_aspect_ratio_padding,
    IMAGENET_MEAN,
    IMAGENET_STD
)


class GradCAM:
    """
    High-Precision Layer-CAM & Adaptive Thresholded Grad-CAM specifically optimized for EfficientNetB3.
    Hooks into deep spatial MBConv blocks (features[7]) rather than 1x1 pointwise convs,
    and applies element-wise positive gradient weighting (Layer-CAM) to preserve fine-grained
    hairline fractures without gradient averaging dilution.
    """
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        use_layercam: bool = True
    ):
        self.model = model
        self.model.eval()
        self.use_layercam = use_layercam

        # Select target layer:
        # Crucial fix: features[7] is the final deep spatial MBConv stage (5x5 kernel),
        # whereas features[8] is merely a 1x1 pointwise projection layer.
        if target_layer is None:
            if hasattr(self.model, "features"):
                if len(self.model.features) >= 8:
                    self.target_layer = self.model.features[7]
                else:
                    self.target_layer = self.model.features[-1]
            else:
                raise ValueError("Model does not have 'features' attribute.")
        else:
            self.target_layer = target_layer

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Attaches forward and backward hooks to target feature layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h1, h2])

    def remove_hooks(self) -> None:
        """Removes registered PyTorch hooks."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
        threshold_ratio: float = 0.25
    ) -> np.ndarray:
        """
        Computes pinpoint Class Activation Map for input tensor.
        Args:
            input_tensor: [1, 3, H, W] normalized PyTorch tensor.
            target_class_idx: Class index (1 for fracture).
            threshold_ratio: Minimum fraction of peak activation to retain (filters out background noise).
        Returns:
            heatmap: 2D numpy array [H, W] normalized to [0.0, 1.0].
        """
        self.model.zero_grad()
        logits, _ = self.model(input_tensor)

        # Binary logit handling
        if logits.dim() == 1 or logits.size(-1) == 1:
            target_score = logits.squeeze()
        else:
            idx = target_class_idx if target_class_idx is not None else 1
            target_score = logits[0, idx]

        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations during Grad-CAM pass.")

        if self.use_layercam:
            # Layer-CAM: Element-wise positive gradient weighting
            # Preserves localized micro-cracks without global averaging dilution
            pos_grads = F.relu(self.gradients)
            pos_acts = F.relu(self.activations)
            cam = torch.sum(pos_grads * pos_acts, dim=1, keepdim=True)
        else:
            # Standard Grad-CAM with global pooling
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam)

        # Bilinear interpolation to input tensor resolution
        cam = F.interpolate(
            cam,
            size=(input_tensor.size(2), input_tensor.size(3)),
            mode='bilinear',
            align_corners=False
        )

        heatmap = cam.squeeze().cpu().numpy()

        # Dynamic Thresholding: Discard diffuse background haze below threshold_ratio
        h_max = np.max(heatmap)
        if h_max > 1e-7:
            norm_map = heatmap / h_max
            # Zero out background noise
            clean_map = np.where(norm_map >= threshold_ratio, norm_map, 0.0)
            if np.max(clean_map) > 0:
                clean_map = (clean_map - threshold_ratio) / (1.0 - threshold_ratio + 1e-8)
                clean_map = np.clip(clean_map, 0.0, 1.0)
            heatmap = clean_map
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap

    def overlay_heatmap(
        self,
        original_image: Union[Image.Image, np.ndarray],
        heatmap: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.55
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Overlays the normalized heatmap on top of the original radiographic crop.
        Uses dynamic alpha masking: zero-activation areas (healthy bone/air) remain
        100% transparent (no blue haze), and only true fracture hotspots glow vividly.
        Returns:
            colored_heatmap_pil: PIL image of heatmap
            overlay_pil: PIL image of blended radiograph + heatmap
        """
        if isinstance(original_image, Image.Image):
            base_rgb = np.array(original_image.convert('RGB'))
        else:
            base_rgb = original_image.copy()

        h, w = base_rgb.shape[:2]

        # Resize heatmap to match image dimensions
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply colormap
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        # Dynamic Alpha Blending:
        # Ensures normal bone and background are completely transparent with 0 blue tint
        alpha_mask = (heatmap_resized[:, :, None] ** 0.85) * alpha
        overlay = np.uint8(alpha_mask * colored_heatmap_rgb + (1.0 - alpha_mask) * base_rgb)

        return Image.fromarray(colored_heatmap_rgb), Image.fromarray(overlay)

    def extract_brightest_region_box(
        self,
        base_image: Union[Image.Image, np.ndarray],
        heatmap: np.ndarray,
        constraint_box: Optional[Tuple[int, int, int, int]] = None,
        intensity_threshold_ratio: float = 0.65,
        padding: int = 10,
        min_box_dim: int = 35,
        is_fractured: bool = True,
        draw_constraint_box: bool = True,
        constraint_box_label: str = "Bone ROI (Step 2)"
    ) -> Tuple[Tuple[int, int, int, int], Image.Image, float]:
        """
        Stage 5: Pinpoints the exact bone micro-region where Grad-CAM attention is the brightest.
        Extracts a localized bounding box around the peak activation hotspot and draws a
        high-visibility targeting box with crosshair corners on the image.

        Strict Spatial Constraint:
            When constraint_box (the box made in Step 2) is provided, the search space and
            the extracted small box are strictly masked and clamped to lie 100% inside
            the constraint_box boundaries.

        Returns:
            box: (x1, y1, x2, y2) coordinates within base_image
            focused_image: PIL Image with the targeting bounding box drawn
            peak_val: maximum activation score [0.0, 1.0]
        """
        from PIL import ImageDraw

        if isinstance(base_image, Image.Image):
            pil_img = base_image.convert("RGB")
        else:
            pil_img = Image.fromarray(base_image).convert("RGB")

        w, h = pil_img.size

        # Clamp constraint_box to valid image bounds if provided
        if constraint_box is not None:
            cb_raw_x1, cb_raw_y1, cb_raw_x2, cb_raw_y2 = constraint_box
            cb_x1 = max(0, min(w, int(cb_raw_x1)))
            cb_y1 = max(0, min(h, int(cb_raw_y1)))
            cb_x2 = max(cb_x1 + 1, min(w, int(cb_raw_x2)))
            cb_y2 = max(cb_y1 + 1, min(h, int(cb_raw_y2)))
        else:
            cb_x1, cb_y1, cb_x2, cb_y2 = 0, 0, w, h

        # Resize heatmap to match image dimensions
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        # STRICT MASKING: Restrict search space to lie ONLY inside the box made in Step 2
        search_map = np.zeros_like(heatmap_resized)
        search_map[cb_y1:cb_y2, cb_x1:cb_x2] = heatmap_resized[cb_y1:cb_y2, cb_x1:cb_x2]

        peak_val = float(np.max(search_map))

        target_img = pil_img.copy()
        draw = ImageDraw.Draw(target_img)

        # Visual colors
        step2_color = "#00e5ff"  # Cyan outline matching Step 2 detector
        box_color = "#ff1744" if is_fractured else "#22c55e"
        glow_color = "#ff8a80" if is_fractured else "#86efac"
        label_text = f"Pathology Focus ({peak_val*100:.0f}%)" if is_fractured else f"Intact ROI ({peak_val*100:.0f}%)"

        # Draw Step 2 reference box if constraint was passed and occupies a subregion
        if draw_constraint_box and constraint_box is not None and (cb_x2 - cb_x1 < w or cb_y2 - cb_y1 < h):
            draw.rectangle([cb_x1, cb_y1, cb_x2, cb_y2], outline=step2_color, width=2)
            s2_y = max(0, cb_y1 - 15)
            s2_bbox = draw.textbbox((cb_x1, s2_y), constraint_box_label)
            draw.rectangle([s2_bbox[0] - 2, s2_bbox[1] - 1, s2_bbox[2] + 2, s2_bbox[3] + 1], fill=step2_color)
            draw.text((cb_x1, s2_y), constraint_box_label, fill="#000000")

        if peak_val < 1e-4:
            # Heatmap has zero activation inside constraint box; create centered safe box inside Step 2
            half_w = max(10, (cb_x2 - cb_x1) // 4)
            half_h = max(10, (cb_y2 - cb_y1) // 4)
            mid_x = (cb_x1 + cb_x2) // 2
            mid_y = (cb_y1 + cb_y2) // 2
            default_box = (
                max(cb_x1, mid_x - half_w),
                max(cb_y1, mid_y - half_h),
                min(cb_x2, mid_x + half_w),
                min(cb_y2, mid_y + half_h)
            )
            draw.rectangle(list(default_box), outline=box_color, width=2)
            return default_box, target_img, 0.0

        # Find coordinate of global maximum strictly within Step 2 box
        yp, xp = np.unravel_index(np.argmax(search_map), search_map.shape)

        # Threshold to isolate the brightest core region (values >= threshold_ratio of peak)
        threshold_val = peak_val * intensity_threshold_ratio
        mask = (search_map >= threshold_val).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            best_contour = None
            min_dist = float("inf")
            for cnt in contours:
                dist = cv2.pointPolygonTest(cnt, (float(xp), float(yp)), True)
                if dist >= 0:
                    best_contour = cnt
                    break
                else:
                    if abs(dist) < min_dist:
                        min_dist = abs(dist)
                        best_contour = cnt

            if best_contour is None:
                best_contour = max(contours, key=cv2.contourArea)

            bx, by, bw, bh = cv2.boundingRect(best_contour)
            x1 = max(cb_x1, bx - padding)
            y1 = max(cb_y1, by - padding)
            x2 = min(cb_x2, bx + bw + padding)
            y2 = min(cb_y2, by + bh + padding)
        else:
            half_dim = min_box_dim // 2
            x1 = max(cb_x1, xp - half_dim)
            y1 = max(cb_y1, yp - half_dim)
            x2 = min(cb_x2, xp + half_dim)
            y2 = min(cb_y2, yp + half_dim)

        # Enforce minimum box dimensions, strictly bounded inside the Step 2 box
        box_w = x2 - x1
        box_h = y2 - y1
        max_allow_w = cb_x2 - cb_x1
        max_allow_h = cb_y2 - cb_y1
        eff_min_w = min(min_box_dim, max_allow_w)
        eff_min_h = min(min_box_dim, max_allow_h)

        if box_w < eff_min_w:
            diff = (eff_min_w - box_w) // 2
            x1 = max(cb_x1, x1 - diff)
            x2 = min(cb_x2, x2 + diff)
            if x2 - x1 < eff_min_w:
                if x1 == cb_x1:
                    x2 = min(cb_x2, x1 + eff_min_w)
                elif x2 == cb_x2:
                    x1 = max(cb_x1, x2 - eff_min_w)

        if box_h < eff_min_h:
            diff = (eff_min_h - box_h) // 2
            y1 = max(cb_y1, y1 - diff)
            y2 = min(cb_y2, y2 + diff)
            if y2 - y1 < eff_min_h:
                if y1 == cb_y1:
                    y2 = min(cb_y2, y1 + eff_min_h)
                elif y2 == cb_y2:
                    y1 = max(cb_y1, y2 - eff_min_h)

        # Final strict clamp inside constraint box
        x1 = max(cb_x1, min(cb_x2 - 1, int(round(x1))))
        y1 = max(cb_y1, min(cb_y2 - 1, int(round(y1))))
        x2 = max(x1 + 1, min(cb_x2, int(round(x2))))
        y2 = max(y1 + 1, min(cb_y2, int(round(y2))))

        box = (int(x1), int(y1), int(x2), int(y2))

        # High-contrast targeting box with glow line
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
        draw.rectangle([max(cb_x1, x1 - 1), max(cb_y1, y1 - 1), min(cb_x2, x2 + 1), min(cb_y2, y2 + 1)], outline=glow_color, width=1)

        # Targeting crosshairs / corner brackets
        corner_len = min(12, max(5, (x2 - x1) // 3), max(5, (y2 - y1) // 3))
        # Top-left corner
        draw.line([(x1, y1), (x1 + corner_len, y1)], fill="#ffffff", width=2)
        draw.line([(x1, y1), (x1, y1 + corner_len)], fill="#ffffff", width=2)
        # Top-right corner
        draw.line([(x2, y1), (x2 - corner_len, y1)], fill="#ffffff", width=2)
        draw.line([(x2, y1), (x2, y1 + corner_len)], fill="#ffffff", width=2)
        # Bottom-left corner
        draw.line([(x1, y2), (x1 + corner_len, y2)], fill="#ffffff", width=2)
        draw.line([(x1, y2), (x1, y2 - corner_len)], fill="#ffffff", width=2)
        # Bottom-right corner
        draw.line([(x2, y2), (x2 - corner_len, y2)], fill="#ffffff", width=2)
        draw.line([(x2, y2), (x2 - corner_len, y2)], fill="#ffffff", width=2)

        # Tag label banner for the small box
        tag_y = y1 - 18 if y1 - 18 >= cb_y1 else y2 + 2
        text_bbox = draw.textbbox((x1, tag_y), label_text)
        draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 1, text_bbox[2] + 4, text_bbox[3] + 2], fill=box_color)
        draw.text((x1 + 1, tag_y), label_text, fill="#ffffff")

        return box, target_img, peak_val
