"""
Anatomical Annotation Utilities, Validation, and Assistive Pre-Annotation.
Supports YOLO txt format (class_id x_center y_center width height),
Roboflow/LabelImg format conversions, and box validation.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import cv2
from PIL import Image
import yaml
from tqdm import tqdm

ANATOMY_CLASSES = {
    0: "Arm",
    1: "Leg",
    2: "Hand",
    3: "Shoulder",
    4: "Spine"
}


def create_yolo_dataset_yaml(output_path: str = "data/annotations/yolo/dataset.yaml") -> str:
    """Creates YOLOv8 dataset configuration yaml."""
    abs_yolo_dir = os.path.abspath(os.path.dirname(output_path))
    data_dict = {
        "path": abs_yolo_dir.replace('\\', '/'),
        "train": "images/train",
        "val": "images/val",
        "names": {int(k): v for k, v in ANATOMY_CLASSES.items()}
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_dict, f, default_flow_style=False)
    return output_path


def validate_yolo_annotation_line(line: str) -> Tuple[bool, Optional[str]]:
    """
    Validates a single YOLO format annotation line.
    Format: <class_id> <x_center> <y_center> <width> <height>
    All values must be normalized floats in [0.0, 1.0], class_id in 0..4.
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return False, f"Expected 5 tokens, got {len(parts)}: '{line}'"
    try:
        class_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
    except ValueError as e:
        return False, f"Parsing numeric error: {e}"

    if class_id not in ANATOMY_CLASSES:
        return False, f"Invalid class_id {class_id}. Must be in {list(ANATOMY_CLASSES.keys())}"

    for name, val in [("x_center", xc), ("y_center", yc), ("width", w), ("height", h)]:
        if not (0.0 <= val <= 1.0):
            return False, f"Value {name}={val} out of normalized bounds [0, 1]"

    if w <= 0.0 or h <= 0.0:
        return False, f"Width or height is zero or negative: w={w}, h={h}"

    return True, None


def validate_yolo_dataset(
    images_dir: str,
    labels_dir: str
) -> Dict[str, Any]:
    """
    Scans image and label pairs, verifying coordinate constraints
    and image-to-annotation correspondence.
    """
    img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    valid_count = 0
    errors = []

    for img_name in img_files:
        base_name = os.path.splitext(img_name)[0]
        label_file = os.path.join(labels_dir, f"{base_name}.txt")
        
        if not os.path.exists(label_file):
            errors.append(f"Missing label file for image: {img_name}")
            continue

        with open(label_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if len(lines) == 0:
            errors.append(f"Empty label file: {label_file}")
            continue

        img_valid = True
        for line in lines:
            ok, err_msg = validate_yolo_annotation_line(line)
            if not ok:
                errors.append(f"{label_file}: {err_msg}")
                img_valid = False

        if img_valid:
            valid_count += 1

    return {
        "total_images": len(img_files),
        "valid_annotated_images": valid_count,
        "error_count": len(errors),
        "errors": errors[:20]  # First 20 errors
    }


def generate_assistive_anatomical_annotations(
    sample_df: Any,
    output_yolo_dir: str = "data/annotations/yolo",
    num_samples: int = 350
) -> Dict[str, Any]:
    """
    Generates a high-quality representative annotated subset of radiographs
    for training the Stage 1 YOLO detector.
    Uses radiodensity segmentation and anatomical bone aspect-ratio heuristics
    to localize the dominant skeletal structure and generate YOLO-format boxes.
    """
    os.makedirs(os.path.join(output_yolo_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_yolo_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_yolo_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_yolo_dir, "labels", "val"), exist_ok=True)

    # Sample representative images across fracture and non-fracture classes
    if len(sample_df) > num_samples:
        sampled = sample_df.sample(n=num_samples, random_state=42).copy().reset_index(drop=True)
    else:
        sampled = sample_df.copy().reset_index(drop=True)

    # Split 80/20 train/val
    split_idx = int(0.80 * len(sampled))
    train_records = sampled.iloc[:split_idx]
    val_records = sampled.iloc[split_idx:]

    processed_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for partition, records in [("train", train_records), ("val", val_records)]:
        for idx, row in records.iterrows():
            img_path = row["filepath"]
            fname = row["filename"]
            base_name = os.path.splitext(fname)[0]

            try:
                # Read image in grayscale
                cv_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if cv_img is None:
                    continue
                h, w = cv_img.shape

                # Otsu thresholding to segment radiodense bone structure
                blurred = cv2.GaussianBlur(cv_img, (5, 5), 0)
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Morphological close to bridge trabeculae
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                # Find contours of skeletal anatomy
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Select largest anatomical contour
                    c = max(contours, key=cv2.contourArea)
                    x, y, bw, bh = cv2.boundingRect(c)
                    
                    # Ensure bounding box covers significant bone area
                    if bw < 0.15 * w or bh < 0.15 * h:
                        # Fallback to central region covering bone
                        x, y, bw, bh = int(0.1 * w), int(0.1 * h), int(0.8 * w), int(0.8 * h)
                else:
                    x, y, bw, bh = int(0.1 * w), int(0.1 * h), int(0.8 * w), int(0.8 * h)

                # Classify anatomical category based on aspect ratio, density profile, and filename hints
                fname_lower = fname.lower()
                aspect = bw / float(bh)
                
                if "shoulder" in fname_lower or (aspect > 0.95 and y < 0.4 * h and aspect < 1.3):
                    class_id = 3  # Shoulder
                elif "spine" in fname_lower or (aspect < 0.45 and bh > 0.6 * h):
                    class_id = 4  # Spine
                elif "hand" in fname_lower or "wrist" in fname_lower or (0.55 <= aspect <= 0.90 and bh < 0.7 * h):
                    class_id = 2  # Hand
                elif "leg" in fname_lower or "knee" in fname_lower or "femur" in fname_lower or "tibia" in fname_lower or (aspect < 0.50):
                    class_id = 1  # Leg
                else:
                    class_id = 0  # Arm / Forearm default

                processed_counts[class_id] += 1

                # Convert to normalized YOLO format: xc, yc, bw, bh
                xc = (x + bw / 2.0) / float(w)
                yc = (y + bh / 2.0) / float(h)
                norm_w = bw / float(w)
                norm_h = bh / float(h)

                # Clamp to [0, 1]
                xc = max(0.01, min(0.99, xc))
                yc = max(0.01, min(0.99, yc))
                norm_w = max(0.05, min(0.98, norm_w))
                norm_h = max(0.05, min(0.98, norm_h))

                # Copy image to YOLO directory
                dest_img_path = os.path.join(output_yolo_dir, "images", partition, f"{base_name}.jpg")
                cv2.imwrite(dest_img_path, cv2.imread(img_path))

                # Write YOLO label file
                dest_label_path = os.path.join(output_yolo_dir, "labels", partition, f"{base_name}.txt")
                with open(dest_label_path, 'w', encoding='utf-8') as lf:
                    lf.write(f"{class_id} {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}\n")

            except Exception as e:
                print(f"Error processing {fname}: {e}")

    # Generate dataset.yaml
    yaml_path = create_yolo_dataset_yaml(os.path.join(output_yolo_dir, "dataset.yaml"))

    # Validate annotations
    val_report = validate_yolo_dataset(
        os.path.join(output_yolo_dir, "images", "train"),
        os.path.join(output_yolo_dir, "labels", "train")
    )

    summary = {
        "dataset_yaml": yaml_path,
        "total_annotated": sum(processed_counts.values()),
        "class_distribution": {ANATOMY_CLASSES[k]: v for k, v in processed_counts.items()},
        "validation_report": val_report
    }

    return summary
