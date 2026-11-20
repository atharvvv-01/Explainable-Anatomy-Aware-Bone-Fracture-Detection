"""
Dataset Cleaning and Integrity Verification Module.
Handles corruption checks, exact duplicate detection (SHA-256),
and perceptual near-duplicate detection (dHash).
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
from PIL import Image
import pandas as pd
from tqdm import tqdm


def compute_sha256(filepath: str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """
    Compute difference hash (dHash) for perceptual near-duplicate detection.
    """
    # Resize to (hash_size + 1, hash_size) grayscale
    resized = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(resized.getdata())
    
    difference = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = pixels[row * (hash_size + 1) + col]
            pixel_right = pixels[row * (hash_size + 1) + col + 1]
            difference.append(pixel_left > pixel_right)
            
    decimal_value = 0
    for idx, val in enumerate(difference):
        if val:
            decimal_value += 2 ** idx
    return decimal_value


def hamming_distance(hash1: int, hash2: int) -> int:
    """Compute Hamming distance between two integer hashes."""
    x = hash1 ^ hash2
    return bin(x).count('1')


def scan_and_clean_dataset(
    dataset_dir: str = "Dataset/Dataset",
    save_report_path: str = "reports/cleaning_report.json"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Scans the dataset directory, inspects all images for integrity,
    identifies exact and near-duplicates, and returns a clean catalog dataframe.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_dir}")

    records: List[Dict[str, Any]] = []
    corrupted_files: List[Dict[str, str]] = []
    sha256_map: Dict[str, str] = {}
    exact_duplicates: List[Dict[str, str]] = []
    dhash_map: Dict[str, Tuple[str, int]] = {}
    near_duplicates: List[Dict[str, Any]] = []

    # Map subfolder names to binary fracture labels
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    total_scanned = 0
    for class_dir in class_dirs:
        class_name = class_dir.name
        # Standardize fracture label
        if "non" in class_name.lower():
            fracture_label = "Non-fractured"
            fracture_id = 0
        else:
            fracture_label = "Fractured"
            fracture_id = 1

        image_files = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        
        for img_path in tqdm(image_files, desc=f"Scanning {class_name}"):
            total_scanned += 1
            file_str = str(img_path)
            
            # Step 1: Integrity check
            try:
                # File hash
                file_hash = compute_sha256(file_str)
                
                # Check exact duplicate
                is_exact_dup = False
                original_dup_file = None
                if file_hash in sha256_map:
                    is_exact_dup = True
                    original_dup_file = sha256_map[file_hash]
                    exact_duplicates.append({
                        "duplicate_file": file_str,
                        "original_file": original_dup_file,
                        "class": fracture_label,
                        "sha256": file_hash
                    })
                else:
                    sha256_map[file_hash] = file_str

                # PIL verification
                with Image.open(file_str) as img:
                    img.verify()
                
                # Open again for reading properties
                with Image.open(file_str) as img:
                    width, height = img.size
                    channels = len(img.getbands())
                    mode = img.mode
                    img_format = img.format or img_path.suffix.upper().replace('.', '')
                    perceptual_hash = compute_dhash(img)

                # Check near-duplicates (Hamming distance <= 2)
                is_near_dup = False
                near_dup_file = None
                if not is_exact_dup:
                    for exist_path, (exist_orig, exist_hash) in dhash_map.items():
                        dist = hamming_distance(perceptual_hash, exist_hash)
                        if dist <= 2:
                            is_near_dup = True
                            near_dup_file = exist_path
                            near_duplicates.append({
                                "file_a": file_str,
                                "file_b": exist_path,
                                "hamming_distance": dist,
                                "class": fracture_label
                            })
                            break
                    dhash_map[file_str] = (file_str, perceptual_hash)

                records.append({
                    "filepath": file_str,
                    "filename": img_path.name,
                    "folder_label": class_name,
                    "fracture_label": fracture_label,
                    "fracture_id": fracture_id,
                    "format": img_format,
                    "mode": mode,
                    "channels": channels,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 4),
                    "sha256": file_hash,
                    "dhash": perceptual_hash,
                    "is_exact_duplicate": is_exact_dup,
                    "exact_duplicate_of": original_dup_file,
                    "is_near_duplicate": is_near_dup,
                    "near_duplicate_of": near_dup_file,
                    "is_valid": True
                })

            except Exception as exc:
                corrupted_files.append({
                    "filepath": file_str,
                    "error": str(exc),
                    "class": fracture_label
                })

    df = pd.DataFrame(records)
    
    summary = {
        "total_scanned": total_scanned,
        "valid_images": len(df),
        "corrupted_count": len(corrupted_files),
        "exact_duplicates_count": len(exact_duplicates),
        "near_duplicates_count": len(near_duplicates),
        "corrupted_files": corrupted_files,
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates
    }

    os.makedirs(os.path.dirname(save_report_path), exist_ok=True)
    with open(save_report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return df, summary


if __name__ == "__main__":
    df, summary = scan_and_clean_dataset()
    print("\n--- Data Cleaning Complete ---")
    print(f"Total Scanned: {summary['total_scanned']}")
    print(f"Valid: {summary['valid_images']}")
    print(f"Corrupted: {summary['corrupted_count']}")
    print(f"Exact Duplicates: {summary['exact_duplicates_count']}")
    print(f"Near Duplicates: {summary['near_duplicates_count']}")
