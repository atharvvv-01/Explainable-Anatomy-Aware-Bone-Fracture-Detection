"""
Packages a lightweight, standalone Production API Deployment Bundle into a single .zip file.
Excludes the multi-gigabyte training dataset and cache files, keeping only
necessary models, code, configurations, and API server scripts.
"""

import os
import zipfile

def create_api_zip(output_zip: str = "fracture_detection_api_bundle.zip"):
    print(f"Creating Production API Deployment Archive: {output_zip}...")
    
    # Items to include in the package
    include_files = [
        "api_server.py",
        "choose_image_and_detect.py",
        "app.py",
        "requirements.txt",
        "config/best_threshold.json",
        "models/detector/yolov8n_anatomy.pt",
        "models/checkpoints/best_model.pth",
        "models/checkpoints/enhanced_anatomy_guided_best.pth",
        "samples/sample_fracture_arm.png",
        "samples/sample_healthy_spine.jpg",
    ]

    include_dirs = [
        "src",
    ]

    total_files = 0
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add discrete files
        for f in include_files:
            if os.path.exists(f):
                zipf.write(f, arcname=f)
                total_files += 1
                print(f"  + Added: {f}")
            else:
                print(f"  - Warning: file not found: {f}")

        # Add source directories recursively (excluding __pycache__)
        for d in include_dirs:
            for root, dirs, files in os.walk(d):
                if "__pycache__" in root:
                    continue
                for file in files:
                    if file.endswith((".py", ".json", ".yaml", ".yml", ".md")):
                        full_path = os.path.join(root, file)
                        arcname = os.path.relpath(full_path, ".")
                        zipf.write(full_path, arcname=arcname)
                        total_files += 1

    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print("\n" + "=" * 60)
    print(f"[SUCCESS] API Bundle created: {output_zip}")
    print(f"Total files packed: {total_files}")
    print(f"Total bundle size:  {size_mb:.2f} MB (vs 2.5 GB full repo)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    create_api_zip()
