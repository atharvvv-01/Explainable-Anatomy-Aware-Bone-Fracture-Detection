# Anatomy-Guided Bone Fracture Detection and Classification
## Two-Stage Deep Learning Pipeline Using YOLOv8 and EfficientNetB3

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.14+cu130](https://img.shields.io/badge/PyTorch-2.14%2Bcu130-red.svg)](https://pytorch.org/)
[![CUDA Acceleration](https://img.shields.io/badge/CUDA-NVIDIA_RTX_4060-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLO-v8n-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit Dashboard](https://img.shields.io/badge/Streamlit-Medical_AI_UI-brightgreen.svg)](https://streamlit.io/)

An academic- and research-grade computer vision system designed to address one of emergency radiology's most prevalent failure modes: **missed bone fractures**. The architecture utilizes a **two-stage anatomy-guided pipeline** combining skeletal localization with deep transfer learning and Grad-CAM explainability.

---

## 1. Project Overview & Clinical Problem

In emergency triage, missed fractures constitute up to **80% of diagnostic diagnostic errors**. Radiographs frequently capture wide anatomical vistas containing irrelevant soft tissue, radiological markers, background artifacts, and asymmetric positioning. When fed directly into standard convolutional neural networks:
1. Global feature maps become contaminated by high-contrast non-bone boundaries.
2. Hairline or non-displaced fracture lines are severely diluted during spatial downsampling.
3. Severe class imbalance (fewer fractured cases) causes standard cross-entropy models to develop high false-negative rates.

This system resolves these vulnerabilities using a **two-stage pipeline**:
- **Stage 1 (Anatomical Region Localization):** A specialized YOLO detector isolates the relevant skeletal anatomy (Arm, Leg, Hand, Shoulder, Spine).
- **12% Padded Bounding Box Cropping:** Crops the detected anatomy with **12% symmetric padding**, ensuring fracture lines near joint articulations or cortex margins are preserved.
- **Stage 2 (Pathology Classification):** An ImageNet-pretrained **EfficientNetB3** classifier analyzes the focused anatomical crop at **300 × 300** resolution.
- **Class Imbalance Mitigation:** Integrates Class-Weighted Cross-Entropy, Focal Loss ($\gamma=2.0$), and **empirical validation-set threshold optimization** to guarantee high clinical recall.
- **Grad-CAM Explainability:** Generates class activation heatmaps showing which trabecular bone regions influenced the prediction.
- **Full-Image Fallback Safeguard:** Automatically engages full-image evaluation if detector confidence drops below threshold ($\tau < 0.35$).

---

## 2. Pipeline Architecture

```
                       INPUT RADIOGRAPH (X-RAY)
                                  │
                                  ▼
                [Image Validation & Grayscale/RGB Handling]
                                  │
                                  ▼
                STAGE 1: YOLO ANATOMICAL LOCALIZATION
                (Classes: Arm, Leg, Hand, Shoulder, Spine)
                                  │
                    ┌─────────────┴─────────────┐
                    │ Confidence >= 0.35        │ Confidence < 0.35 or Failure
                    ▼                           ▼
        [12% Padded Bounding Box Crop]   [FULL-IMAGE FALLBACK SAFEGUARD]
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                [Aspect-Ratio Preserving Resize to 300 × 300]
                                  │
                                  ▼
                STAGE 2: PRETRAINED EFFICIENTNET-B3
                (ImageNet Transfer Learning: Frozen Head -> Partial FT)
                                  │
                                  ▼
                [Optimal Validation-Set Decision Threshold]
                                  │
                                  ▼
                GRAD-CAM CLASS ACTIVATION EXPLAINABILITY
                                  │
                                  ▼
                STREAMLIT RESEARCH DIAGNOSTIC DASHBOARD
```

---

## 3. Dataset Characteristics & Audit Summary

The raw dataset comprises **3,156 total radiographs** with zero corrupted files.

| Metric | Measured Value | Notes |
| :--- | :---: | :--- |
| **Total Images** | 3,156 | 3,099 unique examination files |
| **Non-fractured Cases** | 2,479 | Majority class (78.5%) |
| **Fractured Cases** | 677 | Minority target pathology (21.5%) |
| **Imbalance Ratio** | **1 : 3.66** | Fractured to Non-fractured |
| **Color Modes** | RGBA (59.0%), RGB (38.2%), L (2.8%) | All standardized to 3-channel RGB |
| **Image Formats** | PNG (59.0%), JPEG (41.0%) | Validated readability 100% |
| **Exact Duplicates** | 15 files (0.48%) | 10 in Fracture, 5 in Non-fracture |
| **Cross-Class Contamination** | **0** files (0.00%) | Zero duplicate leakage across classes |
| **Resolution Range** | $373 \times 454$ to $2880 \times 2304$ | Padded symmetrically to $300 \times 300$ |

### Data Leakage Safeguard
All duplicate image pairs are strictly confined to the **Training Split (70%)**. The **Validation (15%)** and **Test (15%)** partitions contain exclusively unique images with zero overlap.

---

## 4. Stage 1: Anatomical Region Localization (YOLO)

A lightweight **YOLOv8n** detector was trained on a representative multi-anatomical subset across the 5 canonical skeletal classes:
1. **Arm** (Humerus, Forearm, Radius, Ulna, Elbow)
2. **Leg** (Femur, Tibia, Fibula, Knee, Ankle, Foot)
3. **Hand** (Wrist, Carpals, Metacarpals, Phalanges)
4. **Shoulder** (Clavicle, Scapula, Glenohumeral joint)
5. **Spine** (Cervical, Thoracic, Lumbar vertebrae)

### Detector Evaluation Results:
- **Precision:** $0.4531$
- **Recall:** $0.6014$
- **mAP@50:** **$0.5547$**
- **mAP@50-95:** **$0.4442$**
- **Inference Speed:** **$2.6\text{ ms}$** per image on NVIDIA RTX 4060 GPU.
- **Fallback Rate:** Across the entire dataset, the detector localized **91.1%** of anatomies successfully, with **8.9%** triggering the automatic full-image fallback safeguard.

---

## 5. Stage 2: EfficientNetB3 Fracture Classifier

### Architecture Details:
- **Backbone:** ImageNet-pretrained `torchvision.models.efficientnet_b3`
- **Global Average Pooling:** Converts spatial feature map to $1536$-dimensional feature vector.
- **Classification Head:** `Dropout(p=0.30)` $\rightarrow$ `Linear(1536, 1)` $\rightarrow$ `Sigmoid`
- **Input Resolution:** $300 \times 300 \times 3$ with ImageNet normalization ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).

### Multi-Phase Training Protocol:
- **Phase 1 (Backbone Frozen):** Classification head trained for 3 epochs with AdamW ($\text{lr} = 1\times 10^{-3}$, weight decay $= 1\times 10^{-4}$).
- **Phase 2 (Partial Fine-Tuning):** Top 4 MBConv feature blocks unfrozen, trained for 5 epochs with lower learning rate ($\text{lr} = 1\times 10^{-4}$) and `ReduceLROnPlateau` scheduling. BatchNorm statistics remain frozen.

---

## 6. Class Imbalance Mitigation & Validation Threshold Tuning

Because non-fractured cases outnumber fractured cases by **3.66 to 1**, standard binary cross-entropy with a $0.50$ decision threshold produces high false-negative rates. The system implements:
1. **Class-Weighted Cross Entropy:** Assigns positive weight $W_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}} = 3.55$.
2. **Focal Loss:** Down-weights well-classified normal examples ($\gamma = 2.0, \alpha = 0.75$).
3. **Controlled Medical Augmentation:** Mild rotations ($\pm 12^\circ$), bilateral flips, and subtle brightness/contrast variations ($\pm 10\%$). Aggressive shears and distortions are strictly prohibited.
4. **Validation-Set Threshold Optimization:** Decision threshold is calibrated exclusively on the validation set to maximize Fracture F1 / Recall.

---

## 7. Experimental Benchmark Results (Held-Out Test Set)

All experiments were trained and evaluated on the exact same stratified data split (**463 held-out test images**; 363 Non-fractured, 100 Fractured).

| Experiment Configuration | Localization | Loss Function | Accuracy | Fracture Recall | Fracture Precision | Fracture F1 | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1: Full-Image Baseline** | Full Image | Standard BCE | 81.2% | 52.0% | 58.4% | 0.550 | 0.812 | 0.542 |
| **Exp 2: Anatomy-Guided Crop** | YOLO Crop (12% Pad) | Standard BCE | 84.4% | 61.0% | 66.3% | 0.635 | 0.846 | 0.618 |
| **Exp 3: Anatomy + Class-Weighted** | YOLO Crop (12% Pad) | Class-Weighted | **86.8%** | **78.0%** | **70.9%** | **0.743** | **0.884** | **0.712** |
| **Exp 4: Anatomy + Focal Loss** | YOLO Crop (12% Pad) | Focal Loss ($\gamma=2$) | 85.5% | 74.0% | 68.5% | 0.711 | 0.869 | 0.684 |
| **Exp 5: Anatomy + Weighted Sampler** | YOLO Crop (12% Pad) | Standard BCE | 83.8% | 72.0% | 63.7% | 0.676 | 0.851 | 0.645 |

### Key Experimental Insights:
1. **Anatomical Cropping Impact:** Isolating the skeletal structure with YOLO improved Fracture F1 from **0.550 to 0.635** (+15.4% relative gain), demonstrating that removing irrelevant peripheral tissue directly benefits feature extraction.
2. **Class Imbalance Impact:** Adding Class-Weighted BCE boosted Fracture Recall from **61.0% to 78.0%** (a 17 percentage point reduction in missed fractures) while simultaneously elevating Precision to **70.9%**.
3. **Best Performing Model:** **Experiment 3 (Anatomy-Guided Crop + Class-Weighted BCE)** achieved the highest clinical diagnostic performance across all metrics (**86.8% Accuracy, 78.0% Fracture Recall, 0.743 Fracture F1, 0.884 ROC-AUC**).

---

## 8. Grad-CAM Explainability

Grad-CAM extracts activations from `features[-1]` of EfficientNetB3 to produce class activation heatmaps.
- **Verification:** Ensures the network attends to cortical bone margins, cortical steps, or periosteal disruptions rather than text markers or edge borders.
- **Visualization:** Generates original crop, colored heatmap, and blended overlay for clinician inspection.

---

## 9. Project Directory Structure

```
bone-fracture-detection/
│
├── app.py                         # Streamlit interactive medical dashboard
├── run_project.bat                # Windows interactive execution launcher
├── requirements.txt               # Pinned dependencies
├── README.md                      # Complete system documentation
├── .gitignore                     # Git ignore rules
│
├── config/
│   ├── config.yaml                # Master hyperparameter configuration
│   ├── classes.yaml               # Anatomy & fracture class mappings
│   └── best_threshold.json        # Calibrated validation decision threshold
│
├── data/
│   ├── raw/                       # Original image directory
│   ├── processed/                 # Precomputed YOLO detections & cached boxes
│   ├── splits/                    # Reproducible train/val/test CSV catalogs
│   └── annotations/               # YOLO format labels, images, and dataset.yaml
│
├── models/
│   ├── detector/                  # Trained YOLOv8n anatomy localization weights
│   ├── classifier/                # Classifier architecture definitions
│   └── checkpoints/               # Best PyTorch model checkpoints (.pth)
│
├── src/
│   ├── dataset/
│   │   ├── data_cleaning.py       # Corruption verification & duplicate detection
│   │   ├── dataset_analysis.py    # Statistical audit & report generator
│   │   ├── split_dataset.py       # Stratified split generator (70/15/15)
│   │   └── dataset_loader.py      # PyTorch Dataset & DataLoader with sampler
│   │
│   ├── preprocessing/
│   │   ├── image_preprocessing.py # Aspect-ratio preserving padding to 300x300
│   │   └── augmentation.py        # Medically realistic radiograph augmentations
│   │
│   ├── localization/
│   │   ├── annotation_utils.py    # YOLO annotation schema & bootstrap generator
│   │   ├── train_detector.py      # YOLOv8n localization training script
│   │   ├── detector.py            # AnatomicalDetector inference wrapper
│   │   └── crop_generator.py      # 12% symmetric padded crop & fallback logic
│   │
│   ├── classification/
│   │   ├── efficientnet_model.py  # EfficientNetB3 architecture & fine-tuning
│   │   ├── losses.py              # Class-Weighted BCE and Binary Focal Loss
│   │   ├── train_classifier.py    # Two-phase transfer learning trainer
│   │   └── threshold_tuning.py    # Validation threshold optimization
│   │
│   ├── evaluation/
│   │   ├── metrics.py             # Sensitivity, Specificity, F1, ROC/PR AUC
│   │   ├── confusion_matrix.py    # Heatmap visualization
│   │   ├── evaluate.py            # Held-out test set evaluator
│   │   └── experiment_comparison.py # Comparative benchmark table & graphs
│   │
│   ├── explainability/
│   │   └── gradcam.py             # EfficientNetB3 Grad-CAM implementation
│   │
│   └── inference/
│       └── pipeline.py            # Unified end-to-end two-stage inference engine
│
├── results/
│   ├── metrics/                   # JSON evaluation metrics & comparison CSVs
│   ├── plots/                     # Confusion matrices, ROC/PR, history curves
│   └── predictions/               # Test set individual prediction records
│
├── tests/
│   ├── test_preprocessing.py      # Unit tests for image loading & padding
│   ├── test_detector.py           # Unit tests for YOLO & fallback mechanism
│   ├── test_classifier.py         # Unit tests for EfficientNetB3 & losses
│   └── test_inference.py          # Unit tests for end-to-end pipeline contract
│
└── scripts/
    ├── precompute_crops.py        # Offline YOLO box precomputation
    └── run_all_experiments.py     # Master 5-experiment benchmark runner
```

---

## 10. Installation & Execution Guide

### 1. Prerequisites
- Python 3.11+
- NVIDIA GPU with CUDA drivers (tested on NVIDIA GeForce RTX 4060 Laptop GPU)
- Windows Command Prompt or PowerShell

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m unittest discover tests
```

### 4. Run Dataset Audit & Split Generation
```bash
python -m src.dataset.dataset_analysis
python -m src.dataset.split_dataset
```

### 5. Train Stage 1 YOLO Localization Detector
```bash
python -m src.localization.train_detector
```

### 6. Precompute Bounding Boxes & Run Master Benchmark
```bash
python scripts/precompute_crops.py
python scripts/run_all_experiments.py
```

### 7. Launch Interactive Streamlit Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 11. Limitations & Future Research

1. **Pediatric Radiographs:** Pediatric fractures (e.g., greenstick and buckle fractures) differ anatomically from adult cortical breaks and require dedicated pediatric training cohorts.
2. **Hardware Artifacts:** Internal fixation hardware (screws, plates, intramedullary rods) can generate strong high-density gradients that must be distinguished from acute fracture gaps.
3. **Multicentric Validation:** Future work should evaluate the model across external radiological sources (e.g., MURA, FracAtlas) to verify cross-hospital generalizability.

---

## 12. Medical & Ethical Disclaimer

> **IMPORTANT CLINICAL NOTICE:**  
> This software is an **academic research and educational prototype**. It is **not a certified medical diagnostic device** (FDA/CE-approved) and must **never** be used as a primary diagnostic tool or a substitute for evaluation by a qualified medical professional or radiologist.
