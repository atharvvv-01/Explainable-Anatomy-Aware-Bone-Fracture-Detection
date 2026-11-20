# Model Experiment Comparison on Test Partition

| Experiment | Architecture | Loss Function | Train Acc (%) | Val Acc (%) | Test Acc (%) | Fracture Recall (%) | Fracture Precision (%) | Fracture F1 | Specificity (%) | ROC-AUC | PR-AUC | Optimal Threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Multi-Model Fusion Ensemble (Best Overall)** | EfficientNet-B3 (Full + Anatomy Crop Fusion) | Multi-Loss + TTA | **97.79** | **95.03** | **95.03** | 84.0 | 92.31 | 0.8796 | 98.07 | 0.9582 | 0.9124 | 0.38 |
| Dual-Stream Anatomy-Aware EffNet | EfficientNet-B3 (Dual Branch) | Label-Smooth BCE + Aux | **98.34** | **89.42** | **88.55** | 61.0 | 81.33 | 0.6971 | 96.14 | 0.8833 | 0.8032 | 0.49 |
| Dual-Stream Ensemble (Full+Crop) | EfficientNet-B3 (Linear Dual) | Class-Weighted + TTA | **96.50** | **89.85** | **89.63** | 63.0 | 85.14 | 0.7241 | 96.97 | 0.9143 | 0.8154 | 0.46 |
| Enhanced Full-Image (12 Epochs) | EfficientNet-B3 (Full Image) | Class-Weighted + TTA | **95.80** | **88.34** | **88.55** | 66.0 | 77.65 | 0.7135 | 94.77 | 0.9188 | 0.8176 | 0.45 |
| CLAHE Anatomy-Guided (12 Epochs) | EfficientNet-B3 (YOLO Crop) | CLAHE + Class-Weighted | **95.12** | **88.55** | **88.34** | 57.0 | 83.82 | 0.6786 | 96.97 | 0.8906 | 0.7724 | 0.49 |
| CLAHE Full-Image (12 Epochs) | EfficientNet-B3 (Full Image) | CLAHE + Class-Weighted | **95.20** | **87.69** | **87.90** | 66.0 | 75.00 | 0.7021 | 93.94 | 0.9207 | 0.8149 | 0.50 |
| Enhanced Anatomy-Guided (12 Epochs) | EfficientNet-B3 (YOLO Crop) | Class-Weighted + TTA | **94.60** | **87.47** | **87.26** | 58.0 | 77.33 | 0.6629 | 95.32 | 0.8901 | 0.7718 | 0.46 |
| Baseline Full Image (3 Epochs) | EfficientNet-B3 (Full Image) | Class-Weighted | **91.20** | **87.26** | **87.69** | 68.0 | 73.12 | 0.7047 | 93.11 | 0.8955 | 0.7921 | 0.22 |
