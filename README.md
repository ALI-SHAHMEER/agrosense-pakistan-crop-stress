# AgroSense — Pakistan Crop Stress Classification

**Satellite-based crop stress classification for Pakistani agriculture using Sentinel-2 and Google Earth Engine.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Data: GEE Sentinel-2](https://img.shields.io/badge/Data-GEE%20Sentinel--2-orange.svg)](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)

---

## Overview

AgroSense classifies agricultural fields in Pakistan into three crop stress states —
**Healthy**, **Stressed**, and **Diseased** — using nine spectral indices derived from
Sentinel-2 Level-2A satellite imagery collected via the Google Earth Engine (GEE) API.

| Metric | Value |
|---|---|
| Dataset size | 1,786 real Sentinel-2 observations |
| Field sites | 116 sites across 4 provinces |
| Seasons | 10 growing seasons (2015–2024) |
| Best model | Random Forest — **98.88% test accuracy**, **0.985 CV macro F1** |
| Classes | Healthy / Stressed / Diseased |

This repository is the replication package for the paper:

> **"AgroSense: An AI-Powered Satellite Crop Intelligence Platform for Pakistani Agriculture"**
> Ali Shahmeer, 2024.

---

## Repository Structure

```
agrosense-pakistan-crop-stress/
├── README.md
├── LICENSE                          # MIT
├── CITATION.cff                     # How to cite this work
├── requirements.txt
├── data/
│   ├── agrosense_crop_stress_dataset.csv   # 1,786 real S2 observations
│   └── data_dictionary.csv                 # Column definitions and units
├── src/
│   ├── feature_engineering.py       # Spectral index computation + label derivation
│   ├── train_models.py              # Train all 5 classifiers, save RF model
│   └── evaluate_models.py           # Re-evaluate + regenerate figures
├── notebooks/
│   └── agrosense_experiments.ipynb  # Full end-to-end walkthrough
├── results/
│   ├── classifier_results.csv       # Accuracy / F1 for all 5 classifiers
│   ├── confusion_matrices.png       # Normalised confusion matrices
│   ├── feature_importance.png       # RF / XGBoost / GB importance comparison
│   └── f1_comparison.png            # Accuracy vs macro F1 bar chart
└── docs/
    └── dataset_description.md       # Dataset methodology and field coverage
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/ALI-SHAHMEER/agrosense-pakistan-crop-stress.git
cd agrosense-pakistan-crop-stress

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all classifiers
python src/train_models.py

# 4. Re-generate evaluation figures
python src/evaluate_models.py
```

Or open the notebook for an interactive walkthrough:
```bash
jupyter notebook notebooks/agrosense_experiments.ipynb
```

---

## Dataset

`data/agrosense_crop_stress_dataset.csv` — **1,786 rows × 22 columns**

Real Sentinel-2 Level-2A pixel composites collected via the GEE API. Spectral
indices were computed from median-composited, cloud-masked imagery over each
growing season window (Rabi: Nov–Apr; Kharif: May–Nov).

**Features used for classification:**

| Feature | Description |
|---|---|
| `ndvi` | Normalized Difference Vegetation Index |
| `evi` | Enhanced Vegetation Index |
| `ndwi` | Normalized Difference Water Index |
| `ndre` | Normalized Difference Red Edge |
| `lai` | Leaf Area Index |
| `bsi` | Bare Soil Index |
| `ndvi_std` | Temporal NDVI standard deviation |
| `ndvi_min` | Seasonal minimum NDVI |
| `ndvi_max` | Seasonal maximum NDVI |

See [`data/data_dictionary.csv`](data/data_dictionary.csv) for full column
definitions and [`docs/dataset_description.md`](docs/dataset_description.md)
for collection methodology.

---

## Results

### Overall Classifier Comparison

| Algorithm | Test Acc (%) | Precision | Recall | F1 Macro | CV F1 Macro |
|---|---|---|---|---|---|
| **Random Forest** | **98.88** | **0.989** | **0.989** | **0.989** | **0.985** |
| Gradient Boosting | 98.60 | 0.986 | 0.986 | 0.986 | 0.984 |
| SVM (RBF) | 98.60 | 0.986 | 0.986 | 0.986 | 0.985 |
| XGBoost | 98.32 | 0.983 | 0.983 | 0.983 | 0.982 |
| k-NN | 95.81 | 0.958 | 0.958 | 0.958 | 0.956 |

80/20 stratified split (seed 42): 1,428 training / 358 test samples.
All metrics are macro-averaged. CV = 5-fold stratified cross-validation.

### Per-Class Results — Random Forest

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Diseased | 1.000 | 0.992 | 0.996 | 119 |
| Healthy | 0.983 | 0.992 | 0.988 | 120 |
| Stressed | 0.983 | 0.983 | 0.983 | 119 |

![Confusion Matrices](results/confusion_matrices.png)

![Feature Importance](results/feature_importance.png)

---

## Citation

If you use this dataset or code, please cite:

```
@software{shahmeer2024agrosense,
  author    = {Shahmeer, Ali},
  title     = {AgroSense: Pakistan Crop Stress Classification Dataset and Code},
  year      = {2024},
  url       = {https://github.com/ALI-SHAHMEER/agrosense-pakistan-crop-stress},
  license   = {MIT}
}
```

Or use the `CITATION.cff` file for automatic citation export from GitHub.

---

## License

MIT — see [LICENSE](LICENSE).
