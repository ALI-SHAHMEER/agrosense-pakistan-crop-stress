# AgroSense — Pakistan Crop-Condition Proxy Classification

**Sentinel-2 spectral proxy classification for Pakistani agricultural
sampling locations using Google Earth Engine and scikit-learn.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg)](tests/)

---

## ⚠ Label-Circularity Disclosure

Class labels (Healthy / Stressed / Diseased) are derived from a vegetation health
composite **0.5·NDVI + 0.3·NDRE + 0.2·EVI** via tertile split.  
**NDVI, NDRE, and EVI are also model input features.**

This means classifiers partly learn the deterministic rule that created the labels.
Reported performance is an **upper bound on spectral proxy separability**, not
independently validated crop disease detection.

The "Diseased" label identifies the lowest vegetation-health tertile spectrally;
it does **not** indicate a pathology-confirmed diagnosis.

---

## Overview

AgroSense classifies 1,786 Sentinel-2 observations from **99 georeferenced
agricultural sampling locations** (district/city-centroid buffers) across all four
provinces of Pakistan into three spectrally derived crop-condition proxy classes:
**Healthy**, **Stressed**, and **Diseased**.

| Fact | Value |
|---|---|
| Observations | 1,786 |
| Columns | 21 |
| Sampling locations | 99 (district/city-centroid buffers, ~1 km radius) |
| Provinces | Punjab (34), KPK (23), Balochistan (22), Sindh (20) |
| Seasons | 19 (Kharif 2015–2024, Rabi 2016–2024) |
| Classes | Healthy 596 / Stressed 595 / Diseased 595 |
| Best held-out test accuracy | SVM (RBF) — **98.34%** |
| Best nested CV macro F1 | XGBoost — **0.983 ± 0.006** |
| Evaluation | GroupShuffleSplit + nested StratifiedGroupKFold CV |
| Feature scaling | StandardScaler in Pipeline (SVM, k-NN) |

This repository is the replication package for:

> **"AgroSense: Satellite-Based Crop Stress Classification and Smart Farming
> Decision Support for Pakistan Smallholder Agriculture"**  
> Ali Shahmeer, Basit Hassan, Kabir Ghoto — 2026.

---

## Repository Structure

```
agrosense-pakistan-crop-stress/
├── README.md
├── LICENSE                              # MIT 2026
├── CITATION.cff
├── requirements.txt                     # pinned versions
├── requirements-gee.txt                 # earthengine-api (optional)
├── data/
│   ├── agrosense_crop_stress_dataset.csv   # 1,786 × 21
│   └── data_dictionary.csv
├── src/
│   ├── collect_gee_data.py      # Step 1: GEE extraction pipeline
│   ├── build_dataset.py         # Step 2: post-process GEE output → CSV
│   ├── feature_engineering.py   # Spectral index computation + label derivation
│   ├── train_models.py          # Step 3: nested CV + GridSearchCV + all outputs
│   ├── evaluate_models.py       # Reload saved models, regenerate figures
│   └── validate_dataset.py      # Dataset integrity audit
├── notebooks/
│   └── agrosense_experiments.ipynb
├── results/
│   ├── classifier_results.csv
│   ├── per_class_results.csv
│   ├── nested_cv_results.csv
│   ├── best_params.json
│   ├── experiment_meta.json
│   ├── split_manifest.csv
│   ├── feature_importance.csv
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   └── f1_comparison.png
├── models/
│   └── random_forest.pkl
├── tests/
│   └── test_repository_consistency.py
└── docs/
    └── dataset_description.md
```

---

## Quick Start

```bash
git clone https://github.com/ALI-SHAHMEER/agrosense-pakistan-crop-stress.git
cd agrosense-pakistan-crop-stress
pip install -r requirements.txt

# Validate the dataset
python src/validate_dataset.py

# Train all classifiers (nested CV, ~10 min on 8 cores)
python src/train_models.py

# Regenerate figures from saved models
python src/evaluate_models.py

# Run consistency tests
pytest tests/ -v
```

### Full pipeline from GEE (optional)

```bash
pip install -r requirements-gee.txt
earthengine authenticate
python src/collect_gee_data.py --out data/gee_collected.csv
python src/build_dataset.py --gee_csv data/gee_collected.csv
python src/train_models.py
```

---

## Dataset

`data/agrosense_crop_stress_dataset.csv` — **1,786 rows × 21 columns**

Sentinel-2 Level-2A spectral composites from
`COPERNICUS/S2_SR_HARMONIZED` via the GEE API, cloud-masked (QA60),
median-composited per season window, extracted with a 1 km buffer at
**99 district/city-centroid sampling locations** — not individually
surveyed farm fields.

**Nine classification features** (six derived indices + three temporal statistics):

| Feature | Formula / Description |
|---|---|
| `ndvi` | (B8−B4)/(B8+B4) |
| `evi` | 2.5·(B8−B4)/(B8+6·B4−7.5·B2+1) |
| `ndwi` | (B3−B8)/(B3+B8) |
| `ndre` | (B8A−B5)/(B8A+B5) — requires native S2 red-edge bands |
| `lai` | 3.618·EVI − 0.118 |
| `bsi` | ((B11+B4)−(B8+B2))/((B11+B4)+(B8+B2)) |
| `ndvi_std` | Temporal NDVI standard deviation across composite images |
| `ndvi_min` | Seasonal minimum NDVI |
| `ndvi_max` | Seasonal maximum NDVI |

> **Imputed columns:** `temp_celsius`, `rainfall_mm`, and `soil_moisture` are
> climatology-based estimates, not in-situ observations. See
> [`src/build_dataset.py`](src/build_dataset.py) for exact imputation rules.

### Province and crop coverage *(from CSV, not manually maintained)*

| Province    | Locations | Crops |
|-------------|----------:|-------|
| Punjab      | 34        | wheat, cotton, rice, sugarcane |
| KPK         | 23        | wheat, rice, sugarcane, mango |
| Balochistan | 22        | wheat, mango, cotton |
| Sindh       | 20        | cotton, wheat, sugarcane, rice |

| Crop      | Count | % |
|-----------|------:|---|
| wheat     | 883   | 49.4% |
| rice      | 308   | 17.2% |
| cotton    | 272   | 15.2% |
| sugarcane | 216   | 12.1% |
| mango     | 107   | 6.0% |

---

## Results

### Evaluation methodology

- **Split:** `GroupShuffleSplit(location_id, test_size=0.20, seed=42)`
  → 1,425 train obs (79 locations) / 361 test obs (20 locations)
- **Nested CV:** 5-fold outer `StratifiedGroupKFold` with 3-fold inner
  `GridSearchCV` for hyperparameter selection within each fold
- **Feature scaling:** `StandardScaler` inside `Pipeline` for SVM and k-NN
- **See:** `results/split_manifest.csv` for the exact location→partition mapping

### Overall classifier comparison

| Algorithm | Test Acc | Test P | Test R | Test F1 | Nested CV F1 ± std |
|---|---|---|---|---|---|
| SVM (RBF) | **0.9834** | **0.9804** | **0.9835** | **0.9819** | 0.969 ± 0.017 |
| XGBoost | 0.9751 | 0.9707 | 0.9780 | 0.9740 | **0.983 ± 0.006** |
| Gradient Boosting | 0.9723 | 0.9706 | 0.9744 | 0.9723 | 0.981 ± 0.005 |
| **Random Forest** | 0.9640 | 0.9613 | 0.9689 | 0.9646 | 0.980 ± 0.007 |
| k-NN | 0.9612 | 0.9542 | 0.9610 | 0.9569 | 0.956 ± 0.008 |

All metrics macro-averaged. Random Forest selected as production model for
interpretability. See `results/best_params.json` for winning hyperparameters.

### Per-class results — Random Forest (production model, 361 test obs)

| Proxy class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Diseased | 0.9829 | 0.9503 | 0.9663 | 181 |
| Healthy | 1.0000 | 0.9884 | 0.9942 | 86 |
| Stressed | 0.9010 | 0.9681 | 0.9333 | 94 |
| **Macro** | **0.9613** | **0.9689** | **0.9646** | **361** |

![Confusion Matrices](results/confusion_matrices.png)

![Feature Importance](results/feature_importance.png)

> **Note on feature importance:** NDVI, LAI, and NDRE dominate across tree models.
> Because NDVI, NDRE, and EVI are used in the label-generation rule, native
> impurity-based importance values are affected by this circularity and should
> not be interpreted as causal biological importance.

---

## Sampling location note

Coordinates are **district and city centroids** from publicly available
administrative data, with a 1 km GEE buffer for pixel extraction.
These are **not** individually surveyed or GPS-verified farm fields.
Location-level grouping prevents repeated seasonal observations of the
same centroid from inflating accuracy estimates.

`PB15 Hafizabad` and `PB34 Hafizabad2` share the same latitude but differ
by ~6.8 km in longitude and represent different crops (rice vs wheat);
they are treated as distinct sampling locations.

---

## Citation

```bibtex
@software{shahmeer2026agrosense,
  author  = {Shahmeer, Ali and Hassan, Basit and Ghoto, Kabir},
  title   = {AgroSense: Pakistan Crop-Condition Proxy Classification},
  year    = {2026},
  url     = {https://github.com/ALI-SHAHMEER/agrosense-pakistan-crop-stress},
  license = {MIT}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
