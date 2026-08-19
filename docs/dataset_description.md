# Dataset Description — AgroSense Pakistan Crop-Condition Proxy Classification

## Overview

`data/agrosense_crop_stress_dataset.csv` contains **1,786 Sentinel-2
Level-2A observations** collected via the Google Earth Engine (GEE) API from
**99 district/city-centroid sampling locations** — ~1 km buffer around each
administrative centroid — across all four provinces of Pakistan
(Punjab, Sindh, Balochistan, KPK), spanning **19 crop-season windows**
(Kharif 2015–2024 and Rabi 2016–2024).

### Important: sampling location definition

Coordinates are **publicly available district/city centroids**, not
individually surveyed or GPS-verified farm field polygons.
Results represent location-grouped performance across administrative-unit
centroid buffers, not individual farm-field generalisation.

**No cropland mask** is applied during extraction. The 1 km buffer may include
mixed land-cover pixels (roads, built-up areas, bare soil, water bodies, or
non-agricultural vegetation) that contribute to the extracted spectral values.
This is a known limitation of the current sampling methodology.

**`crop_type` is a nominal primary-crop category** assigned to each sampling
location for contextual analysis. Crop identity was **not independently
verified for each location-season observation**; it is a static label per
location derived from dominant regional cropping patterns.

---

## ⚠ Label Circularity

The `crop_stress_label` column is derived **from the same spectral indices
(NDVI, NDRE, EVI) that are used as model input features**. This means:

- Classifiers partly learn the deterministic rule that created the labels.
- Reported accuracy is an **upper bound on spectral proxy separability**,
  not independently validated disease detection accuracy.
- "Diseased" identifies the lowest vegetation-health tertile spectrally.
  It is a **proxy class name**, not a pathology-confirmed diagnosis.

Use language like **"spectrally derived crop-condition proxy class"**, not
"disease detection" or "field-verified stress classification".

---

## Collection Methodology

Satellite data was retrieved using `COPERNICUS/S2_SR_HARMONIZED` via the
GEE Python API (`src/collect_gee_data.py`). For each location and season:

1. Filter by season date range:
   - **Rabi**: November 15 (previous year) – April 30
   - **Kharif**: May 1 – November 14
2. Discard images with >30% cloud cover (`CLOUDY_PIXEL_PERCENTAGE`)
3. Per-image cloud masking via QA60 bit 10 (opaque) and bit 11 (cirrus)
4. Compute **median composite** over all cloud-free images
5. Derive spectral indices (see below) from composite bands
6. Extract pixel values via `reduceRegion`, scale=10 m, 1 km buffer

Spectral indices derived **in GEE** using native Sentinel-2 bands:

| Index | Formula | Bands used |
|---|---|---|
| NDVI | (B8−B4)/(B8+B4) | B4 (Red), B8 (NIR broad) |
| EVI | 2.5·(B8−B4)/(B8+6·B4−7.5·B2+1) | B2, B4, B8 |
| NDWI | (B3−B8)/(B3+B8) | B3 (Green), B8 |
| **NDRE** | **(B8A−B5)/(B8A+B5)** | **B5 (705 nm), B8A (865 nm)** |
| LAI | 3.618·EVI − 0.118 | — |
| BSI | ((B11+B4)−(B8+B2))/((B11+B4)+(B8+B2)) | B2, B4, B8, B11 |

> NDRE uses the native S2 red-edge bands B5 (705 nm) and B8A (865 nm).
> The local `compute_indices()` function in `feature_engineering.py` was
> corrected to require these bands and raise an error if absent.

---

## Derived / Imputed Columns

Three columns in the published CSV are **imputed**, not observed:

| Column | Method | Note |
|---|---|---|
| `soil_moisture` | Linear rescaling of NDWI to [16, 28]% | Proxy, not in-situ |
| `temp_celsius` | Climatology default ± noise: rabi=19°C, kharif=30°C | Imputed |
| `rainfall_mm` | Climatology default ± noise: rabi=120mm, kharif=280mm | Imputed |

Exact imputation rules are in `src/build_dataset.py`.

---

## Province and Crop Coverage *(derived from CSV)*

| Province    | Locations | Observations | Crops |
|-------------|----------:|-------------:|-------|
| Punjab      | 34        | 619          | wheat, cotton, rice, sugarcane |
| KPK         | 23        | 411          | wheat, rice, sugarcane, mango |
| Balochistan | 22        | 396          | wheat, mango, cotton |
| Sindh       | 20        | 360          | cotton, wheat, sugarcane, rice |

| Crop      | Count | % |
|-----------|------:|---|
| wheat     | 883   | 49.4% |
| rice      | 308   | 17.2% |
| cotton    | 272   | 15.2% |
| sugarcane | 216   | 12.1% |
| mango     | 107   | 6.0% |

### Hafizabad note

`PB15 Hafizabad` (lat=32.0711, lon=73.6883, crop=rice) and
`PB34 Hafizabad2` (lat=32.0711, lon=73.7500, crop=wheat) share the same
latitude but differ by ~6.8 km longitude and represent different crops.
They are retained as distinct sampling locations with different seasonal
spectral signatures.

---

## Label Derivation

```
score = 0.5 × NDVI + 0.3 × NDRE + 0.2 × EVI
```

Tertile thresholds derived from the **full released dataset** (not from training data):

| Proxy class | Score range | Spectral interpretation |
|---|---|---|
| Healthy | top tertile (≥ q₆₆ = 0.1915) | Strong photosynthetic activity |
| Stressed | middle tertile | Reduced vegetation vigour |
| Diseased | bottom tertile (< q₃₃ = 0.1403) | Severe chlorophyll reduction |

**Class distribution:** Healthy=596, Stressed=595, Diseased=595 (≈balanced)

Exact threshold values are stored in `data/label_definition.json`.

**⚠ Threshold leakage note:** Because q₃₃ and q₆₆ were computed from the full
dataset before any train/test split, the global thresholds encode information
about the held-out partition. This is a methodological limitation of the
released benchmark labels. For fair inductive evaluation, use the
training-derived threshold baseline in `src/rule_baseline.py`, which recomputes
thresholds from training data only and achieves held-out macro F1 = 0.929
(nested CV = 0.964 ± 0.011).

---

## Classification Features (9 total)

```
ndvi, evi, ndwi, ndre, lai, bsi, ndvi_std, ndvi_min, ndvi_max
```

Six spectral indices + three temporal NDVI statistics.

---

## Train/Test Split (location-grouped)

```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=df["location_id"]))
# → 1,425 train obs / 79 locations | 361 test obs / 20 locations
```

See `results/split_manifest.csv` for the exact location-to-partition mapping.

**Do not use `train_test_split` without groups** — seasonal observations
from the same centroid are correlated; a random split leaks this information.

---

## Seasons

19 unique season labels:
- **Kharif** (May–Nov): 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024
- **Rabi** (Nov–Apr): 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024

Observations span 2015–2024 across both growing seasons,
not "ten growing seasons" as sometimes stated (there are 19 season windows).

---

## Citation

See `CITATION.cff` at the repository root.
