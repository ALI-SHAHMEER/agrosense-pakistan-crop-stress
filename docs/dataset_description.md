# Dataset Description — AgroSense Pakistan Crop Stress

## Overview

`agrosense_crop_stress_dataset.csv` contains **1,786 real Sentinel-2 Level-2A
observations** collected via the Google Earth Engine (GEE) API from **116 field
sites across all four provinces of Pakistan** (Punjab, Sindh, Balochistan, KPK)
over **ten growing seasons (2015–2024)**.

## Important: Label Circularity

The `crop_stress_label` column is derived **from the same spectral indices
(NDVI, NDRE, EVI) that are used as model input features**. This means:

- Classifiers will partly learn the deterministic rule that created the labels.
- Reported accuracy is an **upper bound for spectral separability**, not an
  independently validated disease-detection benchmark.
- Results should be described as *spectral proxy classification*, not
  *crop disease detection*.

The correct framing: "classifies Sentinel-2 observations into spectrally derived
crop-condition proxy classes (Healthy, Stressed, Diseased). These labels are
generated from a vegetation-health composite and have not yet been validated
against independent field diagnoses."

## Collection Methodology

Satellite data was retrieved using the `COPERNICUS/S2_SR_HARMONIZED` image
collection through the GEE Python API (`src/collect_gee_data.py`).

For each site and season window:
1. Images are filtered by season date range (Rabi: Nov 15 – Apr 30; Kharif: May 1 – Nov 14)
2. Images with >30% cloud cover are discarded (CLOUDY_PIXEL_PERCENTAGE filter)
3. Per-image cloud masking via QA60 bit flags (opaque clouds bit 10, cirrus bit 11)
4. A **median composite** is computed over all cloud-free images
5. Spectral indices are derived from the composite bands:
   - NDVI = (B8−B4) / (B8+B4)
   - EVI = 2.5 × (B8−B4) / (B8+6·B4−7.5·B2+1)
   - NDWI = (B3−B8) / (B3+B8)
   - **NDRE = (B8A−B5) / (B8A+B5)** using native S2 red-edge bands B5 (705 nm) and B8A (865 nm)
   - LAI = 3.618 × EVI − 0.118
   - BSI = ((B11+B4)−(B8+B2)) / ((B11+B4)+(B8+B2))
6. Temporal NDVI statistics (std, min, max) computed across all composite images
7. Pixel values extracted via `reduceRegion` with a **1 km buffer** around each district centroid, scale=10 m

## Province and Crop Coverage

*Generated directly from `agrosense_crop_stress_dataset.csv`.*

| Province    | Sites | Crops | Observations |
|-------------|------:|-------|-------------:|
| Punjab      | 34    | wheat, cotton, rice, sugarcane | 619 |
| Balochistan | 22    | wheat, mango, cotton | 396 |
| KPK         | 23    | wheat, rice, sugarcane, mango | 411 |
| Sindh       | 20    | cotton, wheat, sugarcane, rice | 360 |

| Crop      | Observations | % |
|-----------|-------------:|---|
| Wheat     | 756 | 42.3% |
| Rice      | 308 | 17.2% |
| Cotton    | 272 | 15.2% |
| Mango     | 217 | 12.1% |
| Sugarcane | 233 | 13.1% |

## Label Derivation

The `crop_stress_label` column is derived from a vegetation health composite:

```
score = 0.5 × NDVI + 0.3 × NDRE + 0.2 × EVI
```

Tertile thresholds split the score distribution into three classes:

| Label    | Score range   | Agronomic interpretation           |
|----------|---------------|------------------------------------|
| Healthy  | top tertile   | Strong photosynthetic activity     |
| Stressed | middle tertile| Reduced vigour / moisture stress   |
| Diseased | bottom tertile| Severe chlorophyll loss            |

**⚠ These labels are spectrally derived, not ground-truthed.**  
Ground-truth validation is an active area of ongoing work.

## Class Balance

| Class    | Count | Fraction |
|----------|------:|---------:|
| Healthy  | 596   | 33.4%    |
| Stressed | 595   | 33.3%    |
| Diseased | 595   | 33.3%    |
| **Total**| **1,786** | |

## Train / Test Split

The **correct** split for this dataset is **field-grouped**:

```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=df["location_id"]))
```

This ensures **no field site appears in both training and test sets**, which would
otherwise cause optimistic accuracy estimates from seasonal observations of the
same location. Observations of the same site across different seasons are
correlated, so a random split leaks this information.

**Do not use `train_test_split` without groups on this dataset.**

Cross-validation should use `StratifiedGroupKFold`, not plain `StratifiedKFold`.

## Feature Columns

The nine features used for classification (six derived indices + three temporal statistics):

```
ndvi, evi, ndwi, ndre, lai, bsi, ndvi_std, ndvi_min, ndvi_max
```

See `data/data_dictionary.csv` for full column definitions.

## Historical Archive

A broader 4,730-sample archive spanning Landsat 7 (2000–2012), Landsat 8
(2013–2014), and Sentinel-2 (2015–2024) is available on request for long-term
temporal trend analysis.

## Citation

If you use this dataset, please cite using `CITATION.cff` at the repository root.
