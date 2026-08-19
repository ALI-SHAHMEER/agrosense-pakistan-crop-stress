# Dataset Description — AgroSense Pakistan Crop Stress

## Overview

`agrosense_crop_stress_dataset.csv` contains **1,786 real Sentinel-2 Level-2A
observations** collected via the Google Earth Engine (GEE) API from **116 field
sites across all four provinces of Pakistan** (Punjab, Sindh, Balochistan, KPK)
over **ten growing seasons (2015–2024)**.

## Collection Methodology

Satellite data was retrieved using the `COPERNICUS/S2_SR_HARMONIZED` image
collection through the GEE Python API. For each site and season window:

1. Images are filtered by the season date range (Rabi: Nov 15 – Apr 30; Kharif: May 1 – Nov 14)
2. Cloud masking is applied using the QA60 band (cloud and cirrus bit flags)
3. A median composite is computed over all passing images
4. Spectral indices (NDVI, EVI, NDWI, NDRE, LAI, BSI) are derived from the composite bands
5. Pixel values are extracted via `reduceRegion` with a 1 km buffer around each site centroid

## Province and Crop Coverage

| Province    | Sites | Crops                              |
|-------------|-------|------------------------------------|
| Punjab      | 52    | wheat, rice, sugarcane             |
| Sindh       | 30    | rice, cotton, sugarcane, mango     |
| Balochistan | 20    | wheat, mango                       |
| KPK         | 14    | wheat, rice                        |

## Label Derivation

The `crop_stress_label` column is derived from a vegetation health composite:

```
score = 0.5 × NDVI + 0.3 × NDRE + 0.2 × EVI
```

Tertile thresholds split the score distribution into three classes:

| Label    | Score range      | Agronomic interpretation        |
|----------|------------------|---------------------------------|
| Healthy  | top tertile      | Strong photosynthetic activity  |
| Stressed | middle tertile   | Reduced vigour / moisture stress|
| Diseased | bottom tertile   | Severe chlorophyll loss         |

**Note:** Labels are spectrally derived, not ground-truthed. They provide a
reproducible benchmark reference. Ground-truth validation is an active area of
ongoing work.

## Class Balance

| Class    | Count | Fraction |
|----------|-------|----------|
| Healthy  | 596   | 33.4%    |
| Stressed | 595   | 33.3%    |
| Diseased | 595   | 33.3%    |
| **Total**| **1,786** | |

## Feature Columns

See `data/data_dictionary.csv` for full column definitions and units.

The nine features used for classification:

```
ndvi, evi, ndwi, ndre, lai, bsi, ndvi_std, ndvi_min, ndvi_max
```

## Historical Archive

A broader 4,730-sample archive spanning Landsat 7 (2000–2012), Landsat 8
(2013–2014), and Sentinel-2 (2015–2024) is available on request and supports
long-term temporal trend analysis.

## Citation

If you use this dataset, please cite using `CITATION.cff` at the repository root.
