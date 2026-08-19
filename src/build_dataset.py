"""
build_dataset.py
Post-process raw GEE output into the published 21-column CSV.

Workflow
--------
    collect_gee_data.py  →  data/gee_collected.csv
    build_dataset.py     →  data/agrosense_crop_stress_dataset.csv

Derived columns and their exact rules
--------------------------------------
soil_moisture
    Linearly rescaled from NDWI to the range [16, 28] %.
    Formula:
        ndwi_min = NDWI.min()
        ndwi_max = NDWI.max()
        soil_moisture = 16 + (NDWI - ndwi_min) / (ndwi_max - ndwi_min) * 12
    Rationale: NDWI is a proxy for canopy / surface water; this linear
    mapping produces a plausible soil-moisture range for Pakistani cropland.
    These are IMPUTED values, not in-situ measurements.

temp_celsius
    Season-type imputed from Open-Meteo Pakistani climatology:
        rabi   season → 19 °C + Gaussian noise N(0, 2)
        kharif season → 30 °C + Gaussian noise N(0, 2)
    Clipped to [5, 45] °C.
    These are IMPUTED values, not observed measurements.

rainfall_mm
    Season-type imputed from CHIRPS climatology:
        rabi   season → 120 mm + Gaussian noise N(0, 20)
        kharif season → 280 mm + Gaussian noise N(0, 30)
    Clipped to [10, 600] mm.
    These are IMPUTED values, not observed measurements.

crop_stress_label
    Tertile split on:   score = 0.5*NDVI + 0.3*NDRE + 0.2*EVI
    Top tertile    → 'Healthy'
    Middle tertile → 'Stressed'
    Bottom tertile → 'Diseased'
    IMPORTANT: NDVI, NDRE and EVI are ALSO classification features.
    Labels are spectrally derived proxy classes, not ground-truth diagnoses.

Usage
-----
    python src/build_dataset.py \\
        --gee_csv  data/gee_collected.csv \\
        --out_csv  data/agrosense_crop_stress_dataset.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42

KEEP_COLS = [
    "location_id", "location_name", "province", "season", "season_type",
    "crop_type", "lat", "lon",
    "ndvi", "evi", "ndwi", "ndre", "lai", "bsi",
    "ndvi_std", "ndvi_min", "ndvi_max",
    "temp_celsius", "rainfall_mm", "soil_moisture",
    "crop_stress_label",
]


def impute_soil_moisture(df: pd.DataFrame) -> pd.Series:
    ndwi = df["ndwi"]
    mn, mx = ndwi.min(), ndwi.max()
    return (16 + (ndwi - mn) / (mx - mn + 1e-9) * 12).round(2)


def impute_temp(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    temp = np.where(
        df["season_type"] == "rabi",
        19 + rng.normal(0, 2, len(df)),
        30 + rng.normal(0, 2, len(df)),
    )
    return pd.Series(np.clip(temp, 5, 45), index=df.index).round(1)


def impute_rainfall(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    rain = np.where(
        df["season_type"] == "rabi",
        120 + rng.normal(0, 20, len(df)),
        280 + rng.normal(0, 30, len(df)),
    )
    return pd.Series(np.clip(rain, 10, 600), index=df.index).round(1)


def derive_stress_label(df: pd.DataFrame) -> pd.Series:
    score = 0.5 * df["ndvi"] + 0.3 * df["ndre"] + 0.2 * df["evi"]
    q33, q66 = score.quantile([1/3, 2/3])
    return pd.Series(
        np.where(score >= q66, "Healthy",
        np.where(score >= q33, "Stressed", "Diseased")),
        index=df.index,
    )


def build(gee_csv: str, out_csv: str) -> pd.DataFrame:
    print(f"Reading GEE output: {gee_csv}")
    df = pd.read_csv(gee_csv)
    print(f"  Raw rows: {len(df)}")

    # Drop rows with nulls in spectral columns
    spec_cols = ["ndvi", "evi", "ndwi", "ndre", "lai", "bsi",
                 "ndvi_std", "ndvi_min", "ndvi_max"]
    before = len(df)
    df = df.dropna(subset=spec_cols).reset_index(drop=True)
    print(f"  After dropping null spectral rows: {len(df)} (dropped {before-len(df)})")

    rng = np.random.default_rng(RANDOM_SEED)
    df["soil_moisture"]  = impute_soil_moisture(df)
    df["temp_celsius"]   = impute_temp(df, rng)
    df["rainfall_mm"]    = impute_rainfall(df, rng)
    df["crop_stress_label"] = derive_stress_label(df)

    # Keep only published columns
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns from GEE output: {missing}")

    df = df[KEEP_COLS]
    df.to_csv(out_csv, index=False)
    print(f"  Saved {len(df)} rows × {len(df.columns)} columns → {out_csv}")
    print(f"\n  ⚠  temp_celsius, rainfall_mm, and soil_moisture are IMPUTED.")
    print(f"  ⚠  crop_stress_label is a SPECTRAL PROXY, not ground truth.")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build published dataset from raw GEE output")
    parser.add_argument("--gee_csv", default="data/gee_collected.csv")
    parser.add_argument("--out_csv",
                        default="data/agrosense_crop_stress_dataset.csv")
    args = parser.parse_args()
    build(args.gee_csv, args.out_csv)
