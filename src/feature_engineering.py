"""
feature_engineering.py
Compute Sentinel-2 spectral indices from raw band reflectances.

Required input bands (surface reflectance, scale 0–1):
  B2  Blue      B3  Green     B4  Red
  B5  Red-Edge  B8  NIR       B8A Narrow NIR / Red-Edge 4
  B11 SWIR-1

NOTE on NDRE: The standard formulation is (B8A - B5) / (B8A + B5).
B5 (705 nm) is the first Sentinel-2 red-edge band; B8A (865 nm) is the
narrow-NIR band.  These are DIFFERENT from B4 (Red) and B8 (Broad NIR).
Do not substitute B4 for B5 — that would replicate NDVI, not NDRE.
"""

import numpy as np
import pandas as pd


def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add spectral indices to a dataframe containing raw Sentinel-2 band columns.

    Required columns: b2, b3, b4, b5, b8, b8a, b11
    Raises KeyError if b5 or b8a are absent (silent fallback is disabled to
    prevent accidentally computing NDRE as NDVI).
    """
    required = {"b2", "b3", "b4", "b5", "b8", "b8a", "b11"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"compute_indices() is missing required band columns: {sorted(missing)}. "
            "Ensure your GEE extraction includes B5 and B8A for NDRE."
        )

    b4  = df["b4"]    # Red (665 nm)
    b3  = df["b3"]    # Green (560 nm)
    b2  = df["b2"]    # Blue (490 nm)
    b5  = df["b5"]    # Red-Edge 1 (705 nm)
    b8  = df["b8"]    # NIR Broad (842 nm)
    b8a = df["b8a"]   # NIR Narrow / Red-Edge 4 (865 nm)
    b11 = df["b11"]   # SWIR-1 (1610 nm)

    eps = 1e-9
    df = df.copy()
    df["ndvi"]  = (b8  - b4)  / (b8  + b4  + eps)
    df["evi"]   = 2.5 * (b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1 + eps)
    df["ndwi"]  = (b3  - b8)  / (b3  + b8  + eps)
    df["ndre"]  = (b8a - b5)  / (b8a + b5  + eps)   # correct: B8A / B5
    df["lai"]   = 3.618 * df["evi"] - 0.118
    df["bsi"]   = ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2) + eps)
    return df


def derive_stress_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Derive spectrally-based crop-condition proxy labels via tertile split on a
    vegetation health composite (0.5·NDVI + 0.3·NDRE + 0.2·EVI).

    IMPORTANT — label circularity:
    These labels are derived from the same spectral indices (NDVI, NDRE, EVI)
    that are used as model features.  Classifiers trained on these labels will
    partly learn the deterministic rule that created them, which inflates
    accuracy relative to independently validated disease labels.
    Treat reported accuracy as an upper bound for spectral separability, not
    as a validated disease-detection benchmark.

    Returns: array of strings — 'Healthy', 'Stressed', or 'Diseased'
    """
    score = 0.5 * df["ndvi"] + 0.3 * df["ndre"] + 0.2 * df["evi"]
    q33, q66 = score.quantile([1 / 3, 2 / 3])
    return np.where(score >= q66, "Healthy",
           np.where(score >= q33, "Stressed", "Diseased"))


# Nine spectral features used for classification:
# six derived indices + three temporal NDVI statistics
FEATURE_COLS = ["ndvi", "evi", "ndwi", "ndre", "lai", "bsi",
                "ndvi_std", "ndvi_min", "ndvi_max"]
