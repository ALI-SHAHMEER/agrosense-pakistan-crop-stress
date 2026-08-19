"""
feature_engineering.py
Compute Sentinel-2 spectral indices from raw band reflectances.

Input bands (surface reflectance, scale 0-1):
  B2  Blue    B3  Green    B4  Red
  B8  NIR     B8A Red-Edge B11 SWIR-1
"""

import numpy as np
import pandas as pd


def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Add spectral indices to a dataframe that contains raw band columns."""
    b4  = df["b4"]   # Red
    b3  = df["b3"]   # Green
    b2  = df["b2"]   # Blue
    b8  = df["b8"]   # NIR
    b8a = df.get("b8a", b8)  # Red-Edge (fallback to NIR if absent)
    b11 = df["b11"]  # SWIR-1

    eps = 1e-9

    df = df.copy()
    df["ndvi"]  = (b8  - b4)  / (b8  + b4  + eps)
    df["evi"]   = 2.5 * (b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1 + eps)
    df["ndwi"]  = (b3  - b8)  / (b3  + b8  + eps)
    df["ndre"]  = (b8a - b4)  / (b8a + b4  + eps)
    df["lai"]   = 3.618 * df["evi"] - 0.118
    df["bsi"]   = ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2) + eps)
    return df


def derive_stress_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Tertile split on vegetation health composite (NDVI + NDRE + EVI).
    Returns: 'Healthy' / 'Stressed' / 'Diseased'
    """
    score = 0.5 * df["ndvi"] + 0.3 * df["ndre"] + 0.2 * df["evi"]
    q33, q66 = score.quantile([1 / 3, 2 / 3])
    return np.where(score >= q66, "Healthy",
           np.where(score >= q33, "Stressed", "Diseased"))


FEATURE_COLS = ["ndvi", "evi", "ndwi", "ndre", "lai", "bsi",
                "ndvi_std", "ndvi_min", "ndvi_max"]
