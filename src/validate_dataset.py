"""
validate_dataset.py
Programmatic audit of agrosense_crop_stress_dataset.csv.
Fails with a non-zero exit code if any critical assertion fails.

Usage:
    python src/validate_dataset.py
"""

import sys
import pandas as pd

DATA_PATH = "data/agrosense_crop_stress_dataset.csv"
FEATURE_COLS = ["ndvi", "evi", "ndwi", "ndre", "lai", "bsi",
                "ndvi_std", "ndvi_min", "ndvi_max"]
EXPECTED_CLASSES = {"Healthy", "Stressed", "Diseased"}
EXPECTED_PROVINCES = {"Punjab", "Sindh", "Balochistan", "KPK"}
EXPECTED_CROPS = {"wheat", "rice", "cotton", "sugarcane", "mango"}

errors = []

def check(condition, msg):
    if not condition:
        errors.append(f"  FAIL: {msg}")
    else:
        print(f"  OK  : {msg}")


def main():
    print(f"\nLoading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    print("\n=== Shape ===")
    check(len(df) == 1786, f"row count == 1786 (got {len(df)})")
    check(len(df.columns) == 21, f"column count == 21 (got {len(df.columns)})")

    print("\n=== No missing values ===")
    nulls = df.isnull().sum().sum()
    check(nulls == 0, f"zero null values (got {nulls})")

    print("\n=== No exact duplicates ===")
    dupes = df.duplicated().sum()
    check(dupes == 0, f"zero duplicate rows (got {dupes})")

    print("\n=== Unique sampling locations ===")
    n_locs = df["location_id"].nunique()
    check(n_locs == 99, f"99 unique location_ids (got {n_locs})")

    # Province-level location counts must sum to total
    per_prov = df.groupby("province")["location_id"].nunique()
    sum_prov = per_prov.sum()
    check(sum_prov == n_locs,
          f"province location counts sum to {n_locs} (got {sum_prov})")

    print("\n=== Provinces ===")
    check(set(df["province"].unique()) == EXPECTED_PROVINCES,
          f"provinces == {EXPECTED_PROVINCES}")
    print(f"       Punjab={per_prov.get('Punjab',0)}, "
          f"KPK={per_prov.get('KPK',0)}, "
          f"Balochistan={per_prov.get('Balochistan',0)}, "
          f"Sindh={per_prov.get('Sindh',0)}")

    print("\n=== Crops ===")
    check(set(df["crop_type"].unique()) == EXPECTED_CROPS,
          f"crop types == {EXPECTED_CROPS}")

    print("\n=== Class distribution ===")
    class_counts = df["crop_stress_label"].value_counts()
    check(set(class_counts.index) == EXPECTED_CLASSES,
          f"classes == {EXPECTED_CLASSES}")
    check(class_counts.sum() == len(df),
          f"class counts sum to {len(df)}")
    for cls in EXPECTED_CLASSES:
        print(f"       {cls}: {class_counts.get(cls, 0)}")

    print("\n=== Seasons ===")
    seasons = df["season"].unique()
    check(len(seasons) == 19,
          f"19 unique seasons (got {len(seasons)})")
    kharif = [s for s in seasons if "Kharif" in s]
    rabi   = [s for s in seasons if "Rabi" in s]
    print(f"       Kharif seasons: {len(kharif)}, Rabi seasons: {len(rabi)}")

    print("\n=== Feature columns ===")
    for col in FEATURE_COLS:
        check(col in df.columns, f"feature column '{col}' present")
    check(len(FEATURE_COLS) == 9, f"exactly 9 feature columns defined")

    print("\n=== Observations per site ===")
    obs = df.groupby("location_id").size()
    check(obs.min() >= 17, f"min obs/site >= 17 (got {obs.min()})")
    check(obs.max() <= 19, f"max obs/site <= 19 (got {obs.max()})")

    # Numeric sanity checks
    print("\n=== Numeric ranges ===")
    for col in ["ndvi", "evi", "ndwi", "ndre"]:
        mn, mx = df[col].min(), df[col].max()
        check(-1.5 <= mn and mx <= 1.5,
              f"{col} in plausible range [-1.5, 1.5] (min={mn:.3f}, max={mx:.3f})")

    print("\n" + "="*50)
    if errors:
        print(f"VALIDATION FAILED — {len(errors)} error(s):")
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}, "
              f"Locations: {n_locs}, Seasons: {len(seasons)}")
        crop_pct = (df['crop_type'].value_counts() / len(df) * 100).round(1)
        print("  Crop distribution:")
        for c, p in crop_pct.items():
            print(f"    {c}: {df['crop_type'].value_counts()[c]} ({p}%)")


if __name__ == "__main__":
    main()
