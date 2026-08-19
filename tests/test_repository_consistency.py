"""
test_repository_consistency.py
Lightweight automated checks that key repository claims match the actual
dataset and results files.

Run with:
    pytest tests/test_repository_consistency.py -v
or:
    python tests/test_repository_consistency.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def df():
    return pd.read_csv(ROOT / "data" / "agrosense_crop_stress_dataset.csv")


@pytest.fixture(scope="module")
def results():
    p = ROOT / "results" / "classifier_results.csv"
    if not p.exists():
        pytest.skip("classifier_results.csv not found — run train_models.py")
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def manifest():
    p = ROOT / "results" / "split_manifest.csv"
    if not p.exists():
        pytest.skip("split_manifest.csv not found — run train_models.py")
    return pd.read_csv(p)


# ── Dataset tests ─────────────────────────────────────────────────────────────

def test_row_count(df):
    assert len(df) == 1786, f"Expected 1786 rows, got {len(df)}"


def test_column_count(df):
    assert len(df.columns) == 21, f"Expected 21 columns, got {len(df.columns)}"


def test_unique_locations(df):
    n = df["location_id"].nunique()
    assert n == 99, f"Expected 99 unique locations, got {n}"


def test_province_location_sum(df):
    per_prov = df.groupby("province")["location_id"].nunique()
    assert per_prov.sum() == df["location_id"].nunique()


def test_class_distribution(df):
    counts = df["crop_stress_label"].value_counts()
    assert set(counts.index) == {"Healthy", "Stressed", "Diseased"}
    assert counts.sum() == len(df)
    # Near-balanced classes
    for cls in counts.index:
        assert 580 <= counts[cls] <= 610, \
            f"Class {cls} count {counts[cls]} outside expected range [580, 610]"


def test_nine_feature_columns(df):
    feat_cols = ["ndvi", "evi", "ndwi", "ndre", "lai", "bsi",
                 "ndvi_std", "ndvi_min", "ndvi_max"]
    for col in feat_cols:
        assert col in df.columns, f"Feature column '{col}' missing from dataset"


def test_no_missing_values(df):
    assert df.isnull().sum().sum() == 0


def test_no_duplicate_rows(df):
    assert df.duplicated().sum() == 0


def test_season_count(df):
    n = df["season"].nunique()
    assert n == 19, f"Expected 19 seasons, got {n}"


def test_province_set(df):
    assert set(df["province"].unique()) == {"Punjab", "Sindh", "Balochistan", "KPK"}


def test_crop_set(df):
    assert set(df["crop_type"].unique()) == {
        "wheat", "rice", "cotton", "sugarcane", "mango"}


# ── Results tests ─────────────────────────────────────────────────────────────

def test_all_algorithms_present(results):
    expected = {"Random Forest", "XGBoost", "Gradient Boosting",
                "SVM (RBF)", "k-NN"}
    assert set(results["Algorithm"]) == expected


def test_no_null_cv_metrics(results):
    assert results["Nested_CV_F1_Macro"].isnull().sum() == 0, \
        "Nested CV F1 contains null values"


def test_accuracy_in_range(results):
    for _, row in results.iterrows():
        assert 0 <= row["Test_Accuracy"] <= 1, \
            f"{row['Algorithm']} Test_Accuracy out of [0,1]: {row['Test_Accuracy']}"


def test_f1_in_range(results):
    for col in ["Test_F1_Macro", "Nested_CV_F1_Macro"]:
        assert results[col].between(0, 1).all(), \
            f"{col} has values outside [0, 1]"


# ── Split manifest tests ──────────────────────────────────────────────────────

def test_no_location_leakage(df, manifest):
    train_locs = set(manifest[manifest["partition"] == "train"]["location_id"])
    test_locs  = set(manifest[manifest["partition"] == "test"]["location_id"])
    overlap = train_locs & test_locs
    assert len(overlap) == 0, \
        f"Location leakage: {overlap} appear in both train and test"


def test_manifest_covers_all_locations(df, manifest):
    all_locs = set(df["location_id"].unique())
    manifest_locs = set(manifest["location_id"])
    assert all_locs == manifest_locs, \
        f"Manifest missing locations: {all_locs - manifest_locs}"


# ── Collect_gee_data.py tests ─────────────────────────────────────────────────

def test_locations_count():
    sys.path.insert(0, str(ROOT / "src"))
    from collect_gee_data import LOCATIONS
    assert len(LOCATIONS) == 99, \
        f"LOCATIONS list has {len(LOCATIONS)} entries, expected 99"


def test_no_duplicate_location_ids():
    sys.path.insert(0, str(ROOT / "src"))
    from collect_gee_data import LOCATIONS
    ids = [l[0] for l in LOCATIONS]
    assert len(ids) == len(set(ids)), \
        f"Duplicate location IDs in LOCATIONS: {[x for x in ids if ids.count(x)>1]}"


# ── Feature importance artefact ───────────────────────────────────────────────

def test_feature_importance_csv_has_nine_rows():
    p = ROOT / "results" / "feature_importance.csv"
    if not p.exists():
        pytest.skip("feature_importance.csv not found")
    imp = pd.read_csv(p, index_col=0)
    assert len(imp) == 9, f"Expected 9 feature rows, got {len(imp)}"


if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        cwd=str(ROOT))
    sys.exit(result.returncode)
