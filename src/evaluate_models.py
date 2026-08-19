"""
evaluate_models.py
Load the saved Random Forest model and regenerate a subset of outputs
without retraining or re-running GridSearchCV / nested CV.

This script READS results produced by train_models.py and does NOT
overwrite classifier_results.csv with incomplete data.

For the full experiment (GridSearchCV + nested CV), run train_models.py.
For updated paper figures, run build_paper_figures.py.

Usage:
    python src/evaluate_models.py

What this script does:
    - Loads the saved Random Forest model (models/random_forest.pkl)
    - Prints the RF per-class classification report
    - Saves results/per_class_report.txt
    - Saves results/confusion_rf.png  (RF confusion matrix only)
    - Regenerates results/f1_comparison.png from saved classifier_results.csv
    - Regenerates results/feature_importance.png from saved feature_importance.csv

What this script does NOT do:
    - Does NOT retrain any model
    - Does NOT recompute CV metrics
    - Does NOT regenerate the multi-model confusion_matrices.png
      (that requires all trained models in memory, which train_models.py holds)
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from feature_engineering import FEATURE_COLS
from train_models import grouped_split

DATA_PATH   = Path("data/agrosense_crop_stress_dataset.csv")
RESULTS_DIR = Path("results")
MODELS_DIR  = Path("models")


def load_results():
    req_files = [
        RESULTS_DIR / "classifier_results.csv",
        RESULTS_DIR / "nested_cv_results.csv",
        RESULTS_DIR / "per_class_results.csv",
        RESULTS_DIR / "best_params.json",
        MODELS_DIR  / "random_forest.pkl",
    ]
    missing = [str(f) for f in req_files if not f.exists()]
    if missing:
        print("ERROR: Run train_models.py first. Missing:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    results  = pd.read_csv(RESULTS_DIR / "classifier_results.csv")
    nested   = pd.read_csv(RESULTS_DIR / "nested_cv_results.csv")
    perclass = pd.read_csv(RESULTS_DIR / "per_class_results.csv")
    with open(RESULTS_DIR / "best_params.json") as f:
        best_params = json.load(f)
    return results, nested, perclass, best_params


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print("Loading dataset …")
    df = pd.read_csv(DATA_PATH)
    le = LabelEncoder()
    X      = df[FEATURE_COLS].values
    y      = le.fit_transform(df["crop_stress_label"])
    groups = df["location_id"].values
    classes = le.classes_

    X_train, X_test, y_train, y_test, g_train, g_test, _, _ = \
        grouped_split(X, y, groups)
    print(f"  Test set: {len(X_test)} obs / {len(set(g_test))} sampling locations")

    # Load results from disk
    results, nested, perclass, best_params = load_results()
    print("\n=== Classifier Results (from saved CSV) ===")
    print(results.to_string(index=False))

    # Load production model
    rf_bundle = joblib.load(MODELS_DIR / "random_forest.pkl")
    rf_model  = rf_bundle["model"]

    # Print classification report for RF
    y_pred_rf = rf_model.predict(X_test)
    report = classification_report(y_test, y_pred_rf,
                                   target_names=classes, zero_division=0)
    print(f"\n=== Random Forest Per-Class Report ===\n{report}")
    (RESULTS_DIR / "per_class_report.txt").write_text(
        "Random Forest — Per-Class Classification Report\n"
        f"Test set: {len(X_test)} obs / {len(set(g_test))} held-out locations\n\n"
        + report
    )
    print("  Saved per_class_report.txt")

    # ── Figures ───────────────────────────────────────────────────────────────
    # Confusion matrices — requires all trained models (use RF only if others missing)
    rf_preds = rf_model.predict(X_test)
    cm = confusion_matrix(y_test, rf_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues")
    ax.set_title("Random Forest (Production Model)\nNormalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_rf.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved confusion_rf.png")

    # F1 comparison from saved CSV (correct scale: 0–1)
    plot_df = results.set_index("Algorithm")[
        ["Test_Accuracy", "Test_F1_Macro", "Nested_CV_F1_Macro"]
    ].rename(columns={
        "Test_Accuracy": "Test Accuracy",
        "Test_F1_Macro": "Test F1 Macro",
        "Nested_CV_F1_Macro": "Nested CV F1",
    })
    plot_df.plot(kind="bar", figsize=(9, 4))
    plt.title("Classifier Comparison — All metrics on 0–1 scale")
    plt.ylabel("Score (0–1)")
    plt.xticks(rotation=25, ha="right")
    bottom = max(0.85, plot_df.min().min() - 0.02)
    plt.ylim(bottom, 1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "f1_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved f1_comparison.png")

    # Feature importance from saved CSV
    if (RESULTS_DIR / "feature_importance.csv").exists():
        imp = pd.read_csv(RESULTS_DIR / "feature_importance.csv", index_col=0)
        imp.plot(kind="bar", figsize=(10, 4))
        plt.title("Native Feature Importance — Tree-Based Models\n"
                  "(impurity-based; affected by label-generation rule using NDVI/NDRE/EVI)")
        plt.ylabel("Importance")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "feature_importance.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved feature_importance.png (from saved CSV)")

    print(f"\nAll outputs in {RESULTS_DIR}/")
    print("NOTE: To recompute CV metrics, run train_models.py")


if __name__ == "__main__":
    main()
