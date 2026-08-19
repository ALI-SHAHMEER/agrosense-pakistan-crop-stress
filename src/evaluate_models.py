"""
evaluate_models.py
Re-evaluate trained models and regenerate all result figures.

Usage:
    python src/evaluate_models.py

Outputs (appended to, not overwriting, existing classifier_results.csv):
    results/classifier_results.csv
    results/confusion_matrices.png
    results/feature_importance.png
    results/f1_comparison.png
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier

from feature_engineering import FEATURE_COLS
from train_models import BASE_ESTIMATORS, PARAM_GRIDS, field_grouped_split, tune_and_train

DATA_PATH   = Path("data/agrosense_crop_stress_dataset.csv")
RESULTS_DIR = Path("results")


def load_data():
    df  = pd.read_csv(DATA_PATH)
    le  = LabelEncoder()
    X      = df[FEATURE_COLS].values
    y      = le.fit_transform(df["crop_stress_label"])
    groups = df["location_id"].values
    return X, y, groups, le.classes_, df


def plot_confusion_matrices(trained, X_test, y_test, classes):
    n = len(trained)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, (name, model) in zip(axes, trained.items()):
        cm = confusion_matrix(y_test, model.predict(X_test), normalize="true")
        ConfusionMatrixDisplay(cm, display_labels=classes).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved confusion_matrices.png")


def plot_feature_importance(trained):
    tree_models = {k: v for k, v in trained.items()
                   if hasattr(v, "feature_importances_")}
    imp = pd.DataFrame(
        {k: v.feature_importances_ for k, v in tree_models.items()},
        index=FEATURE_COLS
    ).sort_values("Random Forest", ascending=False)

    imp.plot(kind="bar", figsize=(10, 4))
    plt.title("Feature Importance Comparison")
    plt.ylabel("Importance")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved feature_importance.png")


def plot_f1_comparison(results_df):
    # Convert accuracy to 0–1 scale so axes are comparable with F1
    plot_df = results_df.copy()
    plot_df["Accuracy"] = plot_df["Accuracy_pct"] / 100.0
    plot_df = plot_df.set_index("Algorithm")[
        ["Accuracy", "F1_Macro", "CV_F1_Macro"]
    ]
    plot_df.plot(kind="bar", figsize=(9, 4))
    plt.title("Classifier Comparison — Accuracy & Macro F1 (0–1 scale)")
    plt.ylabel("Score")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0.85, 1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "f1_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved f1_comparison.png")


def main():
    X, y, groups, classes, df = load_data()
    X_train, X_test, y_train, y_test, g_train, g_test = \
        field_grouped_split(X, y, groups)

    print(f"Train: {len(X_train)} obs / {len(set(g_train))} sites")
    print(f"Test : {len(X_test)} obs / {len(set(g_test))} sites\n")

    trained, rows = {}, []
    for name in BASE_ESTIMATORS:
        print(f"Training {name}…")
        model = tune_and_train(name, X_train, y_train, g_train)
        trained[name] = model

        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0)

        print(classification_report(y_test, y_pred, target_names=classes))
        rows.append(dict(
            Algorithm=name,
            Accuracy_pct=round(acc * 100, 2),
            Precision_Macro=round(p, 3),
            Recall_Macro=round(r, 3),
            F1_Macro=round(f1, 3),
            CV_F1_Macro=None,   # run train_models.py for grouped CV scores
        ))

    RESULTS_DIR.mkdir(exist_ok=True)
    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "classifier_results.csv", index=False)
    print(f"\nResults saved → {RESULTS_DIR}/classifier_results.csv")
    print("(Re-run train_models.py to populate CV_F1_Macro with grouped CV scores)")

    plot_confusion_matrices(trained, X_test, y_test, classes)
    plot_feature_importance(trained)
    plot_f1_comparison(results_df)


if __name__ == "__main__":
    main()
