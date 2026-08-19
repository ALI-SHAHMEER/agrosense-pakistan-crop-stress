"""
evaluate_models.py
Re-evaluate a trained model and regenerate all result figures.

Usage:
    python src/evaluate_models.py

Outputs:
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
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier

from feature_engineering import FEATURE_COLS

DATA_PATH    = Path("data/agrosense_crop_stress_dataset.csv")
RESULTS_DIR  = Path("results")
CLASS_NAMES  = ["Diseased", "Healthy", "Stressed"]

CLASSIFIERS = {
    "Random Forest":     RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost":           XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                       use_label_encoder=False, eval_metric="mlogloss",
                                       random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                     learning_rate=0.1, random_state=42),
    "SVM (RBF)":         SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
    "k-NN":              KNeighborsClassifier(n_neighbors=5),
}


def load_data():
    df  = pd.read_csv(DATA_PATH)
    le  = LabelEncoder().fit(df["crop_stress_label"])
    X   = df[FEATURE_COLS].values
    y   = le.transform(df["crop_stress_label"])
    return X, y, le.classes_


def plot_confusion_matrices(trained, X_test, y_test, classes):
    fig, axes = plt.subplots(1, len(trained), figsize=(18, 3.5))
    for ax, (name, model) in zip(axes, trained.items()):
        cm = confusion_matrix(y_test, model.predict(X_test), normalize="true")
        ConfusionMatrixDisplay(cm, display_labels=classes).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=9)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close()
    print("Saved confusion_matrices.png")


def plot_feature_importance(trained):
    tree_models = {k: v for k, v in trained.items()
                   if hasattr(v, "feature_importances_")}
    imp = pd.DataFrame({k: v.feature_importances_
                        for k, v in tree_models.items()}, index=FEATURE_COLS)
    imp = imp.sort_values("Random Forest", ascending=False)

    imp.plot(kind="bar", figsize=(10, 4))
    plt.title("Feature Importance Comparison")
    plt.ylabel("Importance")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print("Saved feature_importance.png")


def plot_f1_comparison(results_df):
    ax = results_df.set_index("Algorithm")[["Accuracy_pct", "F1_Macro", "CV_F1_Macro"]].plot(
        kind="bar", figsize=(9, 4))
    plt.title("Classifier Comparison — Accuracy & Macro F1")
    plt.ylabel("Score")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0.85, 1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "f1_comparison.png", dpi=150)
    plt.close()
    print("Saved f1_comparison.png")


def main():
    X, y, classes = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)

    trained, rows = {}, []
    for name, clf in CLASSIFIERS.items():
        clf.fit(X_train, y_train)
        trained[name] = clf
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rep = classification_report(y_test, y_pred, output_dict=True)
        rows.append(dict(
            Algorithm=name,
            Accuracy_pct=round(acc * 100, 2),
            F1_Macro=round(rep["macro avg"]["f1-score"], 3),
            CV_F1_Macro=None,
        ))
        print(f"\n{name}\n{classification_report(y_test, y_pred, target_names=classes)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "classifier_results.csv", index=False)

    plot_confusion_matrices(trained, X_test, y_test, classes)
    plot_feature_importance(trained)
    plot_f1_comparison(results_df)


if __name__ == "__main__":
    main()
