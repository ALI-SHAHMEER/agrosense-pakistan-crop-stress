"""
train_models.py
Train five classifiers on the AgroSense crop stress dataset and save results.

Usage:
    python src/train_models.py

Outputs:
    results/classifier_results.csv  — per-model accuracy / F1
    models/random_forest.pkl        — production model (Random Forest)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from xgboost import XGBClassifier

from feature_engineering import FEATURE_COLS

DATA_PATH    = Path("data/agrosense_crop_stress_dataset.csv")
RESULTS_PATH = Path("results/classifier_results.csv")
MODELS_DIR   = Path("models")

CLASSIFIERS = {
    "Random Forest":    RandomForestClassifier(n_estimators=200, max_depth=None,
                                               random_state=42, n_jobs=-1),
    "XGBoost":          XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                      use_label_encoder=False, eval_metric="mlogloss",
                                      random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                     learning_rate=0.1, random_state=42),
    "SVM (RBF)":        SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                             random_state=42),
    "k-NN":             KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples — {df['location_id'].nunique()} sites, "
          f"{df['province'].nunique()} provinces, {df['season'].nunique()} seasons")
    print(df["crop_stress_label"].value_counts().to_string(), "\n")
    X = df[FEATURE_COLS].values
    le = LabelEncoder()
    y  = le.fit_transform(df["crop_stress_label"])
    return X, y, le.classes_


def evaluate(name, model, X_train, X_test, y_train, y_test, X_all, y_all):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro",
                                                   zero_division=0)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    from sklearn.model_selection import cross_val_score
    cv_f1 = cross_val_score(model, X_all, y_all, cv=cv,
                             scoring="f1_macro").mean()

    print(f"{name:20s}  Acc={acc:.4f}  P={p:.3f}  R={r:.3f}  "
          f"F1={f1:.3f}  CV-F1={cv_f1:.3f}")
    return dict(Algorithm=name, Accuracy_pct=round(acc * 100, 2),
                Precision_Macro=round(p, 3), Recall_Macro=round(r, 3),
                F1_Macro=round(f1, 3), CV_F1_Macro=round(cv_f1, 3))


def main():
    X, y, classes = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    print(f"Train: {len(X_train)}  Test: {len(X_test)}\n")

    rows = []
    for name, clf in CLASSIFIERS.items():
        row = evaluate(name, clf, X_train, X_test, y_train, y_test, X, y)
        rows.append(row)

    results = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Save production model (Random Forest)
    rf = CLASSIFIERS["Random Forest"]
    rf.fit(X_train, y_train)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": rf, "classes": classes, "features": FEATURE_COLS},
                MODELS_DIR / "random_forest.pkl")
    print(f"Production model saved to {MODELS_DIR}/random_forest.pkl")


if __name__ == "__main__":
    main()
