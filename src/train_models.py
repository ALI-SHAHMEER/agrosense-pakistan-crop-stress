"""
train_models.py
Train five classifiers on the AgroSense crop stress dataset with:
  - Field-grouped train/test split  (GroupShuffleSplit on location_id)
  - Grouped cross-validation        (StratifiedGroupKFold, 5 folds)
  - Hyperparameter tuning           (GridSearchCV with grouped inner CV)

Usage:
    python src/train_models.py

Outputs:
    results/classifier_results.csv  — per-model accuracy / F1
    models/random_forest.pkl        — production model (best Random Forest)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedGroupKFold,
    GridSearchCV, cross_validate,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from xgboost import XGBClassifier

from feature_engineering import FEATURE_COLS

DATA_PATH    = Path("data/agrosense_crop_stress_dataset.csv")
RESULTS_PATH = Path("results/classifier_results.csv")
MODELS_DIR   = Path("models")

# ── Hyperparameter search grids ───────────────────────────────────────────────

PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth":    [None, 20],
        "min_samples_split": [2, 5],
    },
    "XGBoost": {
        "n_estimators":  [100, 200],
        "max_depth":     [4, 6],
        "learning_rate": [0.05, 0.1],
    },
    "Gradient Boosting": {
        "n_estimators":  [100, 200],
        "max_depth":     [3, 5],
        "learning_rate": [0.05, 0.1],
    },
    "SVM (RBF)": {
        "C":     [1, 10, 100],
        "gamma": ["scale", "auto"],
    },
    "k-NN": {
        "n_neighbors": [3, 5, 7],
        "metric":      ["euclidean", "manhattan"],
    },
}

BASE_ESTIMATORS = {
    "Random Forest":     RandomForestClassifier(random_state=42, n_jobs=-1),
    "XGBoost":           XGBClassifier(use_label_encoder=False,
                                       eval_metric="mlogloss",
                                       random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM (RBF)":         SVC(kernel="rbf", random_state=42),
    "k-NN":              KNeighborsClassifier(),
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} observations — "
          f"{df['location_id'].nunique()} sites, "
          f"{df['province'].nunique()} provinces, "
          f"{df['season'].nunique()} seasons")
    print(df["crop_stress_label"].value_counts().to_string(), "\n")
    X      = df[FEATURE_COLS].values
    groups = df["location_id"].values
    le     = LabelEncoder()
    y      = le.fit_transform(df["crop_stress_label"])
    return X, y, groups, le.classes_


def field_grouped_split(X, y, groups, test_size=0.20, random_state=42):
    """80/20 split at field (location_id) level — no site leaks between sets."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return (X[train_idx], X[test_idx],
            y[train_idx], y[test_idx],
            groups[train_idx], groups[test_idx])


def tune_and_train(name, X_train, y_train, groups_train):
    """Run GridSearchCV with grouped 5-fold inner CV, return best estimator."""
    inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=BASE_ESTIMATORS[name],
        param_grid=PARAM_GRIDS[name],
        cv=inner_cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_train, y_train, groups=groups_train)
    print(f"  Best params : {gs.best_params_}")
    print(f"  Inner CV F1 : {gs.best_score_:.4f}")
    return gs.best_estimator_


def evaluate(name, model, X_train, y_train, groups_train,
             X_test, y_test, X_all, y_all, groups_all):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)

    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_res   = cross_validate(model, X_all, y_all, cv=outer_cv,
                               groups=groups_all, scoring="f1_macro")
    cv_f1    = cv_res["test_score"].mean()

    print(f"  Test  Acc={acc:.4f}  P={p:.3f}  R={r:.3f}  "
          f"F1={f1:.3f}  CV-F1={cv_f1:.3f}")
    return dict(
        Algorithm=name,
        Accuracy_pct=round(acc * 100, 2),
        Precision_Macro=round(p, 3),
        Recall_Macro=round(r, 3),
        F1_Macro=round(f1, 3),
        CV_F1_Macro=round(cv_f1, 3),
    )


def main():
    X, y, groups, classes = load_data()
    X_train, X_test, y_train, y_test, g_train, g_test = \
        field_grouped_split(X, y, groups)

    print(f"Train: {len(X_train)} obs / "
          f"{len(set(g_train))} sites\n"
          f"Test : {len(X_test)} obs / "
          f"{len(set(g_test))} sites\n")

    rows, trained = [], {}
    for name in BASE_ESTIMATORS:
        print(f"\n{'─'*50}\n{name}")
        best = tune_and_train(name, X_train, y_train, g_train)
        trained[name] = best
        row = evaluate(name, best, X_train, y_train, g_train,
                       X_test, y_test, X, y, groups)
        rows.append(row)

    results = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved → {RESULTS_PATH}")
    print(results.to_string(index=False))

    MODELS_DIR.mkdir(exist_ok=True)
    rf = trained["Random Forest"]
    joblib.dump(
        {"model": rf, "classes": classes, "features": FEATURE_COLS,
         "split": "GroupShuffleSplit(location_id)",
         "cv": "StratifiedGroupKFold(5)"},
        MODELS_DIR / "random_forest.pkl",
    )
    print(f"Production model saved → {MODELS_DIR}/random_forest.pkl")


if __name__ == "__main__":
    main()
