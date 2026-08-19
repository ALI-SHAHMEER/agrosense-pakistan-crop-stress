"""
train_models.py
Train five classifiers on the AgroSense crop-condition proxy dataset.

Methodology
-----------
* Labels are spectrally derived proxy classes (Healthy / Stressed / Diseased)
  from a vegetation health composite (0.5·NDVI + 0.3·NDRE + 0.2·EVI).
  NDVI, NDRE and EVI are ALSO input features → reported performance is an
  upper bound on spectral separability, not validated disease detection.

* Split: GroupShuffleSplit on location_id (seed 42) ensures no sampling
  location appears in both train and test partitions.

* Hyperparameters: GridSearchCV with StratifiedGroupKFold inner CV (3 folds)
  fitted on the training partition only.

* Nested CV: 5-fold outer StratifiedGroupKFold on the full dataset to obtain
  an unbiased estimate of generalisation performance; inner 3-fold
  StratifiedGroupKFold for per-fold hyperparameter selection.

* Feature scaling: StandardScaler applied inside Pipeline for SVM and k-NN;
  tree-based models receive raw features.

Usage
-----
    cd agrosense-pakistan-crop-stress
    python src/train_models.py

Outputs (written to results/)
------------------------------
    classifier_results.csv   — held-out and nested-CV metrics
    per_class_results.csv    — per-class P/R/F1 for every model
    best_params.json         — winning hyperparameters
    split_manifest.csv       — location_id ↔ partition mapping
    nested_cv_results.csv    — per-fold nested CV F1
    feature_importance.csv   — tree-model native importances
    confusion_matrices.png
    feature_importance.png
    f1_comparison.png

    models/random_forest.pkl — production RF model bundle
"""

import json
import warnings
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    GridSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier

from feature_engineering import FEATURE_COLS

warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH   = Path("data/agrosense_crop_stress_dataset.csv")
RESULTS_DIR = Path("results")
MODELS_DIR  = Path("models")
RANDOM_SEED = 42

# ── Estimator definitions (Pipelines where scaling is needed) ─────────────────

def make_estimators():
    return {
        "Random Forest": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  SVC(kernel="rbf", random_state=RANDOM_SEED, probability=True)),
        ]),
        "k-NN": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  KNeighborsClassifier()),
        ]),
    }

# ── Hyperparameter grids ──────────────────────────────────────────────────────

PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators":      [100, 200],
        "max_depth":         [None, 20],
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
    "SVM (RBF)": {          # pipeline keys
        "model__C":     [1, 10, 100],
        "model__gamma": ["scale", "auto"],
    },
    "k-NN": {               # pipeline keys
        "model__n_neighbors": [3, 5, 7],
        "model__metric":      ["euclidean", "manhattan"],
    },
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} observations — "
          f"{df['location_id'].nunique()} sampling locations, "
          f"{df['province'].nunique()} provinces, "
          f"{df['season'].nunique()} seasons (2015–2024)")
    print("Class distribution:")
    print(df["crop_stress_label"].value_counts().to_string())
    print()
    le     = LabelEncoder()
    X      = df[FEATURE_COLS].values
    y      = le.fit_transform(df["crop_stress_label"])
    groups = df["location_id"].values
    return X, y, groups, le.classes_, df


# ── Grouped train / test split ────────────────────────────────────────────────

def grouped_split(X, y, groups, test_size=0.20, random_state=RANDOM_SEED):
    """
    80/20 split at sampling-location level.
    No location appears in both partitions.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                            random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return (X[train_idx], X[test_idx],
            y[train_idx], y[test_idx],
            groups[train_idx], groups[test_idx],
            train_idx, test_idx)


# ── Hyperparameter tuning ─────────────────────────────────────────────────────

def tune(name, estimator, X_train, y_train, g_train, inner_folds=3):
    inner_cv = StratifiedGroupKFold(n_splits=inner_folds,
                                    shuffle=True, random_state=RANDOM_SEED)
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=PARAM_GRIDS[name],
        cv=inner_cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_train, y_train, groups=g_train)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


# ── Nested grouped cross-validation ──────────────────────────────────────────

def nested_cv(name, estimator_factory, X, y, groups,
              outer_folds=5, inner_folds=3):
    """
    Outer: StratifiedGroupKFold(5) — unbiased generalisation estimate.
    Inner: GridSearchCV with StratifiedGroupKFold(3) — hyperparameter selection.
    Returns list of outer-fold macro F1 scores.
    """
    outer_cv = StratifiedGroupKFold(n_splits=outer_folds,
                                    shuffle=True, random_state=RANDOM_SEED)
    fold_scores = []
    for fold, (tr_idx, val_idx) in enumerate(
            outer_cv.split(X, y, groups=groups), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        g_tr = groups[tr_idx]

        best_est, _, _ = tune(name, estimator_factory(), X_tr, y_tr, g_tr,
                              inner_folds=inner_folds)
        y_pred = best_est.predict(X_val)
        fold_f1 = f1_score(y_val, y_pred, average="macro")
        fold_scores.append(fold_f1)
        print(f"    Fold {fold}: F1={fold_f1:.4f}")
    return fold_scores


# ── Figure helpers ────────────────────────────────────────────────────────────

def plot_confusion_matrices(trained, X_test, y_test, classes):
    n = len(trained)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, (name, model) in zip(axes, trained.items()):
        cm = confusion_matrix(y_test, model.predict(X_test), normalize="true")
        ConfusionMatrixDisplay(cm, display_labels=classes).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved confusion_matrices.png")


def plot_feature_importance(trained, classes):
    tree_models = {k: v for k, v in trained.items()
                   if hasattr(v, "feature_importances_")}
    imp_df = pd.DataFrame(
        {k: v.feature_importances_ for k, v in tree_models.items()},
        index=FEATURE_COLS
    ).sort_values(list(tree_models.keys())[0], ascending=False)

    imp_df.to_csv(RESULTS_DIR / "feature_importance.csv")
    print("  Saved feature_importance.csv")

    imp_df.plot(kind="bar", figsize=(10, 4))
    plt.title("Native Feature Importance — Tree-Based Models\n"
              "(impurity-based; affected by label-generation rule using NDVI/NDRE/EVI)")
    plt.ylabel("Importance")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved feature_importance.png")
    return imp_df


def plot_f1_comparison(results_df):
    plot_df = results_df.set_index("Algorithm")[
        ["Test_Accuracy", "Test_F1_Macro", "Nested_CV_F1_Macro"]
    ].rename(columns={
        "Test_Accuracy": "Test Accuracy",
        "Test_F1_Macro": "Test F1 Macro",
        "Nested_CV_F1_Macro": "Nested CV F1",
    })
    # All values on 0–1 scale
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    X, y, groups, classes, df = load_data()

    # ── 1. Held-out grouped split ─────────────────────────────────────────────
    (X_train, X_test, y_train, y_test,
     g_train, g_test, tr_idx, te_idx) = grouped_split(X, y, groups)

    train_locs = sorted(set(g_train))
    test_locs  = sorted(set(g_test))
    print(f"Train: {len(X_train)} obs / {len(train_locs)} sampling locations")
    print(f"Test : {len(X_test)} obs / {len(test_locs)} sampling locations\n")

    # Save split manifest
    manifest = pd.DataFrame({
        "location_id": list(g_train) + list(g_test),
        "partition":   ["train"] * len(g_train) + ["test"] * len(g_test),
    }).drop_duplicates("location_id").sort_values("location_id")
    manifest.to_csv(RESULTS_DIR / "split_manifest.csv", index=False)
    print("  Saved split_manifest.csv")

    # ── 2. Tune, train, evaluate each model ──────────────────────────────────
    overall_rows, perclass_rows = [], []
    best_params_all, trained = {}, {}
    nested_cv_records = []

    for name in make_estimators():
        print(f"\n{'─'*54}\n{name}")

        # 2a. Hyperparameter tuning on training partition
        best_est, best_params, inner_f1 = tune(
            name, make_estimators()[name], X_train, y_train, g_train)
        best_params_all[name] = {
            "best_params": best_params,
            "inner_cv_f1": round(float(inner_f1), 4),
        }
        print(f"  Best params : {best_params}")
        print(f"  Inner CV F1 : {inner_f1:.4f}")
        trained[name] = best_est

        # 2b. Held-out test evaluation
        y_pred = best_est.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0)
        print(f"  Held-out    : Acc={acc:.4f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")

        # 2c. Per-class metrics
        rep = classification_report(y_test, y_pred,
                                    target_names=classes,
                                    output_dict=True, zero_division=0)
        for cls in classes:
            perclass_rows.append({
                "Algorithm": name,
                "Class":     cls,
                "Precision": round(rep[cls]["precision"], 4),
                "Recall":    round(rep[cls]["recall"], 4),
                "F1":        round(rep[cls]["f1-score"], 4),
                "Support":   int(rep[cls]["support"]),
            })

        # 2d. Nested grouped CV (full dataset, unbiased estimate)
        print(f"  Nested CV (5×3 folds):")
        fold_f1s = nested_cv(name, lambda n=name: make_estimators()[n],
                             X, y, groups)
        nested_mean = float(np.mean(fold_f1s))
        nested_std  = float(np.std(fold_f1s))
        print(f"  Nested CV F1: {nested_mean:.4f} ± {nested_std:.4f}")
        for fold_i, fv in enumerate(fold_f1s, 1):
            nested_cv_records.append({
                "Algorithm": name, "Fold": fold_i,
                "F1_Macro": round(fv, 4),
            })

        overall_rows.append({
            "Algorithm":          name,
            "Test_Accuracy":      round(acc, 4),
            "Test_Precision_Macro": round(p, 4),
            "Test_Recall_Macro":  round(r, 4),
            "Test_F1_Macro":      round(f1, 4),
            "Nested_CV_F1_Macro": round(nested_mean, 4),
            "Nested_CV_F1_Std":   round(nested_std, 4),
        })

    # ── 3. Save tabular results ───────────────────────────────────────────────
    results_df  = pd.DataFrame(overall_rows)
    perclass_df = pd.DataFrame(perclass_rows)
    nested_df   = pd.DataFrame(nested_cv_records)

    results_df.to_csv(RESULTS_DIR / "classifier_results.csv", index=False)
    perclass_df.to_csv(RESULTS_DIR / "per_class_results.csv", index=False)
    nested_df.to_csv(RESULTS_DIR / "nested_cv_results.csv", index=False)

    with open(RESULTS_DIR / "best_params.json", "w") as f:
        json.dump(best_params_all, f, indent=2)

    meta = {
        "random_seed": RANDOM_SEED,
        "train_obs": int(len(X_train)),
        "test_obs":  int(len(X_test)),
        "train_locations": int(len(train_locs)),
        "test_locations":  int(len(test_locs)),
        "train_location_ids": train_locs,
        "test_location_ids":  test_locs,
        "feature_cols": FEATURE_COLS,
        "classes": list(classes),
        "label_generation": "0.5*NDVI + 0.3*NDRE + 0.2*EVI, tertile split",
        "note": ("Labels are spectrally derived proxy classes. NDVI, NDRE, "
                 "and EVI are also input features (circularity). "
                 "Performance measures spectral proxy separability only."),
    }
    with open(RESULTS_DIR / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n\n=== RESULTS SUMMARY ===")
    print(results_df.to_string(index=False))

    # ── 4. Figures ────────────────────────────────────────────────────────────
    print("\n--- Generating figures ---")
    plot_confusion_matrices(trained, X_test, y_test, classes)
    imp_df = plot_feature_importance(trained, classes)
    plot_f1_comparison(results_df)

    # ── 5. Save production model ──────────────────────────────────────────────
    rf_bundle = {
        "model": trained["Random Forest"],
        "classes": classes,
        "features": FEATURE_COLS,
        "best_params": best_params_all["Random Forest"]["best_params"],
        "split": "GroupShuffleSplit(location_id, test_size=0.20, seed=42)",
        "cv":    "StratifiedGroupKFold(outer=5, inner=3) nested CV",
        "note":  ("Spectrally derived proxy labels — not independently "
                  "validated disease detection."),
    }
    joblib.dump(rf_bundle, MODELS_DIR / "random_forest.pkl")
    print(f"\n  Production model saved → {MODELS_DIR}/random_forest.pkl")
    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
