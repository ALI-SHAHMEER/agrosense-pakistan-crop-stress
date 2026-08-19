"""
rule_baseline.py
Evaluate the deterministic spectral-rule baseline against the proxy labels.

Two clearly distinguished concepts:

1. Benchmark-definition rule
   Uses the globally fixed q33/q66 from data/label_definition.json.
   By construction it produces near-perfect label recovery on the full dataset.
   This is NOT a predictive model — it is the definition of the benchmark itself.

2. Training-derived threshold baseline
   Recomputes q33/q66 from the outer-fold training partition only,
   then applies them to the held-out observations.
   This is an inductive rule that respects the train/test boundary.

Usage
-----
    python src/rule_baseline.py

Outputs
-------
    results/rule_baseline_results.csv
    data/label_definition.json  (if not already present)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_PATH   = Path("data/agrosense_crop_stress_dataset.csv")
RESULTS_DIR = Path("results")
DEFN_PATH   = Path("data/label_definition.json")
RANDOM_SEED = 42

LABEL_ORDER = {"Diseased": 0, "Stressed": 1, "Healthy": 2}
CLASS_NAMES = ["Diseased", "Stressed", "Healthy"]


def compute_score(df: pd.DataFrame) -> np.ndarray:
    return (0.5 * df["ndvi"] + 0.3 * df["ndre"] + 0.2 * df["evi"]).values


def apply_rule(scores: np.ndarray, q33: float, q66: float) -> np.ndarray:
    return np.where(scores < q33, 0, np.where(scores < q66, 1, 2))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    scores = compute_score(df)
    y_true = df["crop_stress_label"].map(LABEL_ORDER).values
    groups = df["location_id"].values

    # ── 1. Benchmark-definition rule ─────────────────────────────────────────
    # Thresholds from the global dataset — these define the released labels.
    q33_global = float(np.percentile(scores, 100 / 3))
    q66_global = float(np.percentile(scores, 200 / 3))

    # Save / update label definition file
    defn = {
        "score_formula": "0.5*NDVI + 0.3*NDRE + 0.2*EVI",
        "q33": round(q33_global, 6),
        "q66": round(q66_global, 6),
        "scope": "full released dataset (1786 observations)",
        "purpose": "benchmark label construction",
        "note": (
            "These thresholds were derived from the FULL dataset before any "
            "train/test split. The class labels in the released CSV are fixed by "
            "these thresholds. They are NOT derived from training data alone. "
            "A training-derived threshold baseline must recompute q33/q66 from "
            "training folds only to avoid threshold leakage."
        ),
    }
    with open(DEFN_PATH, "w") as f:
        json.dump(defn, f, indent=2)
    print(f"Label definition: q33={q33_global:.6f}  q66={q66_global:.6f}")

    y_pred_global = apply_rule(scores, q33_global, q66_global)
    global_acc = accuracy_score(y_true, y_pred_global)
    global_f1  = f1_score(y_true, y_pred_global, average="macro")
    print(f"\n[1] Benchmark-definition rule (global thresholds, full dataset)")
    print(f"    Accuracy={global_acc:.4f}  Macro-F1={global_f1:.4f}")
    print(f"    NOTE: Near-perfect recovery is EXPECTED — this rule defines the labels.")
    print(f"    It is not an independent predictive model.\n")

    # ── 2. Training-derived threshold baseline — held-out evaluation ──────────
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(
        scores.reshape(-1, 1), y_true, groups=groups))

    q33_tr = float(np.percentile(scores[train_idx], 100 / 3))
    q66_tr = float(np.percentile(scores[train_idx], 200 / 3))
    print(f"[2] Training-derived thresholds: q33={q33_tr:.6f}  q66={q66_tr:.6f}")

    y_pred_tr = apply_rule(scores[test_idx], q33_tr, q66_tr)
    held_acc = accuracy_score(y_true[test_idx], y_pred_tr)
    held_f1  = f1_score(y_true[test_idx], y_pred_tr, average="macro")
    print(f"    Held-out: Accuracy={held_acc:.4f}  Macro-F1={held_f1:.4f}")
    print(classification_report(y_true[test_idx], y_pred_tr,
                                target_names=CLASS_NAMES, zero_division=0))

    # ── 3. Training-derived threshold baseline — nested CV ───────────────────
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                    random_state=RANDOM_SEED)
    fold_f1s = []
    print("[3] Nested CV (5-fold outer, training-derived thresholds per fold):")
    for fold, (tr_idx, val_idx) in enumerate(
            outer_cv.split(scores.reshape(-1, 1), y_true, groups=groups), 1):
        q33_f = float(np.percentile(scores[tr_idx], 100 / 3))
        q66_f = float(np.percentile(scores[tr_idx], 200 / 3))
        y_pred_f = apply_rule(scores[val_idx], q33_f, q66_f)
        f1_f = f1_score(y_true[val_idx], y_pred_f, average="macro")
        fold_f1s.append(f1_f)
        print(f"    Fold {fold}: F1={f1_f:.4f}")

    nested_mean = float(np.mean(fold_f1s))
    nested_std  = float(np.std(fold_f1s))
    print(f"    Nested CV F1: {nested_mean:.4f} ± {nested_std:.4f}\n")

    # ── 4. Save results ───────────────────────────────────────────────────────
    rows = [
        {
            "Baseline": "Benchmark-definition rule",
            "Evaluation": "Full dataset (all 1786 obs)",
            "Accuracy": round(global_acc, 4),
            "Macro_F1": round(global_f1, 4),
            "Nested_CV_F1_Mean": None,
            "Nested_CV_F1_Std": None,
            "Note": (
                "Uses global q33/q66 that DEFINE the benchmark labels. "
                "Near-perfect recovery is expected by construction — "
                "this is NOT a predictive model."
            ),
        },
        {
            "Baseline": "Training-derived threshold rule",
            "Evaluation": "Held-out location-grouped test set",
            "Accuracy": round(held_acc, 4),
            "Macro_F1": round(held_f1, 4),
            "Nested_CV_F1_Mean": None,
            "Nested_CV_F1_Std": None,
            "Note": (
                "q33/q66 derived from training partition only (no leakage). "
                "Applied to held-out locations unseen during threshold fitting."
            ),
        },
        {
            "Baseline": "Training-derived threshold rule",
            "Evaluation": "Nested CV (5-fold outer StratifiedGroupKFold)",
            "Accuracy": None,
            "Macro_F1": None,
            "Nested_CV_F1_Mean": round(nested_mean, 4),
            "Nested_CV_F1_Std": round(nested_std, 4),
            "Note": (
                "Thresholds recomputed from outer-fold training data each fold. "
                "Provides an unbiased estimate of inductive rule generalisation."
            ),
        },
    ]
    out_df = pd.DataFrame(rows)
    out_df.to_csv(RESULTS_DIR / "rule_baseline_results.csv", index=False)
    print(f"Saved results/rule_baseline_results.csv")


if __name__ == "__main__":
    main()
