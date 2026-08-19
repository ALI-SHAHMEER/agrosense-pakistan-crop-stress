"""
build_paper_figures.py
Synchronise manuscript figures in paper/ with current artefacts in results/.

Run this after train_models.py to ensure paper figures never drift from results.

Usage
-----
    python src/build_paper_figures.py

Actions
-------
1. Generate results/perclass_f1_heatmap.png from results/per_class_results.csv.
2. Copy:
       results/confusion_matrices.png  → paper/fig3_confusion_matrices.png
       results/feature_importance.png  → paper/fig4_feature_importance.png
       results/f1_comparison.png       → paper/fig5_accuracy_f1_comparison.png
       results/perclass_f1_heatmap.png → paper/fig6_perclass_f1_heatmap.png
"""

import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path("results")
PAPER_DIR   = Path("paper")


def generate_perclass_heatmap() -> Path:
    """Build per-class F1 heatmap from saved CSV."""
    per_class = pd.read_csv(RESULTS_DIR / "per_class_results.csv")

    pivot = per_class.pivot(index="Algorithm", columns="Class", values="F1")
    # Consistent column order
    for col in ["Diseased", "Stressed", "Healthy"]:
        if col not in pivot.columns:
            pivot[col] = float("nan")
    pivot = pivot[["Diseased", "Stressed", "Healthy"]]
    # Rename to show proxy nature
    pivot.columns = ["Diseased\n(proxy)", "Stressed\n(proxy)", "Healthy\n(proxy)"]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGn",
        vmin=0.88,
        vmax=1.0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "F1-score"},
    )
    ax.set_title(
        "Per-Class F1-Score Heatmap — All Five Classifiers\n"
        "(proxy classes derived from spectral indices; not independently validated)",
        fontsize=9,
    )
    ax.set_xlabel("Proxy Class")
    ax.set_ylabel("")
    plt.tight_layout()

    out = RESULTS_DIR / "perclass_f1_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Generated {out}")
    return out


def copy_to_paper(src: Path, dst: Path) -> None:
    shutil.copy2(src, dst)
    print(f"  {src} -> {dst}")


def main() -> None:
    PAPER_DIR.mkdir(exist_ok=True)

    print("--- Generating per-class F1 heatmap ---")
    heatmap_path = generate_perclass_heatmap()

    print("\n--- Copying figures to paper/ ---")
    mapping = [
        (RESULTS_DIR / "confusion_matrices.png",   PAPER_DIR / "fig3_confusion_matrices.png"),
        (RESULTS_DIR / "feature_importance.png",   PAPER_DIR / "fig4_feature_importance.png"),
        (RESULTS_DIR / "f1_comparison.png",        PAPER_DIR / "fig5_accuracy_f1_comparison.png"),
        (RESULTS_DIR / "perclass_f1_heatmap.png",  PAPER_DIR / "fig6_perclass_f1_heatmap.png"),
    ]
    for src, dst in mapping:
        if not src.exists():
            print(f"  WARNING: {src} not found — run train_models.py first")
            continue
        copy_to_paper(src, dst)

    print("\nAll paper figures are now in sync with results/")


if __name__ == "__main__":
    main()
