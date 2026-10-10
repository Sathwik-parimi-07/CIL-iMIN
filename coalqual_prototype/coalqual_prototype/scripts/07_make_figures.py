#!/usr/bin/env python3
"""
Step 7 - regenerate every deck figure from the saved models.

    python scripts/07_make_figures.py --data outputs/tables/dataset_synthetic.csv

Useful after training targets one at a time, or when only the styling of a
figure needs to change.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from coalqual.config import TABLE_DIR, TARGETS
from coalqual.pipeline import rebuild_store
from coalqual.visualize import (plot_benchmark_heatmap, plot_feature_importance,
                                plot_scatter_grid, plot_spectra,
                                plot_target_distributions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/tables/dataset_synthetic.csv")
    ap.add_argument("--test-fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    synthetic = bool(df.get("is_synthetic", pd.Series([False])).any())
    n_lab = int(df[[t for t in TARGETS if t in df.columns]].notna().all(axis=1).sum())
    tf = args.test_fraction if args.test_fraction is not None else (0.25 if n_lab >= 40 else 0.0)

    store = rebuild_store(df, targets=TARGETS, seed=args.seed, test_fraction=tf)
    print(f"rebuilt artefacts for {list(store)}")

    print("wrote", plot_scatter_grid(store, synthetic=synthetic))
    full = Path(TABLE_DIR) / "benchmark_full.csv"
    if full.exists():
        print("wrote", plot_benchmark_heatmap(pd.read_csv(full), synthetic=synthetic))
    for t in store:
        try:
            print("wrote", plot_feature_importance(store, t, synthetic=synthetic))
        except Exception as exc:                                       # noqa: BLE001
            print(f"(importance {t} skipped: {exc})")
    if "Ash" in df.columns:
        print("wrote", plot_spectra(df, colour_by="Ash", synthetic=synthetic))
        print("wrote", plot_target_distributions(df, synthetic=synthetic))


if __name__ == "__main__":
    main()
