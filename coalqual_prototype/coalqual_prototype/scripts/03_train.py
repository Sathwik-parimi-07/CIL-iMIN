#!/usr/bin/env python3
"""
Step 3 - train and benchmark all five quality indices.

    python scripts/03_train.py --data outputs/tables/dataset_synthetic.csv

Writes
    outputs/models/model_<TARGET>.joblib
    outputs/tables/model_summary.csv      one row per index: winner + metrics
    outputs/tables/benchmark_full.csv     every selector x model combination
    outputs/figures/*.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from coalqual.config import DEFAULT_BLOCKS, FeatureBlocks, TABLE_DIR, TARGETS
from coalqual.pipeline import run_all
from coalqual.visualize import (plot_benchmark_heatmap, plot_feature_importance,
                                plot_scatter_grid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/tables/dataset_synthetic.csv")
    ap.add_argument("--targets", nargs="*", default=TARGETS)
    ap.add_argument("--selectors", nargs="*",
                    default=["CC", "Boruta", "CARS", "SPA", "ALL"])
    ap.add_argument("--no-xrf", action="store_true",
                    help="spectra only, for the ablation slide")
    ap.add_argument("--no-spectra", action="store_true",
                    help="XRF + colour only, for the ablation slide")
    ap.add_argument("--fast", action="store_true",
                    help="skip Boruta and CARS (they dominate runtime)")
    ap.add_argument("--test-fraction", type=float, default=None,
                    help="SPXY hold-out fraction; 0 = LOOCV only. Default 0.25, "
                         "or 0 automatically when n < 40")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    synthetic = bool(df.get("is_synthetic", pd.Series([False])).any())
    print(f"{len(df)} samples from {args.data}"
          f"{'  [SYNTHETIC]' if synthetic else '  [REAL LAB LABELS]'}")

    blocks = FeatureBlocks(**DEFAULT_BLOCKS.as_dict())
    if args.no_xrf:
        blocks.xrf = False
    if args.no_spectra:
        blocks.spectra = blocks.derivative = blocks.band_depths = blocks.ratios = False
    print(f"feature blocks: {blocks.as_dict()}")

    selectors = (["CC", "SPA", "ALL"] if args.fast else args.selectors)
    n_lab = int(df[[t for t in args.targets if t in df.columns]].notna().all(axis=1).sum())
    tf = args.test_fraction
    if tf is None:
        tf = 0.25 if n_lab >= 40 else 0.0
        if tf == 0.0:
            print(f"only {n_lab} labelled samples: reporting LOOCV, no hold-out split")
    summary, full, store = run_all(df, targets=args.targets, blocks=blocks,
                                   selectors=tuple(selectors), seed=args.seed,
                                   test_fraction=tf)

    print("\n================ SUMMARY ================")
    cols = ["target", "best_selector", "best_model", "n_features",
            "cv_R2", "cv_RMSE", "pred_R2", "pred_RMSE", "pred_RPD", "pred_RPIQ"]
    print(summary[cols].round(3).to_string(index=False))
    if synthetic:
        print("\n!! These numbers come from SIMULATED labels. They demonstrate that")
        print("!! the pipeline runs end to end; they are not measured accuracy.")

    print("\nwrote", plot_scatter_grid(store, synthetic=synthetic))
    print("wrote", plot_benchmark_heatmap(full, synthetic=synthetic))
    for t in args.targets:
        if t in store:
            try:
                print("wrote", plot_feature_importance(store, t, synthetic=synthetic))
            except Exception as exc:                               # noqa: BLE001
                print(f"(importance {t} skipped: {exc})")


if __name__ == "__main__":
    main()
