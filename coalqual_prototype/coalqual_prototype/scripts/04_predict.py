#!/usr/bin/env python3
"""
Step 4 - score new scan locations with the trained models.

    python scripts/04_predict.py --data outputs/tables/real_scan_features.csv

Writes
    outputs/tables/predictions.csv   five indices + CIL grade per scan location
    outputs/figures/quality_log.png  the down-core quality log
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from coalqual.config import MODEL_DIR, TABLE_DIR, TARGET_META
from coalqual.pipeline import predict
from coalqual.visualize import plot_quality_log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/tables/real_scan_features.csv")
    ap.add_argument("--models", default=str(MODEL_DIR))
    ap.add_argument("--out", default=str(Path(TABLE_DIR) / "predictions.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    pred = predict(df, model_dir=Path(args.models))
    pred.to_csv(args.out, index=False)
    print(f"scored {len(pred)} locations -> {args.out}\n")

    show = ["sample_id", "borehole_depth_m"] + \
           [f"pred_{t}" for t in TARGET_META if f"pred_{t}" in pred.columns] + \
           [c for c in ("pred_grade", "ad_tier") if c in pred.columns]
    print(pred[show].round(2).head(30).to_string(index=False))

    if "pred_grade" in pred.columns:
        print("\ndeclared grade mix:")
        print(pred["pred_grade"].value_counts().sort_index().to_string())
    if "ad_tier" in pred.columns:
        t = pred["ad_tier"].value_counts()
        print(f"\napplicability domain: {t.get('in', 0)} inside, {t.get('edge', 0)} "
              f"extrapolated, {t.get('out', 0)} outside -> send those to the lab")

    if "borehole_depth_m" in pred.columns and pred["borehole_depth_m"].notna().any():
        # Models trained on synthetic labels must not produce an unstamped plot.
        import joblib
        syn = any(
            joblib.load(p).get("trained_on_synthetic", False)
            for p in Path(args.models).glob("model_*.joblib")
        )
        print("\nwrote", plot_quality_log(pred, synthetic=syn))


if __name__ == "__main__":
    main()
