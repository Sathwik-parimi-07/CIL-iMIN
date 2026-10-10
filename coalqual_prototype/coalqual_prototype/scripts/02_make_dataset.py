#!/usr/bin/env python3
"""
Step 2 - build the modelling dataset.

While the laboratory proximate analysis is running, generate a placeholder
population anchored on the real BoxScan endmembers::

    python scripts/02_make_dataset.py --n 200

Once the lab results arrive, merge them onto the real scans instead::

    python scripts/02_make_dataset.py --real outputs/tables/real_scan_features.csv \
                                      --labels outputs/tables/LAB_LABEL_TEMPLATE.csv

Everything downstream is identical in both cases.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from coalqual.config import TABLE_DIR, TARGETS
from coalqual.simulate import generate_dataset
from coalqual.visualize import plot_spectra, plot_target_distributions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of synthetic samples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seams", type=int, default=5)
    ap.add_argument("--real", default="outputs/tables/real_scan_features.csv",
                    help="real scan features, used to anchor the endmembers")
    ap.add_argument("--labels", default=None,
                    help="filled LAB_LABEL_TEMPLATE.csv; switches to real mode")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    real = None
    if args.real and Path(args.real).exists():
        real = pd.read_csv(args.real)
        print(f"anchored on {len(real)} real scan locations from {args.real}")
    else:
        print("no real scans found - falling back to literature-shaped endmembers")

    # ---------------- real mode ----------------------------------------
    if args.labels:
        if real is None:
            raise SystemExit("--labels needs --real")
        lab = pd.read_csv(args.labels)
        keep = ["sample_id"] + [t for t in TARGETS if t in lab.columns]
        df = real.merge(lab[keep], on="sample_id", how="left")
        df["is_synthetic"] = False
        n_lab = int(df[TARGETS].notna().all(axis=1).sum())
        print(f"merged laboratory labels: {n_lab}/{len(df)} rows fully labelled")
        out = Path(args.out or Path(TABLE_DIR) / "dataset_real.csv")
    # ---------------- synthetic mode -----------------------------------
    else:
        df = generate_dataset(args.n, real_df=real, seed=args.seed,
                              n_seams=args.seams)
        print(f"generated {len(df)} synthetic samples across {args.seams} seams")
        print(df[TARGETS].describe().round(2).to_string())
        print("\ngrade distribution:")
        print(df["grade"].value_counts().sort_index().to_string())
        out = Path(args.out or Path(TABLE_DIR) / "dataset_synthetic.csv")

    df.to_csv(out, index=False)
    print(f"\nwrote {out}")

    syn = bool(df.get("is_synthetic", pd.Series([False])).any())
    if "Ash" in df.columns and df["Ash"].notna().any():
        print("wrote", plot_spectra(df, colour_by="Ash", synthetic=syn))
        print("wrote", plot_target_distributions(df, synthetic=syn))


if __name__ == "__main__":
    main()
