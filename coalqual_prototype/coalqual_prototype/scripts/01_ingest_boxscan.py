#!/usr/bin/env python3
"""
Step 1 - ingest real Geotek BoxScan sessions.

    python scripts/01_ingest_boxscan.py --root data/raw

Produces
    outputs/tables/real_scan_features.csv   one row per scan location
    outputs/tables/LAB_LABEL_TEMPLATE.csv   hand this to the lab; paste the
                                            proximate + GCV results into it
    outputs/figures/real_spectra.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from coalqual.config import TABLE_DIR, WAVELENGTHS
from coalqual.io_boxscan import load_sessions, write_label_template
from coalqual.preprocess import noise_level
from coalqual.visualize import plot_spectra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw",
                    help="folder containing the unzipped BoxScan sessions")
    ap.add_argument("--track", type=int, default=None,
                    help="keep only this track number (coal chip tray)")
    args = ap.parse_args()

    df = load_sessions(args.root)
    print(f"loaded {len(df)} scan locations from {df['session'].nunique()} session(s)")
    print(f"  sessions : {sorted(df['session'].unique())}")
    if "track" in df:
        print(f"  tracks   : {sorted(df['track'].dropna().unique().tolist())}")

    if args.track is not None:
        df = df[df["track"] == args.track].reset_index(drop=True)
        print(f"  filtered to track {args.track}: {len(df)} locations")

    X = df[[f"R{w}" for w in WAVELENGTHS]].to_numpy(float)
    print(f"  spectra  : {X.shape[0]} x {X.shape[1]} bands, "
          f"{WAVELENGTHS.min()}-{WAVELENGTHS.max()} nm")
    print(f"  reflect. : {np.nanmin(X):.2f}-{np.nanmax(X):.2f} %")
    nl = noise_level(X)
    print(f"  noise    : median {np.nanmedian(nl)*100:.3f} %, "
          f"{int((nl > 0.01).sum())} bands above the 1 % screen")

    xrf = [c for c in df.columns if c.startswith("xrf_")]
    print(f"  pXRF     : {len(xrf)} channels, "
          f"{int(df[xrf].notna().all(axis=1).sum())}/{len(df)} complete rows")

    out = Path(TABLE_DIR) / "real_scan_features.csv"
    df.to_csv(out, index=False)
    tpl = write_label_template(df, Path(TABLE_DIR) / "LAB_LABEL_TEMPLATE.csv")
    print(f"\nwrote {out}")
    print(f"wrote {tpl}   <- fill MC / Ash / VM / FC / CV from the lab")

    try:
        fig = plot_spectra(df, colour_by="col_L", synthetic=False,
                           title=f"Real BoxScan spectra, n={len(df)} (colour = L*)")
        print(f"wrote {fig}")
    except Exception as exc:                                      # noqa: BLE001
        print(f"(figure skipped: {exc})")


if __name__ == "__main__":
    main()
