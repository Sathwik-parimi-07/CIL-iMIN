#!/usr/bin/env python3
"""
Step 5 - ablation: what does each sensor contribute?

    python scripts/05_ablation.py --data outputs/tables/dataset_synthetic.csv

Trains three feature configurations per index with a fixed, fast selector
and compares held-out RMSE:

    spectra   VIS-SWIR reflectance + derivative + band depths + ratios
    xrf       pXRF chemistry + colour + magnetic susceptibility
    fused     everything

Writes
    outputs/tables/ablation.csv
    outputs/figures/ablation.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coalqual.config import FIG_DIR, TABLE_DIR, TARGET_META, TARGETS, FeatureBlocks
from coalqual.pipeline import run_target

CONFIGS = {
    "spectra": FeatureBlocks(spectra=True, derivative=True, band_depths=True,
                             ratios=True, xrf=False, colour=False),
    "xrf":     FeatureBlocks(spectra=False, derivative=False, band_depths=False,
                             ratios=False, xrf=True, colour=True),
    "fused":   FeatureBlocks(spectra=True, derivative=True, band_depths=True,
                             ratios=True, xrf=True, colour=True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/tables/dataset_synthetic.csv")
    ap.add_argument("--targets", nargs="*", default=TARGETS)
    ap.add_argument("--selector", default="CC")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    synthetic = bool(df.get("is_synthetic", pd.Series([False])).any())

    rows = []
    for t in args.targets:
        for name, blocks in CONFIGS.items():
            res, table, _ = run_target(df, t, blocks=blocks,
                                       selectors=(args.selector,), seed=args.seed,
                                       verbose=False)
            rows.append(dict(target=t, config=name, model=res.best_model,
                             n_features=res.n_features,
                             R2_cv=res.cv["R2"], RMSE_cv=res.cv["RMSE"],
                             R2_pred=res.pred["R2"], RMSE_pred=res.pred["RMSE"],
                             RPD_pred=res.pred["RPD"]))
            print(f"{t:<4} {name:<8} {res.best_model:<11} n_feat={res.n_features:<4} "
                  f"pred R2={res.pred['R2']:.3f} RMSE={res.pred['RMSE']:.3f} "
                  f"RPD={res.pred['RPD']:.2f}")

    out = pd.DataFrame(rows)
    path_csv = Path(TABLE_DIR) / "ablation.csv"
    if path_csv.exists():                       # merge so targets can run in batches
        old = pd.read_csv(path_csv)
        out = pd.concat([old[~old["target"].isin(out["target"].unique())], out],
                        ignore_index=True)
    out.to_csv(path_csv, index=False)

    # ---- figure: RMSE relative to the fused model, per index --------------
    order = [t for t in TARGETS if t in set(out["target"])]
    piv = out.pivot(index="target", columns="config", values="RMSE_pred").loc[order]
    rel = piv.div(piv["fused"], axis=0)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(piv))
    w = 0.26
    colours = {"spectra": "#7a8fa6", "xrf": "#c9a227", "fused": "#B4CF3A"}
    for i, cfg in enumerate(["spectra", "xrf", "fused"]):
        ax.bar(x + (i - 1) * w, rel[cfg], w, label=cfg, color=colours[cfg],
               edgecolor="#2b2b2b", lw=0.4)
        for xi, v, a in zip(x + (i - 1) * w, rel[cfg], piv[cfg]):
            ax.text(xi, v + 0.02, f"{a:.2f}", ha="center", fontsize=6.5, rotation=90)
    ax.axhline(1.0, color="#555", lw=0.8, ls="--")
    ax.set_xticks(x, [f"{t}\n({TARGET_META[t]['unit']})" for t in piv.index], fontsize=8)
    ax.set_ylabel("held-out RMSE / fused RMSE")
    ax.set_ylim(0, max(1.05, float(rel.values.max()) * 1.25))
    ax.set_title("Sensor ablation: fusion vs. each sensor alone (labels = absolute RMSE)",
                 fontsize=9)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    if synthetic:
        fig.text(0.995, 0.005, "SIMULATED DATA - pipeline demonstration only",
                 ha="right", va="bottom", fontsize=7, color="#c0392b")
    fig.tight_layout()
    path = Path(FIG_DIR) / "ablation.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"\nwrote {TABLE_DIR / 'ablation.csv'}\nwrote {path}")


if __name__ == "__main__":
    main()
