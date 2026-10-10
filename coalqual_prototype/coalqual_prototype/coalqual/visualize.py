"""
Figures.

Every figure carries a SIMULATED DATA stamp when it is produced from the
synthetic dataset. Do not remove the stamp for a submission deck - a judge
who spots an unmarked simulated result will discount the whole entry.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DIAGNOSTIC_FEATURES, FIG_DIR, TARGET_META, WAVELENGTHS

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 200, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})

ACCENT = "#B4CF3A"      # deck green
INK = "#2b2b2b"


def _stamp(fig, synthetic: bool) -> None:
    if not synthetic:
        return
    fig.text(0.995, 0.005, "SIMULATED DATA - pipeline demonstration only",
             ha="right", va="bottom", fontsize=7, color="#c0392b", alpha=0.9)


def _save(fig, path: Path, synthetic: bool) -> Path:
    _stamp(fig, synthetic)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
def plot_spectra(df: pd.DataFrame, colour_by: str = "Ash", synthetic: bool = True,
                 out: Path | None = None, title: str | None = None) -> Path:
    """Reflectance spectra coloured by a quality index, with diagnostic bands."""
    out = Path(out or FIG_DIR / f"spectra_by_{colour_by}.png")
    X = df[[f"R{w}" for w in WAVELENGTHS]].to_numpy(float)

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    if colour_by in df.columns:
        v = df[colour_by].to_numpy(float)
        norm = plt.Normalize(np.nanpercentile(v, 2), np.nanpercentile(v, 98))
        cmap = plt.get_cmap("inferno")
        for row, val in zip(X, v):
            ax.plot(WAVELENGTHS, row, lw=0.6, alpha=0.65, color=cmap(norm(val)))
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
        meta = TARGET_META.get(colour_by, {})
        cb.set_label(f"{meta.get('label', colour_by)} ({meta.get('unit', '')})")
    else:
        ax.plot(WAVELENGTHS, X.T, lw=0.6, alpha=0.5, color=INK)

    ymax = np.nanpercentile(X, 99.5)
    for centre, (lo, hi, name, _c) in DIAGNOSTIC_FEATURES.items():
        if centre < 1100:
            continue
        ax.axvspan(lo, hi, color=ACCENT, alpha=0.16, lw=0)
        ax.text(centre, ymax * 1.02, str(centre), ha="center", fontsize=6.5, color="#555")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance (%)")
    ax.set_title(title or f"VIS-SWIR reflectance, n = {len(df)} scan locations")
    ax.set_xlim(WAVELENGTHS.min(), WAVELENGTHS.max())
    return _save(fig, out, synthetic)


def plot_scatter_grid(store: dict, synthetic: bool = True,
                      out: Path | None = None) -> Path:
    """Measured vs predicted for every index, calibration + held-out set."""
    out = Path(out or FIG_DIR / "predicted_vs_measured.png")
    targets = list(store.keys())
    ncol = min(3, len(targets))
    nrow = int(np.ceil(len(targets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, t in zip(axes, targets):
        a = store[t]
        meta = TARGET_META.get(t, {})
        has_pred = len(a["y_pred"]) > 0
        ax.scatter(a["y_cal"], a["y_cal_hat"], s=16, marker="s",
                   facecolor="none", edgecolor=INK, lw=0.7, label="calibration (CV)")
        if has_pred:
            ax.scatter(a["y_pred"], a["y_pred_hat"], s=22, color="#1f9bd1",
                       alpha=0.85, label="prediction (held out)")
        ys = np.concatenate([a["y_cal"], a["y_pred"]]) if has_pred else a["y_cal"]
        lo, hi = float(np.min(ys)), float(np.max(ys))
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", lw=1, color="#888")

        from .validation import regression_metrics
        if has_pred:
            m = regression_metrics(a["y_pred"], a["y_pred_hat"]); tag = "p"
        else:
            m = regression_metrics(a["y_cal"], a["y_cal_hat"]); tag = "cv"
        ax.set_title(f"{meta.get('label', t)} ({meta.get('unit','')})", fontsize=9)
        ax.text(0.04, 0.96,
                f"$R^2_{{{tag}}}$={m['R2']:.3f}\nRMSE={m['RMSE']:.3g}\nRPD={m['RPD']:.2f}",
                transform=ax.transAxes, va="top", fontsize=8)
        ax.set_xlabel("Laboratory reference")
        ax.set_ylabel("Predicted")
    for ax in axes[len(targets):]:
        ax.axis("off")
    axes[0].legend(fontsize=7, loc="lower right", frameon=False)
    return _save(fig, out, synthetic)


def plot_benchmark_heatmap(full: pd.DataFrame, metric: str = "RMSE_cv",
                           synthetic: bool = True, out: Path | None = None) -> Path:
    """Selector x model grid, one panel per index, shaded by CV error."""
    out = Path(out or FIG_DIR / "benchmark_heatmap.png")
    targets = list(dict.fromkeys(full["target"]))
    fig, axes = plt.subplots(1, len(targets), figsize=(2.9 * len(targets), 3.3))
    axes = np.atleast_1d(axes)

    for ax, t in zip(axes, targets):
        sub = full[full.target == t]
        piv = sub.pivot_table(index="selector", columns="model", values=metric)
        im = ax.imshow(piv.values, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=7)
        best = np.unravel_index(np.nanargmin(piv.values), piv.shape)
        ax.scatter(best[1], best[0], marker="*", s=90, color="w", edgecolor="k", lw=0.5)
        ax.set_title(t, fontsize=9)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    fig.suptitle(f"Cross-validated {metric} - star marks the selected pipeline; "
                 "blank cells = tree ensembles are not run on the full-spectrum table",
                 fontsize=9)
    return _save(fig, out, synthetic)


def plot_quality_log(pred: pd.DataFrame, depth_col: str = "borehole_depth_m",
                     synthetic: bool = True, out: Path | None = None) -> Path:
    """Continuous down-core quality log - the deliverable a mine planner reads."""
    out = Path(out or FIG_DIR / "quality_log.png")
    cols = [c for c in ("pred_MC", "pred_Ash", "pred_VM", "pred_FC", "pred_CV")
            if c in pred.columns]
    d = pred.sort_values(depth_col)
    fig, axes = plt.subplots(1, len(cols), figsize=(1.75 * len(cols), 6.0), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, c in zip(axes, cols):
        t = c.replace("pred_", "")
        meta = TARGET_META.get(t, {})
        ax.plot(d[c], d[depth_col], "-o", ms=2.4, lw=1.0, color=INK)
        ax.fill_betweenx(d[depth_col], d[c].min(), d[c], alpha=0.18, color=ACCENT)
        ax.set_xlabel(f"{meta.get('label', t)}\n({meta.get('unit','')})", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Borehole depth (m)")
    axes[0].invert_yaxis()
    fig.suptitle("Predicted coal quality log", fontsize=10)
    return _save(fig, out, synthetic)


def plot_feature_importance(store: dict, target: str, top_n: int = 20,
                            synthetic: bool = True, out: Path | None = None) -> Path:
    """Permutation importance of the retained features for one index."""
    from sklearn.inspection import permutation_importance
    out = Path(out or FIG_DIR / f"importance_{target}.png")
    a = store[target]
    est, names = a["estimator"], a["features"]
    Xc = a.get("X_cal_sel")
    if Xc is None:
        raise ValueError("artefacts lack X_cal_sel; re-run run_target()")

    r = permutation_importance(est, Xc, a["y_cal"], n_repeats=20,
                               random_state=0, scoring="neg_root_mean_squared_error")
    order = np.argsort(r.importances_mean)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(5.2, 0.26 * len(order) + 1.2))
    ax.barh(range(len(order)), r.importances_mean[order][::-1],
            xerr=r.importances_std[order][::-1], color=ACCENT, edgecolor=INK, lw=0.4)
    ax.set_yticks(range(len(order)), [names[i] for i in order][::-1], fontsize=7)
    ax.set_xlabel("Increase in RMSE when permuted")
    ax.set_title(f"{target}: most informative features")
    return _save(fig, out, synthetic)


def plot_target_distributions(df: pd.DataFrame, synthetic: bool = True,
                              out: Path | None = None) -> Path:
    """Histogram of the five indices - shows the calibration range covered."""
    out = Path(out or FIG_DIR / "target_distributions.png")
    targets = [t for t in TARGET_META if t in df.columns]
    fig, axes = plt.subplots(1, len(targets), figsize=(2.3 * len(targets), 2.6))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, targets):
        meta = TARGET_META[t]
        ax.hist(df[t].dropna(), bins=24, color=ACCENT, edgecolor=INK, lw=0.4)
        ax.set_title(f"{meta['label']}", fontsize=8.5)
        ax.set_xlabel(meta["unit"], fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("count")
    fig.suptitle(f"Reference distribution, n = {len(df)}", fontsize=9)
    return _save(fig, out, synthetic)
