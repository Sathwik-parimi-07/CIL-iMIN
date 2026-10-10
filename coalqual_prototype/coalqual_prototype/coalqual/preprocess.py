"""
Spectral pre-processing.

All functions take/return ``X`` of shape (n_samples, n_bands) in percent
reflectance, aligned to a wavelength vector ``wl``.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull

from .config import EXCLUDED_REGIONS, NOISE_LEVEL_MAX


# --------------------------------------------------------------------------
# Smoothing / scatter correction
# --------------------------------------------------------------------------
def sg_smooth(X: np.ndarray, window: int = 9, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay convolutional smoothing (Savitzky & Golay 1964)."""
    window = max(poly + 2, window | 1)  # force odd, > poly
    return savgol_filter(X, window_length=window, polyorder=poly, axis=1)


def first_derivative(X: np.ndarray, wl: np.ndarray, window: int = 11, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay first derivative, scaled to per-nm units."""
    window = max(poly + 2, window | 1)
    step = float(np.median(np.diff(wl)))
    return savgol_filter(X, window_length=window, polyorder=poly, deriv=1, delta=step, axis=1)


def snv(X: np.ndarray) -> np.ndarray:
    """Standard normal variate: per-spectrum centre and scale."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


class MSC:
    """Multiplicative scatter correction.

    Stateful so that the *training* mean spectrum is reused at prediction
    time; applying MSC with a per-batch mean is a classic source of leakage.
    """

    def __init__(self) -> None:
        self.reference_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MSC":
        self.reference_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.reference_ is None:
            raise RuntimeError("MSC.fit must be called before transform")
        ref = self.reference_
        out = np.empty_like(X, dtype=float)
        for i, row in enumerate(X):
            ok = np.isfinite(row) & np.isfinite(ref)
            if ok.sum() < 2:
                out[i] = row
                continue
            slope, intercept = np.polyfit(ref[ok], row[ok], 1)
            out[i] = (row - intercept) / (slope if slope != 0 else 1.0)
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------
# Continuum removal
# --------------------------------------------------------------------------
def continuum_removal(X: np.ndarray, wl: np.ndarray) -> np.ndarray:
    """Upper convex-hull continuum removal; returns hull-normalised reflectance.

    Values are <= 1, with absorption features appearing as dips below 1.
    """
    out = np.ones_like(X, dtype=float)
    for i, y in enumerate(X):
        ok = np.isfinite(y)
        if ok.sum() < 3:
            continue
        x_, y_ = wl[ok], y[ok]
        pts = np.column_stack([x_, y_])
        try:
            hull = ConvexHull(pts)
            idx = np.sort(hull.vertices)
            # Keep only the upper hull: walk left-to-right, monotone x.
            upper = [idx[0]]
            for k in idx[1:]:
                if x_[k] > x_[upper[-1]]:
                    upper.append(k)
            if x_[upper[-1]] < x_[-1]:
                upper.append(len(x_) - 1)
            cont = np.interp(x_, x_[upper], y_[upper])
        except Exception:                                    # degenerate spectrum
            cont = np.maximum.accumulate(y_)
        cont[cont == 0] = 1e-9
        out[i, ok] = y_ / cont
    return np.clip(out, 0, 1.5)


def absorption_depth(Xcr: np.ndarray, wl: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Maximum depth (1 - CR) of the deepest feature inside [lo, hi] nm."""
    m = (wl >= lo) & (wl <= hi)
    if m.sum() == 0:
        return np.zeros(len(Xcr))
    return 1.0 - np.nanmin(Xcr[:, m], axis=1)


def absorption_area(Xcr: np.ndarray, wl: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Integrated area of the absorption feature inside [lo, hi] nm."""
    m = (wl >= lo) & (wl <= hi)
    if m.sum() < 2:
        return np.zeros(len(Xcr))
    return np.trapezoid(1.0 - Xcr[:, m], wl[m], axis=1)


def absorption_position(Xcr: np.ndarray, wl: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Wavelength of the deepest point inside [lo, hi] nm (feature asymmetry proxy)."""
    m = (wl >= lo) & (wl <= hi)
    if m.sum() == 0:
        return np.full(len(Xcr), np.nan)
    sub = Xcr[:, m]
    return wl[m][np.nanargmin(sub, axis=1)]


# --------------------------------------------------------------------------
# Band screening
# --------------------------------------------------------------------------
def noise_level(X: np.ndarray, window: int = 5, poly: int = 2) -> np.ndarray:
    """Per-band relative noise level (Mao et al. 2024, eqs. 3-5).

    ``noise = median_col(|X - SG(X)|) / mean_col(X)``
    """
    Xsg = savgol_filter(X, window_length=window | 1, polyorder=poly, axis=1)
    num = np.nanmedian(np.abs(X - Xsg), axis=0)
    den = np.nanmean(X, axis=0)
    den[den == 0] = np.nan
    return num / den


def screen_bands(
    X: np.ndarray,
    wl: np.ndarray,
    excluded_regions=EXCLUDED_REGIONS,
    noise_max: float = NOISE_LEVEL_MAX,
) -> np.ndarray:
    """Boolean keep-mask over bands: drops detector edges, joins and noisy bands."""
    keep = np.ones(len(wl), bool)
    for lo, hi in excluded_regions:
        keep &= ~((wl >= lo) & (wl <= hi))
    nl = noise_level(X)
    keep &= np.nan_to_num(nl, nan=np.inf) <= noise_max
    if keep.sum() < 20:                                      # never strip everything
        keep = np.ones(len(wl), bool)
        for lo, hi in excluded_regions:
            keep &= ~((wl >= lo) & (wl <= hi))
    return keep


# --------------------------------------------------------------------------
# Convenience bundle
# --------------------------------------------------------------------------
PREPROCESSORS = {
    "RAW": lambda X, wl: X,
    "SG": lambda X, wl: sg_smooth(X),
    "SNV": lambda X, wl: snv(sg_smooth(X)),
    "FD": lambda X, wl: first_derivative(X, wl),
    "CR": lambda X, wl: continuum_removal(sg_smooth(X), wl),
}
