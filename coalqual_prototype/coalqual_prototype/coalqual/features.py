"""
Feature engineering.

Four feature families are built from a BoxScan scan location:

1. **Spectral**  - MSC reflectance at retained bands, first derivative,
   continuum-removed absorption depth / area / position at the diagnostic
   wavelengths listed in :data:`config.DIAGNOSTIC_FEATURES`.
2. **Geochemical** - pXRF element concentrations, normalised, plus mineral
   proxies (clay, pyrite, carbonate, light-element/organic fraction).
3. **Colour** - CIE L*a*b*, greyscale reflectance, chroma.
4. **Texture** (optional) - GLCM + histogram statistics from the RGB core
   image patch around the scan location.

The XRF block is what lifts this pipeline above a spectra-only model: the
light-element channel (LE) is a direct proxy for the organic matrix, and
Si+Al+Fe+Ca are a direct proxy for mineral matter, hence ash.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DIAGNOSTIC_FEATURES, WAVELENGTHS
from .preprocess import (
    MSC, absorption_area, absorption_depth, absorption_position,
    continuum_removal, first_derivative, screen_bands, sg_smooth, snv,
)

# --------------------------------------------------------------------------
# Spectral features
# --------------------------------------------------------------------------
def spectral_matrix(df: pd.DataFrame, wl: np.ndarray = WAVELENGTHS) -> np.ndarray:
    """Pull the R<nm> columns out of a feature table as a float array."""
    cols = [f"R{w}" for w in wl]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing reflectance columns, e.g. {missing[:5]}")
    return df[cols].to_numpy(float)


def diagnostic_features(X: np.ndarray, wl: np.ndarray) -> pd.DataFrame:
    """Continuum-removed depth / area / position at each diagnostic feature."""
    Xcr = continuum_removal(sg_smooth(X), wl)
    out = {}
    for centre, (lo, hi, name, _chem) in DIAGNOSTIC_FEATURES.items():
        out[f"d_{name}"] = absorption_depth(Xcr, wl, lo, hi)
        out[f"a_{name}"] = absorption_area(Xcr, wl, lo, hi)
        out[f"p_{name}"] = absorption_position(Xcr, wl, lo, hi) - centre
    return pd.DataFrame(out)


def band_ratio_features(X: np.ndarray, wl: np.ndarray) -> pd.DataFrame:
    """Physically motivated ratios and slopes.

    Ratios are scatter-insensitive, which matters for chip trays where
    particle size and surface orientation vary from point to point.
    """
    def R(nm):
        return X[:, int(np.argmin(np.abs(wl - nm)))]

    eps = 1e-9
    out = {
        # organic vs mineral character
        "r_2300_2200": R(2300) / (R(2200) + eps),
        "r_2150_2200": R(2150) / (R(2200) + eps),
        "r_1700_2200": R(1700) / (R(2200) + eps),
        # moisture
        "r_1900_1800": R(1900) / (R(1800) + eps),
        "r_1400_1300": R(1400) / (R(1300) + eps),
        # iron
        "r_900_650":  R(900) / (R(650) + eps),
        # overall brightness and spectral slope (ash tracks brightness)
        "mean_vis":   X[:, (wl >= 450) & (wl <= 700)].mean(axis=1),
        "mean_nir":   X[:, (wl >= 1000) & (wl <= 1700)].mean(axis=1),
        "mean_swir":  X[:, (wl >= 2000) & (wl <= 2400)].mean(axis=1),
        "slope_vnir": (R(1000) - R(500)) / 500.0,
        "slope_swir": (R(2400) - R(1600)) / 800.0,
        "log_bright": np.log10(np.clip(X[:, (wl >= 500) & (wl <= 2400)].mean(axis=1), 1e-3, None)),
    }
    out["r_swir_vis"] = out["mean_swir"] / (out["mean_vis"] + eps)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Geochemical features
# --------------------------------------------------------------------------
# Stoichiometric oxide factors, element -> oxide mass ratio.
_OXIDE_FACTOR = {
    "Si": 2.1393, "Al": 1.8895, "Fe": 1.4297, "Ca": 1.3992, "K": 1.2046,
    "Mg": 1.6582, "Ti": 1.6681, "Mn": 1.2912, "P": 2.2916, "S": 2.4972,
}
_MAJORS = ["Si", "Al", "Fe", "Ca", "K", "Mg", "Ti", "S", "P", "Mn"]


def xrf_features(df: pd.DataFrame) -> pd.DataFrame:
    """Element concentrations in wt%, plus mineral-matter proxies.

    ``LE`` is the pXRF light-element channel (everything below Mg that the
    detector cannot resolve: C, H, O, N). In coal it is dominated by the
    organic matrix, so ``LE`` anti-correlates strongly with ash.
    """
    out = pd.DataFrame(index=df.index)
    have = {}
    for el in _MAJORS + ["LE"]:
        col = f"xrf_{el}"
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce") / 1e4      # ppm -> wt%
            out[f"x_{el}"] = v
            have[el] = v

    if not have:
        return out

    def g(el):
        return have.get(el, pd.Series(0.0, index=df.index)).fillna(0.0)

    # mineral-matter proxies
    out["x_clay"] = _OXIDE_FACTOR["Si"] * g("Si") + _OXIDE_FACTOR["Al"] * g("Al")
    out["x_pyrite"] = 1.0 * g("Fe") + 1.0 * g("S")
    out["x_carb"] = _OXIDE_FACTOR["Ca"] * g("Ca") + _OXIDE_FACTOR["Mg"] * g("Mg")
    out["x_mineral_sum"] = sum(
        _OXIDE_FACTOR[e] * g(e) for e in _MAJORS if e in _OXIDE_FACTOR
    )
    out["x_si_al"] = g("Si") / (g("Al") + 1e-6)          # quartz vs clay balance
    out["x_fe_s"] = g("Fe") / (g("S") + 1e-6)            # pyrite vs siderite
    out["x_organic_proxy"] = 100.0 - out["x_mineral_sum"]

    if "LE" in have:
        out["x_LE_frac"] = g("LE") / 100.0
        out["x_ash_proxy"] = 100.0 - g("LE")

    if "magsus" in df.columns:
        out["magsus"] = pd.to_numeric(df["magsus"], errors="coerce")
    return out


def colour_features(df: pd.DataFrame) -> pd.DataFrame:
    """CIE colour metrics; darkness is a first-order ash/rank indicator."""
    out = pd.DataFrame(index=df.index)
    for c in ["col_L", "col_a", "col_b", "col_grey", "col_X", "col_Y", "col_Z"]:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    if {"col_a", "col_b"} <= set(out.columns):
        out["col_chroma"] = np.hypot(out["col_a"], out["col_b"])
        out["col_hue"] = np.degrees(np.arctan2(out["col_b"], out["col_a"]))
    return out


# --------------------------------------------------------------------------
# Texture features (optional; needs the RGB core image)
# --------------------------------------------------------------------------
def glcm_texture(patch: np.ndarray, levels: int = 64, distance: int = 1) -> dict:
    """Six GLCM measures averaged over 0/45/90/135 deg, plus histogram stats.

    ``patch`` is an HxW (grayscale) or HxWx3 (RGB) uint8 image crop centred on
    the scan location.
    """
    from skimage.feature import graycomatrix, graycoprops
    from skimage.color import rgb2gray

    if patch.ndim == 3:
        patch = (rgb2gray(patch) * 255).astype(np.uint8)
    q = (patch.astype(np.float64) / 256 * levels).astype(np.uint8)

    glcm = graycomatrix(
        q, distances=[distance], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels, symmetric=True, normed=True,
    )
    feats = {
        f"t_glcm_{p}": float(graycoprops(glcm, p).mean())
        for p in ("contrast", "dissimilarity", "homogeneity", "correlation",
                  "energy", "ASM")
    }
    p = patch.astype(float)
    hist, _ = np.histogram(p, bins=levels, range=(0, 255), density=True)
    hist = hist + 1e-12
    feats.update({
        "t_hs_mean": p.mean(), "t_hs_var": p.var(),
        "t_hs_skew": float(((p - p.mean()) ** 3).mean() / (p.std() ** 3 + 1e-9)),
        "t_hs_kurt": float(((p - p.mean()) ** 4).mean() / (p.std() ** 4 + 1e-9)),
        "t_hs_energy": float((hist ** 2).sum()),
        "t_hs_entropy": float(-(hist * np.log2(hist)).sum()),
    })
    return feats


def texture_from_boxscan(
    image_path, locations_cm, scale_cm_per_px: float,
    track_y_cm: float, patch_cm: float = 3.0,
) -> pd.DataFrame:
    """Extract GLCM/HS texture for each scan location from the core box image.

    ``locations_cm`` are the along-track positions from ``<box>.locations.csv``
    and ``track_y_cm`` the track offset from ``<box>.curation.csv``.
    """
    from PIL import Image

    img = np.asarray(Image.open(image_path).convert("RGB"))
    half = int(round(patch_cm / 2 / scale_cm_per_px))
    rows = []
    for x_cm in locations_cm:
        cx = int(round(x_cm / scale_cm_per_px))
        cy = int(round(track_y_cm / scale_cm_per_px))
        y0, y1 = max(0, cy - half), min(img.shape[0], cy + half)
        x0, x1 = max(0, cx - half), min(img.shape[1], cx + half)
        patch = img[y0:y1, x0:x1]
        rows.append(glcm_texture(patch) if patch.size else {})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def build_feature_table(
    df: pd.DataFrame,
    blocks,
    wl: np.ndarray = WAVELENGTHS,
    msc: MSC | None = None,
    band_mask: np.ndarray | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, MSC, np.ndarray]:
    """Turn a raw BoxScan table into a model-ready feature matrix.

    Set ``fit=False`` and pass the ``msc``/``band_mask`` learned on the
    calibration set to transform new data without leakage.
    """
    X = spectral_matrix(df, wl)

    if fit:
        band_mask = screen_bands(X, wl)
        msc = MSC().fit(X)
    if band_mask is None or msc is None:
        raise ValueError("band_mask and msc are required when fit=False")

    Xs = sg_smooth(X)
    Xmsc = msc.transform(Xs)

    parts: list[pd.DataFrame] = []
    if blocks.spectra:
        parts.append(pd.DataFrame(
            Xmsc[:, band_mask], columns=[f"S{w}" for w in wl[band_mask]], index=df.index))
    if blocks.derivative:
        fd = first_derivative(Xs, wl)
        parts.append(pd.DataFrame(
            fd[:, band_mask], columns=[f"D{w}" for w in wl[band_mask]], index=df.index))
    if blocks.band_depths:
        parts.append(diagnostic_features(X, wl).set_index(df.index))
    if blocks.ratios:
        parts.append(band_ratio_features(Xs, wl).set_index(df.index))
    if blocks.xrf:
        parts.append(xrf_features(df))
    if blocks.colour:
        parts.append(colour_features(df))
    if blocks.texture:
        tcols = [c for c in df.columns if c.startswith("t_")]
        if tcols:
            parts.append(df[tcols].apply(pd.to_numeric, errors="coerce"))

    F = pd.concat(parts, axis=1)
    F = F.loc[:, ~F.columns.duplicated()]
    return F, msc, band_mask
