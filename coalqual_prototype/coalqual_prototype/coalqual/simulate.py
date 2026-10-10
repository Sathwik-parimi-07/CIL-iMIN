"""
Synthetic coal dataset generator - a *stand-in*, not a result.

Purpose: let the whole pipeline (ingest -> preprocess -> select -> model ->
map -> report) be built, benchmarked and demonstrated at n = 100-300 while
the laboratory proximate analysis is still running. Every synthetic row is
schema-identical to a real BoxScan scan location, so swapping in real data is
a one-line change and nothing downstream is touched.

What is *real* here
-------------------
* the spectral shape, because the endmembers are extracted from the actual
  BoxScan measurements in ``10septestscan01/02``;
* the band-wise noise level, estimated from those same spectra with the
  Savitzky-Golay residual method (Mao et al. 2024, eqs. 3-5);
* the wavelength grid, XRF channel list and column names;
* the proximate-analysis mass balance and the GCV correlation;
* the laboratory repeatability applied to the reference labels.

What is *invented*
------------------
* the coal population itself (ash, moisture, rank, seam identity);
* the exact coupling strengths between chemistry and absorption depth.

So model **rankings and the working pipeline** transfer to the real data;
**absolute R^2 values do not**. Every artefact this module feeds carries an
``is_synthetic`` flag, and the plotting code stamps figures accordingly.

References for the physics
--------------------------
Cloutis (2003) Fuel 82:2239; Fysh et al. (1985) Appl. Spectrosc. 39:354;
Mao et al. (2024) Appl. Sci. 14:7920; Wyman, Sloan & Shirley (2013) JCGT 2(2)
for the analytic CIE colour-matching functions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LAB_REPEATABILITY, TARGETS, WAVELENGTHS, gcv_to_grade
from .preprocess import noise_level, sg_smooth

# --------------------------------------------------------------------------
# CIE colour, so the synthetic L*a*b* is consistent with the synthetic spectrum
# --------------------------------------------------------------------------
def _piecewise_gauss(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def cie_xyz_from_reflectance(R_percent: np.ndarray, wl: np.ndarray) -> np.ndarray:
    """CIE 1931 2-degree XYZ under an equal-energy illuminant, Y in percent.

    Matches the BoxScan convention where a perfect diffuser gives Y = 100 and
    L* = 116 f(Y/100) - 16.
    """
    m = (wl >= 380) & (wl <= 780)
    lam, R = wl[m].astype(float), R_percent[:, m]

    xb = (1.056 * _piecewise_gauss(lam, 599.8, 37.9, 31.0)
          + 0.362 * _piecewise_gauss(lam, 442.0, 16.0, 26.7)
          - 0.065 * _piecewise_gauss(lam, 501.1, 20.4, 26.2))
    yb = (0.821 * _piecewise_gauss(lam, 568.8, 46.9, 40.5)
          + 0.286 * _piecewise_gauss(lam, 530.9, 16.3, 31.1))
    zb = (1.217 * _piecewise_gauss(lam, 437.0, 11.8, 36.0)
          + 0.681 * _piecewise_gauss(lam, 459.0, 26.0, 13.8))

    norm = np.trapezoid(yb, lam)
    X = np.trapezoid(R * xb, lam, axis=1) / norm
    Y = np.trapezoid(R * yb, lam, axis=1) / norm
    Z = np.trapezoid(R * zb, lam, axis=1) / norm
    return np.column_stack([X, Y, Z])


def xyz_to_lab(XYZ: np.ndarray, white=(100.0, 100.0, 100.0)) -> np.ndarray:
    def f(t):
        d = 6 / 29
        return np.where(t > d ** 3, np.cbrt(t), t / (3 * d ** 2) + 4 / 29)

    fx, fy, fz = (f(XYZ[:, i] / white[i]) for i in range(3))
    return np.column_stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)])


# --------------------------------------------------------------------------
# Endmembers learned from the real scans
# --------------------------------------------------------------------------
@dataclass
class SpectralBasis:
    wl: np.ndarray
    organic: np.ndarray       # low-ash, dark endmember
    mineral: np.ndarray       # high-ash, brighter endmember
    ferruginous: np.ndarray   # Fe-rich endmember
    noise_rel: np.ndarray     # per-band relative noise
    n_real: int
    # Residual-shape model learned from the real scans (see learn_basis):
    #   log10 R_real = mixture(endmembers) + shape_pcs^T (B @ chem + eps)
    shape_pcs: np.ndarray | None = None      # (k, n_bands) in log10 units
    shape_B: np.ndarray | None = None        # (k, n_chem+1) ridge coefficients
    shape_eps: np.ndarray | None = None      # (k,) residual std of the scores
    shape_chem: tuple = ("le", "fe", "sial", "ca")
    shape_mean: np.ndarray | None = None


def learn_basis(real_df: pd.DataFrame | None, wl: np.ndarray = WAVELENGTHS) -> SpectralBasis:
    """Extract organic / mineral / ferruginous endmembers from real BoxScan points.

    Ranking uses the pXRF light-element channel: high LE means an
    organic-dominated (low-ash) target, low LE a mineral-dominated one.
    Falls back to literature-shaped synthetic endmembers if no real data is
    supplied, so the generator still runs standalone.
    """
    if real_df is None or len(real_df) < 4:
        return _fallback_basis(wl)

    cols = [f"R{w}" for w in wl]
    X = real_df[cols].to_numpy(float)
    ok = np.isfinite(X).all(axis=1)
    X = X[ok]
    sub = real_df.loc[ok]

    if "xrf_LE" in sub.columns and sub["xrf_LE"].notna().sum() >= 4:
        score = sub["xrf_LE"].astype(float).to_numpy()        # high = organic
    else:                                                      # darker = more coal
        score = -X[:, (wl >= 500) & (wl <= 2400)].mean(axis=1)
    fe = (sub["xrf_Fe"].astype(float).to_numpy()
          if "xrf_Fe" in sub.columns else np.zeros(len(sub)))

    good = np.isfinite(score)
    k = max(2, int(0.25 * good.sum()))
    order = np.argsort(np.where(good, score, -np.inf))

    organic = np.nanmean(X[order[-k:]], axis=0)
    mineral = np.nanmean(X[order[:k]], axis=0)
    fe_order = np.argsort(np.nan_to_num(fe))
    ferruginous = np.nanmean(X[fe_order[-2:]], axis=0)        # the Fe-rich extreme

    organic, mineral, ferruginous = (
        sg_smooth(np.atleast_2d(v))[0] for v in (organic, mineral, ferruginous)
    )
    # Guarantee the ordering the physics assumes: mineral matter is brighter.
    if mineral.mean() < organic.mean():
        organic, mineral = mineral, organic

    nrel = np.nan_to_num(noise_level(X), nan=2e-3)
    nrel = np.clip(nrel, 3e-4, 2e-2)
    basis = SpectralBasis(wl, organic, mineral, ferruginous, nrel, int(len(X)))
    _learn_shape_model(basis, X, sub)
    return basis


def _chem_vector(le_pct, fe_pct, sial_pct, ca_pct):
    """Chemistry regressors for the shape model, on comparable scales."""
    return np.column_stack([
        (np.asarray(le_pct, float) - 75.0) / 10.0,
        np.log1p(np.asarray(fe_pct, float)),
        (np.asarray(sial_pct, float) - 15.0) / 8.0,
        np.log1p(np.asarray(ca_pct, float)),
        np.ones(len(np.atleast_1d(le_pct))),
    ])


def _learn_shape_model(basis: SpectralBasis, X: np.ndarray, sub: pd.DataFrame,
                       n_pcs: int = 6, ridge: float = 1.0) -> None:
    """Learn the spectral-shape variation the endmember mixture cannot explain.

    Each real spectrum is projected onto the organic-mineral axis in log10
    space; the residual is what the two-endmember model misses (the 900 nm
    Fe feature, VNIR curvature, feature asymmetry, detector-join artefacts).
    Residuals are compressed with PCA and their scores regressed on the real
    pXRF chemistry, so a synthetic sample with a given LE / Fe / Si+Al / Ca
    inherits the shape a real chip of that chemistry shows, plus the scatter
    real chips show around it.
    """
    need = {"xrf_LE", "xrf_Fe", "xrf_Si", "xrf_Al", "xrf_Ca"}
    if not need <= set(sub.columns) or len(X) < 8:
        return
    ok = sub[list(need)].notna().all(axis=1).to_numpy()
    if ok.sum() < 8:
        return
    Xl = np.log10(np.clip(X[ok], 0.2, None))
    lo = np.log10(np.clip(basis.organic, 0.2, None))
    lm = np.log10(np.clip(basis.mineral, 0.2, None))
    axis = lm - lo
    # per-sample mixing fraction by least squares along the endmember axis
    f = np.clip(((Xl - lo) @ axis) / (axis @ axis), -0.2, 1.3)
    resid = Xl - (lo + f[:, None] * axis)

    k = int(min(n_pcs, ok.sum() - 2))
    mu = resid.mean(0)
    U, S_, Vt = np.linalg.svd(resid - mu, full_matrices=False)
    pcs = Vt[:k]
    scores = (resid - mu) @ pcs.T

    x = _chem_vector(sub.loc[ok, "xrf_LE"] / 1e4, sub.loc[ok, "xrf_Fe"] / 1e4,
                     (sub.loc[ok, "xrf_Si"] + sub.loc[ok, "xrf_Al"]) / 1e4,
                     sub.loc[ok, "xrf_Ca"] / 1e4)
    B = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ scores).T   # (k, p)
    eps = (scores - x @ B.T).std(0, ddof=1) + 1e-6

    basis.shape_pcs = pcs
    basis.shape_B = B
    basis.shape_eps = eps
    basis.shape_mean = mu


def _fallback_basis(wl: np.ndarray) -> SpectralBasis:
    """Literature-shaped endmembers for when no real scan is available."""
    x = (wl - wl.min()) / (wl.max() - wl.min())
    organic = 2.0 + 6.0 * x ** 1.3
    mineral = 5.0 + 18.0 * x ** 1.1
    ferruginous = organic * (1 + 0.35 * np.exp(-0.5 * ((wl - 900) / 180) ** 2))
    noise = np.full_like(wl, 2e-3, dtype=float)
    return SpectralBasis(wl, organic, mineral, ferruginous, noise, 0)


# --------------------------------------------------------------------------
# Chemistry population
# --------------------------------------------------------------------------
def sample_chemistry(n: int, rng: np.random.Generator, n_seams: int = 5,
                     parting_frac: float = 0.08) -> pd.DataFrame:
    """Draw a realistic Indian (Gondwana) coal population.

    Samples are grouped into seams so the dataset has the clustered structure
    a real sampling campaign produces. Clustering is what makes a random
    train/test split optimistic and an SPXY or seam-grouped split honest.
    """
    seam_id = rng.integers(0, n_seams, n)
    # Stratify rather than draw uniformly: a real sampling campaign
    # deliberately targets a spread of grades, and a lucky draw of five
    # low-ash seams would leave the calibration blind above G10.
    seam_ash = np.linspace(20, 46, n_seams) + rng.normal(0, 2.0, n_seams)
    seam_rank = np.linspace(0.30, 0.75, n_seams) + rng.normal(0, 0.05, n_seams)
    order = rng.permutation(n_seams)
    seam_ash, seam_rank = seam_ash[order], seam_rank[order]

    # rank latent: 0 = low rank / high volatile, 1 = higher rank
    rank = np.clip(seam_rank[seam_id] + rng.normal(0, 0.12, n), 0.02, 0.98)

    ash = np.clip(seam_ash[seam_id] + rng.normal(0, 6.0, n)
                  + rng.standard_t(6, n) * 2.0, 8.0, 58.0)
    # Every real seam log carries shaly partings and roof/floor contacts; a
    # screening tool that has never seen one will mis-grade them silently.
    parting = rng.random(n) < parting_frac
    ash = np.where(parting, rng.uniform(55.0, 82.0, n), ash)

    # inherent moisture falls with rank; air-dried basis
    mc = np.clip(1.2 + 9.0 * (1 - rank) ** 1.5 * rng.uniform(0.5, 1.4, n)
                 + rng.normal(0, 0.5, n), 0.5, 13.0)

    vm_daf = np.clip(22.0 + 24.0 * (1 - rank) + rng.normal(0, 2.0, n), 14.0, 48.0)
    vm = vm_daf * (100.0 - ash - mc) / 100.0
    fc = 100.0 - ash - mc - vm

    # keep the mass balance physical (partings legitimately have FC ~ 5-15)
    floor = np.where(parting, 4.0, 12.0)
    bad = fc < floor
    if bad.any():
        vm[bad] = np.maximum(100.0 - ash[bad] - mc[bad] - floor[bad], 1.0)
        fc = 100.0 - ash - mc - vm

    # Gross calorific value, air-dried basis (kcal/kg).
    # Base term is the standard Indian proximate correlation
    #   GCV = 85.6 [100 - (1.1 A + M)] - 60 M
    # with a rank premium because carbon-rich coal has a higher GCV at equal ash.
    gcv = (85.6 * (100.0 - (1.1 * ash + mc)) - 60.0 * mc
           + 260.0 * (rank - 0.5) + rng.normal(0, 90.0, n))
    gcv = np.clip(gcv, 600, 7200)

    # mineralogy split of the mineral matter (drives XRF and the SWIR features)
    # clay, quartz, pyrite, carbonate, siderite/Fe-oxide. Gondwana coals hold
    # most of their iron in siderite and oxides, not pyrite, so Fe and S are
    # only loosely coupled - exactly what the real pXRF shows.
    w = rng.dirichlet([6.0, 2.0, 0.5, 0.8, 1.2], n)
    fe_band = rng.random(n) < 0.08
    w[fe_band, 4] = rng.uniform(0.3, 0.7, fe_band.sum())
    w = w / w.sum(axis=1, keepdims=True)
    return pd.DataFrame({
        "seam": np.where(parting, "PARTING", [f"SEAM_{s+1}" for s in seam_id]),
        "rank_latent": rank,
        "MC": mc, "Ash": ash, "VM": vm, "FC": fc, "CV": gcv,
        "f_clay": w[:, 0], "f_quartz": w[:, 1], "f_pyrite": w[:, 2], "f_carb": w[:, 3],
        "f_sid": w[:, 4],
    })


# --------------------------------------------------------------------------
# Chemistry -> spectrum
# --------------------------------------------------------------------------
def _gauss(wl, centre, width):
    return np.exp(-0.5 * ((wl - centre) / width) ** 2)


def synth_spectra(chem: pd.DataFrame, basis: SpectralBasis,
                  rng: np.random.Generator,
                  representativeness_sd: float = 2.5,
                  maceral_sd: float = 0.09) -> np.ndarray:
    """Build reflectance spectra from the sampled chemistry.

    Mixing is linear in log-reflectance, which approximates an intimate
    (rather than areal) mixture for dark, strongly absorbing particulates.
    Diagnostic absorptions are then superimposed as Gaussian depth terms
    scaled by the responsible constituent, and finally the scan-to-scan
    nuisances that any real chip-tray measurement carries are applied.

    Two of those nuisances set the realistic accuracy ceiling and are the
    reason the demo does not report R^2 = 0.999:

    ``representativeness_sd``
        The spectrometer illuminates a ~1 cm spot; the laboratory crushes and
        splits a whole ~1 kg increment. On a chip tray those two never see
        exactly the same ash. This sub-sampling mismatch, not detector noise,
        dominates the residual in every field deployment of this technique.
    ``maceral_sd``
        Inertinite-rich coal is measurably brighter than vitrinite-rich coal
        at identical ash, so brightness alone cannot resolve ash. This forces
        the models onto the actual absorption features instead.
    """
    wl = basis.wl
    n = len(chem)

    lo = np.log10(np.clip(basis.organic, 0.2, None))
    lm = np.log10(np.clip(basis.mineral, 0.2, None))
    lf = np.log10(np.clip(basis.ferruginous, 0.2, None))

    # what the optical spot actually sees, not the bulk lab value
    ash_local = np.clip(chem["Ash"].to_numpy()
                        + rng.normal(0, representativeness_sd, n), 3.0, 88.0)
    mm = np.clip(1.1 * ash_local / 100.0, 0.0, 0.92)                # mineral matter
    fpy = chem["f_pyrite"].to_numpy() * mm
    ffe = (chem["f_pyrite"].to_numpy() + chem["f_sid"].to_numpy()) * mm

    base = (1 - mm)[:, None] * lo + mm[:, None] * lm
    base = base + (1.2 * ffe)[:, None] * (lf - lo)
    # maceral composition: brightness varies at constant ash
    base = base + rng.normal(0.0, maceral_sd, n)[:, None]

    # shape variation learned from the real scans, driven by the same
    # chemistry the synthetic pXRF will report (see _learn_shape_model)
    if basis.shape_pcs is not None:
        clay_pct = mm * 100 * chem["f_clay"].to_numpy()
        qtz_pct = mm * 100 * chem["f_quartz"].to_numpy()
        fe_pct = 0.466 * fpy * 100 + 0.48 * chem["f_sid"].to_numpy() * mm * 100
        sial = 0.21 * clay_pct + 0.467 * qtz_pct + 0.20 * clay_pct
        ca_pct = 0.32 * chem["f_carb"].to_numpy() * mm * 100
        le_pct = 100 - (sial + fe_pct + ca_pct) * 1.15
        xchem = _chem_vector(le_pct, fe_pct, sial, ca_pct)
        scores = xchem @ basis.shape_B.T + rng.normal(0, 1, (n, len(basis.shape_eps))) * basis.shape_eps
        base = base + basis.shape_mean + scores @ basis.shape_pcs
    R = 10 ** base

    mc = np.clip(chem["MC"].to_numpy() + rng.normal(0, 0.45, n), 0.1, None)
    ash = ash_local
    denom = np.maximum(100 - ash - mc, 1e-6)
    vm_daf = np.clip(chem["VM"].to_numpy() + rng.normal(0, 1.1, n), 1, None) / denom * 100
    fcd = np.clip(chem["FC"].to_numpy() + rng.normal(0, 1.4, n), 1, None) / denom * 100
    clay = chem["f_clay"].to_numpy() * mm
    carb = chem["f_carb"].to_numpy() * mm

    # depth of each diagnostic feature, as a fraction of local reflectance
    depth = np.zeros_like(R)
    def add(centre, width, strength):
        nonlocal depth
        depth = depth + np.clip(strength, 0, None)[:, None] * _gauss(wl, centre, width)

    add(1400, 32, 0.010 * mc)                      # OH / H2O overtone
    add(1900, 45, 0.019 * mc)                      # H2O combination
    add(2200, 22, 0.011 * clay * 100 / 30)         # Al-OH  -> clay -> ash
    add(2160, 18, 0.004 * clay * 100 / 30)         # Al-OH doublet shoulder
    add(2300, 30, 0.00042 * vm_daf)                # CH2/CH3 asym stretch + bend
    add(1700, 34, 0.00030 * vm_daf)                # CH first overtone
    add(2150, 24, 0.00028 * fcd)                   # aromatic C-H + C-C
    add(2350, 30, 0.010 * carb * 100 / 30)         # carbonate
    add(900, 160, 0.030 * ffe)                     # Fe electronic
    add(1150, 90, 0.014 * ffe)

    R = R * (1.0 - np.clip(depth, 0, 0.6))

    # ---- acquisition nuisances -----------------------------------------
    mult = rng.lognormal(0.0, 0.11, n)[:, None]
    offset = rng.normal(0.0, 0.10, n)[:, None]
    tilt = rng.normal(0.0, 0.22, n)[:, None] * ((wl - wl.mean()) / (np.ptp(wl) / 2))[None, :]
    R = R * mult + offset + tilt

    # ---- instrument noise, band-wise, from the real scans ---------------
    R = R * (1.0 + rng.normal(0.0, 1.0, R.shape) * basis.noise_rel[None, :])
    return np.clip(R, 0.05, None)


def synth_xrf(chem: pd.DataFrame, rng: np.random.Generator,
              representativeness_sd: float = 3.0) -> pd.DataFrame:
    """pXRF element concentrations (ppm) consistent with the mineral matter.

    Clay is treated as kaolinite/illite, quartz as SiO2, pyrite as FeS2 and
    carbonate as calcite/dolomite. ``LE`` is the light-element remainder
    (C, H, O, N), i.e. essentially the organic matrix.

    The pXRF window is a few millimetres across and sits at a different spot
    from the spectrometer, so it gets its *own* sub-sampling offset. Spectral
    and geochemical evidence therefore disagree slightly, which is exactly
    why fusing them beats either alone.
    """
    n = len(chem)
    ash_local = np.clip(chem["Ash"].to_numpy()
                        + rng.normal(0, representativeness_sd, n), 3.0, 88.0)
    mm_pct = np.clip(1.1 * ash_local, 0, 95)
    clay = mm_pct * chem["f_clay"].to_numpy()
    qtz = mm_pct * chem["f_quartz"].to_numpy()
    pyr = mm_pct * chem["f_pyrite"].to_numpy()
    carb = mm_pct * chem["f_carb"].to_numpy()
    sid = mm_pct * chem["f_sid"].to_numpy()

    # element wt% from stoichiometry
    si = 0.21 * clay + 0.467 * qtz
    al = 0.20 * clay
    k = 0.045 * clay * rng.uniform(0.4, 1.4, n)
    fe = 0.466 * pyr + 0.48 * sid + 0.01 * clay
    s = 0.534 * pyr + 0.35 * rng.uniform(0.1, 0.9, n)     # organic + pyritic S
    ca = 0.32 * carb
    mg = 0.06 * carb + 0.004 * clay + 0.02 * sid
    ti = 0.010 * clay * rng.uniform(0.5, 1.6, n)
    mn = 0.004 * carb
    p = 0.002 * mm_pct * rng.uniform(0.2, 2.0, n)

    els = dict(Si=si, Al=al, K=k, Fe=fe, S=s, Ca=ca, Mg=mg, Ti=ti, Mn=mn, P=p)
    out = {}
    for el, v in els.items():
        rel = rng.normal(1.0, 0.06, n)                    # pXRF counting error
        out[f"xrf_{el}"] = np.clip(v * rel, 0, None) * 1e4     # wt% -> ppm

    total = sum(out.values())
    out["xrf_LE"] = np.clip(1e6 - total * rng.normal(1.0, 0.02, n), 3e5, 9.8e5)
    out["xrf_live_time"] = rng.uniform(16, 20, n)
    out["magsus"] = np.clip(0.4 * fe * 10 + rng.normal(0, 2.0, n), -5, 200)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Top-level generator
# --------------------------------------------------------------------------
def generate_dataset(
    n: int = 200,
    real_df: pd.DataFrame | None = None,
    seed: int = 42,
    n_seams: int = 5,
    add_label_noise: bool = True,
) -> pd.DataFrame:
    """Return ``n`` synthetic scan locations in BoxScan schema, with labels.

    Parameters
    ----------
    n : int
        Number of samples. 150-250 is a sensible prototype size.
    real_df : DataFrame, optional
        Output of :func:`io_boxscan.load_sessions`; used to anchor the
        spectral endmembers and the noise model.
    add_label_noise : bool
        Add IS 1350 / IS 1350-2 repeatability to the reference labels so the
        achievable R^2 is bounded by the lab, not by the model.
    """
    rng = np.random.default_rng(seed)
    basis = learn_basis(real_df)
    chem = sample_chemistry(n, rng, n_seams=n_seams)

    R = synth_spectra(chem, basis, rng)
    xrf = synth_xrf(chem, rng)

    XYZ = cie_xyz_from_reflectance(R, basis.wl)
    Lab = xyz_to_lab(XYZ)

    df = pd.DataFrame({
        "sample_id": [f"SYN_{i:04d}" for i in range(n)],
        "session": "SYNTHETIC",
        "measurement_id": [f"SYN_{i:04d}" for i in range(n)],
        "hole_id": chem["seam"],
        "box_id": (np.arange(n) // 15) + 1,
        "track": 2,
        "borehole_depth_m": np.round(np.cumsum(rng.uniform(0.03, 0.09, n)) + 250.0, 3),
        "section_depth_cm": np.round(rng.uniform(2, 95, n), 1),
        "seam": chem["seam"],
    })

    for j, w in enumerate(basis.wl):
        df[f"R{w}"] = R[:, j]

    df["col_X"], df["col_Y"], df["col_Z"] = XYZ.T
    df["col_L"], df["col_a"], df["col_b"] = Lab.T
    df["col_grey"] = XYZ[:, 1]

    df = pd.concat([df, xrf], axis=1)

    # ---- reference labels ----------------------------------------------
    for t in TARGETS:
        v = chem[t].to_numpy(float)
        if add_label_noise:
            v = v + rng.normal(0.0, LAB_REPEATABILITY[t], n)
        df[t] = v
    # re-impose the proximate closure after noise so the mass balance holds
    df["FC"] = 100.0 - df["Ash"] - df["MC"] - df["VM"]
    df["CV_MJkg"] = df["CV"] * 0.0041868
    df["grade"] = [gcv_to_grade(v) for v in df["CV"]]

    df["is_synthetic"] = True
    df["basis_n_real_spectra"] = basis.n_real
    return df
