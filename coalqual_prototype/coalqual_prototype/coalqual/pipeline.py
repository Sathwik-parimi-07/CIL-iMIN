"""
End-to-end training / benchmarking.

For each of the five quality indices the pipeline:

1. builds the feature table (MSC fitted on the calibration fold only);
2. removes persistent outliers with Monte-Carlo cross-validation;
3. splits calibration / prediction with SPXY;
4. benchmarks every (selector x model) pair with an inner cross-validated
   hyper-parameter search *inside* the calibration set;
5. picks the winner on cross-validated RMSE, never on the held-out set;
6. scores that single winner once on the untouched prediction set;
7. persists the fitted object together with the exact feature contract
   needed to reproduce the inputs at inference time.

Step 5/6 ordering is the part worth defending: choosing the best of ~30
candidate pipelines *on the test set* inflates R^2 by 0.03-0.08 at this
sample size, which is exactly the gap between an honest and a flattering
result slide.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV

from .config import (DEFAULT_BLOCKS, MODEL_DIR, N_MCCV_OUTLIER, OUTLIER_SIGMA,
                     RANDOM_STATE, TABLE_DIR, TARGETS, TEST_FRACTION, WAVELENGTHS)
from .features import build_feature_table
from .models import model_zoo
from .selection import SELECTORS
from .validation import (ApplicabilityDomain, cv_predict, mccv_outliers,
                         regression_metrics, spxy_split, summarise)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


TREE_MODELS = {"RF", "ExtraTrees", "GBR"}


@dataclass
class TargetResult:
    target: str
    best_model: str
    best_selector: str
    n_features: int
    n_cal: int
    n_pred: int
    n_outliers: int
    cv: dict
    pred: dict
    params: dict


# --------------------------------------------------------------------------
def _search(est, grid, X, y, cv_splits: int, seed: int, n_random=None):
    """Hyper-parameter search inside the calibration fold.

    Exhaustive for the cheap linear/kernel models, randomised for the tree
    ensembles where a full grid costs minutes for a third decimal place.
    """
    n = len(y)
    k = min(cv_splits, max(3, n // 8)) if n >= 24 else min(5, n)
    cv = KFold(n_splits=max(3, k), shuffle=True, random_state=seed)
    n_jobs = -1 if (os.cpu_count() or 1) > 1 else 1   # loky is pure overhead on 1 core
    kw = dict(scoring="neg_root_mean_squared_error", cv=cv, n_jobs=n_jobs,
              refit=True, error_score=np.nan)

    n_combos = int(np.prod([len(v) for v in grid.values()])) if grid else 1
    if n_random and n_combos > n_random:
        gs = RandomizedSearchCV(est, grid, n_iter=n_random,
                                random_state=seed, **kw)
    else:
        gs = GridSearchCV(est, grid, **kw)
    gs.fit(X, y)
    return gs


def run_target(
    df: pd.DataFrame,
    target: str,
    blocks=DEFAULT_BLOCKS,
    selectors=("CC", "Boruta", "CARS", "SPA", "ALL"),
    models=None,
    test_fraction: float = TEST_FRACTION,
    seed: int = RANDOM_STATE,
    remove_outliers: bool = True,
    max_features: int = 60,
    max_tree_features: int = 150,
    verbose: bool = True,
) -> tuple[TargetResult, pd.DataFrame, dict]:
    """Benchmark one quality index. Returns (winner, full table, artefacts)."""
    t0 = time.time()
    work = df[df[target].notna()].reset_index(drop=True)
    if len(work) < 20:
        raise ValueError(f"{target}: only {len(work)} labelled samples; need >= 20")

    y_all = work[target].to_numpy(float)

    # ---- outlier screen on a quick, fully-refit model -------------------
    n_out = 0
    if remove_outliers and len(work) >= 40:
        F0, _, _ = build_feature_table(work, blocks, WAVELENGTHS, fit=True)
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        quick = Pipeline([("s", StandardScaler()),
                          ("e", PLSRegression(n_components=min(8, F0.shape[1] - 1)))])
        is_out, _ = mccv_outliers(F0.to_numpy(float), y_all, quick,
                                  n_iter=N_MCCV_OUTLIER, sigma=OUTLIER_SIGMA,
                                  random_state=seed)
        n_out = int(is_out.sum())
        if 0 < n_out < 0.15 * len(work):
            work = work.loc[~is_out].reset_index(drop=True)
            y_all = work[target].to_numpy(float)
        else:
            n_out = 0

    # ---- SPXY split on raw spectra --------------------------------------
    # test_fraction=0 keeps every sample in calibration and reports LOOCV
    # only - the right choice below ~40 labelled samples, where a 7-sample
    # hold-out set says nothing.
    X_spec = work[[f"R{w}" for w in WAVELENGTHS]].to_numpy(float)
    if test_fraction and test_fraction > 0:
        cal_idx, pred_idx = spxy_split(X_spec, y_all, test_size=test_fraction,
                                       random_state=seed)
    else:
        cal_idx, pred_idx = np.arange(len(work)), np.array([], dtype=int)
    cal, prd = work.iloc[cal_idx], work.iloc[pred_idx]
    has_pred = len(pred_idx) > 0

    # ---- features, fitted on calibration only ---------------------------
    F_cal, msc, band_mask = build_feature_table(cal, blocks, WAVELENGTHS, fit=True)
    if has_pred:
        F_prd, _, _ = build_feature_table(prd, blocks, WAVELENGTHS, msc=msc,
                                          band_mask=band_mask, fit=False)
        F_prd = F_prd.reindex(columns=F_cal.columns)
    else:
        F_prd = F_cal.iloc[:0].copy()

    med = F_cal.median(numeric_only=True)
    Xc = F_cal.fillna(med).to_numpy(float)
    Xp = F_prd.fillna(med).to_numpy(float)
    yc, yp = y_all[cal_idx], y_all[pred_idx]

    zoo = models or model_zoo(random_state=seed, n_samples=len(yc))
    rows, fitted = [], {}

    for sel_name in selectors:
        sel_fn = SELECTORS[sel_name]
        try:
            kw = {"max_features": max_features} if sel_name in ("CC", "Boruta", "CARS") else {}
            if sel_name == "SPA":
                kw = {"n_features": min(25, Xc.shape[1])}
            mask = sel_fn(Xc, yc, **kw)
        except Exception as exc:                                     # noqa: BLE001
            if verbose:
                print(f"    [{target}] selector {sel_name} failed: {exc}")
            continue
        if mask.sum() == 0:
            continue
        names = list(F_cal.columns[mask])
        Xcs, Xps = Xc[:, mask], Xp[:, mask]

        for mdl_name, (est, grid, n_random) in zoo.items():
            # Tree ensembles on the full ~470-column table are the slowest
            # cell of the grid and never win against the same model on a
            # selected subset; keep the full-spectrum baseline linear/kernel.
            if mask.sum() > max_tree_features and mdl_name in TREE_MODELS:
                continue
            try:
                gs = _search(est, grid, Xcs, yc, cv_splits=10, seed=seed,
                             n_random=n_random)
                # The search already cross-validated; reuse its score rather
                # than paying for a second full CV pass per candidate.
                rmse_cv = float(-gs.best_score_)
                var = float(np.var(yc, ddof=1))
                r2_cv = 1 - rmse_cv ** 2 / var if var > 0 else np.nan
                m_pr = (regression_metrics(yp, gs.best_estimator_.predict(Xps))
                        if has_pred else regression_metrics([], []))
            except Exception as exc:                                 # noqa: BLE001
                if verbose:
                    print(f"    [{target}] {sel_name}/{mdl_name} failed: {exc}")
                continue

            key = f"{sel_name}|{mdl_name}"
            fitted[key] = dict(estimator=gs.best_estimator_, features=names,
                               mask=mask, params=gs.best_params_)
            rows.append(dict(
                target=target, selector=sel_name, model=mdl_name,
                n_features=int(mask.sum()),
                R2_cv=r2_cv, RMSE_cv=rmse_cv,
                R2_pred=m_pr["R2"], RMSE_pred=m_pr["RMSE"],
                RPD_pred=m_pr["RPD"], RPIQ_pred=m_pr["RPIQ"], bias_pred=m_pr["bias"],
            ))
            if verbose:
                print(f"    {target:<4} {sel_name:<7}/{mdl_name:<11} "
                      f"CV R2={r2_cv:.3f} RMSE={rmse_cv:.3f} | "
                      f"PRED R2={m_pr['R2']:.3f} RMSE={m_pr['RMSE']:.3f} "
                      f"RPD={m_pr['RPD']:.2f}")

    if not rows:
        raise RuntimeError(f"{target}: every candidate pipeline failed")

    table = pd.DataFrame(rows)
    # Selection on cross-validated RMSE only - the prediction set stays clean.
    best = table.sort_values("RMSE_cv").iloc[0]
    key = f"{best.selector}|{best.model}"
    art = fitted[key]

    yp_hat = (art["estimator"].predict(Xp[:, art["mask"]]) if has_pred
              else np.array([]))
    yc_hat = cv_predict(art["estimator"], Xc[:, art["mask"]], yc, random_state=seed)

    result = TargetResult(
        target=target, best_model=str(best.model), best_selector=str(best.selector),
        n_features=int(best.n_features), n_cal=len(yc), n_pred=len(yp),
        n_outliers=n_out,
        cv=regression_metrics(yc, yc_hat), pred=regression_metrics(yp, yp_hat),
        params={k: str(v) for k, v in art["params"].items()},
    )

    ad = ApplicabilityDomain().fit(Xc[:, art["mask"]])

    artefacts = dict(
        estimator=art["estimator"], features=art["features"], mask=art["mask"],
        ad=ad,
        msc=msc, band_mask=band_mask, blocks=blocks, medians=med,
        columns=list(F_cal.columns),
        X_cal_sel=Xc[:, art["mask"]], X_pred_sel=Xp[:, art["mask"]],
        cal_idx=cal_idx, pred_idx=pred_idx,
        y_cal=yc, y_cal_hat=yc_hat, y_pred=yp, y_pred_hat=yp_hat,
        cal_ids=cal["sample_id"].tolist(), pred_ids=prd["sample_id"].tolist(),
        elapsed_s=round(time.time() - t0, 1),
    )
    if verbose:
        if has_pred:
            print(f"  -> {target}: winner {key}  "
                  f"pred R2={result.pred['R2']:.3f} RMSE={result.pred['RMSE']:.3f} "
                  f"RPD={result.pred['RPD']:.2f}  ({artefacts['elapsed_s']}s)")
        else:
            print(f"  -> {target}: winner {key}  "
                  f"LOOCV R2={result.cv['R2']:.3f} RMSE={result.cv['RMSE']:.3f} "
                  f"RPD={result.cv['RPD']:.2f}  ({artefacts['elapsed_s']}s)")
    return result, table, artefacts


# --------------------------------------------------------------------------
def run_all(
    df: pd.DataFrame,
    targets=TARGETS,
    blocks=DEFAULT_BLOCKS,
    out_dir: Path = MODEL_DIR,
    table_dir: Path = TABLE_DIR,
    seed: int = RANDOM_STATE,
    verbose: bool = True,
    **kw,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Train all five indices; save models and the benchmark tables."""
    out_dir, table_dir = Path(out_dir), Path(table_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    summaries, tables, store = [], [], {}
    synthetic = bool(df.get("is_synthetic", pd.Series([False])).any())

    for t in targets:
        if verbose:
            print(f"\n[{t}]")
        res, table, art = run_target(df, t, blocks=blocks, seed=seed,
                                     verbose=verbose, **kw)
        tables.append(table)
        store[t] = art

        joblib.dump(
            dict(target=t, estimator=art["estimator"], features=art["features"],
                 mask=art["mask"], ad=art["ad"],
                 msc=art["msc"], band_mask=art["band_mask"],
                 blocks=asdict(blocks), medians=art["medians"],
                 columns=art["columns"], wavelengths=WAVELENGTHS,
                 trained_on_synthetic=synthetic, metrics=asdict(res)),
            out_dir / f"model_{t}.joblib",
        )

        s = asdict(res)
        flat = {k: v for k, v in s.items() if k not in ("cv", "pred", "params")}
        flat.update({f"cv_{k}": v for k, v in s["cv"].items()})
        flat.update({f"pred_{k}": v for k, v in s["pred"].items()})
        flat["params"] = json.dumps(s["params"])
        summaries.append(flat)

    summary = pd.DataFrame(summaries)
    full = pd.concat(tables, ignore_index=True)
    # Merge with any earlier run so targets can be trained one at a time.
    for name, new in (("model_summary.csv", summary), ("benchmark_full.csv", full)):
        path = table_dir / name
        if path.exists():
            old = pd.read_csv(path)
            old = old[~old["target"].isin(new["target"].unique())]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(path, index=False)
    summary = pd.read_csv(table_dir / "model_summary.csv")
    full = pd.read_csv(table_dir / "benchmark_full.csv")

    meta = dict(trained_on_synthetic=synthetic, n_samples=int(len(df)),
                seed=seed, targets=list(targets), blocks=asdict(blocks))
    (table_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return summary, summarise(full.to_dict("records")), store


# --------------------------------------------------------------------------
def predict(df: pd.DataFrame, model_dir: Path = MODEL_DIR,
            targets=TARGETS) -> pd.DataFrame:
    """Apply saved models to new BoxScan scan locations."""
    from .config import gcv_to_grade

    out = df[[c for c in ("sample_id", "session", "measurement_id", "box_id",
                          "track", "borehole_depth_m", "section_depth_cm")
              if c in df.columns]].copy()

    in_dom = []
    for t in targets:
        path = Path(model_dir) / f"model_{t}.joblib"
        if not path.exists():
            continue
        m = joblib.load(path)
        from .config import FeatureBlocks
        blocks = FeatureBlocks(**m["blocks"])
        F, _, _ = build_feature_table(df, blocks, m["wavelengths"],
                                      msc=m["msc"], band_mask=m["band_mask"],
                                      fit=False)
        F = F.reindex(columns=m["columns"]).fillna(m["medians"])
        Xsel = F.to_numpy(float)[:, m["mask"]]
        raw = m["estimator"].predict(Xsel)
        lo, hi = PHYSICAL_BOUNDS[t]
        out[f"pred_{t}"] = np.clip(raw, lo, hi)
        out[f"clipped_{t}"] = (raw < lo) | (raw > hi)
        if m.get("ad") is not None:
            ad = m["ad"].score(Xsel)
            out[f"ad_{t}"] = np.maximum(ad["ad_t2"], ad["ad_q"]).round(2).to_numpy()
            in_dom.append(ad["in_domain"].to_numpy())

    if in_dom:
        out["in_domain"] = np.logical_and.reduce(in_dom)
        out["ad_score"] = out[[c for c in out.columns if c.startswith("ad_")
                               and c != "ad_score"]].max(axis=1)
        # Graded verdict for the operator. ad_score is the worst-case ratio
        # to the calibration limit across the five models:
        #   <= 1  inside the calibration cloud       -> quote the grade
        #   <= 5  mild extrapolation                  -> quote with caution
        #   >  5  a material the models never saw    -> send to the lab
        out["ad_tier"] = pd.cut(out["ad_score"], [-np.inf, 1.0, 5.0, np.inf],
                                labels=["in", "edge", "out"]).astype(str)

    # enforce the proximate closure and derive the declared grade
    if {"pred_Ash", "pred_MC", "pred_VM"} <= set(out.columns):
        out["pred_FC_closure"] = (100 - out["pred_Ash"] - out["pred_MC"]
                                  - out["pred_VM"]).clip(lower=0)
    if "pred_CV" in out.columns:
        out["pred_grade"] = [gcv_to_grade(v) for v in out["pred_CV"]]
        if "ad_tier" in out.columns:
            out["grade_flag"] = out["ad_tier"].map(
                {"in": "", "edge": "extrapolated", "out": "outside calibration domain"})
        out["pred_CV_MJkg"] = out["pred_CV"] * 0.0041868
    return out


PHYSICAL_BOUNDS = {
    "MC": (0.0, 40.0), "Ash": (0.0, 95.0), "VM": (0.0, 60.0),
    "FC": (0.0, 95.0), "CV": (0.0, 8500.0),
}


# --------------------------------------------------------------------------
def rebuild_store(df: pd.DataFrame, model_dir: Path = MODEL_DIR,
                  targets=TARGETS, seed: int = RANDOM_STATE,
                  test_fraction: float = TEST_FRACTION,
                  remove_outliers: bool = True) -> dict:
    """Recreate the per-target artefacts (splits + predictions) from saved models.

    Every step of :func:`run_target` that decides *which* rows are calibration
    and prediction is deterministic given ``seed``, and the MSC reference and
    band mask are stored in the model file, so figures can be regenerated
    long after training without touching the estimator.
    """
    from .config import FeatureBlocks
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    store = {}
    for t in targets:
        path = Path(model_dir) / f"model_{t}.joblib"
        if not path.exists():
            continue
        m = joblib.load(path)
        blocks = FeatureBlocks(**m["blocks"])
        work = df[df[t].notna()].reset_index(drop=True)
        y_all = work[t].to_numpy(float)

        if remove_outliers and len(work) >= 40:
            F0, _, _ = build_feature_table(work, blocks, m["wavelengths"], fit=True)
            quick = Pipeline([("s", StandardScaler()),
                              ("e", PLSRegression(n_components=min(8, F0.shape[1] - 1)))])
            is_out, _ = mccv_outliers(F0.to_numpy(float), y_all, quick,
                                      n_iter=N_MCCV_OUTLIER, sigma=OUTLIER_SIGMA,
                                      random_state=seed)
            if 0 < is_out.sum() < 0.15 * len(work):
                work = work.loc[~is_out].reset_index(drop=True)
                y_all = work[t].to_numpy(float)

        X_spec = work[[f"R{w}" for w in m["wavelengths"]]].to_numpy(float)
        if test_fraction and test_fraction > 0:
            cal_idx, pred_idx = spxy_split(X_spec, y_all, test_size=test_fraction,
                                           random_state=seed)
        else:
            cal_idx, pred_idx = np.arange(len(work)), np.array([], dtype=int)

        def _X(sub):
            F, _, _ = build_feature_table(sub, blocks, m["wavelengths"], msc=m["msc"],
                                          band_mask=m["band_mask"], fit=False)
            F = F.reindex(columns=m["columns"]).fillna(m["medians"])
            return F.to_numpy(float)[:, m["mask"]]

        Xc = _X(work.iloc[cal_idx])
        yc = y_all[cal_idx]
        yc_hat = cv_predict(m["estimator"], Xc, yc, random_state=seed)
        if len(pred_idx):
            Xp = _X(work.iloc[pred_idx])
            yp, yp_hat = y_all[pred_idx], m["estimator"].predict(Xp)
        else:
            Xp, yp, yp_hat = Xc[:0], np.array([]), np.array([])

        store[t] = dict(estimator=m["estimator"], features=m["features"],
                        X_cal_sel=Xc, X_pred_sel=Xp, y_cal=yc, y_cal_hat=yc_hat,
                        y_pred=yp, y_pred_hat=yp_hat, cal_idx=cal_idx, pred_idx=pred_idx)
    return store
