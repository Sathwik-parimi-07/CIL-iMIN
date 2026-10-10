"""
Splitting, outlier screening and scoring.

Two choices here matter for defending the numbers in front of judges:

* **SPXY splitting** (Galvao et al. 2005) spans the joint X-y space so the
  prediction set is representative rather than lucky. Random splits on
  n < 100 swing R^2 by 0.1 or more between seeds.
* **Every transform is fitted inside the calibration fold.** MSC references,
  scalers and wavelength selection all leak if fitted on the full dataset,
  and leakage is the single most common reason a 0.99 lab model collapses on
  a conveyor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold, cross_val_predict


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    """R^2, RMSE, MAE, bias, RPD and RPIQ.

    RPD = SD(y) / RMSE. Convention (Zornoza et al. 2008): RPD > 3 with
    R^2 > 0.90 is an excellent quantitative model; 2-3 is usable for
    screening; < 2 is qualitative only.
    RPIQ = IQR(y) / RMSE is more robust for skewed distributions.
    """
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) < 2:
        return {k: np.nan for k in ("R2", "RMSE", "MAE", "bias", "RPD", "RPIQ", "n")}

    resid = y_true - y_pred
    sse = float((resid ** 2).sum())
    sst = float(((y_true - y_true.mean()) ** 2).sum())
    rmse = float(np.sqrt(sse / len(y_true)))
    iqr = float(np.subtract(*np.percentile(y_true, [75, 25])))
    return {
        "R2": 1 - sse / sst if sst > 0 else np.nan,
        "RMSE": rmse,
        "MAE": float(np.abs(resid).mean()),
        "bias": float(resid.mean()),
        "RPD": float(y_true.std(ddof=1) / rmse) if rmse > 0 else np.inf,
        "RPIQ": float(iqr / rmse) if rmse > 0 else np.inf,
        "n": int(len(y_true)),
    }


# --------------------------------------------------------------------------
# SPXY split
# --------------------------------------------------------------------------
def spxy_split(X, y, test_size: float = 0.25, random_state: int = 0):
    """Sample-set Partitioning based on joint X-Y distances.

    Returns ``(calibration_idx, prediction_idx)``. The calibration set is
    grown greedily to maximise coverage of the combined feature/target space;
    the remainder becomes the held-out prediction set.
    """
    X = np.nan_to_num(np.asarray(X, float))
    y = np.asarray(y, float).reshape(-1, 1)
    n = len(X)
    n_cal = max(2, int(round(n * (1 - test_size))))

    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ys = (y - y.mean()) / (y.std() + 1e-12)

    dx = cdist(Xs, Xs)
    dy = cdist(ys, ys)
    d = dx / (dx.max() + 1e-12) + dy / (dy.max() + 1e-12)

    i, j = np.unravel_index(np.argmax(d), d.shape)
    selected = [int(i), int(j)]
    remaining = [k for k in range(n) if k not in selected]

    while len(selected) < n_cal and remaining:
        sub = d[np.ix_(remaining, selected)]
        nxt = remaining[int(np.argmax(sub.min(axis=1)))]
        selected.append(nxt)
        remaining.remove(nxt)

    return np.array(sorted(selected)), np.array(sorted(remaining))


# --------------------------------------------------------------------------
# Monte-Carlo CV outlier removal
# --------------------------------------------------------------------------
def mccv_outliers(X, y, estimator, n_iter: int = 200, test_frac: float = 0.25,
                  sigma: float = 3.0, random_state: int = 0):
    """Flag samples that are persistently mis-predicted when held out.

    Repeatedly refits on a random 75 % and records the residual for each
    held-out sample. A sample is an outlier when its mean absolute residual
    exceeds ``sigma`` robust standard deviations (1.4826 x MAD) of the pooled
    residuals - the usual cause is a mislabelled lab result or a scan that
    caught the core-box timber rather than the coal.
    """
    X = np.nan_to_num(np.asarray(X, float))
    y = np.asarray(y, float)
    n = len(y)
    rng = np.random.default_rng(random_state)
    acc = np.zeros(n)
    cnt = np.zeros(n)

    for _ in range(n_iter):
        idx = rng.permutation(n)
        k = max(1, int(round(test_frac * n)))
        te, tr = idx[:k], idx[k:]
        if len(tr) < 5:
            continue
        try:
            est = estimator.__class__(**estimator.get_params())
            est.fit(X[tr], y[tr])
            r = y[te] - est.predict(X[te])
        except Exception:
            continue
        acc[te] += np.abs(r)
        cnt[te] += 1

    mean_abs = np.divide(acc, np.maximum(cnt, 1))
    med = np.median(mean_abs)
    mad = 1.4826 * np.median(np.abs(mean_abs - med)) + 1e-12
    is_out = mean_abs > med + sigma * mad
    return is_out, mean_abs


# --------------------------------------------------------------------------
# Cross-validated prediction helper
# --------------------------------------------------------------------------
def cv_predict(estimator, X, y, n_splits: int = 10, random_state: int = 0):
    """Out-of-fold predictions; falls back to LOOCV on very small sets."""
    X = np.nan_to_num(np.asarray(X, float))
    y = np.asarray(y, float)
    n = len(y)
    n_splits = n if n < 30 else min(n_splits, n)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return cross_val_predict(estimator, X, y, cv=cv, n_jobs=None).ravel()


def summarise(rows: list[dict]) -> pd.DataFrame:
    """Tidy a list of metric dicts into a sorted results table."""
    df = pd.DataFrame(rows)
    sort_cols = [c for c in ("target", "RMSE_pred") if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df


# --------------------------------------------------------------------------
# Applicability domain
# --------------------------------------------------------------------------
class ApplicabilityDomain:
    """Flags scans that lie outside the calibration cloud.

    Fits PCA on the standardised calibration features, then scores new
    samples on two classic chemometric statistics:

    * **Hotelling T^2** - distance *within* the model plane (an unusual but
      explainable combination of known features);
    * **Q residual**    - distance *off* the plane (spectral shape the model
      has never seen, e.g. a rock parting or a wet, muddy chip).

    Limits are the 97.5th percentile of the calibration set for each. Any
    prediction on a flagged scan is still returned, but the software must
    show it as unverified rather than let it flow into a grade declaration.
    A model that cannot say "I don't know" is not deployable on a conveyor.
    """

    def __init__(self, var_explained: float = 0.95, quantile: float = 0.975):
        self.var_explained = var_explained
        self.quantile = quantile

    def fit(self, X: np.ndarray) -> "ApplicabilityDomain":
        from sklearn.decomposition import PCA
        X = np.nan_to_num(np.asarray(X, float))
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0) + 1e-9
        Z = (X - self.mu_) / self.sd_
        k = max(2, min(Z.shape[0] - 1, Z.shape[1]))
        pca = PCA(n_components=k).fit(Z)
        cum = np.cumsum(pca.explained_variance_ratio_)
        k_keep = int(np.searchsorted(cum, self.var_explained) + 1)
        self.pca_ = PCA(n_components=max(2, min(k_keep, k))).fit(Z)
        t2, q = self._stats(Z)
        self.t2_lim_ = float(np.quantile(t2, self.quantile))
        self.q_lim_ = float(np.quantile(q, self.quantile))
        return self

    def _stats(self, Z):
        T = self.pca_.transform(Z)
        lam = self.pca_.explained_variance_ + 1e-12
        t2 = ((T ** 2) / lam).sum(1)
        Zhat = self.pca_.inverse_transform(T)
        q = ((Z - Zhat) ** 2).sum(1)
        return t2, q

    def score(self, X: np.ndarray) -> pd.DataFrame:
        X = np.nan_to_num(np.asarray(X, float))
        Z = (X - self.mu_) / self.sd_
        t2, q = self._stats(Z)
        return pd.DataFrame({
            "ad_t2": t2 / self.t2_lim_,          # > 1 means outside
            "ad_q": q / self.q_lim_,
            "in_domain": (t2 <= self.t2_lim_) & (q <= self.q_lim_),
        })
