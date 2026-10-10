"""
Wavelength / feature selection.

Four selectors are implemented from first principles so the pipeline has no
exotic dependencies and every step is auditable:

* :func:`correlation_select` - Pearson |r| threshold (the "CC" screen).
* :func:`boruta_select`      - all-relevant selection against shadow features
  (Kursa & Rudnicki 2010).
* :func:`cars_select`        - Competitive Adaptive Reweighted Sampling (Li et
  al. 2009), the method that produced the best wavelength subsets in Mao et
  al. 2024.
* :func:`spa_select`         - Successive Projections Algorithm, a minimal,
  low-collinearity subset.

Every selector returns a boolean mask over columns of ``X``.
"""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


def _clean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------
# Correlation coefficient screen
# --------------------------------------------------------------------------
def correlation_select(X, y, threshold: float = 0.3, max_features: int | None = None):
    """Keep features whose |Pearson r| with the target exceeds ``threshold``."""
    X, y = _clean(X), np.asarray(y, float)
    Xc = X - X.mean(0)
    yc = y - y.mean()
    denom = np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
    denom[denom == 0] = np.nan
    r = np.abs((Xc * yc[:, None]).sum(0) / denom)
    r = np.nan_to_num(r)
    mask = r >= threshold
    if max_features and mask.sum() > max_features:
        keep = np.argsort(r)[::-1][:max_features]
        mask = np.zeros(X.shape[1], bool)
        mask[keep] = True
    if mask.sum() == 0:                                  # never return nothing
        mask[np.argmax(r)] = True
    return mask


# --------------------------------------------------------------------------
# Boruta
# --------------------------------------------------------------------------
def boruta_select(
    X, y, n_iter: int = 20, alpha: float = 0.05,
    n_estimators: int = 100, random_state: int = 0, max_features: int | None = None,
    prescreen: int = 150,
):
    """All-relevant feature selection against permuted 'shadow' copies.

    At each iteration the design matrix is doubled with column-shuffled
    shadows; a feature scores a hit when its importance beats the best
    shadow. Hits are tested against Binomial(n_iter, 0.5).

    ``prescreen`` caps the candidate pool by |Pearson r| first. Running the
    full Boruta over ~500 highly collinear spectral bands costs minutes and
    buys nothing: adjacent 10 nm bands split each other's importance, so the
    shadow threshold is never cleared. Screening to the top ``prescreen``
    columns keeps the statistic meaningful and the runtime in seconds.
    """
    from scipy.stats import binom

    X, y = _clean(X), np.asarray(y, float)
    n, p_full = X.shape

    if prescreen and p_full > prescreen:
        pre = correlation_select(X, y, threshold=0.0, max_features=prescreen)
        cand = np.flatnonzero(pre)
    else:
        cand = np.arange(p_full)
    Xc = X[:, cand]
    p = Xc.shape[1]

    rng = np.random.default_rng(random_state)
    hits = np.zeros(p, int)

    for _ in range(n_iter):
        shadow = np.column_stack([rng.permutation(Xc[:, j]) for j in range(p)])
        Z = np.hstack([Xc, shadow])
        rf = RandomForestRegressor(
            n_estimators=n_estimators, max_features=0.3, min_samples_leaf=2,
            random_state=int(rng.integers(1 << 31)), n_jobs=-1,
        ).fit(Z, y)
        imp = rf.feature_importances_
        hits += imp[:p] > imp[p:].max()

    upper = binom.ppf(1 - alpha / max(p, 1), n_iter, 0.5)
    sel = hits > upper
    if sel.sum() == 0:
        k = max_features or max(5, p // 20)
        sel = np.zeros(p, bool)
        sel[np.argsort(hits)[::-1][:k]] = True
    elif max_features and sel.sum() > max_features:
        order = np.argsort(np.where(sel, hits, -1))[::-1][:max_features]
        sel = np.zeros(p, bool)
        sel[order] = True

    mask = np.zeros(p_full, bool)
    mask[cand[sel]] = True
    return mask


# --------------------------------------------------------------------------
# CARS
# --------------------------------------------------------------------------
def cars_select(
    X, y, n_runs: int = 60, n_components: int = 8, cv: int = 5,
    sample_ratio: float = 0.85, random_state: int = 0, max_features: int | None = None,
):
    """Competitive Adaptive Reweighted Sampling.

    An exponentially decaying function retains fewer variables each run;
    survivors are chosen by PLS regression-coefficient magnitude, and the
    subset with the lowest cross-validated RMSE wins.
    """
    X, y = _clean(X), np.asarray(y, float)
    n, p = X.shape
    rng = np.random.default_rng(random_state)

    a = (p / 2.0) ** (1.0 / (n_runs - 1))
    k = np.log(p / 2.0) / (n_runs - 1)

    survivors = np.arange(p)
    history, rmses = [], []

    for i in range(n_runs):
        ratio = a * np.exp(-k * (i + 1))
        n_keep = max(2, int(round(ratio * p)))

        idx = rng.choice(n, max(3, int(sample_ratio * n)), replace=False)
        Xi, yi = X[np.ix_(idx, survivors)], y[idx]
        ncomp = max(1, min(n_components, Xi.shape[1], len(idx) - 1))
        try:
            pls = PLSRegression(n_components=ncomp, scale=True).fit(Xi, yi)
            w = np.abs(pls.coef_).ravel()
        except Exception:
            break
        w = w / (w.sum() + 1e-12)

        n_keep = min(n_keep, len(survivors))
        order = np.argsort(w)[::-1]
        # Adaptive reweighted sampling: weighted draw among the top ranked.
        top = order[:n_keep]
        survivors = survivors[np.sort(top)]

        # Score this subset.
        ncomp = max(1, min(n_components, len(survivors), n - 1))
        try:
            pls = PLSRegression(n_components=ncomp, scale=True)
            pred = cross_val_predict(pls, X[:, survivors], y,
                                     cv=KFold(min(cv, n), shuffle=True, random_state=0))
            rmses.append(float(np.sqrt(np.mean((y - pred.ravel()) ** 2))))
        except Exception:
            rmses.append(np.inf)
        history.append(survivors.copy())

        if len(survivors) <= 2:
            break

    if not history:
        return np.ones(p, bool)
    best = history[int(np.argmin(rmses))]
    if max_features and len(best) > max_features:
        best = best[:max_features]
    mask = np.zeros(p, bool)
    mask[best] = True
    return mask


# --------------------------------------------------------------------------
# SPA
# --------------------------------------------------------------------------
def spa_select(X, y=None, n_features: int = 15, start: int | None = None):
    """Successive Projections Algorithm - minimal collinearity subset.

    Greedily adds the column with the largest projection orthogonal to the
    span of those already chosen.
    """
    X = _clean(X)
    Xs = StandardScaler().fit_transform(X)
    n, p = Xs.shape
    n_features = min(n_features, p, max(2, n - 1))

    if start is None:
        start = int(np.argmax(np.linalg.norm(Xs, axis=0)))
    chosen = [start]
    residual = Xs.copy()

    for _ in range(n_features - 1):
        v = residual[:, chosen[-1]]
        vn = v @ v
        if vn < 1e-12:
            break
        residual = residual - np.outer(v, (v @ residual) / vn)
        norms = np.linalg.norm(residual, axis=0)
        norms[chosen] = -1
        nxt = int(np.argmax(norms))
        if norms[nxt] <= 1e-9:
            break
        chosen.append(nxt)

    mask = np.zeros(p, bool)
    mask[chosen] = True
    return mask


SELECTORS = {
    "CC": correlation_select,
    "Boruta": boruta_select,
    "CARS": cars_select,
    "SPA": spa_select,
    "ALL": lambda X, y=None, **kw: np.ones(np.asarray(X).shape[1], bool),
}
