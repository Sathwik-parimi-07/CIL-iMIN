"""
Regression models.

Includes a from-scratch **LSSVR** (Suykens & Vandewalle 1999) because it is
the best performer for coal proximate analysis in the literature and has no
scikit-learn equivalent under that name. LSSVR replaces the epsilon-insensitive
loss of an SVR with a squared loss and equality constraints, which collapses
the quadratic program into a single linear system::

    | 0    1^T | | b |   | 0 |
    |          | |   | = |   |
    | 1  K+I/g | | a |   | y |

so fitting is one dense solve - fast and deterministic, which matters when
hyper-parameters are searched thousands of times on a small dataset.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class LSSVR(BaseEstimator, RegressorMixin):
    """Least-Squares Support Vector Regression.

    Parameters
    ----------
    C : float
        Regularisation (gamma in the original formulation). Larger C means
        less regularisation.
    kernel : {'rbf', 'linear'}
    gamma : float or 'scale'
        RBF kernel width. ``'scale'`` uses ``1 / (n_features * X.var())``.
    """

    def __init__(self, C: float = 10.0, kernel: str = "rbf", gamma="scale"):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

    # -- internals -------------------------------------------------------
    def _kernel(self, A, B):
        if self.kernel == "linear":
            return linear_kernel(A, B)
        return rbf_kernel(A, B, gamma=self._gamma)

    # -- API -------------------------------------------------------------
    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self.X_ = X
        self._gamma = (
            1.0 / (X.shape[1] * X.var()) if self.gamma == "scale" or self.gamma is None
            else float(self.gamma)
        )
        n = X.shape[0]
        K = self._kernel(X, X)
        A = np.zeros((n + 1, n + 1))
        A[0, 1:] = 1.0
        A[1:, 0] = 1.0
        A[1:, 1:] = K + np.eye(n) / self.C
        rhs = np.concatenate([[0.0], y])
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        self.b_, self.alpha_ = sol[0], sol[1:]
        return self

    def predict(self, X):
        check_is_fitted(self, "alpha_")
        X = check_array(X)
        return self._kernel(X, self.X_) @ self.alpha_ + self.b_


# --------------------------------------------------------------------------
# Model zoo
# --------------------------------------------------------------------------
def make_pipeline(estimator, scale: bool = True) -> Pipeline:
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("est", estimator))
    return Pipeline(steps)


def model_zoo(random_state: int = 42, n_samples: int = 100) -> dict:
    """Return ``{name: (pipeline, param_grid, n_random)}`` for the benchmark.

    Grids are deliberately compact. With 100-300 samples an exhaustive search
    overfits the validation fold faster than it helps, and the tree models
    dominate runtime, so ``n_random`` caps them at a randomised subset.
    ``n_random=None`` means search the grid exhaustively (cheap models).
    """
    max_pls = max(2, min(15, n_samples // 6))

    return {
        "PLSR": (
            make_pipeline(PLSRegression(scale=False), scale=True),
            {"est__n_components": list(range(2, max_pls + 1))},
            None,
        ),
        "ENet": (
            make_pipeline(ElasticNet(max_iter=5000, random_state=random_state)),
            {"est__alpha": [0.001, 0.01, 0.1, 1.0],
             "est__l1_ratio": [0.1, 0.5, 0.9]},
            None,
        ),
        "RF": (
            make_pipeline(RandomForestRegressor(random_state=random_state, n_jobs=1),
                          scale=False),
            {"est__n_estimators": [150],
             "est__max_depth": [None, 10],
             "est__max_features": [0.3, 0.6],
             "est__min_samples_leaf": [1, 2]},
            4,
        ),
        "ExtraTrees": (
            make_pipeline(ExtraTreesRegressor(random_state=random_state, n_jobs=1),
                          scale=False),
            {"est__n_estimators": [150],
             "est__max_features": [0.3, 0.6],
             "est__min_samples_leaf": [1, 2]},
            3,
        ),
        "GBR": (
            make_pipeline(GradientBoostingRegressor(random_state=random_state), scale=False),
            {"est__n_estimators": [200, 400],
             "est__learning_rate": [0.03, 0.08],
             "est__max_depth": [2, 3],
             "est__subsample": [0.8, 1.0]},
            4,
        ),
        "LSSVR": (
            make_pipeline(LSSVR()),
            {"est__C": [1, 10, 50, 200, 1000],
             "est__gamma": ["scale", 1e-4, 1e-3, 1e-2, 1e-1],
             "est__kernel": ["rbf"]},
            None,
        ),
    }
