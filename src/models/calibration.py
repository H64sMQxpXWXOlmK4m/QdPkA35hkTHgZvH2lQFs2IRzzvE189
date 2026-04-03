"""
Advanced probability calibration.

Methods implemented:
1. Platt scaling (logistic sigmoid fit)
2. Isotonic regression
3. Beta calibration (fits a Beta CDF to the prediction distribution)
4. Temperature scaling
5. Ensemble calibration (weighted average of multiple methods)

All calibrators share a sklearn-compatible API: fit(y_prob, y_true) / predict(y_prob).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from typing import List, Optional


# ── Base class ────────────────────────────────────────────────────────────────

class BaseCalibrator:
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "BaseCalibrator":
        raise NotImplementedError

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_predict(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self.fit(y_prob, y_true)
        return self.predict(y_prob)


# ── Platt scaling ─────────────────────────────────────────────────────────────

class PlattCalibrator(BaseCalibrator):
    """Logistic regression on raw probabilities (Platt 1999)."""

    def __init__(self):
        self._lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        self._lr.fit(y_prob.reshape(-1, 1), y_true)
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(y_prob.reshape(-1, 1))[:, 1]


# ── Isotonic regression ───────────────────────────────────────────────────────

class IsotonicCalibrator(BaseCalibrator):
    """Isotonic regression calibrator."""

    def __init__(self):
        self._ir = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self._ir.fit(y_prob, y_true)
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        return self._ir.predict(y_prob)


# ── Beta calibration ──────────────────────────────────────────────────────────

class BetaCalibrator(BaseCalibrator):
    """
    Beta calibration (Kull et al. 2017).
    Fits log(p/(1-p)) ~ a*log(p) + b*log(1-p) + c via logistic regression.
    More flexible than Platt scaling for asymmetric distributions.
    """

    def __init__(self):
        self._lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=1000)

    def _transform(self, y_prob: np.ndarray) -> np.ndarray:
        eps = 1e-7
        p = np.clip(y_prob, eps, 1 - eps)
        return np.column_stack([np.log(p), np.log(1 - p)])

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "BetaCalibrator":
        X = self._transform(y_prob)
        self._lr.fit(X, y_true)
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        X = self._transform(y_prob)
        return self._lr.predict_proba(X)[:, 1]


# ── Temperature scaling ───────────────────────────────────────────────────────

class TemperatureScaling(BaseCalibrator):
    """
    Temperature scaling: p_cal = sigmoid(logit(p) / T).
    Optimizes T to minimize NLL on calibration set.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "TemperatureScaling":
        eps = 1e-7
        logits = np.log(np.clip(y_prob, eps, 1 - eps) / (1 - np.clip(y_prob, eps, 1 - eps)))

        def nll(log_t):
            T = np.exp(log_t[0])
            p_cal = expit(logits / T)
            p_cal = np.clip(p_cal, eps, 1 - eps)
            return -np.mean(
                y_true * np.log(p_cal) + (1 - y_true) * np.log(1 - p_cal)
            )

        res = minimize(nll, x0=[0.0], method="Nelder-Mead")
        self.temperature = np.exp(res.x[0])
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        eps = 1e-7
        logits = np.log(np.clip(y_prob, eps, 1 - eps) / (1 - np.clip(y_prob, eps, 1 - eps)))
        return expit(logits / self.temperature)


# ── Ensemble calibrator ───────────────────────────────────────────────────────

class EnsembleCalibrator(BaseCalibrator):
    """
    Weighted average of multiple calibration methods.
    Weights are optimized to minimize NLL on calibration set.
    """

    def __init__(self, methods: Optional[List[str]] = None):
        self.methods = methods or ["platt", "isotonic", "beta", "temperature"]
        self._calibrators: List[BaseCalibrator] = []
        self._weights: np.ndarray = np.array([])

    def _make(self, name: str) -> BaseCalibrator:
        return {
            "platt": PlattCalibrator,
            "isotonic": IsotonicCalibrator,
            "beta": BetaCalibrator,
            "temperature": TemperatureScaling,
        }[name]()

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "EnsembleCalibrator":
        self._calibrators = [self._make(m) for m in self.methods]
        cal_preds = []
        for cal in self._calibrators:
            try:
                cal.fit(y_prob, y_true)
                cal_preds.append(cal.predict(y_prob))
            except Exception:
                cal_preds.append(y_prob.copy())  # fallback: identity

        # Optimize weights via NLL minimization
        cal_preds = np.array(cal_preds)  # shape: (n_methods, n_samples)
        eps = 1e-7

        def nll(log_w):
            w = np.exp(log_w)
            w = w / w.sum()
            p_blend = np.clip(np.dot(w, cal_preds), eps, 1 - eps)
            return -np.mean(
                y_true * np.log(p_blend) + (1 - y_true) * np.log(1 - p_blend)
            )

        n_m = len(self.methods)
        res = minimize(nll, x0=np.zeros(n_m), method="Nelder-Mead")
        raw_w = np.exp(res.x)
        self._weights = raw_w / raw_w.sum()
        self._cal_preds_train = cal_preds
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        cal_preds = []
        for cal in self._calibrators:
            try:
                cal_preds.append(cal.predict(y_prob))
            except Exception:
                cal_preds.append(y_prob.copy())
        cal_preds = np.array(cal_preds)
        return np.clip(np.dot(self._weights, cal_preds), 1e-7, 1 - 1e-7)


# ── Factory ───────────────────────────────────────────────────────────────────

def make_calibrator(method: str = "ensemble") -> BaseCalibrator:
    """Factory function."""
    mapping = {
        "platt": PlattCalibrator,
        "isotonic": IsotonicCalibrator,
        "beta": BetaCalibrator,
        "temperature": TemperatureScaling,
        "ensemble": EnsembleCalibrator,
    }
    if method not in mapping:
        raise ValueError(f"Unknown calibration method: {method}. Choose from {list(mapping)}")
    return mapping[method]()


def cross_val_calibrate(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    calibrator: BaseCalibrator,
    n_folds: int = 5,
) -> np.ndarray:
    """
    Fit calibrator using cross-validation to avoid in-sample overfitting.
    Returns out-of-fold calibrated probabilities (same shape as y_prob).

    Note: For time series, folds are sequential (no shuffling).
    """
    n = len(y_prob)
    fold_size = n // n_folds
    cal_probs = y_prob.copy()

    for i in range(n_folds):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < n_folds - 1 else n
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_start:val_end] = False

        cal = type(calibrator)()  # fresh instance
        try:
            cal.fit(y_prob[train_mask], y_true[train_mask])
            cal_probs[val_start:val_end] = cal.predict(y_prob[val_start:val_end])
        except Exception:
            pass  # keep raw if calibration fails

    return cal_probs
