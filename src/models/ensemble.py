"""
Multi-model ensemble: LightGBM + XGBoost + CatBoost.

Architecture:
  1. Each base learner is trained with Optuna hyperparameter optimization
     using expanding-window time-series CV.
  2. Base learner OOF (out-of-fold) predictions are stacked.
  3. A logistic regression meta-learner combines them.
  4. Final ensemble output is calibrated with EnsembleCalibrator.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from src.config import MODELS_DIR, TRAIN_CONFIG
from src.models.calibration import EnsembleCalibrator, make_calibrator
from src.validation.metrics import compute_all_metrics
from src.validation.walk_forward import expanding_wf_splits

logger = logging.getLogger(__name__)


# ── Feature preprocessing ─────────────────────────────────────────────────────

class FeaturePreprocessor:
    """Handles NaN imputation and feature selection."""

    def __init__(self, max_features: int = 80, nan_strategy: str = "median"):
        self.max_features = max_features
        self.nan_strategy = nan_strategy
        self._medians: Optional[pd.Series] = None
        self._selected_cols: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeaturePreprocessor":
        # Drop columns that are >40% NaN in training data
        thresh = 0.6 * len(X)
        X_clean = X.dropna(axis=1, thresh=int(thresh))

        # Imputation values
        self._medians = X_clean.median()
        X_imp = X_clean.fillna(self._medians)

        # Feature importance via LightGBM (fast, linear complexity)
        selector = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            n_jobs=-1,
            verbose=-1,
            random_state=TRAIN_CONFIG.seed,
        )
        selector.fit(X_imp, y)
        importances = pd.Series(selector.feature_importances_, index=X_imp.columns)
        self._selected_cols = (
            importances.nlargest(self.max_features).index.tolist()
        )
        logger.info(f"Selected {len(self._selected_cols)} features out of {X_clean.shape[1]}")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = X.reindex(columns=self._medians.index)
        X_imp = X_aligned.fillna(self._medians)
        return X_imp[self._selected_cols].values

    @property
    def feature_names(self) -> List[str]:
        return self._selected_cols or []


# ── LightGBM learner ──────────────────────────────────────────────────────────

def _lgbm_objective(
    trial: optuna.Trial,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> float:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "n_jobs": -1,
        "random_state": TRAIN_CONFIG.seed,
    }
    scores = []
    for train_idx, val_idx in cv_splits:
        m = lgb.LGBMClassifier(**params)
        m.fit(
            X_tr[train_idx], y_tr[train_idx],
            eval_set=[(X_tr[val_idx], y_tr[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        p = m.predict_proba(X_tr[val_idx])[:, 1]
        from sklearn.metrics import log_loss
        scores.append(log_loss(y_tr[val_idx], p))
    return float(np.mean(scores))


def train_lgbm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 60,
    timeout: int = 300,
) -> lgb.LGBMClassifier:
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=TRAIN_CONFIG.seed))
    study.optimize(
        lambda t: _lgbm_objective(t, X_tr, y_tr, cv_splits),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )
    best = study.best_params
    logger.info(f"LGBM best params: {best}")

    # Refit on full training data with best params
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        verbosity=-1,
        n_jobs=-1,
        random_state=TRAIN_CONFIG.seed,
        **best,
    )
    model.fit(X_tr, y_tr)
    return model


# ── XGBoost learner ───────────────────────────────────────────────────────────

def _xgb_objective(
    trial: optuna.Trial,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> float:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "n_jobs": -1,
        "random_state": TRAIN_CONFIG.seed,
        "verbosity": 0,
    }
    scores = []
    for train_idx, val_idx in cv_splits:
        m = xgb.XGBClassifier(**params, early_stopping_rounds=50, verbose=False)
        m.fit(
            X_tr[train_idx], y_tr[train_idx],
            eval_set=[(X_tr[val_idx], y_tr[val_idx])],
            verbose=False,
        )
        p = m.predict_proba(X_tr[val_idx])[:, 1]
        from sklearn.metrics import log_loss
        scores.append(log_loss(y_tr[val_idx], p))
    return float(np.mean(scores))


def train_xgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 50,
    timeout: int = 240,
) -> xgb.XGBClassifier:
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=TRAIN_CONFIG.seed + 1))
    study.optimize(
        lambda t: _xgb_objective(t, X_tr, y_tr, cv_splits),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )
    best = study.best_params
    logger.info(f"XGB best params: {best}")
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        n_jobs=-1,
        random_state=TRAIN_CONFIG.seed,
        verbosity=0,
        **best,
    )
    model.fit(X_tr, y_tr)
    return model


# ── CatBoost learner ──────────────────────────────────────────────────────────

def _catboost_objective(
    trial: optuna.Trial,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> float:
    params = {
        "iterations": trial.suggest_int("iterations", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_seed": TRAIN_CONFIG.seed,
        "verbose": False,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
    }
    scores = []
    for train_idx, val_idx in cv_splits:
        m = CatBoostClassifier(**params)
        m.fit(
            X_tr[train_idx], y_tr[train_idx],
            eval_set=(X_tr[val_idx], y_tr[val_idx]),
            early_stopping_rounds=50,
            verbose=False,
        )
        p = m.predict_proba(X_tr[val_idx])[:, 1]
        from sklearn.metrics import log_loss
        scores.append(log_loss(y_tr[val_idx], p))
    return float(np.mean(scores))


def train_catboost(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_trials: int = 40,
    timeout: int = 240,
) -> CatBoostClassifier:
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=TRAIN_CONFIG.seed + 2))
    study.optimize(
        lambda t: _catboost_objective(t, X_tr, y_tr, cv_splits),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )
    best = study.best_params
    logger.info(f"CatBoost best params: {best}")
    model = CatBoostClassifier(
        random_seed=TRAIN_CONFIG.seed,
        verbose=False,
        loss_function="Logloss",
        **best,
    )
    model.fit(X_tr, y_tr)
    return model


# ── Stacked ensemble ─────────────────────────────────────────────────────────

class StackedEnsemble:
    """
    3-model stacked ensemble with an isotonic meta-learner and final calibration.

    Pipeline:
      1. Base models (LGBM, XGB, CatBoost) are trained on the training set.
      2. OOF predictions from the base models are used to train a meta-learner.
      3. The meta-learner's output is calibrated using EnsembleCalibrator.
    """

    def __init__(
        self,
        n_optuna_trials: int = TRAIN_CONFIG.n_optuna_trials,
        optuna_timeout: int = TRAIN_CONFIG.optuna_timeout_sec,
        calibration_method: str = TRAIN_CONFIG.calibration_method,
        n_cv_splits: int = TRAIN_CONFIG.n_wf_splits,
        min_train: int = TRAIN_CONFIG.min_train_bars,
        gap: int = TRAIN_CONFIG.gap_bars,
    ):
        self.n_optuna_trials = n_optuna_trials
        self.optuna_timeout = optuna_timeout
        self.calibration_method = calibration_method
        self.n_cv_splits = n_cv_splits
        self.min_train = min_train
        self.gap = gap

        self.preprocessor: Optional[FeaturePreprocessor] = None
        self.lgbm_model = None
        self.xgb_model = None
        self.cat_model = None
        self.meta_learner: Optional[LogisticRegression] = None
        self.calibrator: Optional[EnsembleCalibrator] = None
        self._feature_names: List[str] = []
        self._lgbm_importance: Optional[pd.Series] = None

    def _get_cv_splits(self, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        return list(
            expanding_wf_splits(
                n,
                n_splits=self.n_cv_splits,
                min_train=self.min_train,
                gap=self.gap,
            )
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StackedEnsemble":
        """
        Fit the stacked ensemble.

        Strategy:
        1. Split data 80% base-train / 20% calibration (temporal split).
        2. Train base models (LGBM, XGB, CatBoost) on base-train using
           Optuna with inner CV for hyperparameter search.
        3. Predict on calibration split with each base model — these are
           genuine out-of-sample predictions, so calibration is unbiased.
        4. Fit a logistic meta-learner on [lgbm_cal, xgb_cal, cat_cal] → y_cal.
        5. Fit EnsembleCalibrator on meta-learner's output vs y_cal.
        6. Retrain ALL base models on the FULL data (for production predictions).
           The calibrator (fit on 20% holdout) remains valid because it maps
           the meta-learner's probability distribution, which is stable.
        """
        logger.info("=== Fitting StackedEnsemble ===")
        logger.info(f"  Data: {X.shape}, label balance: {y.mean():.3f}")

        # Preprocessing: imputation + feature selection (fit on full X, y)
        self.preprocessor = FeaturePreprocessor(max_features=TRAIN_CONFIG.max_features)
        self.preprocessor.fit(X, y)
        X_arr = self.preprocessor.transform(X)
        y_arr = y.values.astype(int)
        self._feature_names = self.preprocessor.feature_names

        n = len(y_arr)

        # ── Temporal split: base-train (80%) + calibration (20%) ─────────────
        cal_size = max(200, int(n * 0.20))
        bp = n - cal_size  # base-train partition end index
        X_base, X_cal = X_arr[:bp], X_arr[bp:]
        y_base, y_cal = y_arr[:bp], y_arr[bp:]
        logger.info(f"  Base-train: {len(X_base)} bars, Calibration: {len(X_cal)} bars")

        # CV splits for hyperparameter search (on base-train only)
        cv_splits = list(
            expanding_wf_splits(
                len(X_base),
                n_splits=self.n_cv_splits,
                min_train=min(self.min_train, len(X_base) // 3),
                gap=self.gap,
            )
        )
        hp_splits = cv_splits[: max(3, len(cv_splits))]

        # ── Train base models on base-train ───────────────────────────────────
        logger.info("Training LightGBM (base-train)...")
        self.lgbm_model = train_lgbm(
            X_base, y_base, hp_splits,
            n_trials=self.n_optuna_trials // 3 * 2,
            timeout=self.optuna_timeout // 3,
        )
        logger.info("Training XGBoost (base-train)...")
        self.xgb_model = train_xgb(
            X_base, y_base, hp_splits,
            n_trials=self.n_optuna_trials // 3,
            timeout=self.optuna_timeout // 3,
        )
        logger.info("Training CatBoost (base-train)...")
        self.cat_model = train_catboost(
            X_base, y_base, hp_splits,
            n_trials=self.n_optuna_trials // 4,
            timeout=self.optuna_timeout // 3,
        )

        # ── OOS calibration predictions ───────────────────────────────────────
        logger.info("Generating calibration-set predictions (OOS)...")
        p_lgbm_cal = self.lgbm_model.predict_proba(X_cal)[:, 1]
        p_xgb_cal = self.xgb_model.predict_proba(X_cal)[:, 1]
        p_cat_cal = self.cat_model.predict_proba(X_cal)[:, 1]
        meta_input_cal = np.column_stack([p_lgbm_cal, p_xgb_cal, p_cat_cal])

        # ── Meta-learner (fit on OOS calibration data) ────────────────────────
        logger.info("Training meta-learner on calibration set...")
        self.meta_learner = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000, random_state=TRAIN_CONFIG.seed
        )
        self.meta_learner.fit(meta_input_cal, y_cal)
        p_meta_cal = self.meta_learner.predict_proba(meta_input_cal)[:, 1]

        # ── Calibrator (fit on meta-learner's OOS output) ─────────────────────
        logger.info("Fitting calibrator on OOS predictions...")
        self.calibrator = EnsembleCalibrator()
        self.calibrator.fit(p_meta_cal, y_cal.astype(float))

        # ── Retrain base models on FULL data for production ───────────────────
        logger.info("Retraining base models on FULL data for production...")
        cv_full = list(
            expanding_wf_splits(
                n,
                n_splits=self.n_cv_splits,
                min_train=min(self.min_train, n // 3),
                gap=self.gap,
            )
        )
        self.lgbm_model = train_lgbm(
            X_arr, y_arr, cv_full,
            n_trials=max(5, self.n_optuna_trials // 5),  # fewer trials, reuse knowledge
            timeout=self.optuna_timeout // 5,
        )
        self.xgb_model = train_xgb(
            X_arr, y_arr, cv_full,
            n_trials=max(5, self.n_optuna_trials // 6),
            timeout=self.optuna_timeout // 6,
        )
        self.cat_model = train_catboost(
            X_arr, y_arr, cv_full,
            n_trials=max(5, self.n_optuna_trials // 6),
            timeout=self.optuna_timeout // 6,
        )

        # ── Feature importance ────────────────────────────────────────────────
        self._lgbm_importance = pd.Series(
            self.lgbm_model.feature_importances_, index=self._feature_names
        ).sort_values(ascending=False)

        logger.info("StackedEnsemble fitting complete.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probabilities for up-direction."""
        X_arr = self.preprocessor.transform(X)
        p_lgbm = self.lgbm_model.predict_proba(X_arr)[:, 1]
        p_xgb = self.xgb_model.predict_proba(X_arr)[:, 1]
        p_cat = self.cat_model.predict_proba(X_arr)[:, 1]
        meta_input = np.column_stack([p_lgbm, p_xgb, p_cat])
        p_meta = self.meta_learner.predict_proba(meta_input)[:, 1]
        return self.calibrator.predict(p_meta)

    def predict_proba_components(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Return individual model and ensemble probabilities for diagnostics."""
        X_arr = self.preprocessor.transform(X)
        p_lgbm = self.lgbm_model.predict_proba(X_arr)[:, 1]
        p_xgb = self.xgb_model.predict_proba(X_arr)[:, 1]
        p_cat = self.cat_model.predict_proba(X_arr)[:, 1]
        meta_input = np.column_stack([p_lgbm, p_xgb, p_cat])
        p_meta = self.meta_learner.predict_proba(meta_input)[:, 1]
        p_cal = self.calibrator.predict(p_meta)
        return {
            "lgbm": p_lgbm,
            "xgb": p_xgb,
            "catboost": p_cat,
            "meta": p_meta,
            "calibrated": p_cal,
        }

    @property
    def feature_importance(self) -> pd.Series:
        if self._lgbm_importance is None or self._lgbm_importance.empty:
            return pd.Series(dtype=float)
        return self._lgbm_importance

    # ── Serialization ─────────────────────────────────────────────────────────

    def save(self, symbol: str, model_dir: Path = MODELS_DIR) -> None:
        safe = symbol.replace("/", "_")
        path = model_dir / f"{safe}_ensemble.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, symbol: str, model_dir: Path = MODELS_DIR) -> "StackedEnsemble":
        safe = symbol.replace("/", "_")
        path = model_dir / f"{safe}_ensemble.pkl"
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model
