"""Forecasting models behind one small interface.

Each model returns a dict mapping quantile -> predictions, so point and probabilistic
models are compared on the same footing.
"""

from __future__ import annotations

from typing import Protocol

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from eupm.config import ModelConfig

Quantiles = dict[float, np.ndarray]


class FeatureModel(Protocol):
    name: str

    def fit(self, x: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, x: np.ndarray) -> Quantiles: ...


def _sort_quantiles(q: Quantiles) -> Quantiles:
    """Stop quantile crossing (e.g. P10 > P50) by sorting across quantiles per row."""
    keys = sorted(q)
    stacked = np.sort(np.vstack([q[k] for k in keys]), axis=0)
    return {k: stacked[i] for i, k in enumerate(keys)}


class LightGBMQuantile:
    name = "LightGBM"

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.models: dict[float, lgb.LGBMRegressor] = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        for q in self.cfg.quantiles:
            m = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.cfg.n_estimators,
                learning_rate=self.cfg.learning_rate,
                num_leaves=self.cfg.num_leaves,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                random_state=self.cfg.random_state,
                verbose=-1,
            )
            m.fit(x, y)
            self.models[q] = m

    def predict(self, x: np.ndarray) -> Quantiles:
        return _sort_quantiles({q: np.asarray(m.predict(x)) for q, m in self.models.items()})

    def contributions(self, x: np.ndarray, q: float = 0.5) -> np.ndarray:
        """SHAP values (TreeSHAP, computed natively by LightGBM) for the ``q`` model.

        Returns shape (rows, features + 1): each feature's push away from the average
        forecast in EUR/MWh, with the average (base value) in the last column. A row
        sums to that model's prediction."""
        return np.asarray(self.models[q].predict(x, pred_contrib=True))


class XGBoostQuantile:
    """One XGBoost model fitted to all quantiles at once (``reg:quantileerror``)."""

    name = "XGBoost"

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.model: xgb.XGBRegressor | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=np.array(self.cfg.quantiles),
            n_estimators=self.cfg.n_estimators,
            learning_rate=self.cfg.learning_rate,
            max_depth=self.cfg.max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=self.cfg.random_state,
        )
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> Quantiles:
        assert self.model is not None, "call fit first"
        pred = np.asarray(self.model.predict(x)).reshape(len(x), -1)
        return _sort_quantiles({q: pred[:, i] for i, q in enumerate(self.cfg.quantiles)})
