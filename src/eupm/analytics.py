"""European market coverage: data-quality checks, a cross-zone battery arbitrage atlas,
and day-ahead price forecasting driven by fundamentals for any ENTSO-E bidding zone.

The forecast is issued on D-1 before the 12:00 CET SDAC gate closure. At that point:

* all prices for D-1 are known (published on D-2), so price lags of at least 24 hours are safe;
* day-ahead load, wind and solar forecasts for D are published, so they can be used at
  the same timestamp. That is the main gain over a price-only model.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import cast

import numpy as np
import polars as pl

from eupm.battery import optimise_day, settle
from eupm.config import BatteryConfig, Settings
from eupm.models import FeatureModel, LightGBMQuantile, XGBoostQuantile

log = logging.getLogger(__name__)
HOURS = 24
FUNDAMENTALS = ["load_fc", "solar_fc", "wind_fc", "residual_load_fc"]


@dataclass
class BacktestResult:
    predictions: pl.DataFrame  # start_time, model, actual, p10, p50, p90
    daily: pl.DataFrame  # day, model, revenue, perfect_revenue
    summary: pl.DataFrame  # one row per model


def pinball(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    diff = y - q_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def summarise(pred: pl.DataFrame, daily: pl.DataFrame) -> pl.DataFrame:
    """Accuracy, calibration and money: MAE, pinball, P10-P90 coverage, revenue capture."""
    rows = []
    for name in pred["model"].unique(maintain_order=True):
        p = pred.filter(pl.col("model") == name)
        d = daily.filter(pl.col("model") == name)
        y, p10, p50, p90 = (p[c].to_numpy() for c in ("actual", "p10", "p50", "p90"))
        rev = cast(float, d["revenue"].sum())
        perfect = cast(float, d["perfect_revenue"].sum())
        rows.append(
            {
                "model": name,
                "mae": float(np.mean(np.abs(y - p50))),
                "rmse": float(np.sqrt(np.mean((y - p50) ** 2))),
                "pinball_avg": float(
                    np.mean([pinball(y, p10, 0.1), pinball(y, p50, 0.5), pinball(y, p90, 0.9)])
                ),
                "p10_p90_coverage": float(np.mean((y >= p10) & (y <= p90))),
                "revenue_eur": rev,
                "capture_pct": 100 * rev / perfect if perfect else float("nan"),
                "worst_day_eur": d["revenue"].min(),
                "p5_day_eur": float(np.percentile(d["revenue"].to_numpy(), 5)),
            }
        )
    return pl.DataFrame(rows).sort("capture_pct", descending=True)


# --------------------------------------------------------------------------- quality


def quality_report(frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """One row per zone: coverage and gaps per dataset, plus basic price sanity checks."""
    rows = []
    for zone, df in frames.items():
        if df.is_empty():
            rows.append({"zone": zone, "hours_expected": 0})
            continue
        t0 = cast(datetime, df["start_time"].min())
        t1 = cast(datetime, df["start_time"].max())
        expected = int((t1 - t0).total_seconds() // 3600) + 1
        price = df["price"]
        present = df.filter(pl.col("price").is_not_null())["start_time"].sort()
        gaps = present.diff().dt.total_hours().fill_null(1)
        row = {
            "zone": zone,
            "hours_expected": expected,
            "price_coverage_pct": 100 * price.drop_nulls().len() / expected,
            "longest_price_gap_h": int(cast(float, gaps.max()) or 1) - 1,
            "duplicate_hours": df.height - df["start_time"].n_unique(),
            "negative_price_pct": 100 * float(cast(float, (price < 0).mean()) or 0),
            "price_min": price.min(),
            "price_max": price.max(),
        }
        for col in FUNDAMENTALS[:3]:
            row[f"{col}_coverage_pct"] = 100 * df[col].drop_nulls().len() / expected
        rows.append(row)
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- atlas


def _full_days(df: pl.DataFrame, col: str = "price") -> pl.DataFrame:
    df = df.filter(pl.col(col).is_not_null()).with_columns(
        pl.col("start_time").dt.date().alias("day")
    )
    ok = df.group_by("day").len().filter(pl.col("len") == HOURS)["day"]
    return df.filter(pl.col("day").is_in(ok.implode())).sort("start_time")


def arbitrage_atlas(
    frames: dict[str, pl.DataFrame], battery: BatteryConfig | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """How much is day-ahead arbitrage worth in each zone?

    For every full day, a perfect-foresight LP schedules a battery (1 MW / 2 MWh by
    default) on hourly prices. Perfect foresight gives the upper bound a trader works
    towards. Returns (summary per zone, daily values per zone)."""
    cfg = battery or BatteryConfig()
    daily_rows = []
    for zone, df in frames.items():
        full = _full_days(df)
        for day, g in full.group_by("day", maintain_order=True):
            p = g["price"].to_numpy()
            value = settle(optimise_day(p, cfg, dt_h=1.0), p, cfg, dt_h=1.0)
            daily_rows.append(
                {
                    "zone": zone,
                    "day": day[0],
                    "bess_value": value,
                    "spread": float(p.max() - p.min()),
                    "mean_price": float(p.mean()),
                    "neg_hours": int((p < 0).sum()),
                }
            )
    daily = pl.DataFrame(daily_rows)
    summary = (
        daily.group_by("zone")
        .agg(
            pl.len().alias("days"),
            pl.col("mean_price").mean().alias("avg_price"),
            pl.col("spread").mean().alias("avg_daily_spread"),
            pl.col("bess_value").mean().alias("avg_daily_value"),
            (pl.col("bess_value").mean() * 365 / cfg.power_mw).alias("value_per_mw_year"),
            pl.col("bess_value").quantile(0.05).alias("p5_daily_value"),
            (pl.col("neg_hours").sum() / (pl.len() * HOURS) * 100).alias("negative_hours_pct"),
        )
        .sort("value_per_mw_year", descending=True)
    )
    return summary, daily


def price_correlation(frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Hourly price correlation between zones. High values point to coupled markets."""
    wide = None
    for zone, df in frames.items():
        s = df.select("start_time", pl.col("price").alias(zone))
        wide = s if wide is None else wide.join(s, on="start_time", how="inner")
    if wide is None:
        return pl.DataFrame()
    return (
        wide.drop("start_time")
        .corr()
        .insert_column(0, pl.Series("zone", [c for c in wide.columns if c != "start_time"]))
    )


# --------------------------------------------------------------------------- forecasting


def eu_features(df: pl.DataFrame, tz: str = "Europe/Brussels") -> pl.DataFrame:
    local = pl.col("start_time").dt.convert_time_zone(tz)
    lags = [24, 48, 168]
    out = df.sort("start_time").with_columns(
        local.dt.hour().alias("hour"),
        local.dt.weekday().alias("dow"),
        local.dt.month().alias("month"),
        (local.dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
        (2 * math.pi * local.dt.ordinal_day() / 365.25).sin().alias("doy_sin"),
        (2 * math.pi * local.dt.ordinal_day() / 365.25).cos().alias("doy_cos"),
        *[pl.col("price").shift(lag).alias(f"price_lag{lag}") for lag in lags],
        pl.col("price").shift(24).rolling_mean(24).alias("price_rmean24"),
        pl.col("price").shift(24).rolling_mean(168).alias("price_rmean168"),
        pl.col("price").shift(24).rolling_std(168).alias("price_rstd168"),
        (pl.col("residual_load_fc") - pl.col("residual_load_fc").shift(24)).alias("resid_chg24"),
        (pl.col("residual_load_fc") / pl.col("load_fc")).alias("resid_share"),
    )
    return out.drop_nulls(subset=["price", "price_lag168", "price_rmean168"])


def eu_feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"start_time", "price", "day"}]


def run_eu_backtest(
    zone_df: pl.DataFrame, settings: Settings | None = None, tz: str = "Europe/Brussels"
) -> BacktestResult:
    """Walk-forward day-ahead forecast for one zone. Missing fundamentals are passed to
    the tree models as NaN, which both libraries handle natively."""
    s = settings or Settings()
    feats = _full_days(eu_features(zone_df, tz))
    cols = eu_feature_columns(feats)
    days = feats["day"].unique(maintain_order=True).to_list()
    bt = s.backtest
    if len(days) < bt.train_days + bt.test_days:
        raise ValueError(f"need {bt.train_days + bt.test_days} full days, got {len(days)}")
    models: list[FeatureModel] = [LightGBMQuantile(s.model), XGBoostQuantile(s.model)]
    preds, daily_rows = [], []

    for k, day in enumerate(days[-bt.test_days :]):
        idx = days.index(day)
        today = feats.filter(pl.col("day") == day)
        y = today["price"].to_numpy()
        x = today.select(cols).to_numpy().astype(np.float64)
        if k % bt.retrain_every_days == 0:
            train = feats.filter(pl.col("day").is_in(days[max(0, idx - bt.train_days) : idx]))
            for m in models:
                m.fit(train.select(cols).to_numpy().astype(np.float64), train["price"].to_numpy())
        forecasts = {m.name: m.predict(x) for m in models}
        naive = today["price_lag24"].to_numpy()
        forecasts["Seasonal naive"] = {0.1: naive, 0.5: naive, 0.9: naive}

        perfect = settle(optimise_day(y, s.battery, 1.0), y, s.battery, 1.0)
        for name, q in forecasts.items():
            sched = optimise_day(q[0.5], s.battery, 1.0)
            daily_rows.append(
                {
                    "day": day,
                    "model": name,
                    "revenue": settle(sched, y, s.battery, 1.0),
                    "perfect_revenue": perfect,
                }
            )
            preds.append(
                pl.DataFrame(
                    {
                        "start_time": today["start_time"],
                        "model": [name] * len(y),
                        "actual": y,
                        **{
                            f"p{int(qq * 100)}": np.asarray(q[qq], dtype=np.float64)
                            for qq in (0.1, 0.5, 0.9)
                        },
                    }
                )
            )
    pred_df, daily_df = pl.concat(preds), pl.DataFrame(daily_rows)
    return BacktestResult(pred_df, daily_df, summarise(pred_df, daily_df))
