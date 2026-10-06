"""Synthetic European zone data for tests and offline demos.

Not real market data: never report results from this as findings. It exists so the
pipeline, tests and dashboard run without an ENTSO-E token or network access.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

# Rough, made-up zone "personalities" so the EU pipeline has something to chew on offline.
_EU_PROFILES = {
    "DE_LU": {"load": 55_000, "solar": 30_000, "wind": 25_000, "base": 85},
    "FR": {"load": 50_000, "solar": 8_000, "wind": 8_000, "base": 70},
    "NL": {"load": 13_000, "solar": 9_000, "wind": 5_000, "base": 88},
    "BE": {"load": 9_500, "solar": 3_500, "wind": 2_500, "base": 85},
    "ES": {"load": 28_000, "solar": 20_000, "wind": 10_000, "base": 60},
    "PL": {"load": 17_000, "solar": 6_000, "wind": 4_000, "base": 105},
}


def make_synthetic_eu(
    zones: list[str] | None = None, days: int = 400, seed: int = 1
) -> dict[str, pl.DataFrame]:
    """Synthetic hourly frames shaped like ``entsoe.build_zone_frame`` output. NOT real data."""
    rng = np.random.default_rng(seed)
    zones = zones or list(_EU_PROFILES)
    n = days * 24
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    times = [t0 + timedelta(hours=i) for i in range(n)]
    h = np.tile(np.arange(24), days)
    d = np.repeat(np.arange(days), 24)
    season = np.cos(2 * np.pi * d / 365)  # +1 in winter
    common_wind = np.convolve(rng.normal(0, 1, n), np.ones(36) / 36, mode="same")
    out = {}
    for z in zones:
        pr = _EU_PROFILES.get(z, _EU_PROFILES["DE_LU"])
        daily_shape = 1 + 0.12 * np.sin(2 * np.pi * (h - 7) / 24) + 0.06 * season
        load = pr["load"] * daily_shape * (1 + rng.normal(0, 0.02, n))
        sun = np.clip(np.sin(np.pi * (h - 6) / 13), 0, None) * (0.6 - 0.35 * season)
        solar = pr["solar"] * sun * rng.uniform(0.6, 1.0, n)
        wind_idx = np.clip(0.35 + 0.25 * (0.7 * common_wind * 4 + 0.3 * rng.normal(0, 1, n)), 0, 1)
        wind = pr["wind"] * wind_idx
        resid = load - solar - wind
        price = pr["base"] + 0.004 * (resid - 0.6 * pr["load"]) + 15 * season
        price = price + rng.normal(0, 6, n) + (rng.random(n) < 0.005) * rng.gamma(2, 60, n)
        out[z] = pl.DataFrame(
            {
                "start_time": times,
                "price": price.round(2),
                "load_fc": load.round(0),
                "solar_fc": solar.round(0),
                "wind_fc": wind.round(0),
                "residual_load_fc": resid.round(0),
            }
        )
    return out
