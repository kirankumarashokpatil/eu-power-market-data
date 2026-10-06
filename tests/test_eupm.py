"""ENTSO-E parser and EU analytics tests, using hand-built XML in the real API's format."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from eupm.analytics import (
    arbitrage_atlas,
    eu_feature_columns,
    eu_features,
    price_correlation,
    quality_report,
)
from eupm.data.entsoe import EntsoeConfig, build_zone_frame, parse_timeseries_xml, to_hourly
from eupm.data.synthetic import make_synthetic_eu

PRICE_NS = "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"
GL_NS = "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"


def _doc(ns: str, root: str, series: str) -> str:
    return f'<?xml version="1.0" encoding="UTF-8"?><{root} xmlns="{ns}">{series}</{root}>'


def _series(
    start: str,
    end: str,
    res: str,
    points: list[tuple[int, float]],
    tag: str,
    curve: str = "A01",
    psr: str | None = None,
) -> str:
    pts = "".join(f"<Point><position>{p}</position><{tag}>{v}</{tag}></Point>" for p, v in points)
    psr_xml = f"<MktPSRType><psrType>{psr}</psrType></MktPSRType>" if psr else ""
    return (
        f"<TimeSeries><curveType>{curve}</curveType>{psr_xml}<Period><timeInterval>"
        f"<start>{start}</start><end>{end}</end></timeInterval><resolution>{res}"
        f"</resolution>{pts}</Period></TimeSeries>"
    )


def test_parse_hourly_prices():
    xml = _doc(
        PRICE_NS,
        "Publication_MarketDocument",
        _series(
            "2025-01-01T23:00Z",
            "2025-01-02T02:00Z",
            "PT60M",
            [(1, 50.0), (2, 45.5), (3, -3.2)],
            "price.amount",
        ),
    )
    df = parse_timeseries_xml(xml, "price.amount")
    assert df["value"].to_list() == [50.0, 45.5, -3.2]
    assert str(df["start_time"][0]) == "2025-01-01 23:00:00+00:00"
    assert df["resolution_min"].unique().to_list() == [60]


def test_a03_compressed_curve_carries_values_forward():
    xml = _doc(
        PRICE_NS,
        "Publication_MarketDocument",
        _series(
            "2025-06-01T00:00Z",
            "2025-06-01T01:00Z",
            "PT15M",
            [(1, 10.0), (3, 30.0)],
            "price.amount",
            curve="A03",
        ),
    )
    df = parse_timeseries_xml(xml, "price.amount")
    assert df["value"].to_list() == [10.0, 10.0, 30.0, 30.0]


def test_mixed_resolution_resamples_to_hourly():
    q = _series(
        "2025-06-01T00:00Z",
        "2025-06-01T01:00Z",
        "PT15M",
        [(1, 10.0), (2, 20.0), (3, 30.0), (4, 40.0)],
        "price.amount",
    )
    df = parse_timeseries_xml(_doc(PRICE_NS, "Publication_MarketDocument", q), "price.amount")
    hourly = to_hourly(df)
    assert hourly.height == 1 and hourly["value"][0] == pytest.approx(25.0)


def test_acknowledgement_returns_empty():
    xml = (
        '<Acknowledgement_MarketDocument xmlns="urn:x"><Reason><code>999</code>'
        "<text>No matching data found</text></Reason></Acknowledgement_MarketDocument>"
    )
    assert parse_timeseries_xml(xml, "price.amount").is_empty()


def test_wind_solar_split_and_residual_load():
    t = ("2025-06-01T10:00Z", "2025-06-01T12:00Z", "PT60M")
    prices = parse_timeseries_xml(
        _doc(
            PRICE_NS,
            "Publication_MarketDocument",
            _series(*t, [(1, 40.0), (2, 35.0)], "price.amount"),
        ),
        "price.amount",
    )
    load = parse_timeseries_xml(
        _doc(GL_NS, "GL_MarketDocument", _series(*t, [(1, 1000.0), (2, 1100.0)], "quantity")),
        "quantity",
    )
    res = parse_timeseries_xml(
        _doc(
            GL_NS,
            "GL_MarketDocument",
            _series(*t, [(1, 300.0), (2, 350.0)], "quantity", psr="B16")
            + _series(*t, [(1, 100.0), (2, 50.0)], "quantity", psr="B19")
            + _series(*t, [(1, 20.0), (2, 30.0)], "quantity", psr="B18"),
        ),
        "quantity",
    )
    df = build_zone_frame(prices, load, res)
    assert df["solar_fc"].to_list() == [300.0, 350.0]
    assert df["wind_fc"].to_list() == [120.0, 80.0]  # onshore + offshore
    assert df["residual_load_fc"].to_list() == [580.0, 670.0]


def test_unknown_zone_rejected():
    with pytest.raises(ValidationError):
        EntsoeConfig(zones=["ATLANTIS"])


def test_quality_report_flags_gaps():
    df = make_synthetic_eu(["FR"], days=5)["FR"]
    gappy = df.with_columns(
        pl.when(pl.int_range(pl.len()).is_between(10, 14))
        .then(None)
        .otherwise(pl.col("price"))
        .alias("price")
    )
    q = quality_report({"FR": gappy}).row(0, named=True)
    assert q["longest_price_gap_h"] == 5
    assert q["price_coverage_pct"] < 100


def test_atlas_ranks_zones_and_values_are_positive():
    frames = make_synthetic_eu(["DE_LU", "BE"], days=10)
    summary, daily = arbitrage_atlas(frames)
    assert set(summary["zone"]) == {"DE_LU", "BE"}
    assert daily.height == 18  # 10 UTC days = 9 full CET days per zone
    assert (daily["bess_value"] >= -1e-6).all()  # perfect foresight never loses money
    corr = price_correlation(frames)
    assert np.isclose(corr.filter(pl.col("zone") == "BE")["BE"][0], 1.0)


def test_eu_features_use_no_same_day_prices():
    df = make_synthetic_eu(["DE_LU"], days=15)["DE_LU"]
    f = eu_features(df)
    row = f.row(100, named=True)
    t = row["start_time"]
    src = df.filter(pl.col("start_time") == t).row(0, named=True)
    prev = df.filter(pl.col("start_time") == t - timedelta(hours=24)).row(0, named=True)
    assert row["price_lag24"] == prev["price"]
    assert row["load_fc"] == src["load_fc"]  # day-ahead fundamentals at time t are allowed
    assert "price" not in eu_feature_columns(f)


def test_battery_respects_limits_hourly():
    from eupm.battery import optimise_day
    from eupm.config import BatteryConfig

    cfg = BatteryConfig()
    prices = 60 + 50 * np.sin(np.linspace(0, 2 * np.pi, 24))
    s = optimise_day(prices, cfg, dt_h=1.0)
    assert s.charge_mw.max() <= cfg.power_mw + 1e-6
    assert s.soc_mwh.max() <= cfg.energy_mwh + 1e-6
    assert abs(s.soc_mwh[-1] - cfg.initial_soc_frac * cfg.energy_mwh) < 1e-6


def test_backtest_smoke():
    from eupm.analytics import run_eu_backtest
    from eupm.config import Settings

    s = Settings()
    s.backtest.train_days, s.backtest.test_days = 30, 2
    s.model.n_estimators = 30
    res = run_eu_backtest(make_synthetic_eu(["FR"], days=45)["FR"], s)
    assert set(res.summary["model"]) == {"LightGBM", "XGBoost", "Seasonal naive"}
    assert res.predictions.height == 3 * 2 * 24


def test_days_are_market_days_with_dst():
    from eupm.analytics import _full_days

    df = make_synthetic_eu(["DE_LU"], days=320)["DE_LU"]  # 2024: covers both DST switches
    full = _full_days(df)
    first_hour = (
        full.group_by("day")
        .agg(pl.col("start_time").min())
        .with_columns(
            pl.col("start_time").dt.convert_time_zone("Europe/Brussels").dt.hour().alias("h")
        )
    )
    assert (first_hour["h"] == 0).all()  # every day starts at local midnight, not UTC
    sizes = dict(full.group_by("day").len().iter_rows())
    assert sizes[date(2024, 3, 31)] == 23  # clocks forward
    assert sizes[date(2024, 10, 27)] == 25  # clocks back


def test_no_lag_reaches_into_the_delivery_day():
    df = make_synthetic_eu(["DE_LU"], days=320)["DE_LU"]
    f = eu_features(df).filter(pl.col("price_lag24").is_not_null())
    local = pl.col("start_time").dt.convert_time_zone("Europe/Brussels")
    day_start = local.dt.truncate("1d").dt.convert_time_zone("UTC")
    leaks = f.filter(pl.col("start_time") - timedelta(hours=24) >= day_start)
    assert leaks.is_empty()


def test_request_windows_follow_delivery_days():
    from eupm.data.entsoe import _chunks

    winter = _chunks(EntsoeConfig(start=date(2025, 1, 1), end=date(2025, 1, 7)))
    assert [(a.isoformat(), b.isoformat()) for a, b in winter] == [
        ("2024-12-31T23:00:00+00:00", "2025-01-07T23:00:00+00:00")
    ]
    summer = _chunks(EntsoeConfig(start=date(2025, 7, 1), end=date(2025, 7, 1)))
    assert summer[0][0].hour == 22  # CEST is UTC+2
