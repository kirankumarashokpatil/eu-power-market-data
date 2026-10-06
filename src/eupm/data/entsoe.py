"""Client for the ENTSO-E Transparency Platform REST API (European market data).

Free registration gives an API token: create an account on transparency.entsoe.eu, then
email transparency@entsoe.eu with the subject "Restful API access". Put the token in the
``ENTSOE_API_KEY`` environment variable.

Datasets used, all published **day-ahead**, so they are legitimate forecast inputs:

=========  ========================================  ======================
docType    Content                                   Key params
=========  ========================================  ======================
A44        Day-ahead (SDAC) prices                   in/out_Domain = zone
A65 / A01  Day-ahead total load forecast             outBiddingZone_Domain
A69 / A01  Day-ahead wind & solar generation fcst    in_Domain, psrType
=========  ========================================  ======================

The parser handles the awkward parts of the format:

* **Namespaces differ by document type.** Tags are matched with a ``{*}`` wildcard.
* **Mixed resolutions.** SDAC moved to a 15-minute MTU in 2025, so PT15M, PT30M and PT60M
  series can all appear. Everything is normalised to UTC and can be resampled.
* **Curve type A03 (compressed).** Repeated values are omitted, so missing positions
  carry the last value forward.
* **"No matching data"** acknowledgement documents return an empty frame, not an error.
"""

from __future__ import annotations

import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import requests
from pydantic import BaseModel, Field, field_validator

BASE_URL = "https://web-api.tp.entsoe.eu/api"
log = logging.getLogger(__name__)

# Bidding-zone EIC codes (subset; extend as coverage grows)
ZONES: dict[str, str] = {
    "DE_LU": "10Y1001A1001A82H",
    "FR": "10YFR-RTE------C",
    "NL": "10YNL----------L",
    "BE": "10YBE----------2",
    "AT": "10YAT-APG------L",
    "ES": "10YES-REE------0",
    "PT": "10YPT-REN------W",
    "IT_NORD": "10Y1001A1001A73I",
    "PL": "10YPL-AREA-----S",
    "DK1": "10YDK-1--------W",
    "NO2": "10YNO-2--------T",
    "SE3": "10Y1001A1001A46L",
    "FI": "10YFI-1--------U",
}

PSR_TYPES = {"B16": "solar", "B18": "wind_offshore", "B19": "wind_onshore"}
_RES_MIN = {"PT15M": 15, "PT30M": 30, "PT60M": 60, "P1D": 1440}


class EntsoeConfig(BaseModel):
    zones: list[str] = ["DE_LU", "FR", "NL", "BE", "ES", "PL"]
    start: date = date(2024, 1, 1)
    end: date = date(2025, 12, 31)
    api_key: str = Field(default_factory=lambda: os.environ.get("ENTSOE_API_KEY", ""))
    cache_dir: Path = Path("data/entsoe")
    chunk_days: int = Field(default=90, ge=1, le=365)  # API caps most queries at one year
    timeout_s: float = 60.0
    retries: int = 3

    @field_validator("zones")
    @classmethod
    def _known_zones(cls, v: list[str]) -> list[str]:
        unknown = [z for z in v if z not in ZONES]
        if unknown:
            raise ValueError(f"unknown zones {unknown}; known: {sorted(ZONES)}")
        return v


# --------------------------------------------------------------------------- parsing


def _text(el: ET.Element, path: str) -> str | None:
    found = el.find(path)
    return found.text if found is not None else None


def parse_timeseries_xml(xml: str | bytes, value_tag: str) -> pl.DataFrame:
    """Parse any ENTSO-E market document into long format:
    ``start_time`` (UTC), ``value``, ``resolution_min``, ``psr_type``, ``series``.

    ``value_tag`` is ``price.amount`` for prices or ``quantity`` for load/generation.
    """
    root = ET.fromstring(xml)
    if root.tag.endswith("Acknowledgement_MarketDocument"):
        reason = _text(root, ".//{*}Reason/{*}text") or "no data"
        log.info("ENTSO-E acknowledgement: %s", reason)
        return _empty()

    rows: dict[str, list] = {
        k: [] for k in ("start_time", "value", "resolution_min", "psr_type", "series")
    }
    for s_idx, ts in enumerate(root.iterfind(".//{*}TimeSeries")):
        psr = _text(ts, ".//{*}MktPSRType/{*}psrType")
        curve = _text(ts, "{*}curveType") or "A01"
        for period in ts.iterfind("{*}Period"):
            start_s = _text(period, "{*}timeInterval/{*}start")
            end_s = _text(period, "{*}timeInterval/{*}end")
            res_s = _text(period, "{*}resolution")
            if not (start_s and end_s and res_s) or res_s not in _RES_MIN:
                continue
            start = _parse_ts(start_s)
            step = timedelta(minutes=_RES_MIN[res_s])
            n = int((_parse_ts(end_s) - start) / step)
            values: list[float | None] = [None] * n
            for pt in period.iterfind("{*}Point"):
                pos, val = _text(pt, "{*}position"), _text(pt, f"{{*}}{value_tag}")
                if pos is not None and val is not None and 1 <= int(pos) <= n:
                    values[int(pos) - 1] = float(val)
            if curve == "A03":  # compressed curve: carry the last value forward
                last = None
                for i, v in enumerate(values):
                    last = v if v is not None else last
                    values[i] = last
            for i, v in enumerate(values):
                if v is None:
                    continue
                rows["start_time"].append(start + i * step)
                rows["value"].append(v)
                rows["resolution_min"].append(_RES_MIN[res_s])
                rows["psr_type"].append(PSR_TYPES.get(psr or "", psr))
                rows["series"].append(s_idx)
    if not rows["value"]:
        return _empty()
    return pl.DataFrame(rows, schema=_SCHEMA)


_SCHEMA = pl.Schema(
    {
        "start_time": pl.Datetime("us", "UTC"),
        "value": pl.Float64(),
        "resolution_min": pl.Int32(),
        "psr_type": pl.Utf8(),
        "series": pl.Int32(),
    }
)


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=_SCHEMA)


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s.replace("Z", ""), "%Y-%m-%dT%H:%M").replace(tzinfo=UTC)


def to_hourly(df: pl.DataFrame, by: list[str] | None = None) -> pl.DataFrame:
    """Normalise mixed 15/30/60-minute series to hourly means on a UTC grid.
    If the same hour arrives at two resolutions, the mean of the two is kept."""
    if df.is_empty():
        return df.select("start_time", "value")
    keys = by or []
    return (
        df.with_columns(pl.col("start_time").dt.truncate("1h"))
        .group_by(["start_time", *keys])
        .agg(pl.col("value").mean())
        .sort(["start_time", *keys])
    )


# --------------------------------------------------------------------------- fetching


def _fmt(d: datetime) -> str:
    return d.strftime("%Y%m%d%H%M")


def _request(params: dict[str, str], cfg: EntsoeConfig) -> bytes:
    if not cfg.api_key:
        raise RuntimeError("Set ENTSOE_API_KEY (free token from transparency.entsoe.eu)")
    params = {"securityToken": cfg.api_key, **params}
    for attempt in range(cfg.retries):
        resp = requests.get(BASE_URL, params=params, timeout=cfg.timeout_s)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2**attempt * 5)
            continue
        if resp.status_code == 400 and b"No matching data" in resp.content:
            return resp.content  # parsed as an acknowledgement -> empty frame
        resp.raise_for_status()
        return resp.content
    raise RuntimeError(f"ENTSO-E request failed after {cfg.retries} attempts: {params}")


def _chunks(cfg: EntsoeConfig) -> list[tuple[datetime, datetime]]:
    out = []
    cur = datetime.combine(cfg.start, datetime.min.time(), UTC)
    stop = datetime.combine(cfg.end + timedelta(days=1), datetime.min.time(), UTC)
    while cur < stop:
        nxt = min(cur + timedelta(days=cfg.chunk_days), stop)
        out.append((cur, nxt))
        cur = nxt
    return out


def _fetch(base_params: dict[str, str], value_tag: str, cfg: EntsoeConfig) -> pl.DataFrame:
    frames = []
    for a, b in _chunks(cfg):
        xml = _request({**base_params, "periodStart": _fmt(a), "periodEnd": _fmt(b)}, cfg)
        frames.append(parse_timeseries_xml(xml, value_tag))
    return pl.concat(frames) if frames else _empty()


def fetch_prices(zone: str, cfg: EntsoeConfig) -> pl.DataFrame:
    eic = ZONES[zone]
    return _fetch({"documentType": "A44", "in_Domain": eic, "out_Domain": eic}, "price.amount", cfg)


def fetch_load_forecast(zone: str, cfg: EntsoeConfig) -> pl.DataFrame:
    return _fetch(
        {"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": ZONES[zone]},
        "quantity",
        cfg,
    )


def fetch_wind_solar_forecast(zone: str, cfg: EntsoeConfig) -> pl.DataFrame:
    return _fetch(
        {"documentType": "A69", "processType": "A01", "in_Domain": ZONES[zone]}, "quantity", cfg
    )


def build_zone_frame(prices: pl.DataFrame, load: pl.DataFrame, res: pl.DataFrame) -> pl.DataFrame:
    """Join the three datasets into one hourly frame:
    start_time, price, load_fc, solar_fc, wind_fc, residual_load_fc."""
    p = to_hourly(prices).rename({"value": "price"})
    lf = to_hourly(load).rename({"value": "load_fc"})
    out = p.join(lf, on="start_time", how="left")
    if not res.is_empty():
        r = (
            to_hourly(res, by=["psr_type"])
            .with_columns(
                pl.when(pl.col("psr_type") == "solar")
                .then(pl.lit("solar_fc"))
                .otherwise(pl.lit("wind_fc"))
                .alias("kind")
            )
            .group_by("start_time", "kind")
            .agg(pl.col("value").sum())
            .pivot(on="kind", index="start_time", values="value")
        )
        out = out.join(r, on="start_time", how="left")
    for col in ("load_fc", "solar_fc", "wind_fc"):
        if col not in out.columns:
            out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    return out.with_columns(
        (
            pl.col("load_fc") - pl.col("solar_fc").fill_null(0) - pl.col("wind_fc").fill_null(0)
        ).alias("residual_load_fc")
    ).sort("start_time")


def load_or_fetch_zone(zone: str, cfg: EntsoeConfig) -> pl.DataFrame:
    """Cached hourly frame for one bidding zone."""
    path = Path(cfg.cache_dir) / f"{zone}_{cfg.start}_{cfg.end}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    log.info("fetching %s", zone)
    df = build_zone_frame(
        fetch_prices(zone, cfg),
        fetch_load_forecast(zone, cfg),
        fetch_wind_solar_forecast(zone, cfg),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df
