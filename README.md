# European Power Market Data Platform

A data and analytics pipeline for **European day-ahead (SDAC) power markets**, built on the
[ENTSO-E Transparency Platform](https://transparency.entsoe.eu/). It covers:

1. **Ingestion:** day-ahead prices, load forecasts and wind/solar forecasts for 13
   bidding zones, normalised to one tidy hourly table per zone.
2. **Data quality:** coverage, gaps, duplicates and price sanity checks per zone,
   before anyone builds on the data.
3. **Battery arbitrage atlas:** which zones pay a battery most, how spreads change over
   time, and how strongly zones are coupled.
4. **Forecasting:** day-ahead price forecasts driven by fundamentals (LightGBM/XGBoost
   quantiles), scored on **battery revenue capture**, not only on error.

<!-- After running on real data, add a screenshot: docs/dashboard.png -->

## Coverage

| Zone | EIC | | Zone | EIC |
|---|---|---|---|---|
| DE_LU | 10Y1001A1001A82H | | IT_NORD | 10Y1001A1001A73I |
| FR | 10YFR-RTE------C | | PL | 10YPL-AREA-----S |
| NL | 10YNL----------L | | DK1 | 10YDK-1--------W |
| BE | 10YBE----------2 | | NO2 | 10YNO-2--------T |
| AT | 10YAT-APG------L | | SE3 | 10Y1001A1001A46L |
| ES | 10YES-REE------0 | | FI | 10YFI-1--------U |
| PT | 10YPT-REN------W | | | |

Adding a zone means adding one line to `ZONES` in `src/eupm/data/entsoe.py`.

| ENTSO-E document | Content | Why it matters |
|---|---|---|
| A44 | Day-ahead prices | Target and lags |
| A65 / A01 | Day-ahead total load forecast | Demand driver |
| A69 / A01 | Day-ahead wind & solar forecast (B16, B18, B19) | Supply driver → residual load |

## Engineering notes

The ENTSO-E API has quirks that a market-data platform has to handle correctly:

- **Namespaces differ by document type** (publication vs. generation/load). The parser
  matches tags with a `{*}` wildcard, so one parser handles every document.
- **Mixed resolutions.** SDAC moved to a **15-minute MTU in 2025**, so PT15M, PT30M and
  PT60M series can appear in the same query. All are parsed to UTC and resampled
  explicitly.
- **Curve type A03 (compressed)** leaves out repeated points, so missing positions carry
  the last value forward. A naive parser silently loses these hours.
- **"No matching data"** comes back as an Acknowledgement document (sometimes with
  HTTP 400). It's treated as an empty result, not an error.
- **One-year request limits and rate limits**: requests are chunked, retried with
  backoff, and cached as Parquet.
- **No leakage.** Forecasts are issued on D-1 before 12:00 CET gate closure. Price lags
  of at least 24 hours are known by then (D-1 prices were published on D-2), and the
  day-ahead load/RES forecasts for D are already public.

## Quick start

```bash
uv sync --all-groups
export ENTSOE_API_KEY=...        # free: register on transparency.entsoe.eu, then email
                                 # transparency@entsoe.eu with subject "Restful API access"
uv run eupm zones                                    # list supported zones
uv run eupm fetch    --zones DE_LU FR NL BE ES PL    # download + data-quality report
uv run eupm atlas    --zones DE_LU FR NL BE ES PL    # + arbitrage atlas and correlation
uv run eupm backtest --zones DE_LU FR ES --test-days 90
uv run streamlit run app/dashboard.py
```

Offline demo (synthetic data, clearly flagged in the dashboard, not for results):
`uv run eupm backtest --synthetic --zones DE_LU FR ES --test-days 14`

Quality checks (also run in GitHub Actions):
`uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest`

## Results

Fill in after running on real data:

**Arbitrage atlas** (1 MW / 2 MWh, 88% RTE, perfect foresight, hourly)

| Zone | Avg price €/MWh | Avg daily spread | €/MW/year | Negative-price hours |
|---|---|---|---|---|
| DE_LU | | | | |
| ES | | | | |
| FR | | | | |

**Day-ahead forecast back-test** (90 days)

| Zone | Model | MAE €/MWh | P10–P90 coverage | Revenue capture |
|---|---|---|---|---|
| DE_LU | LightGBM | | | |
| DE_LU | Seasonal naive | | | |

**Findings:** _e.g. which zones have the widest spreads and why (solar share, coupling),
how much residual load improves on price-only lags, and how negative-price hours have
grown in solar-heavy zones._

## Project layout

```
src/eupm/
  data/entsoe.py     ENTSO-E client, XML parser, zone registry, Parquet cache
  data/synthetic.py  offline test data (not real)
  analytics.py       quality report, arbitrage atlas, correlation, forecasting back-test
  models.py          LightGBM / XGBoost quantile models
  battery.py         LP dispatch (HiGHS via SciPy) + settlement
  config.py          pydantic settings
  cli.py             `eupm zones|fetch|atlas|backtest`
app/dashboard.py     Streamlit dashboard
tests/               parser fixtures in the real API's XML format, analytics and LP tests
```

## Design choices

- **Revenue capture over MAE.** Battery value depends on the *shape* of the day. A flat
  P50 that understates the spread leads the LP (with degradation cost and efficiency
  losses) to decide not to trade, so capture falls to zero even when MAE looks fine.
- **Perfect foresight as the benchmark.** The atlas gives the upper bound per zone, and
  forecast capture % shows how much of it a model actually earns.
- **Quality report first.** Coverage and gap statistics are produced before any analysis,
  so issues in the data aren't mistaken for findings.

## Next steps

- Run the LP on native 15-minute prices (hourly means understate the spread).
- Add cross-border flows, TTF gas and EUA carbon prices as fundamentals.
- Add intraday (continuous/IDA) and balancing/aFRR data for revenue stacking.
- Extend to all Italian zones, the Nordics and CEE, and orchestrate daily refreshes with
  Prefect into PostgreSQL.

## Author

Kirankumar Ashok Patil: [LinkedIn](https://linkedin.com/in/kirankumarashokpatil)
