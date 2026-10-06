"""European markets: data quality, battery arbitrage atlas and per-zone forecasts.

Generate the data first: ``uv run eupm backtest --zones DE_LU FR NL BE ES PL``
"""

from __future__ import annotations

from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

EU = Path(__file__).resolve().parents[1] / "reports"

st.set_page_config(page_title="EU Power Markets", layout="wide")
st.title("European day-ahead markets")
st.caption("ENTSO-E Transparency Platform · hourly · battery: 1 MW / 2 MWh, 88% RTE")

if not (EU / "atlas.csv").exists():
    st.warning("No EU results yet. Run `uv run eupm atlas` or `uv run eupm backtest`.")
    st.stop()

src = (EU / "SOURCE.txt").read_text().strip() if (EU / "SOURCE.txt").exists() else ""
if src.startswith("synthetic"):
    st.error("These results come from SYNTHETIC demo data, not real ENTSO-E data.")
else:
    st.info(f"Data: {src}")

atlas = pl.read_csv(EU / "atlas.csv")
daily = pl.read_parquet(EU / "atlas_daily.parquet")
quality = pl.read_csv(EU / "quality.csv")

# 1. Arbitrage atlas
st.subheader("Where is battery arbitrage worth most?")
st.caption("Perfect-foresight day-ahead value: the upper bound a trader works towards.")
c1, c2 = st.columns([3, 2])
bar = px.bar(
    atlas.to_pandas(),
    x="zone",
    y="value_per_mw_year",
    labels={"value_per_mw_year": "€ / MW / year", "zone": ""},
    color_discrete_sequence=["#1F4E79"],
)
bar.update_layout(height=360, margin={"t": 10})
c1.plotly_chart(bar, use_container_width=True)
c2.dataframe(
    atlas.select("zone", "avg_price", "avg_daily_spread", "value_per_mw_year", "negative_hours_pct")
    .to_pandas()
    .style.format(precision=1),
    use_container_width=True,
    hide_index=True,
)

# 2. Spread through time
st.subheader("Daily price spread by zone")
zones = atlas["zone"].to_list()
pick = st.multiselect("Zones", zones, default=zones[:4])
roll = (
    daily.filter(pl.col("zone").is_in(pick))
    .sort("day")
    .with_columns(pl.col("spread").rolling_mean(14).over("zone").alias("spread_14d"))
)
line = px.line(
    roll.to_pandas(),
    x="day",
    y="spread_14d",
    color="zone",
    labels={"spread_14d": "€/MWh (14-day mean of daily max − min)", "day": ""},
)
line.update_layout(height=360, margin={"t": 10})
st.plotly_chart(line, use_container_width=True)

# 3. Market coupling
if (EU / "correlation.csv").exists():
    st.subheader("Price correlation (market coupling)")
    corr = pl.read_csv(EU / "correlation.csv")
    z = corr.drop("zone").to_numpy()
    heat = go.Figure(
        go.Heatmap(
            z=z,
            x=corr.columns[1:],
            y=corr["zone"].to_list(),
            zmin=0,
            zmax=1,
            colorscale="Blues",
            text=z.round(2),
            texttemplate="%{text}",
        )
    )
    heat.update_layout(height=420, margin={"t": 10})
    st.plotly_chart(heat, use_container_width=True)

# 4. Forecast back-tests per zone
bts = sorted(p.name.removeprefix("backtest_") for p in EU.glob("backtest_*") if p.is_dir())
if bts:
    st.subheader("Day-ahead forecast back-test")
    zone = st.selectbox("Zone", bts)
    summ = pl.read_csv(EU / f"backtest_{zone}" / "summary.csv")
    st.dataframe(
        summ.to_pandas().style.format(precision=2), use_container_width=True, hide_index=True
    )
    pred = pl.read_parquet(EU / f"backtest_{zone}" / "predictions.parquet")
    days = sorted(pred["start_time"].dt.date().unique().to_list())
    day = st.select_slider("Day", options=days, value=days[-1])
    d = pred.filter((pl.col("model") == summ["model"][0]) & (pl.col("start_time").dt.date() == day))
    fig = go.Figure(
        [
            go.Scatter(x=d["start_time"], y=d["p90"], line={"width": 0}, showlegend=False),
            go.Scatter(
                x=d["start_time"],
                y=d["p10"],
                fill="tonexty",
                line={"width": 0},
                name="P10–P90",
                fillcolor="rgba(31,78,121,0.2)",
            ),
            go.Scatter(
                x=d["start_time"],
                y=d["p50"],
                name=f"{summ['model'][0]} P50",
                line={"color": "#1F4E79"},
            ),
            go.Scatter(x=d["start_time"], y=d["actual"], name="Actual", line={"color": "#C0504D"}),
        ]
    )
    fig.update_layout(yaxis_title="€/MWh", height=360, margin={"t": 10})
    st.plotly_chart(fig, use_container_width=True)

# 5. Data quality
st.subheader("Data quality")
st.dataframe(
    quality.to_pandas().style.format(precision=1), use_container_width=True, hide_index=True
)
