"""Command line: ``eupm fetch`` · ``eupm atlas`` · ``eupm backtest``."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import polars as pl

from eupm.analytics import arbitrage_atlas, price_correlation, quality_report, run_eu_backtest
from eupm.config import Settings
from eupm.data.entsoe import ZONES, EntsoeConfig, load_or_fetch_zone
from eupm.data.synthetic import make_synthetic_eu

DEFAULT_ZONES = ["DE_LU", "FR", "NL", "BE", "ES", "PL"]


def _frames(args: argparse.Namespace) -> dict[str, pl.DataFrame]:
    if args.synthetic:
        return make_synthetic_eu(args.zones, days=args.days)
    cfg = EntsoeConfig(
        zones=args.zones, start=date.fromisoformat(args.start), end=date.fromisoformat(args.end)
    )
    return {z: load_or_fetch_zone(z, cfg) for z in cfg.zones}


def main() -> None:
    ap = argparse.ArgumentParser(prog="eupm", description="European power market data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("zones", help="list supported bidding zones")
    for name in ("fetch", "atlas", "backtest"):
        p = sub.add_parser(name)
        p.add_argument("--zones", nargs="+", default=DEFAULT_ZONES)
        p.add_argument("--start", default="2024-01-01")
        p.add_argument("--end", default="2025-12-31")
        p.add_argument("--synthetic", action="store_true", help="offline demo data (not real)")
        p.add_argument("--days", type=int, default=400, help="synthetic days")
        p.add_argument("--out", default="reports")
        if name == "backtest":
            p.add_argument("--test-days", type=int, default=90)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "zones":
        for z, eic in ZONES.items():
            print(f"{z:8s} {eic}")
        return

    frames = _frames(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = "synthetic" if args.synthetic else f"entsoe {args.start} {args.end}"
    (out / "SOURCE.txt").write_text(source + "\n")

    with pl.Config(tbl_cols=-1, tbl_width_chars=180, float_precision=1):
        qr = quality_report(frames)
        qr.write_csv(out / "quality.csv")
        print(qr)
        if args.cmd == "fetch":
            return
        summary, daily = arbitrage_atlas(frames)
        summary.write_csv(out / "atlas.csv")
        daily.write_parquet(out / "atlas_daily.parquet")
        price_correlation(frames).write_csv(out / "correlation.csv")
        print(summary)
        if args.cmd == "atlas":
            return
        s = Settings()
        s.backtest.test_days = args.test_days
        for z in args.zones:
            res = run_eu_backtest(frames[z], s)
            d = out / f"backtest_{z}"
            d.mkdir(exist_ok=True)
            res.predictions.write_parquet(d / "predictions.parquet")
            res.daily.write_parquet(d / "daily_revenue.parquet")
            res.summary.write_csv(d / "summary.csv")
            print(f"\n=== {z} ===\n{res.summary}")


if __name__ == "__main__":
    main()
