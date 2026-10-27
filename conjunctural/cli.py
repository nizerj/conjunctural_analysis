"""
Command-Line Interface (CLI) for Conjunctural Engine.
"""

import argparse
import sys
import pandas as pd
from conjunctural.data.catalog import DataCatalog, INDICATORS
from conjunctural.data.engine import DataEngine
from conjunctural.timeseries.harmonizer import TimeSeriesHarmonizer
from conjunctural.indicators.inflation import InflationEngine
from conjunctural.indicators.interest_rate import InterestRateEngine
from conjunctural.indicators.technical import TechnicalEngine
from conjunctural.backtest.engine import BacktestEngine
from conjunctural.backtest.strategies import MacroRegimeStrategy, EqualWeightStrategy


def main():
    parser = argparse.ArgumentParser(
        prog="conjunctural",
        description="Conjunctural - Brazilian Macroeconomic Time-Series Backtester & Indicator Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: catalog
    cat_parser = subparsers.add_parser("catalog", help="List all 10 core macroeconomic indicators")

    # Command: fetch
    fetch_parser = subparsers.add_parser("fetch", help="Fetch one or more indicators")
    fetch_parser.add_argument("indicator", type=str, help="Indicator canonical name (e.g. SELIC_OVER, IPCA, USD_BRL, or ALL)")
    fetch_parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    fetch_parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    fetch_parser.add_argument("--rows", type=int, default=10, help="Number of preview rows to display")

    # Command: backtest
    bt_parser = subparsers.add_parser("backtest", help="Run a macroeconomic regime allocation backtest")
    bt_parser.add_argument("--strategy", type=str, default="macro", choices=["macro", "equal"], help="Strategy to test")
    bt_parser.add_argument("--start", type=str, default="2022-01-01", help="Backtest start date")
    bt_parser.add_argument("--end", type=str, default=None, help="Backtest end date")

    args = parser.parse_args()

    if args.command == "catalog" or args.command is None:
        print("\n=== CONJUNCTURAL: CORE BRAZILIAN TIME-SERIES INDICATORS ===")
        df = DataCatalog.to_dataframe()
        print(df.to_string(index=False))
        print("")
        return

    engine = DataEngine()

    if args.command == "fetch":
        ind_name = args.indicator.upper()
        print(f"Fetching indicator '{ind_name}' (start={args.start}, end={args.end})...")
        if ind_name == "ALL":
            df = engine.fetch_all_core(start_date=args.start, end_date=args.end)
        else:
            df = engine.fetch_indicator(ind_name, start_date=args.start, end_date=args.end)

        print(f"\nRetrieved {len(df)} records. Preview (last {args.rows} rows):")
        print(df.tail(args.rows).to_string(index=False))

    elif args.command == "backtest":
        print(f"Running backtest with strategy '{args.strategy}' from {args.start}...")
        # Fetch core data needed
        df_raw = engine.fetch_multiple(
            ["SELIC_OVER", "CDI", "IPCA", "USD_BRL", "EMBI_BRASIL"],
            start_date=args.start,
            end_date=args.end,
        )
        harmonizer = TimeSeriesHarmonizer()
        daily_df = harmonizer.align_to_daily(df_raw)

        # Compute indicators
        daily_df = InflationEngine.compute_all(daily_df)
        daily_df = InterestRateEngine.compute_all(daily_df)
        daily_df = TechnicalEngine.compute_all(daily_df)

        # Create assets dataframe
        prices_df = pd.DataFrame(index=daily_df["date"])
        prices_df["CDI"] = daily_df["CDI_INDEX"].values
        prices_df["USD_BRL"] = daily_df["USD_BRL"].values

        # Select strategy
        if args.strategy == "macro":
            strategy = MacroRegimeStrategy(cdi_asset="CDI", fx_asset="USD_BRL")
        else:
            strategy = EqualWeightStrategy()

        bt = BacktestEngine(initial_capital=100_000.0, transaction_cost_bps=5.0)
        res = bt.run(
            asset_prices=prices_df,
            strategy=strategy,
            signals=daily_df,
            benchmark=prices_df["CDI"],
            risk_free_rate=daily_df["CDI_DAILY_RETURN"],
        )

        print("\n" + res.tearsheet())


if __name__ == "__main__":
    main()
