"""
Example 03: Run end-to-end portfolio backtest comparing Fixed vs Macro Regime allocation.
"""

from pathlib import Path
import pandas as pd
from conjunctural.data.engine import DataEngine
from conjunctural.timeseries.harmonizer import TimeSeriesHarmonizer
from conjunctural.indicators.inflation import InflationEngine
from conjunctural.indicators.interest_rate import InterestRateEngine
from conjunctural.indicators.technical import TechnicalEngine
from conjunctural.backtest.engine import BacktestEngine
from conjunctural.backtest.strategies import (
    FixedWeightStrategy,
    MacroRegimeStrategy,
    EqualWeightStrategy,
)
from conjunctural.backtest.tearsheet import plot_backtest


def main():
    print("=" * 70)
    print("RUNNING MACROECONOMIC REGIME BACKTEST (2022 - 2024)")
    print("=" * 70)

    # 1. Ingest Data
    engine = DataEngine()
    df_raw = engine.fetch_multiple(
        ["SELIC_OVER", "CDI", "IPCA", "USD_BRL", "EMBI_BRASIL"],
        start_date="2022-01-01",
        end_date="2024-06-30",
    )

    # 2. Harmonize & Align
    harmonizer = TimeSeriesHarmonizer()
    daily_df = harmonizer.align_to_daily(df_raw, apply_publication_lags=True, ffill=True)

    # 3. Compute Macro Indicators & Signals
    daily_df = InflationEngine.compute_all(daily_df)
    daily_df = InterestRateEngine.compute_all(daily_df)
    daily_df = TechnicalEngine.compute_all(daily_df)

    # 4. Construct Asset Prices / Total Return Universe
    asset_prices = pd.DataFrame(index=daily_df["date"])
    asset_prices["CDI"] = daily_df["CDI_INDEX"].values
    asset_prices["USD_BRL"] = daily_df["USD_BRL"].values

    # Benchmark: 100% CDI
    benchmark_series = asset_prices["CDI"]
    risk_free_series = daily_df["CDI_DAILY_RETURN"]

    # 5. Initialize Backtest Engine
    bt = BacktestEngine(initial_capital=100_000.0, transaction_cost_bps=5.0)

    # Strategy A: Fixed 70% CDI / 30% USD
    fixed_strat = FixedWeightStrategy({"CDI": 0.70, "USD_BRL": 0.30}, name="Fixed 70/30 CDI-USD")
    res_fixed = bt.run(
        asset_prices=asset_prices,
        strategy=fixed_strat,
        signals=daily_df,
        benchmark=benchmark_series,
        risk_free_rate=risk_free_series,
    )

    # Strategy B: Dynamic Macro Regime Allocation
    macro_strat = MacroRegimeStrategy(cdi_asset="CDI", fx_asset="USD_BRL", name="Dynamic Macro Regime")
    res_macro = bt.run(
        asset_prices=asset_prices,
        strategy=macro_strat,
        signals=daily_df,
        benchmark=benchmark_series,
        risk_free_rate=risk_free_series,
    )

    # 6. Print Tearsheets
    print("\n" + res_fixed.tearsheet())
    print("\n" + res_macro.tearsheet())

    # 7. Generate Plot
    output_img = Path("output/macro_backtest_tearsheet.png")
    output_img.parent.mkdir(parents=True, exist_ok=True)
    plot_backtest(res_macro, title="Macro Regime Backtest: CDI + USD Allocation", save_path=output_img)
    print(f"\nTearsheet plot saved to: {output_img.resolve()}")


if __name__ == "__main__":
    main()
