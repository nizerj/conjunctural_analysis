"""
Example 02: Ingest, harmonize, and compute macroeconomic & financial rolling indicators.
"""

from conjunctural.data.engine import DataEngine
from conjunctural.timeseries.harmonizer import TimeSeriesHarmonizer
from conjunctural.indicators.inflation import InflationEngine
from conjunctural.indicators.interest_rate import InterestRateEngine
from conjunctural.indicators.activity import ActivityEngine
from conjunctural.indicators.technical import TechnicalEngine


def main():
    print("=" * 70)
    print("INGESTING AND HARMONIZING TIME-SERIES")
    print("=" * 70)

    engine = DataEngine()
    indicators_to_fetch = [
        "SELIC_OVER",
        "CDI",
        "IPCA",
        "IGP_M",
        "IBC_BR",
        "USD_BRL",
        "CURRENT_ACCOUNT",
        "EMBI_BRASIL",
    ]

    # Fetch 2022 to mid 2024
    df_raw = engine.fetch_multiple(indicators_to_fetch, start_date="2022-01-01", end_date="2024-06-30")
    print(f"Merged raw dataset rows: {len(df_raw)}")

    # Harmonize to daily business days applying publication lags to prevent lookahead bias
    harmonizer = TimeSeriesHarmonizer()
    df_daily = harmonizer.align_to_daily(df_raw, apply_publication_lags=True, ffill=True)
    print(f"Aligned daily business days: {len(df_daily)}")

    print("\n" + "=" * 70)
    print("COMPUTING MACROECONOMIC INDICATORS")
    print("=" * 70)

    # 1. Inflation Indicators
    df_daily = InflationEngine.compute_all(df_daily)
    # 2. Interest Rate & Real Rate Indicators
    df_daily = InterestRateEngine.compute_all(df_daily, neutral_real_rate=4.5)
    # 3. Activity Indicators
    df_daily = ActivityEngine.compute_all(df_daily)
    # 4. Market Risk & Technical Indicators
    df_daily = TechnicalEngine.compute_all(df_daily)

    # Display indicator snapshot
    cols_to_show = [
        "date",
        "SELIC_OVER",
        "IPCA_12M",
        "REAL_SELIC_EX_POST",
        "MONETARY_STANCE_SPREAD",
        "USD_BRL",
        "USD_BRL_VOL_21D",
        "EMBI_BRASIL",
        "EMBI_REGIME",
    ]
    available_cols = [c for c in cols_to_show if c in df_daily.columns]

    print("\nRecent Indicator Snapshot:")
    print(df_daily[available_cols].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
