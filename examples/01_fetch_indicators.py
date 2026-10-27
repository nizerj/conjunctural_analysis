"""
Example 01: Discover and fetch Brazilian macroeconomic indicators.
"""

from conjunctural.data.catalog import DataCatalog
from conjunctural.data.engine import DataEngine


def main():
    print("=" * 70)
    print("CONJUNCTURAL: BRAZILIAN TIME-SERIES INDICATOR CATALOG")
    print("=" * 70)

    # 1. Print all 10 indicators
    df_catalog = DataCatalog.to_dataframe()
    print(df_catalog.to_string(index=False))

    print("\n" + "=" * 70)
    print("FETCHING CORE INDICATORS")
    print("=" * 70)

    engine = DataEngine()

    # 2. Fetch Selic Over (SGS 1178)
    print("\nFetching SELIC_OVER...")
    df_selic = engine.fetch_indicator("SELIC_OVER", start_date="2024-01-01", end_date="2024-06-30")
    print(df_selic.tail(5))

    # 3. Fetch IPCA (SGS 433) as Polars DataFrame
    print("\nFetching IPCA as Polars DataFrame...")
    df_ipca_pl = engine.fetch_indicator("IPCA", start_date="2023-01-01", as_polars=True)
    print(df_ipca_pl.tail(5))

    # 4. Fetch USD/BRL PTAX (SGS 1)
    print("\nFetching USD/BRL (PTAX)...")
    df_usd = engine.fetch_indicator("USD_BRL", start_date="2024-01-01", end_date="2024-06-30")
    print(df_usd.tail(5))

    # 5. Fetch EMBI+ Brasil (IPEA JPM366_EMBI366 / 40940)
    print("\nFetching EMBI+ Brasil...")
    df_embi = engine.fetch_indicator("EMBI_BRASIL", start_date="2024-01-01", end_date="2024-06-30")
    print(df_embi.tail(5))


if __name__ == "__main__":
    main()
