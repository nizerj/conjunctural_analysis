"""
Unified Macro Data Engine for Brazilian Time-Series Ingestion.
"""

from typing import Optional, List, Dict, Union
import datetime
import pandas as pd
import polars as pl
from conjunctural.data.catalog import INDICATORS, Indicator, IndicatorMeta, ProviderType, DataCatalog
from conjunctural.data.cache import CacheManager
from conjunctural.data.bcb import BCBProvider
from conjunctural.data.ipea import IpeaProvider
from conjunctural.data.b3 import B3DIProvider


class DataEngine:
    """Unified client to fetch and combine macroeconomic and financial indicators."""

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache = cache_manager or CacheManager()
        self.bcb = BCBProvider(cache_manager=self.cache)
        self.ipea = IpeaProvider(cache_manager=self.cache)
        self.b3 = B3DIProvider(cache_manager=self.cache)

    def fetch_indicator(
        self,
        indicator: str | Indicator,
        start_date: Optional[str | datetime.date] = None,
        end_date: Optional[str | datetime.date] = None,
        use_cache: bool = True,
        as_polars: bool = False,
    ) -> pd.DataFrame | pl.DataFrame:
        """Fetch a single indicator by canonical name or enum."""
        meta = DataCatalog.get(indicator)
        if meta.provider == ProviderType.BCB:
            return self.bcb.fetch_series(meta.id, start_date=start_date, end_date=end_date, use_cache=use_cache, as_polars=as_polars)
        elif meta.provider == ProviderType.IPEA:
            return self.ipea.fetch_series(meta.id, start_date=start_date, end_date=end_date, use_cache=use_cache, as_polars=as_polars)
        elif meta.provider == ProviderType.B3:
            raise ValueError(
                "DI_FUTURES is a term structure curve. Use `fetch_di_curve()` for single date or `fetch_di_history()` for vertex time series."
            )
        else:
            raise NotImplementedError(f"Provider {meta.provider} not supported.")

    def fetch_all_core(
        self,
        start_date: Optional[str | datetime.date] = None,
        end_date: Optional[str | datetime.date] = None,
        use_cache: bool = True,
        as_polars: bool = False,
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Fetch all 9 scalar indicators (Selic, CDI, IPCA, IPCA-15, IGP-M, IBC-Br, USD/BRL, Current Account, EMBI+).
        """
        core_keys = [
            "SELIC_OVER",
            "CDI",
            "IPCA",
            "IPCA_15",
            "IGP_M",
            "IBC_BR",
            "USD_BRL",
            "CURRENT_ACCOUNT",
            "EMBI_BRASIL",
        ]
        return self.fetch_multiple(core_keys, start_date=start_date, end_date=end_date, use_cache=use_cache, as_polars=as_polars)

    def fetch_multiple(
        self,
        indicators: List[str | Indicator],
        start_date: Optional[str | datetime.date] = None,
        end_date: Optional[str | datetime.date] = None,
        use_cache: bool = True,
        as_polars: bool = False,
    ) -> pd.DataFrame | pl.DataFrame:
        """Fetch multiple indicators and merge into a single DataFrame indexed by date."""
        dfs: List[pd.DataFrame] = []

        for item in indicators:
            key = item.value if isinstance(item, Indicator) else str(item).upper()
            if key == "DI_FUTURES":
                continue  # Curve handled separately
            meta = DataCatalog.get(key)
            if meta.provider == ProviderType.BCB:
                df_i = self.bcb.fetch_series(meta.id, start_date=start_date, end_date=end_date, use_cache=use_cache, as_polars=False)
            elif meta.provider == ProviderType.IPEA:
                df_i = self.ipea.fetch_series(meta.id, start_date=start_date, end_date=end_date, use_cache=use_cache, as_polars=False)
            else:
                continue

            if df_i is not None and len(df_i) > 0:
                dfs.append(df_i)

        if not dfs:
            result = pd.DataFrame(columns=["date"])
        else:
            result = dfs[0]
            for df_next in dfs[1:]:
                result = pd.merge(result, df_next, on="date", how="outer")
            result = result.sort_values("date").reset_index(drop=True)

        if as_polars:
            return pl.from_pandas(result)
        return result
