"""
Banco Central do Brasil (BCB) SGS Time-Series Data Provider.
"""

from typing import Optional, Dict, List
import datetime
import requests
import pandas as pd
import polars as pl
from conjunctural.config import REQUEST_TIMEOUT, DEFAULT_USER_AGENT
from conjunctural.data.catalog import INDICATORS, Indicator, IndicatorMeta
from conjunctural.data.cache import CacheManager


class BCBProvider:
    """Ingests time-series from the BCB SGS (Sistema Gerenciador de Séries Temporais) API."""

    BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"

    def __init__(self, cache_manager: Optional[CacheManager] = None, timeout: int = REQUEST_TIMEOUT):
        self.cache = cache_manager or CacheManager()
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json",
        })

    def fetch_series(
        self,
        code_or_indicator: str | int | Indicator,
        start_date: Optional[str | datetime.date] = None,
        end_date: Optional[str | datetime.date] = None,
        use_cache: bool = True,
        as_polars: bool = False,
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Fetch a single series from BCB SGS by code or indicator name.

        Parameters
        ----------
        code_or_indicator : str | int | Indicator
            SGS numeric code (e.g. 1178) or canonical indicator name (e.g. 'SELIC_OVER' or Indicator.SELIC_OVER).
        start_date : str | datetime.date, optional
            Start date formatted as 'YYYY-MM-DD' or 'DD/MM/YYYY'.
        end_date : str | datetime.date, optional
            End date formatted as 'YYYY-MM-DD' or 'DD/MM/YYYY'.
        use_cache : bool, default True
            Whether to use local parquet cache.
        as_polars : bool, default False
            Whether to return result as Polars DataFrame.
        """
        series_code, col_name = self._resolve_code_and_name(code_or_indicator)
        cache_key = f"bcb_sgs_{series_code}"

        if use_cache:
            cached = self.cache.get(cache_key, start_date=start_date, end_date=end_date, as_polars=as_polars)
            if cached is not None and len(cached) > 0:
                return cached

        # Build URL and query parameters
        params = {"formato": "json"}
        if start_date:
            dt_start = pd.to_datetime(start_date)
            params["dataInicial"] = dt_start.strftime("%d/%m/%Y")
        if end_date:
            dt_end = pd.to_datetime(end_date)
            params["dataFinal"] = dt_end.strftime("%d/%m/%Y")

        url = self.BASE_URL.format(code=series_code)
        
        # Robust fetch with retry
        resp = None
        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    break
            except (requests.RequestException, TimeoutError):
                if attempt == 2:
                    raise

        if resp is None or resp.status_code != 200:
            status = resp.status_code if resp else "No Response"
            raise ConnectionError(f"Failed to fetch BCB SGS series {series_code}: HTTP {status}")

        data = resp.json()
        if not data:
            df = pd.DataFrame(columns=["date", col_name])
        else:
            df = pd.DataFrame(data)
            # BCB returns columns: 'data' (DD/MM/YYYY) and 'valor' (string)
            df["date"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
            def _parse_val(v):
                if v is None or pd.isna(v):
                    return None
                s = str(v).strip()
                if not s:
                    return None
                if "," in s:
                    s = s.replace(".", "").replace(",", ".")
                try:
                    return float(s)
                except ValueError:
                    return None

            df[col_name] = df["valor"].apply(_parse_val)
            df = df[["date", col_name]].sort_values("date").reset_index(drop=True)

        if use_cache and len(df) > 0:
            self.cache.put(cache_key, df)

        # Apply date filters if not fully handled by API
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        if as_polars:
            return pl.from_pandas(df)
        return df

    def fetch_multiple(
        self,
        indicators: List[str | int | Indicator],
        start_date: Optional[str | datetime.date] = None,
        end_date: Optional[str | datetime.date] = None,
        use_cache: bool = True,
        as_polars: bool = False,
    ) -> pd.DataFrame | pl.DataFrame:
        """Fetch multiple SGS series and merge on date (outer join)."""
        merged_df: Optional[pd.DataFrame] = None
        for item in indicators:
            df_i = self.fetch_series(
                item,
                start_date=start_date,
                end_date=end_date,
                use_cache=use_cache,
                as_polars=False,
            )
            if merged_df is None:
                merged_df = df_i
            else:
                merged_df = pd.merge(merged_df, df_i, on="date", how="outer")

        if merged_df is None:
            merged_df = pd.DataFrame(columns=["date"])
        else:
            merged_df = merged_df.sort_values("date").reset_index(drop=True)

        if as_polars:
            return pl.from_pandas(merged_df)
        return merged_df

    def _resolve_code_and_name(self, code_or_indicator: str | int | Indicator) -> tuple[str, str]:
        if isinstance(code_or_indicator, Indicator):
            meta = INDICATORS[code_or_indicator.value]
            return meta.code, meta.id

        str_val = str(code_or_indicator).upper()
        if str_val in INDICATORS:
            meta = INDICATORS[str_val]
            return meta.code, meta.id

        # Direct numeric SGS code
        if str_val.isdigit():
            # Check if it matches an indicator code in catalog
            for meta in INDICATORS.values():
                if meta.code == str_val:
                    return meta.code, meta.id
            return str_val, f"SGS_{str_val}"

        raise ValueError(f"Could not resolve SGS code or indicator name from '{code_or_indicator}'")
