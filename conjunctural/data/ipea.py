"""
Instituto de Pesquisa Econômica Aplicada (IPEA / Ipeadata) Time-Series Data Provider.
"""

from typing import Optional
import datetime
import requests
import pandas as pd
import polars as pl
from conjunctural.config import REQUEST_TIMEOUT, DEFAULT_USER_AGENT
from conjunctural.data.catalog import INDICATORS, Indicator
from conjunctural.data.cache import CacheManager


class IpeaProvider:
    """Ingests time-series from the Ipeadata OData v4 REST API."""

    BASE_URL = "http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{code}')"

    # Known code aliases
    ALIASES = {
        "40940": "JPM366_EMBI366",
        "EMBI_BRASIL": "JPM366_EMBI366",
        "EMBI": "JPM366_EMBI366",
    }

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
        Fetch a series from Ipeadata by code or indicator name.

        Parameters
        ----------
        code_or_indicator : str | int | Indicator
            Series code (e.g. 'JPM366_EMBI366', 40940) or Indicator.EMBI_BRASIL.
        start_date : str | datetime.date, optional
            Start date.
        end_date : str | datetime.date, optional
            End date.
        use_cache : bool, default True
            Whether to use local parquet cache.
        as_polars : bool, default False
            Whether to return as Polars DataFrame.
        """
        series_code, col_name = self._resolve_code_and_name(code_or_indicator)
        cache_key = f"ipea_{series_code}"

        if use_cache:
            cached = self.cache.get(cache_key, start_date=start_date, end_date=end_date, as_polars=as_polars)
            if cached is not None and len(cached) > 0:
                return cached

        url = self.BASE_URL.format(code=series_code)

        resp = None
        for attempt in range(3):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 200:
                    break
            except (requests.RequestException, TimeoutError):
                if attempt == 2:
                    raise

        if resp is None or resp.status_code != 200:
            status = resp.status_code if resp else "No Response"
            raise ConnectionError(f"Failed to fetch Ipeadata series {series_code}: HTTP {status}")

        payload = resp.json()
        values = payload.get("value", [])

        if not values:
            df = pd.DataFrame(columns=["date", col_name])
        else:
            df = pd.DataFrame(values)
            # Standardize date and values
            df["date"] = pd.to_datetime(df["VALDATA"].str[:10], format="%Y-%m-%d")
            df[col_name] = pd.to_numeric(df["VALVALOR"], errors="coerce")
            df = df[["date", col_name]].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        if use_cache and len(df) > 0:
            self.cache.put(cache_key, df)

        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        if as_polars:
            return pl.from_pandas(df)
        return df

    def _resolve_code_and_name(self, code_or_indicator: str | int | Indicator) -> tuple[str, str]:
        if isinstance(code_or_indicator, Indicator):
            meta = INDICATORS[code_or_indicator.value]
            return meta.code, meta.id

        str_val = str(code_or_indicator).strip().upper()
        if str_val in self.ALIASES:
            return self.ALIASES[str_val], "EMBI_BRASIL"

        if str_val in INDICATORS:
            meta = INDICATORS[str_val]
            return meta.code, meta.id

        return str_val, str_val
