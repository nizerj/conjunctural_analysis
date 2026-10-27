"""
Local disk caching layer for time-series data using Parquet files.
"""

from pathlib import Path
from typing import Optional
import datetime
import pandas as pd
import polars as pl
from conjunctural.config import get_cache_dir


class CacheManager:
    """Manages persistent Parquet caching of time-series series."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else get_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        clean_key = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in key)
        return self.cache_dir / f"{clean_key}.parquet"

    def has(self, key: str) -> bool:
        return self._get_path(key).is_file()

    def get(
        self,
        key: str,
        start_date: Optional[str | datetime.date] = None,
        end_date: Optional[str | datetime.date] = None,
        as_polars: bool = False,
    ) -> Optional[pd.DataFrame | pl.DataFrame]:
        """Retrieve cached series if available and covers the requested date span."""
        path = self._get_path(key)
        if not path.is_file():
            return None

        try:
            df = pd.read_parquet(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")
            elif not isinstance(df.index, pd.DatetimeIndex):
                # Check if index is date
                try:
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                except Exception:
                    pass

            if start_date or end_date:
                # If cached min date is after requested start_date or max date is before end_date,
                # the caller might still want whatever is available or refetch.
                # We filter by the dates:
                if "date" in df.columns:
                    if start_date:
                        df = df[df["date"] >= pd.to_datetime(start_date)]
                    if end_date:
                        df = df[df["date"] <= pd.to_datetime(end_date)]
                elif isinstance(df.index, pd.DatetimeIndex):
                    if start_date:
                        df = df[df.index >= pd.to_datetime(start_date)]
                    if end_date:
                        df = df[df.index <= pd.to_datetime(end_date)]

            if as_polars:
                return pl.from_pandas(df.reset_index() if isinstance(df.index, pd.DatetimeIndex) else df)
            return df
        except Exception:
            return None

    def put(self, key: str, data: pd.DataFrame | pl.DataFrame) -> None:
        """Save series to local parquet cache."""
        path = self._get_path(key)
        if isinstance(data, pl.DataFrame):
            df_pandas = data.to_pandas()
        else:
            df_pandas = data.copy()

        # If data is indexed by DatetimeIndex, move to column or keep
        if isinstance(df_pandas.index, pd.DatetimeIndex) and "date" not in df_pandas.columns:
            df_pandas = df_pandas.reset_index().rename(columns={"index": "date"})

        df_pandas.to_parquet(path, index=False)

    def clear(self, key: Optional[str] = None) -> None:
        """Clear cache for a specific key or all keys."""
        if key:
            path = self._get_path(key)
            if path.is_file():
                path.unlink()
        else:
            for p in self.cache_dir.glob("*.parquet"):
                p.unlink()
