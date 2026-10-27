"""
Mixed-Frequency Time-Series Harmonizer and Lookahead Bias Prevention.
"""

from typing import Optional, List, Dict
import pandas as pd
import polars as pl
from conjunctural.timeseries.calendar import BrazilCalendar
from conjunctural.timeseries.returns import annual_to_daily_rate


class TimeSeriesHarmonizer:
    """
    Harmonizes daily and monthly macroeconomic/financial series into aligned,
    leakage-free datasets for indicators and backtesting.
    """

    # Publication lags (in business days) to prevent lookahead bias
    DEFAULT_LAGS = {
        "IPCA": 10,       # Published around 10th-12th of following month
        "IPCA_15": 15,    # Published mid-month
        "IGP_M": 1,       # Published around end of month or 1st day of next month
        "IBC_BR": 35,     # Published ~5 weeks after reference month
        "CURRENT_ACCOUNT": 25,  # Published late in subsequent month
    }

    def __init__(self, publication_lags: Optional[Dict[str, int]] = None):
        self.lags = publication_lags or self.DEFAULT_LAGS

    def align_to_daily(
        self,
        df: pd.DataFrame | pl.DataFrame,
        apply_publication_lags: bool = True,
        business_days_only: bool = True,
        ffill: bool = True,
        as_polars: bool = False,
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Harmonize mixed daily and monthly series onto a continuous daily business day index.
        """
        if isinstance(df, pl.DataFrame):
            df_pd = df.to_pandas()
        else:
            df_pd = df.copy()

        if "date" not in df_pd.columns:
            raise KeyError("DataFrame must contain a 'date' column.")

        df_pd["date"] = pd.to_datetime(df_pd["date"])
        df_pd = df_pd.sort_values("date").drop_duplicates(subset=["date"])

        min_date = df_pd["date"].min()
        max_date = df_pd["date"].max()

        if business_days_only:
            grid = BrazilCalendar.get_business_day_range(min_date, max_date)
        else:
            grid = pd.date_range(min_date, max_date, freq="D")

        base_df = pd.DataFrame({"date": grid})
        merged = pd.merge(base_df, df_pd, on="date", how="left")

        # Apply publication lags to monthly series before forward-filling
        if apply_publication_lags:
            for col, lag in self.lags.items():
                if col in merged.columns:
                    # Shift the sparse observation by 'lag' business day rows
                    merged[col] = merged[col].shift(lag)

        if ffill:
            merged = merged.ffill()

        if as_polars:
            return pl.from_pandas(merged)
        return merged

    def align_to_monthly(
        self,
        df: pd.DataFrame | pl.DataFrame,
        as_polars: bool = False,
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Harmonize mixed series to monthly frequency (month-end observations).
        Daily rates/prices are sampled at month-end; monthly inflation rates are preserved.
        """
        if isinstance(df, pl.DataFrame):
            df_pd = df.to_pandas()
        else:
            df_pd = df.copy()

        df_pd["date"] = pd.to_datetime(df_pd["date"])
        df_pd = df_pd.sort_values("date").set_index("date")

        # Resample month-end taking last available observation
        monthly_df = df_pd.resample("ME").last().reset_index()

        if as_polars:
            return pl.from_pandas(monthly_df)
        return monthly_df
