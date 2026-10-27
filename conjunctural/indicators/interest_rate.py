"""
Interest rate indicators engine (Selic Over, CDI, Real Interest Rates).
"""

import numpy as np
import pandas as pd
import polars as pl
from conjunctural.config import ANNUAL_BUSINESS_DAYS
from conjunctural.timeseries.returns import annual_to_daily_rate, compound_daily_rates, real_rate


class InterestRateEngine:
    """Computes interest rate metrics, real rates, and compounding benchmark series."""

    @staticmethod
    def real_ex_post_rate(nominal_rate_aa: pd.Series, ipca_12m_pct: pd.Series) -> pd.Series:
        """
        Compute real ex-post interest rate (% a.a.) via the Fisher equation:
        r_real = [(1 + nominal / 100) / (1 + ipca_12m / 100) - 1] * 100
        """
        nom = nominal_rate_aa / 100.0
        inf = ipca_12m_pct / 100.0
        return ((1.0 + nom) / (1.0 + inf) - 1.0) * 100.0

    @staticmethod
    def cdi_index(cdi_annual_pct: pd.Series, initial_value: float = 100.0) -> pd.Series:
        """Compute the total return accumulator index of CDI from annualized daily rates."""
        daily_rates = annual_to_daily_rate(cdi_annual_pct)
        return compound_daily_rates(daily_rates, initial_value=initial_value)

    @classmethod
    def compute_all(
        cls,
        df: pd.DataFrame | pl.DataFrame,
        neutral_real_rate: float = 4.5,
        as_polars: bool = False,
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Compute real interest rates, CDI index, and monetary stance spread.
        """
        if isinstance(df, pl.DataFrame):
            df_pd = df.to_pandas()
        else:
            df_pd = df.copy()

        # Real Selic
        if "SELIC_OVER" in df_pd.columns and "IPCA_12M" in df_pd.columns:
            df_pd["REAL_SELIC_EX_POST"] = cls.real_ex_post_rate(df_pd["SELIC_OVER"], df_pd["IPCA_12M"])
            df_pd["MONETARY_STANCE_SPREAD"] = df_pd["REAL_SELIC_EX_POST"] - neutral_real_rate

        # Real CDI
        if "CDI" in df_pd.columns and "IPCA_12M" in df_pd.columns:
            df_pd["REAL_CDI_EX_POST"] = cls.real_ex_post_rate(df_pd["CDI"], df_pd["IPCA_12M"])

        # CDI Index and daily returns
        if "CDI" in df_pd.columns:
            df_pd["CDI_DAILY_RETURN"] = annual_to_daily_rate(df_pd["CDI"])
            df_pd["CDI_INDEX"] = cls.cdi_index(df_pd["CDI"])

        # Selic - CDI spread (bps)
        if "SELIC_OVER" in df_pd.columns and "CDI" in df_pd.columns:
            df_pd["SELIC_CDI_SPREAD_BPS"] = (df_pd["SELIC_OVER"] - df_pd["CDI"]) * 100.0

        if as_polars:
            return pl.from_pandas(df_pd)
        return df_pd
