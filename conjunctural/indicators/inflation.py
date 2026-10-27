"""
Inflation indicators engine (IPCA, IPCA-15, IGP-M).
"""

from typing import Union
import numpy as np
import pandas as pd
import polars as pl


class InflationEngine:
    """Computes rolling inflation indicators, compounding, and price indices."""

    @staticmethod
    def rolling_12m_cumulative(monthly_rates: pd.Series) -> pd.Series:
        """
        Calculate 12-month accumulated inflation percentage:
        Pi_{12M} = [ prod_{i=0}^{11} (1 + rate_{t-i}/100) - 1 ] * 100
        """
        decimal_factors = 1.0 + (monthly_rates / 100.0)
        # 12-period rolling product
        rolling_prod = decimal_factors.rolling(window=12, min_periods=12).apply(np.prod, raw=True)
        return (rolling_prod - 1.0) * 100.0

    @staticmethod
    def annualized_3m_rate(monthly_rates: pd.Series) -> pd.Series:
        """
        Calculate 3-month annualized inflation rate (momentum indicator used by central banks):
        rate_{3m, saar} = [ prod_{i=0}^2 (1 + rate_{t-i}/100) ]^4 - 1
        """
        decimal_factors = 1.0 + (monthly_rates / 100.0)
        rolling_3m_prod = decimal_factors.rolling(window=3, min_periods=3).apply(np.prod, raw=True)
        return (rolling_3m_prod ** 4 - 1.0) * 100.0

    @staticmethod
    def price_index(monthly_rates: pd.Series, base_value: float = 100.0) -> pd.Series:
        """Build cumulative price index from monthly percentage inflation rates."""
        factors = 1.0 + (monthly_rates.fillna(0.0) / 100.0)
        return base_value * factors.cumprod()

    @classmethod
    def compute_all(cls, df: pd.DataFrame | pl.DataFrame, as_polars: bool = False) -> pd.DataFrame | pl.DataFrame:
        """
        Compute all standard inflation indicators for IPCA, IPCA_15, and IGP_M if present.
        """
        if isinstance(df, pl.DataFrame):
            df_pd = df.to_pandas()
        else:
            df_pd = df.copy()

        for col in ["IPCA", "IPCA_15", "IGP_M"]:
            if col in df_pd.columns:
                clean_s = df_pd[col].dropna()
                if len(clean_s) > 0:
                    df_pd[f"{col}_12M"] = cls.rolling_12m_cumulative(df_pd[col])
                    df_pd[f"{col}_3M_ANN"] = cls.annualized_3m_rate(df_pd[col])
                    df_pd[f"{col}_INDEX"] = cls.price_index(df_pd[col])

        # Wholesale vs consumer inflation spread
        if "IGP_M_12M" in df_pd.columns and "IPCA_12M" in df_pd.columns:
            df_pd["IGPM_IPCA_SPREAD"] = df_pd["IGP_M_12M"] - df_pd["IPCA_12M"]

        if as_polars:
            return pl.from_pandas(df_pd)
        return df_pd
