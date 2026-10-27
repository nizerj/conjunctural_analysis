"""
Market risk and rolling technical indicators (USD/BRL PTAX, EMBI+ Brasil).
"""

import numpy as np
import pandas as pd
import polars as pl
from conjunctural.config import ANNUAL_BUSINESS_DAYS


class TechnicalEngine:
    """Computes rolling volatility, momentum, drawdowns, and country risk metrics."""

    @classmethod
    def compute_all(cls, df: pd.DataFrame | pl.DataFrame, as_polars: bool = False) -> pd.DataFrame | pl.DataFrame:
        if isinstance(df, pl.DataFrame):
            df_pd = df.to_pandas()
        else:
            df_pd = df.copy()

        # USD/BRL PTAX Indicators
        if "USD_BRL" in df_pd.columns:
            # Daily return
            df_pd["USD_BRL_RETURN_1D"] = df_pd["USD_BRL"].pct_change()
            # 21-day (~1 month) return
            df_pd["USD_BRL_RETURN_21D"] = df_pd["USD_BRL"].pct_change(21)
            # 63-day (~3 month) return
            df_pd["USD_BRL_RETURN_63D"] = df_pd["USD_BRL"].pct_change(63)
            # 252-day (~1 year) return
            df_pd["USD_BRL_RETURN_252D"] = df_pd["USD_BRL"].pct_change(252)

            # Annualized Rolling Volatility
            df_pd["USD_BRL_VOL_21D"] = df_pd["USD_BRL_RETURN_1D"].rolling(21).std() * np.sqrt(ANNUAL_BUSINESS_DAYS)
            df_pd["USD_BRL_VOL_63D"] = df_pd["USD_BRL_RETURN_1D"].rolling(63).std() * np.sqrt(ANNUAL_BUSINESS_DAYS)

        # EMBI+ Brasil Risk Indicators
        if "EMBI_BRASIL" in df_pd.columns:
            # Rolling changes in bps
            df_pd["EMBI_DELTA_21D"] = df_pd["EMBI_BRASIL"].diff(21)
            df_pd["EMBI_DELTA_63D"] = df_pd["EMBI_BRASIL"].diff(63)

            # Rolling 252-day Z-Score
            rolling_mean = df_pd["EMBI_BRASIL"].rolling(252, min_periods=63).mean()
            rolling_std = df_pd["EMBI_BRASIL"].rolling(252, min_periods=63).std()
            df_pd["EMBI_ZSCORE"] = (df_pd["EMBI_BRASIL"] - rolling_mean) / (rolling_std + 1e-8)

            # Regime
            def _embi_regime(z):
                if pd.isna(z):
                    return None
                if z > 1.5:
                    return "HIGH_RISK_OFF"
                elif z < -1.0:
                    return "BENIGN_RISK_ON"
                return "NEUTRAL"

            df_pd["EMBI_REGIME"] = df_pd["EMBI_ZSCORE"].apply(_embi_regime)

        if as_polars:
            return pl.from_pandas(df_pd)
        return df_pd
