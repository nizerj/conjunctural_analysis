"""
Economic activity and external sector indicators (IBC-Br, Current Account).
"""

import numpy as np
import pandas as pd
import polars as pl


class ActivityEngine:
    """Computes economic activity momentum and external balance indicators."""

    @classmethod
    def compute_all(cls, df: pd.DataFrame | pl.DataFrame, as_polars: bool = False) -> pd.DataFrame | pl.DataFrame:
        if isinstance(df, pl.DataFrame):
            df_pd = df.to_pandas()
        else:
            df_pd = df.copy()

        # IBC-Br Activity momentum
        if "IBC_BR" in df_pd.columns:
            # 3M growth
            df_pd["IBC_BR_3M_GROWTH"] = df_pd["IBC_BR"].pct_change(3) * 100.0
            # 12M YoY growth
            df_pd["IBC_BR_12M_YOY"] = df_pd["IBC_BR"].pct_change(12) * 100.0
            # 24-month rolling Z-Score of growth
            rolling_mean = df_pd["IBC_BR_12M_YOY"].rolling(24, min_periods=12).mean()
            rolling_std = df_pd["IBC_BR_12M_YOY"].rolling(24, min_periods=12).std()
            df_pd["IBC_BR_CYCLE_ZSCORE"] = (df_pd["IBC_BR_12M_YOY"] - rolling_mean) / (rolling_std + 1e-8)

            def _classify_activity(z):
                if pd.isna(z):
                    return None
                if z > 0.5:
                    return "EXPANSION"
                elif z < -0.5:
                    return "CONTRACTION"
                return "NEUTRAL"

            df_pd["ACTIVITY_REGIME"] = df_pd["IBC_BR_CYCLE_ZSCORE"].apply(_classify_activity)

        # Current Account 12M Trailing Sum
        if "CURRENT_ACCOUNT" in df_pd.columns:
            df_pd["CURRENT_ACCOUNT_12M_SUM"] = df_pd["CURRENT_ACCOUNT"].rolling(12, min_periods=12).sum()
            df_pd["CURRENT_ACCOUNT_12M_DELTA"] = df_pd["CURRENT_ACCOUNT_12M_SUM"].diff(12)

        if as_polars:
            return pl.from_pandas(df_pd)
        return df_pd
