"""
Yield Curve indicators engine (Term structure slopes, curvature, and regimes).
"""

from typing import Optional, Dict
import pandas as pd
import polars as pl


class YieldCurveEngine:
    """Computes fixed-income term structure metrics and curve regimes."""

    @classmethod
    def compute_slopes_and_spreads(
        cls,
        df: pd.DataFrame | pl.DataFrame,
        as_polars: bool = False,
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Compute slope spreads, curvature, and curve shape regimes.
        """
        if isinstance(df, pl.DataFrame):
            df_pd = df.to_pandas()
        else:
            df_pd = df.copy()

        # Slopes
        if "DI_2Y" in df_pd.columns and "DI_1M" in df_pd.columns:
            df_pd["CURVE_SLOPE_2Y_1M"] = df_pd["DI_2Y"] - df_pd["DI_1M"]

        if "DI_5Y" in df_pd.columns and "DI_1Y" in df_pd.columns:
            df_pd["CURVE_SLOPE_5Y_1Y"] = df_pd["DI_5Y"] - df_pd["DI_1Y"]

        if "DI_3Y" in df_pd.columns and "DI_1Y" in df_pd.columns:
            df_pd["CURVE_SLOPE_3Y_1Y"] = df_pd["DI_3Y"] - df_pd["DI_1Y"]

        # Butterfly curvature: 2 * Belly - (Short + Long)
        if all(c in df_pd.columns for c in ["DI_1Y", "DI_2Y", "DI_5Y"]):
            df_pd["CURVE_BUTTERFLY_1_2_5"] = 2.0 * df_pd["DI_2Y"] - (df_pd["DI_1Y"] + df_pd["DI_5Y"])

        # Curve Regime classification
        if "CURVE_SLOPE_2Y_1M" in df_pd.columns:
            def _classify_regime(val):
                if pd.isna(val):
                    return None
                if val < -0.0025:  # Inverted by more than 25 bps
                    return "INVERTED"
                elif val > 0.0050:  # Steep by more than 50 bps
                    return "STEEP"
                else:
                    return "FLAT"

            df_pd["CURVE_REGIME"] = df_pd["CURVE_SLOPE_2Y_1M"].apply(_classify_regime)

        if as_polars:
            return pl.from_pandas(df_pd)
        return df_pd
