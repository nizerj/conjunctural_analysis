"""
B3 DI Futures Curve Provider and Yield Curve Interpolator.
"""

from typing import Optional, List, Dict
import datetime
import numpy as np
import pandas as pd
import polars as pl
from conjunctural.config import ANNUAL_BUSINESS_DAYS
from conjunctural.data.cache import CacheManager

try:
    import pyield as yd
    HAS_PYIELD = True
except ImportError:
    HAS_PYIELD = False


STANDARD_VERTICES_DU: Dict[str, int] = {
    "DI_1M": 21,
    "DI_3M": 63,
    "DI_6M": 126,
    "DI_1Y": 252,
    "DI_2Y": 504,
    "DI_3Y": 756,
    "DI_5Y": 1260,
}


class B3DIProvider:
    """Provides B3 DI Futures term structure data and interpolated constant-maturity vertices."""

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache = cache_manager or CacheManager()

    def fetch_curve(
        self,
        date: str | datetime.date,
        use_cache: bool = True,
        as_polars: bool = False,
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Fetch all traded DI1 futures contracts and settlement rates for a specific trading day.

        Returns DataFrame with columns:
        ['data_referencia', 'codigo_negociacao', 'data_vencimento', 'dias_uteis', 'taxa_ajuste', ...]
        """
        dt_str = pd.to_datetime(date).strftime("%Y-%m-%d")
        cache_key = f"b3_di1_curve_{dt_str}"

        if use_cache:
            cached = self.cache.get(cache_key, as_polars=as_polars)
            if cached is not None and len(cached) > 0:
                return cached

        if not HAS_PYIELD:
            raise ImportError("pyield is required to fetch B3 DI futures. Install via `pip install pyield`.")

        try:
            pl_df = yd.di1.dados(dt_str)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch B3 DI1 curve for date {dt_str}: {e}")

        if use_cache and len(pl_df) > 0:
            self.cache.put(cache_key, pl_df)

        if as_polars:
            return pl_df
        return pl_df.to_pandas()

    def interpolate_vertices(
        self,
        curve_df: pd.DataFrame | pl.DataFrame,
        vertices: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """
        Interpolate standard constant-maturity vertices from a single-day DI curve using flat-forward interpolation.

        Parameters
        ----------
        curve_df : DataFrame
            DI curve for a single trade date containing 'dias_uteis' and 'taxa_ajuste'.
        vertices : dict, optional
            Mapping of vertex name to business days (default: 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y).

        Returns
        -------
        dict[str, float]
            Interpolated rates annualized (e.g. {'DI_1Y': 0.1050}).
        """
        if vertices is None:
            vertices = STANDARD_VERTICES_DU

        if isinstance(curve_df, pd.DataFrame):
            df_pl = pl.from_pandas(curve_df)
        else:
            df_pl = curve_df

        if len(df_pl) == 0:
            return {k: np.nan for k in vertices}

        df_sorted = df_pl.filter(pl.col("taxa_ajuste").is_not_null() & (pl.col("dias_uteis") > 0)).sort("dias_uteis")
        if len(df_sorted) == 0:
            return {k: np.nan for k in vertices}

        target_dus = list(vertices.values())
        target_keys = list(vertices.keys())

        if HAS_PYIELD:
            target_series = pl.Series(target_dus)
            interpolated = yd.di1.interpolar(
                target_series,
                df_sorted["dias_uteis"],
                df_sorted["taxa_ajuste"],
                extrapolar=True,
            )
            rates = interpolated.to_list()
            return dict(zip(target_keys, rates))
        else:
            # Fallback exponential flat forward interpolation
            rates = []
            dus_known = df_sorted["dias_uteis"].to_numpy()
            taxas_known = df_sorted["taxa_ajuste"].to_numpy()

            for du_target in target_dus:
                if du_target <= dus_known[0]:
                    rates.append(float(taxas_known[0]))
                elif du_target >= dus_known[-1]:
                    rates.append(float(taxas_known[-1]))
                else:
                    idx = np.searchsorted(dus_known, du_target)
                    du1, du2 = dus_known[idx - 1], dus_known[idx]
                    r1, r2 = taxas_known[idx - 1], taxas_known[idx]
                    # Compound accumulation factor
                    f1 = (1.0 + r1) ** (du1 / ANNUAL_BUSINESS_DAYS)
                    f2 = (1.0 + r2) ** (du2 / ANNUAL_BUSINESS_DAYS)
                    # Forward rate
                    f_target = f1 * ((f2 / f1) ** ((du_target - du1) / (du2 - du1)))
                    r_target = (f_target ** (ANNUAL_BUSINESS_DAYS / du_target)) - 1.0
                    rates.append(float(r_target))
            return dict(zip(target_keys, rates))

    def fetch_vertices_history(
        self,
        dates: List[str | datetime.date],
        vertices: Optional[Dict[str, int]] = None,
        use_cache: bool = True,
        as_polars: bool = False,
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Fetch and interpolate constant-maturity vertices across a list of dates.
        """
        if vertices is None:
            vertices = STANDARD_VERTICES_DU

        records = []
        for dt in dates:
            dt_obj = pd.to_datetime(dt)
            try:
                curve = self.fetch_curve(dt_obj, use_cache=use_cache)
                rates = self.interpolate_vertices(curve, vertices=vertices)
                row = {"date": dt_obj}
                row.update(rates)
                records.append(row)
            except Exception:
                continue

        if not records:
            cols = ["date"] + list(vertices.keys())
            df = pd.DataFrame(columns=cols)
        else:
            df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

        if as_polars:
            return pl.from_pandas(df)
        return df
