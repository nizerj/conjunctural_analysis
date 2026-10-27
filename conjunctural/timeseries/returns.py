"""
Financial return and rate compounding calculations (Brazilian standard 252 business days).
"""

from typing import Union
import numpy as np
import pandas as pd
from conjunctural.config import ANNUAL_BUSINESS_DAYS


def annual_to_daily_rate(rate_annual_pct: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """
    Convert an annualized rate (% a.a. base 252) to an effective daily decimal rate.
    Formula: r_daily = (1 + r_aa / 100) ** (1 / 252) - 1
    """
    if isinstance(rate_annual_pct, (pd.Series, np.ndarray)):
        return (1.0 + rate_annual_pct / 100.0) ** (1.0 / ANNUAL_BUSINESS_DAYS) - 1.0
    return float((1.0 + rate_annual_pct / 100.0) ** (1.0 / ANNUAL_BUSINESS_DAYS) - 1.0)


def daily_to_annual_rate(rate_daily: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """
    Convert an effective daily rate (decimal or percentage) to annualized rate (% a.a. base 252).
    Formula: r_aa = ((1 + r_daily) ** 252 - 1) * 100
    """
    # If passed as percentage (e.g. 0.05%), check scale
    if isinstance(rate_daily, (pd.Series, np.ndarray)):
        # If mean is small (e.g. 0.0005) it's decimal; if ~0.05 it might be %
        r_dec = np.where(rate_daily > 0.1, rate_daily / 100.0, rate_daily)
        return ((1.0 + r_dec) ** ANNUAL_BUSINESS_DAYS - 1.0) * 100.0
    r_dec = rate_daily / 100.0 if rate_daily > 0.1 else rate_daily
    return float(((1.0 + r_dec) ** ANNUAL_BUSINESS_DAYS - 1.0) * 100.0)


def compound_daily_rates(daily_rates: pd.Series, initial_value: float = 100.0) -> pd.Series:
    """
    Calculate accumulated index series (e.g., CDI accumulator index) from daily rates.
    """
    # Convert daily % to decimal if needed
    rates = daily_rates.copy()
    if (rates.dropna() > 0.05).any():
        rates = rates / 100.0

    growth_factors = 1.0 + rates.fillna(0.0)
    acc = initial_value * growth_factors.cumprod()
    return acc


def simple_returns(prices: pd.Series) -> pd.Series:
    """Calculate simple discrete percentage returns: (P_t - P_{t-1}) / P_{t-1}."""
    return prices.pct_change()


def log_returns(prices: pd.Series) -> pd.Series:
    """Calculate continuously compounded logarithmic returns: ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1))


def real_rate(nominal_pct: Union[float, pd.Series], inflation_pct: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """
    Calculate real interest rate via exact Fisher equation:
    (1 + r_real) = (1 + r_nominal) / (1 + r_inflation)
    Returns percentage.
    """
    nom = nominal_pct / 100.0
    inf = inflation_pct / 100.0
    real = ((1.0 + nom) / (1.0 + inf) - 1.0) * 100.0
    return real
