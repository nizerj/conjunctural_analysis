"""
Time-series alignment, calendar conventions, and return compounding.
"""

from conjunctural.timeseries.calendar import BrazilCalendar
from conjunctural.timeseries.returns import (
    annual_to_daily_rate,
    daily_to_annual_rate,
    compound_daily_rates,
    simple_returns,
    log_returns,
    real_rate,
)
from conjunctural.timeseries.harmonizer import TimeSeriesHarmonizer

__all__ = [
    "BrazilCalendar",
    "annual_to_daily_rate",
    "daily_to_annual_rate",
    "compound_daily_rates",
    "simple_returns",
    "log_returns",
    "real_rate",
    "TimeSeriesHarmonizer",
]
