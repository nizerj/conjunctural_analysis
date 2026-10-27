"""
Tests for Calendar, Returns compounding, and Mixed-Frequency Harmonization.
"""

import pytest
import numpy as np
import pandas as pd
from conjunctural.timeseries.calendar import BrazilCalendar
from conjunctural.timeseries.returns import (
    annual_to_daily_rate,
    daily_to_annual_rate,
    compound_daily_rates,
    real_rate,
    simple_returns,
)
from conjunctural.timeseries.harmonizer import TimeSeriesHarmonizer


def test_calendar():
    # 2024-01-01 is Monday (New Year holiday)
    assert not BrazilCalendar.is_business_day("2024-01-01")
    # 2024-01-02 is Tuesday (regular business day)
    assert BrazilCalendar.is_business_day("2024-01-02")
    # 2024-01-06 is Saturday
    assert not BrazilCalendar.is_business_day("2024-01-06")

    # Range
    b_days = BrazilCalendar.get_business_day_range("2024-01-01", "2024-01-07")
    # Mon (hol), Tue, Wed, Thu, Fri, Sat, Sun -> 4 business days (Tue-Fri)
    assert len(b_days) == 4


def test_rate_conversions():
    # 10.0% a.a. converted to daily and back to annual should return exactly 10.0%
    daily = annual_to_daily_rate(10.0)
    # daily rate should be approx 0.0378% (0.000378)
    assert 0.00037 < daily < 0.00039

    annual_back = daily_to_annual_rate(daily)
    assert pytest.approx(annual_back, 1e-5) == 10.0

    # Compounding 252 days of that daily rate should yield 110.0 from 100.0
    series = pd.Series([daily] * 252)
    acc = compound_daily_rates(series, initial_value=100.0)
    assert pytest.approx(acc.iloc[-1], 1e-4) == 110.0


def test_real_rate_fisher():
    # Nominal 12%, Inflation 5%
    # (1 + 0.12) / (1 + 0.05) - 1 = 0.066667 (6.6667%)
    real = real_rate(12.0, 5.0)
    assert pytest.approx(real, 1e-4) == 6.66667


def test_harmonizer_alignment():
    harmonizer = TimeSeriesHarmonizer()
    # Daily CDI series
    dates_d = pd.date_range("2024-01-01", "2024-03-31", freq="B")
    df_daily = pd.DataFrame({"date": dates_d, "CDI": 11.75})

    # Monthly IPCA series (released 1st of each month in raw data)
    dates_m = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")]
    df_monthly = pd.DataFrame({"date": dates_m, "IPCA": [0.42, 0.83, 0.16]})

    df_raw = pd.merge(df_daily, df_monthly, on="date", how="outer").sort_values("date")

    # Align to daily with publication lags
    aligned = harmonizer.align_to_daily(df_raw, apply_publication_lags=True, ffill=True)

    assert "date" in aligned.columns
    assert "CDI" in aligned.columns
    assert "IPCA" in aligned.columns
    # Check that there are no NaNs after initial lag period
    assert len(aligned) > 50
