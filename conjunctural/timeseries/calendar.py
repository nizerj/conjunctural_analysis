"""
Brazilian Business Day and Calendar utilities.
"""

from typing import Union, List
import datetime
import numpy as np
import pandas as pd
from conjunctural.config import ANNUAL_BUSINESS_DAYS


class BrazilCalendar:
    """Standard Brazilian Financial Calendar (ANBIMA / B3)."""

    ANNUAL_DAYS = ANNUAL_BUSINESS_DAYS

    @staticmethod
    def is_business_day(date: Union[str, datetime.date, pd.Timestamp]) -> bool:
        ts = pd.to_datetime(date)
        # Check weekend
        if ts.weekday() >= 5:
            return False
        # Common Brazilian national fixed holidays
        fixed_holidays = {
            (1, 1),   # Ano Novo
            (4, 21),  # Tiradentes
            (5, 1),   # Dia do Trabalho
            (9, 7),   # Independência
            (10, 12), # Nossa Senhora Aparecida
            (11, 2),  # Finados
            (11, 15), # Proclamação da República
            (11, 20), # Consciência Negra (national from 2024 onwards)
            (12, 25), # Natal
        }
        if (ts.month, ts.day) in fixed_holidays:
            return False
        return True

    @classmethod
    def business_days_between(cls, start_date: str | datetime.date, end_date: str | datetime.date) -> int:
        """Count business days between start_date and end_date (inclusive of start, exclusive of end)."""
        dt_start = pd.to_datetime(start_date)
        dt_end = pd.to_datetime(end_date)
        if dt_start >= dt_end:
            return 0
        date_range = pd.date_range(dt_start, dt_end - pd.Timedelta(days=1))
        return sum(1 for d in date_range if cls.is_business_day(d))

    @classmethod
    def get_business_day_range(cls, start_date: str | datetime.date, end_date: str | datetime.date) -> pd.DatetimeIndex:
        """Generate DatetimeIndex containing only business days."""
        all_dates = pd.date_range(start_date, end_date)
        b_dates = [d for d in all_dates if cls.is_business_day(d)]
        return pd.DatetimeIndex(b_dates)
