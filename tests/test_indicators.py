"""
Tests for Macro Indicators Engines.
"""

import pytest
import numpy as np
import pandas as pd
from conjunctural.indicators.inflation import InflationEngine
from conjunctural.indicators.interest_rate import InterestRateEngine
from conjunctural.indicators.yield_curve import YieldCurveEngine
from conjunctural.indicators.activity import ActivityEngine
from conjunctural.indicators.technical import TechnicalEngine


def test_inflation_engine():
    # 12 months with 0.5% monthly inflation
    # (1 + 0.005)^12 - 1 = 6.1678%
    rates = pd.Series([0.5] * 24)
    cum12 = InflationEngine.rolling_12m_cumulative(rates)
    assert pytest.approx(cum12.iloc[11], 1e-4) == 6.16778
    assert pytest.approx(cum12.iloc[23], 1e-4) == 6.16778

    # 3m annualized: (1 + 0.005)^12 - 1 = 6.1678%
    ann3 = InflationEngine.annualized_3m_rate(rates)
    assert pytest.approx(ann3.iloc[2], 1e-4) == 6.16778


def test_interest_rate_engine():
    df = pd.DataFrame({
        "SELIC_OVER": [13.75, 13.75, 10.50],
        "CDI": [13.65, 13.65, 10.40],
        "IPCA_12M": [4.0, 5.0, 4.0],
    })
    res = InterestRateEngine.compute_all(df, neutral_real_rate=4.5)
    # Real Selic: (1 + 0.1375) / (1 + 0.04) - 1 = 9.375%
    assert pytest.approx(res["REAL_SELIC_EX_POST"].iloc[0], 1e-3) == 9.375
    # Monetary stance spread: 9.375 - 4.5 = 4.875%
    assert pytest.approx(res["MONETARY_STANCE_SPREAD"].iloc[0], 1e-3) == 4.875
    # CDI daily return exists
    assert "CDI_DAILY_RETURN" in res.columns
    assert "CDI_INDEX" in res.columns


def test_yield_curve_engine():
    df = pd.DataFrame({
        "DI_1M": [0.12, 0.10, 0.11],
        "DI_1Y": [0.115, 0.11, 0.11],
        "DI_2Y": [0.11, 0.12, 0.11],
        "DI_5Y": [0.105, 0.13, 0.11],
    })
    res = YieldCurveEngine.compute_slopes_and_spreads(df)
    # Row 0: DI_2Y (11%) < DI_1M (12%) -> Inverted slope: -1.0%
    assert pytest.approx(res["CURVE_SLOPE_2Y_1M"].iloc[0], 1e-4) == -0.01
    assert res["CURVE_REGIME"].iloc[0] == "INVERTED"

    # Row 1: DI_2Y (12%) > DI_1M (10%) -> Steep slope: +2.0%
    assert res["CURVE_REGIME"].iloc[1] == "STEEP"

    # Row 2: DI_2Y == DI_1M -> Flat
    assert res["CURVE_REGIME"].iloc[2] == "FLAT"


def test_activity_and_technical_engines():
    df_act = pd.DataFrame({
        "IBC_BR": np.linspace(100, 110, 36),
        "CURRENT_ACCOUNT": [-2000.0] * 36,
    })
    res_act = ActivityEngine.compute_all(df_act)
    assert "IBC_BR_3M_GROWTH" in res_act.columns
    assert "CURRENT_ACCOUNT_12M_SUM" in res_act.columns

    df_tech = pd.DataFrame({
        "USD_BRL": [5.0 + 0.01 * i for i in range(100)],
        "EMBI_BRASIL": [200.0 + 2.0 * i for i in range(100)],
    })
    res_tech = TechnicalEngine.compute_all(df_tech)
    assert "USD_BRL_VOL_21D" in res_tech.columns
    assert "EMBI_DELTA_21D" in res_tech.columns
    assert "EMBI_ZSCORE" in res_tech.columns
