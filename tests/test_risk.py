"""
Tests for Quantitative Risk Engine (Sharpe, Drawdown, VaR, CVaR).
"""

import pytest
import numpy as np
import pandas as pd
from conjunctural.risk.metrics import RiskEngine


def test_annualized_metrics():
    # 252 days with 0.1% daily return
    # (1 + 0.001)^252 - 1 = 28.6%
    rets = pd.Series([0.001] * 252)
    ann_ret = RiskEngine.annualized_return(rets)
    assert pytest.approx(ann_ret, 1e-3) == 0.2863


def test_drawdowns():
    nav = pd.Series([100.0, 110.0, 120.0, 90.0, 105.0, 130.0])
    dd = RiskEngine.drawdown_series(nav)
    # Peak is 120, trough is 90 -> (90 - 120) / 120 = -0.25 (-25%)
    mdd = RiskEngine.max_drawdown(nav)
    assert pytest.approx(mdd, 1e-4) == -0.25
    assert pytest.approx(dd.iloc[3], 1e-4) == -0.25


def test_sharpe_and_sortino():
    np.random.seed(42)
    rets = pd.Series(np.random.normal(0.0005, 0.01, 500))  # positive drift
    sharpe = RiskEngine.sharpe_ratio(rets, risk_free_rate=0.0)
    sortino = RiskEngine.sortino_ratio(rets, target_return=0.0)
    assert sharpe > 0
    assert sortino > 0


def test_var_and_cvar():
    np.random.seed(42)
    rets = pd.Series(np.random.normal(0.0, 0.02, 1000))
    var_95 = RiskEngine.var_historical(rets, 0.95)
    cvar_95 = RiskEngine.cvar_expected_shortfall(rets, 0.95)
    var_gauss = RiskEngine.var_gaussian(rets, 0.95)
    mvar = RiskEngine.var_cornish_fisher(rets, 0.95)

    # Losses should be positive percentages
    assert var_95 > 0.0
    assert cvar_95 > var_95  # Expected shortfall must be strictly greater than VaR
    assert var_gauss > 0.0
    assert mvar > 0.0


def test_risk_summary():
    rets = pd.Series([0.0002, 0.0003, -0.0001, 0.0005] * 63)
    summary = RiskEngine.compute_summary(rets)
    assert "Sharpe Ratio" in summary
    assert "Max Drawdown (%)" in summary
    assert "VaR 95% (1d, %)" in summary
    assert "CVaR 95% (Expected Shortfall, %)" in summary
