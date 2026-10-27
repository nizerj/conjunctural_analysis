"""
Tests for Portfolio Backtesting Engine and Rebalancing Strategies.
"""

import pytest
import numpy as np
import pandas as pd
from conjunctural.backtest.engine import BacktestEngine, Portfolio
from conjunctural.backtest.strategies import (
    FixedWeightStrategy,
    EqualWeightStrategy,
    MacroRegimeStrategy,
    MomentumRiskStrategy,
)


def test_portfolio_rebalance_turnover():
    port = Portfolio(initial_capital=100_000.0, transaction_cost_bps=10.0)
    # Rebalance to 50/50
    turnover1 = port.rebalance({"A": 0.5, "B": 0.5})
    assert pytest.approx(turnover1, 1e-4) == 0.5
    # Cost = 0.5 * 0.001 * 100,000 = 50 BRL -> NAV = 99,950
    assert pytest.approx(port.current_nav, 1e-1) == 99950.0

    # Rebalance to 100% A
    turnover2 = port.rebalance({"A": 1.0, "B": 0.0})
    assert pytest.approx(turnover2, 1e-4) == 0.5


def test_backtest_fixed_weights():
    dates = pd.date_range("2023-01-01", "2023-06-30", freq="B")
    # Asset A grows +0.05% daily, Asset B flat
    prices = pd.DataFrame(index=dates)
    prices["A"] = 100.0 * (1.0005 ** np.arange(len(dates)))
    prices["B"] = 100.0

    strat = FixedWeightStrategy({"A": 0.6, "B": 0.4}, rebalance_freq="ME")
    engine = BacktestEngine(initial_capital=100_000.0, transaction_cost_bps=5.0)
    res = engine.run(asset_prices=prices, strategy=strat, benchmark=prices["A"])

    assert len(res.nav) == len(dates)
    assert res.nav.iloc[-1] > 100_000.0
    assert "Sharpe Ratio" in res.risk_summary
    assert res.summary() is not None
    tearsheet = res.tearsheet()
    assert "CONJUNCTURAL BACKTEST TEARSHEET" in tearsheet


def test_macro_regime_strategy():
    dates = pd.date_range("2023-01-01", "2023-06-30", freq="B")
    prices = pd.DataFrame(index=dates)
    prices["CDI"] = 100.0 * (1.0004 ** np.arange(len(dates)))
    prices["USD_BRL"] = 5.0 * (1.0 + 0.02 * np.sin(np.arange(len(dates)) / 10.0))

    signals = pd.DataFrame(index=dates)
    signals["EMBI_REGIME"] = ["HIGH_RISK_OFF" if i > 60 else "BENIGN_RISK_ON" for i in range(len(dates))]
    signals["CURVE_REGIME"] = "STEEP"
    signals["MONETARY_STANCE_SPREAD"] = 1.0

    strat = MacroRegimeStrategy(cdi_asset="CDI", fx_asset="USD_BRL", rebalance_freq="ME")
    engine = BacktestEngine(initial_capital=100_000.0)
    res = engine.run(asset_prices=prices, strategy=strat, signals=signals, benchmark=prices["CDI"])

    assert res.nav.iloc[-1] > 0
    assert "USD_BRL" in res.weights.columns
    assert "CDI" in res.weights.columns
