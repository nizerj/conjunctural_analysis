"""
Backtesting, simulation, and tearsheet generation.
"""

from conjunctural.backtest.strategies import (
    Strategy,
    FixedWeightStrategy,
    EqualWeightStrategy,
    MacroRegimeStrategy,
    MomentumRiskStrategy,
)
from conjunctural.backtest.engine import Portfolio, BacktestEngine, BacktestResult
from conjunctural.backtest.tearsheet import plot_backtest

__all__ = [
    "Strategy",
    "FixedWeightStrategy",
    "EqualWeightStrategy",
    "MacroRegimeStrategy",
    "MomentumRiskStrategy",
    "Portfolio",
    "BacktestEngine",
    "BacktestResult",
    "plot_backtest",
]
