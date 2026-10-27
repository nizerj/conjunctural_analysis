"""
Conjunctural - Brazilian Macroeconomic & Financial Time-Series Backtester & Indicator Engine.

A local, lightweight library (using Python/Pandas/Polars) that ingests Brazilian
economic and market series (BCB, B3, IBGE, FGV, IPEA) and computes rolling indicators,
macro regime models, portfolio rebalancing models, and institutional risk metrics.
"""

from conjunctural.data.catalog import Indicator, INDICATORS, DataCatalog
from conjunctural.data.bcb import BCBProvider
from conjunctural.data.ipea import IpeaProvider
from conjunctural.data.b3 import B3DIProvider
from conjunctural.timeseries.harmonizer import TimeSeriesHarmonizer
from conjunctural.indicators.inflation import InflationEngine
from conjunctural.indicators.interest_rate import InterestRateEngine
from conjunctural.indicators.yield_curve import YieldCurveEngine
from conjunctural.indicators.activity import ActivityEngine
from conjunctural.risk.metrics import RiskEngine
from conjunctural.backtest.engine import BacktestEngine, Portfolio
from conjunctural.backtest.strategies import (
    Strategy,
    EqualWeightStrategy,
    FixedWeightStrategy,
    MacroRegimeStrategy,
    MomentumRiskStrategy,
)

__version__ = "0.1.0"
__all__ = [
    "Indicator",
    "INDICATORS",
    "DataCatalog",
    "BCBProvider",
    "IpeaProvider",
    "B3DIProvider",
    "TimeSeriesHarmonizer",
    "InflationEngine",
    "InterestRateEngine",
    "YieldCurveEngine",
    "ActivityEngine",
    "RiskEngine",
    "BacktestEngine",
    "Portfolio",
    "Strategy",
    "EqualWeightStrategy",
    "FixedWeightStrategy",
    "MacroRegimeStrategy",
    "MomentumRiskStrategy",
]
