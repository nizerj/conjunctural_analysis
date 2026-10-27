"""
Core Portfolio Backtesting and Simulation Engine.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from conjunctural.backtest.strategies import Strategy
from conjunctural.risk.metrics import RiskEngine


@dataclass
class BacktestResult:
    """Encapsulates the full results of a backtest simulation."""
    strategy_name: str
    nav: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    risk_summary: Dict[str, float]
    benchmark_nav: Optional[pd.Series] = None
    benchmark_returns: Optional[pd.Series] = None
    trades: Optional[pd.DataFrame] = None

    def summary(self) -> pd.DataFrame:
        """Return risk and performance metrics as a pandas DataFrame."""
        return pd.DataFrame(list(self.risk_summary.items()), columns=["Metric", "Value"])

    def tearsheet(self) -> str:
        """Produce a clean formatted text tearsheet."""
        lines = [
            "=" * 60,
            f"  CONJUNCTURAL BACKTEST TEARSHEET: {self.strategy_name.upper()}",
            "=" * 60,
            f"Period: {self.nav.index[0].strftime('%Y-%m-%d')} to {self.nav.index[-1].strftime('%Y-%m-%d')}",
            f"Trading Days: {len(self.nav)}",
            "-" * 60,
        ]
        for metric, val in self.risk_summary.items():
            if isinstance(val, float):
                if "%" in metric or "Rate" in metric or "Return" in metric or "Vol" in metric or "Alpha" in metric or "Drawdown" in metric or "VaR" in metric:
                    lines.append(f"{metric:<38} : {val:>10.2f}%" if not metric.endswith("(%)") else f"{metric:<38} : {val:>10.2f}")
                else:
                    lines.append(f"{metric:<38} : {val:>10.4f}")
            else:
                lines.append(f"{metric:<38} : {val}")
        lines.append("=" * 60)
        return "\n".join(lines)


class Portfolio:
    """Tracks NAV, asset allocations, cash, and trade transaction costs."""

    def __init__(self, initial_capital: float = 100_000.0, transaction_cost_bps: float = 5.0):
        self.initial_capital = initial_capital
        self.current_nav = initial_capital
        self.transaction_cost_pct = transaction_cost_bps / 10_000.0
        self.current_weights: Dict[str, float] = {}

    def rebalance(self, target_weights: Dict[str, float]) -> float:
        """
        Rebalance portfolio to target weights and compute transaction costs.
        Returns total transaction costs in currency units.
        """
        all_assets = set(self.current_weights.keys()).union(target_weights.keys())
        turnover = sum(
            abs(target_weights.get(a, 0.0) - self.current_weights.get(a, 0.0))
            for a in all_assets
        ) / 2.0  # single-counted turnover

        cost = turnover * self.transaction_cost_pct * self.current_nav
        self.current_nav -= cost
        self.current_weights = {k: v for k, v in target_weights.items() if v > 1e-6}
        return turnover


class BacktestEngine:
    """Historical event-driven portfolio rebalancer and simulation engine."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        transaction_cost_bps: float = 5.0,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps

    def run(
        self,
        asset_prices: pd.DataFrame,
        strategy: Strategy,
        signals: Optional[pd.DataFrame] = None,
        benchmark: Optional[pd.Series] = None,
        risk_free_rate: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """
        Run backtest simulation.

        Parameters
        ----------
        asset_prices : pd.DataFrame
            Prices or total return indices of assets in the universe. Must have DatetimeIndex or 'date' col.
        strategy : Strategy
            Rebalancing strategy implementation.
        signals : pd.DataFrame, optional
            Macro indicators and signals aligned to the same dates.
        benchmark : pd.Series, optional
            Benchmark price/index series (e.g. CDI index).
        risk_free_rate : pd.Series, optional
            Daily risk-free returns for Sharpe/Sortino calculations.
        """
        prices = asset_prices.copy()
        if "date" in prices.columns:
            prices["date"] = pd.to_datetime(prices["date"])
            prices = prices.set_index("date")
        elif not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index)

        prices = prices.sort_index().ffill().dropna(how="all")
        asset_returns = prices.pct_change().fillna(0.0)
        asset_universe = list(prices.columns)

        if signals is not None:
            sig = signals.copy()
            if "date" in sig.columns:
                sig["date"] = pd.to_datetime(sig["date"])
                sig = sig.set_index("date")
            elif not isinstance(sig.index, pd.DatetimeIndex):
                sig.index = pd.to_datetime(sig.index)
            merged_context = pd.concat([prices, sig], axis=1).sort_index().ffill()
        else:
            merged_context = prices

        # Determine rebalancing dates
        rebal_freq = strategy.rebalance_freq
        if rebal_freq == "D":
            rebal_dates = set(prices.index)
        elif rebal_freq == "W":
            # Last business day of each week
            rebal_dates = set(prices.resample("W-FRI").last().index.intersection(prices.index))
        else:
            # Month-end
            rebal_dates = set(prices.resample("ME").last().index.intersection(prices.index))

        portfolio = Portfolio(initial_capital=self.initial_capital, transaction_cost_bps=self.transaction_cost_bps)
        nav_history = []
        turnover_history = []
        weights_history = []

        dates = list(prices.index)
        current_weights = {a: 0.0 for a in asset_universe}

        for i, dt in enumerate(dates):
            # 1. Update valuation with today's asset returns
            if i > 0:
                day_returns = asset_returns.loc[dt]
                port_return = sum(current_weights.get(a, 0.0) * day_returns.get(a, 0.0) for a in asset_universe)
                portfolio.current_nav *= (1.0 + port_return)

            # 2. Check rebalance
            turnover = 0.0
            # If first day or rebalance date, adjust target weights
            if i == 0 or dt in rebal_dates:
                # Use context up to previous day (or current day on day 0) to avoid lookahead bias
                hist_slice = merged_context.loc[:dt] if i == 0 else merged_context.loc[:dates[i - 1]]
                target_weights = strategy.generate_weights(dt, hist_slice, asset_universe)
                turnover = portfolio.rebalance(target_weights)
                current_weights = portfolio.current_weights.copy()

            nav_history.append((dt, portfolio.current_nav))
            turnover_history.append((dt, turnover))
            row_w = {"date": dt}
            row_w.update(current_weights)
            weights_history.append(row_w)

        nav_s = pd.Series([n for _, n in nav_history], index=[d for d, _ in nav_history], name="NAV")
        port_ret = nav_s.pct_change().fillna(0.0)
        turn_s = pd.Series([t for _, t in turnover_history], index=[d for d, _ in turnover_history], name="Turnover")
        weights_df = pd.DataFrame(weights_history).set_index("date")

        # Risk and benchmark stats
        bm_ret = None
        bm_nav = None
        if benchmark is not None:
            bm_clean = benchmark.reindex(nav_s.index).ffill()
            bm_ret = bm_clean.pct_change().fillna(0.0)
            bm_nav = (1.0 + bm_ret).cumprod() * self.initial_capital

        rf_series = risk_free_rate.reindex(nav_s.index).fillna(0.0) if risk_free_rate is not None else 0.0
        risk_summary = RiskEngine.compute_summary(port_ret, risk_free_rate=rf_series, benchmark_returns=bm_ret)

        return BacktestResult(
            strategy_name=strategy.name,
            nav=nav_s,
            returns=port_ret,
            weights=weights_df,
            turnover=turn_s,
            risk_summary=risk_summary,
            benchmark_nav=bm_nav,
            benchmark_returns=bm_ret,
        )
