"""
Portfolio allocation and macroeconomic regime-switching strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class Strategy(ABC):
    """Abstract base class for portfolio rebalancing strategies."""

    def __init__(self, name: str, rebalance_freq: str = "ME"):
        self.name = name
        self.rebalance_freq = rebalance_freq  # 'D' (daily), 'W' (weekly), 'ME' (monthly)

    @abstractmethod
    def generate_weights(
        self,
        date: pd.Timestamp,
        history: pd.DataFrame,
        asset_universe: List[str],
    ) -> Dict[str, float]:
        """
        Compute target asset weights for the portfolio on a rebalance date.
        Must return a dict mapping asset -> weight (summing to <= 1.0).
        """
        pass


class FixedWeightStrategy(Strategy):
    """Static fixed-weights allocation rebalanced periodically."""

    def __init__(self, weights: Dict[str, float], name: str = "Fixed Weights", rebalance_freq: str = "ME"):
        super().__init__(name, rebalance_freq)
        self.weights = weights

    def generate_weights(
        self,
        date: pd.Timestamp,
        history: pd.DataFrame,
        asset_universe: List[str],
    ) -> Dict[str, float]:
        return {k: self.weights.get(k, 0.0) for k in asset_universe}


class EqualWeightStrategy(Strategy):
    """1/N equal weighting across all assets in universe."""

    def __init__(self, name: str = "Equal Weight (1/N)", rebalance_freq: str = "ME"):
        super().__init__(name, rebalance_freq)

    def generate_weights(
        self,
        date: pd.Timestamp,
        history: pd.DataFrame,
        asset_universe: List[str],
    ) -> Dict[str, float]:
        n = len(asset_universe)
        if n == 0:
            return {}
        w = 1.0 / n
        return {asset: w for asset in asset_universe}


class MacroRegimeStrategy(Strategy):
    """
    Macroeconomic Regime-Switching Strategy for Brazilian Assets:
    Allocates dynamically between CDI (Money Market / Cash), Fixed Income / Duration,
    and USD (FX Hedge) based on:
    1. Real Interest Rate Stance (Selic vs IPCA)
    2. DI Term Structure Slope (Inversion vs Steepening)
    3. Country Risk (EMBI+ Brasil) and Inflation Momentum
    """

    def __init__(
        self,
        cdi_asset: str = "CDI",
        fx_asset: str = "USD_BRL",
        fixed_income_asset: Optional[str] = None,
        name: str = "Macro Regime Allocator",
        rebalance_freq: str = "ME",
    ):
        super().__init__(name, rebalance_freq)
        self.cdi_asset = cdi_asset
        self.fx_asset = fx_asset
        self.fixed_income_asset = fixed_income_asset

    def generate_weights(
        self,
        date: pd.Timestamp,
        history: pd.DataFrame,
        asset_universe: List[str],
    ) -> Dict[str, float]:
        if len(history) == 0:
            return {a: 1.0 / len(asset_universe) for a in asset_universe}

        latest = history.iloc[-1]

        # Extract macro signals
        embi_regime = latest.get("EMBI_REGIME", "NEUTRAL")
        curve_regime = latest.get("CURVE_REGIME", "FLAT")
        monetary_stance = latest.get("MONETARY_STANCE_SPREAD", 0.0)

        # Baseline weights
        w_cdi = 0.50
        w_fx = 0.20
        w_fi = 0.30 if self.fixed_income_asset in asset_universe else 0.0

        # Rule 1: High country risk stress -> Overweight USD hedge, reduce duration
        if embi_regime == "HIGH_RISK_OFF":
            w_fx += 0.20
            w_cdi += 0.10
            w_fi = max(0.0, w_fi - 0.30)

        # Rule 2: Inverted yield curve (aggressive tightening) -> Flight to CDI cash
        elif curve_regime == "INVERTED" or monetary_stance > 2.0:
            w_cdi += 0.30
            w_fx = max(0.10, w_fx - 0.10)
            w_fi = max(0.0, w_fi - 0.20)

        # Rule 3: Steep curve & benign country risk -> Overweight fixed income duration / assets
        elif curve_regime == "STEEP" and embi_regime == "BENIGN_RISK_ON":
            if self.fixed_income_asset in asset_universe:
                w_fi += 0.30
                w_cdi = max(0.20, w_cdi - 0.20)
                w_fx = max(0.10, w_fx - 0.10)
            else:
                w_cdi += 0.20

        # Normalize weights
        weights = {}
        if self.cdi_asset in asset_universe:
            weights[self.cdi_asset] = w_cdi
        if self.fx_asset in asset_universe:
            weights[self.fx_asset] = w_fx
        if self.fixed_income_asset and self.fixed_income_asset in asset_universe:
            weights[self.fixed_income_asset] = w_fi

        # Fill remaining assets with equal share if any
        remaining_assets = [a for a in asset_universe if a not in weights]
        total_assigned = sum(weights.values())
        if remaining_assets:
            rem_weight = max(0.0, 1.0 - total_assigned) / len(remaining_assets)
            for a in remaining_assets:
                weights[a] = rem_weight

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights


class MomentumRiskStrategy(Strategy):
    """
    Inverse-Volatility / Risk Parity Strategy with Trend Filter.
    Allocates proportionally to 1 / sigma, filtered by 63-day momentum.
    """

    def __init__(
        self,
        lookback_days: int = 63,
        name: str = "Risk Parity & Momentum",
        rebalance_freq: str = "ME",
    ):
        super().__init__(name, rebalance_freq)
        self.lookback_days = lookback_days

    def generate_weights(
        self,
        date: pd.Timestamp,
        history: pd.DataFrame,
        asset_universe: List[str],
    ) -> Dict[str, float]:
        if len(history) < self.lookback_days:
            n = len(asset_universe)
            return {a: 1.0 / n for a in asset_universe}

        sub = history.iloc[-self.lookback_days:]
        inv_vols = {}

        for asset in asset_universe:
            if asset in sub.columns:
                returns = sub[asset].pct_change().dropna()
                vol = returns.std()
                mom = (sub[asset].iloc[-1] / (sub[asset].iloc[0] + 1e-12)) - 1.0
                if vol > 1e-6 and mom >= 0:
                    inv_vols[asset] = 1.0 / vol
                elif vol > 1e-6:
                    inv_vols[asset] = 0.25 / vol  # penalty for negative trend
                else:
                    inv_vols[asset] = 1.0
            else:
                inv_vols[asset] = 1.0

        total_inv = sum(inv_vols.values())
        if total_inv > 0:
            return {a: inv_vols[a] / total_inv for a in asset_universe}
        return {a: 1.0 / len(asset_universe) for a in asset_universe}
