"""
Risk metrics and quantitative portfolio analytics (Sharpe, Drawdown, VaR, CVaR).
"""

from typing import Union, Optional, Dict
import numpy as np
import pandas as pd
from scipy import stats
from conjunctural.config import ANNUAL_BUSINESS_DAYS


class RiskEngine:
    """Calculates quantitative risk and performance metrics for returns or NAV series."""

    ANNUAL_DAYS = ANNUAL_BUSINESS_DAYS

    @classmethod
    def annualized_return(cls, returns: pd.Series) -> float:
        """Annualized geometric compounded return: prod(1 + r)^(252 / N) - 1."""
        clean = returns.dropna()
        n = len(clean)
        if n == 0:
            return 0.0
        cum_ret = np.prod(1.0 + clean)
        return float(cum_ret ** (cls.ANNUAL_DAYS / n) - 1.0)

    @classmethod
    def annualized_volatility(cls, returns: pd.Series) -> float:
        """Annualized volatility: std(r) * sqrt(252)."""
        clean = returns.dropna()
        if len(clean) < 2:
            return 0.0
        return float(clean.std() * np.sqrt(cls.ANNUAL_DAYS))

    @classmethod
    def sharpe_ratio(
        cls,
        returns: pd.Series,
        risk_free_rate: Union[float, pd.Series] = 0.0,
    ) -> float:
        """
        Calculate annualized Sharpe Ratio.
        If risk_free_rate is a Series (e.g., daily CDI return), calculates exact excess returns.
        If float, assumes annualized percentage or daily rate depending on magnitude.
        """
        clean_ret = returns.dropna()
        if len(clean_ret) < 2:
            return 0.0

        if isinstance(risk_free_rate, pd.Series):
            rf_aligned = risk_free_rate.reindex(clean_ret.index).fillna(0.0)
            excess = clean_ret - rf_aligned
            vol = clean_ret.std() * np.sqrt(cls.ANNUAL_DAYS)
            if vol == 0.0:
                return 0.0
            excess_ann = excess.mean() * cls.ANNUAL_DAYS
            return float(excess_ann / vol)
        else:
            rf_daily = risk_free_rate / cls.ANNUAL_DAYS if risk_free_rate > 0.05 else risk_free_rate
            excess = clean_ret - rf_daily
            vol = clean_ret.std() * np.sqrt(cls.ANNUAL_DAYS)
            if vol == 0.0:
                return 0.0
            excess_ann = excess.mean() * cls.ANNUAL_DAYS
            return float(excess_ann / vol)

    @classmethod
    def sortino_ratio(
        cls,
        returns: pd.Series,
        target_return: Union[float, pd.Series] = 0.0,
    ) -> float:
        """
        Calculate annualized Sortino Ratio (penalizing only downside volatility).
        """
        clean_ret = returns.dropna()
        if len(clean_ret) < 2:
            return 0.0

        if isinstance(target_return, pd.Series):
            target = target_return.reindex(clean_ret.index).fillna(0.0)
        else:
            target = target_return / cls.ANNUAL_DAYS if target_return > 0.05 else target_return

        excess = clean_ret - target
        downside = np.minimum(0.0, excess)
        downside_dev = np.sqrt(np.mean(downside ** 2)) * np.sqrt(cls.ANNUAL_DAYS)

        if downside_dev == 0.0:
            return 0.0

        excess_ann = excess.mean() * cls.ANNUAL_DAYS
        return float(excess_ann / downside_dev)

    @staticmethod
    def drawdown_series(nav_or_prices: pd.Series) -> pd.Series:
        """Calculate underwater drawdown series: (NAV - Peak) / Peak."""
        clean = nav_or_prices.dropna()
        peak = clean.cummax()
        dd = (clean - peak) / peak
        return dd

    @classmethod
    def max_drawdown(cls, nav_or_prices: pd.Series) -> float:
        """Maximum peak-to-trough drawdown (returns negative value or zero)."""
        dd = cls.drawdown_series(nav_or_prices)
        if len(dd) == 0:
            return 0.0
        return float(dd.min())

    @classmethod
    def calmar_ratio(cls, returns: pd.Series, nav: Optional[pd.Series] = None) -> float:
        """Calmar Ratio: Annualized Return / |Max Drawdown|."""
        if nav is None:
            nav = (1.0 + returns.fillna(0.0)).cumprod()
        mdd = abs(cls.max_drawdown(nav))
        if mdd == 0.0:
            return 0.0
        ann_ret = cls.annualized_return(returns)
        return float(ann_ret / mdd)

    @staticmethod
    def var_historical(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Historical Value at Risk (VaR) as a positive loss percentage.
        e.g., at 0.95 confidence, returns the 5th percentile loss.
        """
        clean = returns.dropna()
        if len(clean) == 0:
            return 0.0
        alpha = 1.0 - confidence_level
        quantile = np.percentile(clean, alpha * 100.0)
        return float(-quantile if quantile < 0.0 else 0.0)

    @staticmethod
    def var_gaussian(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Parametric Gaussian VaR assuming normal distribution."""
        clean = returns.dropna()
        if len(clean) < 2:
            return 0.0
        mu = clean.mean()
        sigma = clean.std()
        z = stats.norm.ppf(1.0 - confidence_level)
        var = -(mu + z * sigma)
        return float(var if var > 0.0 else 0.0)

    @staticmethod
    def var_cornish_fisher(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Modified Value at Risk using Cornish-Fisher expansion
        accounting for empirical skewness and kurtosis.
        """
        clean = returns.dropna()
        if len(clean) < 5:
            return 0.0
        mu = clean.mean()
        sigma = clean.std()
        s = stats.skew(clean)
        k = stats.kurtosis(clean)  # excess kurtosis

        z = stats.norm.ppf(1.0 - confidence_level)
        z_cf = (
            z
            + (z**2 - 1.0) * s / 6.0
            + (z**3 - 3.0 * z) * k / 24.0
            - (2.0 * z**3 - 5.0 * z) * (s**2) / 36.0
        )
        mvar = -(mu + z_cf * sigma)
        return float(mvar if mvar > 0.0 else 0.0)

    @staticmethod
    def cvar_expected_shortfall(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Conditional Value at Risk (CVaR / Expected Shortfall):
        Average loss conditioned on loss being strictly worse than VaR.
        """
        clean = returns.dropna()
        if len(clean) == 0:
            return 0.0
        alpha = 1.0 - confidence_level
        cutoff = np.percentile(clean, alpha * 100.0)
        tail_losses = clean[clean <= cutoff]
        if len(tail_losses) == 0:
            return float(-cutoff if cutoff < 0 else 0.0)
        return float(-tail_losses.mean())

    @classmethod
    def compute_summary(
        cls,
        returns: pd.Series,
        risk_free_rate: Union[float, pd.Series] = 0.0,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Generate a complete statistical risk & performance profile."""
        clean = returns.dropna()
        nav = (1.0 + clean).cumprod()

        ann_ret = cls.annualized_return(clean)
        ann_vol = cls.annualized_volatility(clean)
        sharpe = cls.sharpe_ratio(clean, risk_free_rate=risk_free_rate)
        sortino = cls.sortino_ratio(clean, target_return=risk_free_rate)
        mdd = cls.max_drawdown(nav)
        calmar = cls.calmar_ratio(clean, nav=nav)
        var_95 = cls.var_historical(clean, 0.95)
        var_99 = cls.var_historical(clean, 0.99)
        cvar_95 = cls.cvar_expected_shortfall(clean, 0.95)
        mvar_95 = cls.var_cornish_fisher(clean, 0.95)

        summary = {
            "Total Return (%)": float((nav.iloc[-1] - 1.0) * 100.0) if len(nav) > 0 else 0.0,
            "Annualized Return (%)": ann_ret * 100.0,
            "Annualized Volatility (%)": ann_vol * 100.0,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown (%)": mdd * 100.0,
            "Calmar Ratio": calmar,
            "VaR 95% (1d, %)": var_95 * 100.0,
            "VaR 99% (1d, %)": var_99 * 100.0,
            "Modified VaR 95% (%)": mvar_95 * 100.0,
            "CVaR 95% (Expected Shortfall, %)": cvar_95 * 100.0,
        }

        if benchmark_returns is not None:
            bm_clean = benchmark_returns.reindex(clean.index).dropna()
            paired = pd.concat([clean, bm_clean], axis=1).dropna()
            if len(paired) > 10:
                y = paired.iloc[:, 0]
                x = paired.iloc[:, 1]
                cov = np.cov(y, x)
                beta = cov[0, 1] / (cov[1, 1] + 1e-12)
                alpha_ann = (y.mean() - beta * x.mean()) * cls.ANNUAL_DAYS
                tracking_error = (y - x).std() * np.sqrt(cls.ANNUAL_DAYS)
                info_ratio = (ann_ret - cls.annualized_return(x)) / (tracking_error + 1e-12)
                summary["Beta to Benchmark"] = float(beta)
                summary["Annualized Alpha (%)"] = float(alpha_ann * 100.0)
                summary["Tracking Error (%)"] = float(tracking_error * 100.0)
                summary["Information Ratio"] = float(info_ratio)

        return summary
