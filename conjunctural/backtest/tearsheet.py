"""
Visual tearsheet and performance reports generation.
"""

from typing import Optional
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from conjunctural.backtest.engine import BacktestResult
from conjunctural.risk.metrics import RiskEngine


def plot_backtest(
    result: BacktestResult,
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """
    Generate a 3-panel performance tearsheet plot:
    1. Strategy NAV vs Benchmark
    2. Underwater Drawdown (%)
    3. Asset Allocation Weights over time
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 2]})
    
    # 1. NAV Panel
    ax1 = axes[0]
    ax1.plot(result.nav.index, result.nav.values, label=f"Strategy: {result.strategy_name}", color="#0052cc", linewidth=1.8)
    if result.benchmark_nav is not None:
        ax1.plot(result.benchmark_nav.index, result.benchmark_nav.values, label="Benchmark", color="#6b778c", linestyle="--", linewidth=1.4)
    ax1.set_ylabel("Portfolio NAV (BRL)")
    ax1.set_title(title or f"Backtest Tearsheet: {result.strategy_name}", fontsize=14, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # 2. Drawdown Panel
    ax2 = axes[1]
    dd = RiskEngine.drawdown_series(result.nav) * 100.0
    ax2.plot(dd.index, dd.values, color="#de350b", linewidth=1.2)
    ax2.fill_between(dd.index, dd.values, 0, color="#de350b", alpha=0.25)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. Allocation Weights Panel
    ax3 = axes[2]
    weights = result.weights.copy()
    if len(weights.columns) > 0:
        ax3.stackplot(weights.index, [weights[c].values for c in weights.columns], labels=weights.columns, alpha=0.8)
        ax3.set_ylabel("Allocation Weight")
        ax3.set_ylim(0, 1.0)
        ax3.legend(loc="lower left", ncol=min(4, len(weights.columns)), fontsize=8)
    ax3.set_xlabel("Date")
    ax3.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    return fig
