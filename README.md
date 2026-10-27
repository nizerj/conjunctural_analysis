# Conjunctural (Análise Conjuntural)

**Macroeconomic / Financial Time-Series Backtester & Indicator Engine**

`conjunctural` is a local, lightweight Python library (leveraging **Pandas** and **Polars**) designed to ingest public Brazilian economic and financial market series, compute rolling macroeconomic indicators, align mixed-frequency time-series without lookahead bias, and backtest portfolio rebalancing models with institutional-grade risk metrics (Sharpe, Drawdown, VaR, CVaR).

---

## The 10 Core Brazilian Indicators

| Indicator | Frequency | Source Entity | Primary Metric / Unit | SGS / Source Code | Engine Key |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Selic Over** | Daily | BCB | % a.a. (252 b.d.) | SGS `1178` | `SELIC_OVER` |
| **CDI** | Daily | B3 / BCB | % daily / % a.a. | SGS `12` / `4389` | `CDI` |
| **DI Futures** | Daily | B3 | % a.a. per vertex | B3 DI1 curves | `DI_FUTURES` |
| **IPCA** | Monthly | IBGE | % monthly / Index number | SGS `433` | `IPCA` |
| **IPCA-15** | Monthly | IBGE | % monthly | SGS `7478` | `IPCA_15` |
| **IGP-M** | Monthly | FGV | % monthly | SGS `189` | `IGP_M` |
| **IBC-Br** | Monthly | BCB | Index number (saz.) | SGS `24364` | `IBC_BR` |
| **USD/BRL (PTAX)** | Daily | BCB | BRL per USD | SGS `1` (venda) | `USD_BRL` |
| **Current Account**| Monthly | BCB | USD (Millions) | SGS `22708` | `CURRENT_ACCOUNT`|
| **EMBI+ Brasil** | Daily | J.P. Morgan / IPEA | Basis points (bps) | IPEA `JPM366_EMBI366` (`40940`) | `EMBI_BRASIL` |

---

## Key Features

1. **Robust Public Data Ingestion**:
   - **BCB SGS**: Connects directly to the Central Bank's REST API with automated retry and fallback.
   - **IPEA OData v4**: Fetches sovereign risk spread series (EMBI+ Brasil / 40940).
   - **B3 DI Futures Curves**: Interacts with DI1 term structures via `Polars` and flat-forward compounding to extract constant-maturity vertices (1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y).
   - **Local Parquet Caching**: Automatically stores queries locally in Parquet/Arrow format, eliminating external rate limits, network latency, and enabling 100% offline backtests.

2. **Mixed-Frequency Harmonization & Leakage Prevention**:
   - Seamlessly aligns monthly releases (IPCA, IGP-M, IBC-Br) with daily financial series (Selic, CDI, PTAX, EMBI+).
   - Enforces Brazilian financial day count convention (252 business days / year) and ANBIMA compounding formulas:
     $$r_{\text{daily}} = (1 + r_{\text{annual}} / 100)^{1/252} - 1$$
   - Models official IBGE/FGV publication lags (e.g., 10 business days for IPCA) to eliminate lookahead bias in historical simulations.

3. **Macroeconomic Indicators Engine**:
   - **Inflation**: 12-month rolling cumulative inflation, 3-month annualized momentum ($SAAR$), and IGP-M / IPCA wholesale spread.
   - **Interest Rates**: Ex-post real Selic rate via exact Fisher equation:
     $$r_{\text{real}} = \frac{1 + \text{Nominal}}{1 + \text{Inflation}_{12M}} - 1$$
   - **Yield Curve**: Term structure slope ($2Y - 1M$, $5Y - 1Y$), butterfly curvature ($2 \times 2Y - (1Y + 5Y)$), and regime classification (`INVERTED`, `FLAT`, `STEEP`).
   - **Activity & External**: IBC-Br 3M momentum, 12M YoY growth, rolling Z-score cycle regimes (`EXPANSION`, `CONTRACTION`), and Current Account 12M trailing balance.
   - **Market Risk**: USD/BRL rolling volatility, drawdown from peak, and EMBI+ country risk regimes.

4. **Institutional Risk Engine**:
   - Compounded Annualized Return and Volatility.
   - Annualized Sharpe and Sortino Ratios (excess over daily CDI).
   - Maximum Drawdown (MDD), Underwater Drawdown series, and Calmar Ratio.
   - Value at Risk (VaR): Historical, Parametric Gaussian, and Cornish-Fisher Modified VaR (accounting for empirical skewness and kurtosis).
   - Conditional Value at Risk (CVaR / Expected Shortfall).
   - Benchmark Alpha, Beta, Tracking Error, and Information Ratio against CDI.

5. **Portfolio Backtesting & Regime Switching**:
   - Realistic event-driven simulation with turnover tracking and transaction cost models.
   - Flexible strategies: `FixedWeightStrategy`, `EqualWeightStrategy`, `MomentumRiskStrategy`, and `MacroRegimeStrategy` (dynamic allocation across CDI, USD, and duration driven by yield curve slopes, real interest rate stance, and country risk).
   - Automatic ASCII tearsheet and 3-panel visual reporting.

---

## Installation

```bash
# In your virtual environment
pip install -e .
```

Dependencies: `pandas`, `polars`, `numpy`, `scipy`, `requests`, `pyyield`, `pyarrow`, `matplotlib`.

---

## Quickstart

### 1. Inspect the Indicator Catalog

```python
from conjunctural.data.catalog import DataCatalog

catalog_df = DataCatalog.to_dataframe()
print(catalog_df)
```

### 2. Ingest Series with Local Parquet Caching

```python
from conjunctural.data.engine import DataEngine

engine = DataEngine()

# Fetch Selic Over and IPCA as Pandas or Polars DataFrames
df_selic = engine.fetch_indicator("SELIC_OVER", start_date="2023-01-01")
df_ipca = engine.fetch_indicator("IPCA", start_date="2023-01-01", as_polars=True)

# Fetch multiple indicators merged on date
df_macro = engine.fetch_multiple(
    ["SELIC_OVER", "CDI", "IPCA", "USD_BRL", "EMBI_BRASIL"],
    start_date="2022-01-01"
)
```

### 3. Compute Macroeconomic Indicators

```python
from conjunctural.timeseries.harmonizer import TimeSeriesHarmonizer
from conjunctural.indicators.inflation import InflationEngine
from conjunctural.indicators.interest_rate import InterestRateEngine
from conjunctural.indicators.technical import TechnicalEngine

# Align mixed frequencies onto a business-day grid (with publication lag)
harmonizer = TimeSeriesHarmonizer()
df_daily = harmonizer.align_to_daily(df_macro, apply_publication_lags=True)

# Compute rolling indicators
df_daily = InflationEngine.compute_all(df_daily)
df_daily = InterestRateEngine.compute_all(df_daily, neutral_real_rate=4.5)
df_daily = TechnicalEngine.compute_all(df_daily)

print(df_daily[["date", "REAL_SELIC_EX_POST", "MONETARY_STANCE_SPREAD", "EMBI_REGIME"]].tail())
```

### 4. Backtest a Macro Regime Allocation Strategy

```python
import pandas as pd
from conjunctural.backtest.engine import BacktestEngine
from conjunctural.backtest.strategies import MacroRegimeStrategy
from conjunctural.backtest.tearsheet import plot_backtest

# Construct price universe
asset_prices = pd.DataFrame(index=df_daily["date"])
asset_prices["CDI"] = df_daily["CDI_INDEX"].values
asset_prices["USD_BRL"] = df_daily["USD_BRL"].values

# Configure strategy
strategy = MacroRegimeStrategy(cdi_asset="CDI", fx_asset="USD_BRL")

# Run backtest with 5 bps transaction costs
bt = BacktestEngine(initial_capital=100_000.0, transaction_cost_bps=5.0)
result = bt.run(
    asset_prices=asset_prices,
    strategy=strategy,
    signals=df_daily,
    benchmark=asset_prices["CDI"],
    risk_free_rate=df_daily["CDI_DAILY_RETURN"],
)

# Print institutional performance tearsheet
print(result.tearsheet())

# Generate visualization
plot_backtest(result, save_path="output/tearsheet.png")
```

---

## CLI Usage

The library includes an integrated command-line tool `conjunctural`:

```bash
# Display indicator catalog
conjunctural catalog

# Fetch real-time data from BCB / IPEA
conjunctural fetch SELIC_OVER --start 2024-01-01 --rows 10
conjunctural fetch ALL --start 2023-01-01

# Run macroeconomic allocation backtest
conjunctural backtest --strategy macro --start 2022-01-01
```

---

## Running Tests

The test suite covers providers, cache mechanisms, financial compounding, rolling indicators, risk statistics, and simulation engines:

```bash
pytest -v
```
