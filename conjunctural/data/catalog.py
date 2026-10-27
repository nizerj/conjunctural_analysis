"""
Indicator Catalog for Brazilian Macroeconomic and Financial Time-Series.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, List


class Frequency(str, Enum):
    DAILY = "Daily"
    MONTHLY = "Monthly"


class ProviderType(str, Enum):
    BCB = "bcb"
    IPEA = "ipea"
    B3 = "b3"


@dataclass(frozen=True)
class IndicatorMeta:
    id: str
    name: str
    frequency: Frequency
    source_entity: str
    primary_metric: str
    code: str
    provider: ProviderType
    description: str


# Canonical 10 indicators specified in the core architecture
INDICATORS: Dict[str, IndicatorMeta] = {
    "SELIC_OVER": IndicatorMeta(
        id="SELIC_OVER",
        name="Selic Over",
        frequency=Frequency.DAILY,
        source_entity="BCB",
        primary_metric="% a.a. (252 b.d.)",
        code="1178",
        provider=ProviderType.BCB,
        description="Taxa de juros - Selic acumulada no mês anualizada base 252 / Taxa média diária anualizada.",
    ),
    "CDI": IndicatorMeta(
        id="CDI",
        name="CDI",
        frequency=Frequency.DAILY,
        source_entity="B3 / BCB",
        primary_metric="% a.a. (252 b.d.)",
        code="4389",  # 4389: Taxa CDI anualizada base 252; 12: Taxa CDI diária
        provider=ProviderType.BCB,
        description="Taxa de juros dos Certificados de Depósito Interfinanceiro (anualizada base 252).",
    ),
    "CDI_DAILY": IndicatorMeta(
        id="CDI_DAILY",
        name="CDI Diário",
        frequency=Frequency.DAILY,
        source_entity="B3 / BCB",
        primary_metric="% daily",
        code="12",
        provider=ProviderType.BCB,
        description="Taxa diária de juros do CDI (% ao dia).",
    ),
    "DI_FUTURES": IndicatorMeta(
        id="DI_FUTURES",
        name="DI Futures Curve",
        frequency=Frequency.DAILY,
        source_entity="B3",
        primary_metric="% a.a. per vertex",
        code="B3_DI1",
        provider=ProviderType.B3,
        description="Curva a termo de juros futuros de DI1 da B3 com vértices padronizados.",
    ),
    "IPCA": IndicatorMeta(
        id="IPCA",
        name="IPCA",
        frequency=Frequency.MONTHLY,
        source_entity="IBGE",
        primary_metric="% monthly",
        code="433",
        provider=ProviderType.BCB,
        description="Índice Nacional de Preços ao Consumidor Amplo (variação mensal %).",
    ),
    "IPCA_15": IndicatorMeta(
        id="IPCA_15",
        name="IPCA-15",
        frequency=Frequency.MONTHLY,
        source_entity="IBGE",
        primary_metric="% monthly",
        code="7478",
        provider=ProviderType.BCB,
        description="Índice Nacional de Preços ao Consumidor Amplo-15 (variação mensal %).",
    ),
    "IGP_M": IndicatorMeta(
        id="IGP_M",
        name="IGP-M",
        frequency=Frequency.MONTHLY,
        source_entity="FGV",
        primary_metric="% monthly",
        code="189",
        provider=ProviderType.BCB,
        description="Índice Geral de Preços do Mercado (variação mensal %).",
    ),
    "IBC_BR": IndicatorMeta(
        id="IBC_BR",
        name="IBC-Br",
        frequency=Frequency.MONTHLY,
        source_entity="BCB",
        primary_metric="Index number",
        code="24364",
        provider=ProviderType.BCB,
        description="Índice de Atividade Econômica do Banco Central com ajuste sazonal.",
    ),
    "USD_BRL": IndicatorMeta(
        id="USD_BRL",
        name="USD/BRL (PTAX)",
        frequency=Frequency.DAILY,
        source_entity="BCB",
        primary_metric="BRL per USD",
        code="1",  # SGS 1: Câmbio livre - Dólar americano (venda)
        provider=ProviderType.BCB,
        description="Taxa de câmbio comercial do Dólar americano (PTAX venda).",
    ),
    "CURRENT_ACCOUNT": IndicatorMeta(
        id="CURRENT_ACCOUNT",
        name="Current Account",
        frequency=Frequency.MONTHLY,
        source_entity="BCB",
        primary_metric="USD (Millions)",
        code="22708",
        provider=ProviderType.BCB,
        description="Balanço de pagamentos - Transações correntes - Saldo mensal (US$ milhões).",
    ),
    "EMBI_BRASIL": IndicatorMeta(
        id="EMBI_BRASIL",
        name="EMBI+ Brasil",
        frequency=Frequency.DAILY,
        source_entity="J.P. Morgan / IPEA",
        primary_metric="Basis points (bps)",
        code="JPM366_EMBI366",  # Numeric ID 40940 in Ipeadata
        provider=ProviderType.IPEA,
        description="Indicador Risco-Brasil - EMBI+ Brasil (spread soberano em pontos base).",
    ),
}


class Indicator(str, Enum):
    SELIC_OVER = "SELIC_OVER"
    CDI = "CDI"
    CDI_DAILY = "CDI_DAILY"
    DI_FUTURES = "DI_FUTURES"
    IPCA = "IPCA"
    IPCA_15 = "IPCA_15"
    IGP_M = "IGP_M"
    IBC_BR = "IBC_BR"
    USD_BRL = "USD_BRL"
    CURRENT_ACCOUNT = "CURRENT_ACCOUNT"
    EMBI_BRASIL = "EMBI_BRASIL"


class DataCatalog:
    """Catalog helper to discover and inspect macroeconomic indicators."""

    @staticmethod
    def get(indicator: Indicator | str) -> IndicatorMeta:
        key = indicator.value if isinstance(indicator, Indicator) else str(indicator).upper()
        if key not in INDICATORS:
            raise KeyError(f"Unknown indicator '{indicator}'. Available: {list(INDICATORS.keys())}")
        return INDICATORS[key]

    @staticmethod
    def list_all() -> List[IndicatorMeta]:
        return list(INDICATORS.values())

    @staticmethod
    def to_dataframe(library: str = "pandas"):
        records = [
            {
                "Indicator": m.name,
                "Key": m.id,
                "Frequency": m.frequency.value,
                "Source Entity": m.source_entity,
                "Primary Metric / Unit": m.primary_metric,
                "SGS / Code": m.code,
                "Provider": m.provider.value,
            }
            for m in INDICATORS.values()
        ]
        if library == "polars":
            import polars as pl
            return pl.DataFrame(records)
        import pandas as pd
        return pd.DataFrame(records)
