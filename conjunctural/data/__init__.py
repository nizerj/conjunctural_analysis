"""
Data ingestion layer for Conjunctural.
"""

from conjunctural.data.catalog import Indicator, INDICATORS, DataCatalog, Frequency, ProviderType
from conjunctural.data.cache import CacheManager
from conjunctural.data.bcb import BCBProvider
from conjunctural.data.ipea import IpeaProvider
from conjunctural.data.b3 import B3DIProvider, STANDARD_VERTICES_DU
from conjunctural.data.engine import DataEngine

__all__ = [
    "Indicator",
    "INDICATORS",
    "DataCatalog",
    "Frequency",
    "ProviderType",
    "CacheManager",
    "BCBProvider",
    "IpeaProvider",
    "B3DIProvider",
    "STANDARD_VERTICES_DU",
    "DataEngine",
]
