"""
Tests for Data Catalog, Cache, and Providers.
"""

import tempfile
from pathlib import Path
import pytest
import pandas as pd
import polars as pl
from conjunctural.data.catalog import DataCatalog, Indicator, INDICATORS
from conjunctural.data.cache import CacheManager
from conjunctural.data.bcb import BCBProvider
from conjunctural.data.ipea import IpeaProvider
from conjunctural.data.b3 import B3DIProvider, STANDARD_VERTICES_DU


def test_catalog_retrieval():
    assert len(INDICATORS) >= 10
    selic = DataCatalog.get("SELIC_OVER")
    assert selic.name == "Selic Over"
    assert selic.code == "1178"
    assert selic.source_entity == "BCB"

    embi = DataCatalog.get(Indicator.EMBI_BRASIL)
    assert embi.primary_metric == "Basis points (bps)"
    assert embi.provider.value == "ipea"


def test_catalog_dataframe():
    df_pd = DataCatalog.to_dataframe("pandas")
    assert isinstance(df_pd, pd.DataFrame)
    assert len(df_pd) >= 10

    df_pl = DataCatalog.to_dataframe("polars")
    assert isinstance(df_pl, pl.DataFrame)
    assert len(df_pl) >= 10


def test_cache_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CacheManager(cache_dir=Path(tmpdir))
        key = "test_series"

        df_sample = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5),
            "val": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        assert not cm.has(key)
        cm.put(key, df_sample)
        assert cm.has(key)

        retrieved_pd = cm.get(key)
        assert isinstance(retrieved_pd, pd.DataFrame)
        assert len(retrieved_pd) == 5

        retrieved_pl = cm.get(key, as_polars=True)
        assert isinstance(retrieved_pl, pl.DataFrame)
        assert len(retrieved_pl) == 5

        # Test date filtering
        filtered = cm.get(key, start_date="2023-01-03", end_date="2023-01-04")
        assert len(filtered) == 2


def test_bcb_code_resolution():
    bcb = BCBProvider()
    code, name = bcb._resolve_code_and_name("SELIC_OVER")
    assert code == "1178"
    assert name == "SELIC_OVER"

    code_num, name_num = bcb._resolve_code_and_name(433)
    assert code_num == "433"
    assert name_num == "IPCA"


def test_ipea_code_resolution():
    ipea = IpeaProvider()
    code, name = ipea._resolve_code_and_name("EMBI_BRASIL")
    assert code == "JPM366_EMBI366"
    assert name == "EMBI_BRASIL"


def test_b3_di_interpolation_synthetic():
    b3 = B3DIProvider()
    # Create synthetic DI curve with known rates:
    # 21 du: 10.0%, 252 du: 11.0%, 504 du: 12.0%
    curve = pd.DataFrame({
        "codigo_negociacao": ["DI1A", "DI1B", "DI1C"],
        "dias_uteis": [21, 252, 504],
        "taxa_ajuste": [0.10, 0.11, 0.12],
    })

    target_vertices = {"DI_1M": 21, "DI_1Y": 252, "DI_2Y": 504}
    interp = b3.interpolate_vertices(curve, target_vertices)
    assert pytest.approx(interp["DI_1M"], 1e-4) == 0.10
    assert pytest.approx(interp["DI_1Y"], 1e-4) == 0.11
    assert pytest.approx(interp["DI_2Y"], 1e-4) == 0.12
