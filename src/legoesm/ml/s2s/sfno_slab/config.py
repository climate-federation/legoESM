"""Shared configuration for the SFNO subseasonal slab-ocean workflow.

The defaults here define the canonical local setup:
- ChaosBench daily atmosphere files as the training target source
- ARCO-prepared daily SST cache as the ocean forcing source
- the five-variable atmospheric target set ``z,q,t,u,v``
"""

from __future__ import annotations

import os
from typing import NamedTuple


CHAOSBENCH_DATA_DIR = os.environ.get("CHAOSBENCH_DATA_DIR", "data/ChaosBench")
CHAOSBENCH_PRESSURE_LEVELS = (10, 50, 100, 200, 300, 500, 700, 850, 925, 1000)
CHAOSBENCH_ATMOS_VARS = ("z", "q", "t", "u", "v")
DEFAULT_ARCO_SST_CACHE_PATH = os.environ.get(
    "ARCO_SST_CACHE_PATH",
    "data/arco_sst_cache/arco_sst_daily_19790101_20231231.zarr",
)
DEFAULT_ARCO_SST_STATS_PATH = os.environ.get(
    "ARCO_SST_STATS_PATH",
    "data/arco_sst_cache/arco_sst_daily_19790101_20231231_stats.zarr",
)
LAND_SEA_MASK_VAR = "land_sea_mask"


class ChaosBenchS2SConfig(NamedTuple):
    """Configuration for SFNO training and inference on daily ChaosBench-style files."""

    years: tuple[int, ...]
    data_dir: str = CHAOSBENCH_DATA_DIR
    atmosphere_vars: tuple[str, ...] = CHAOSBENCH_ATMOS_VARS
    pressure_levels: tuple[int, ...] = CHAOSBENCH_PRESSURE_LEVELS
    land_vars: tuple[str, ...] = ()
    ocean_vars: tuple[str, ...] = ("sosstsst",)
    ocean_source: str = "oras5"
    arco_sst_cache_path: str = DEFAULT_ARCO_SST_CACHE_PATH
    arco_sst_stats_path: str = DEFAULT_ARCO_SST_STATS_PATH
    n_steps: int = 42
    lead_time: int = 1
    normalize: bool = True
    gaussian_n_max: int = 79
    latitude_name: str = "latitude"
    longitude_name: str = "longitude"
    level_name: str = "level"
