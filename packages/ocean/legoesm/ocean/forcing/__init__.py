"""Ocean atmospheric forcing loaders for climate-scale simulations.

Modules:

* ``jra55_do`` -- JRA55-do reanalysis (Tsujino et al. 2018 / 2020 OMIP-2
  standard); inter-annual forcing 1958-present.
* ``core2`` -- CORE-II Normal-Year Forcing (Large & Yeager 2009),
  perpetual-year climatology for Bryan-style THC spinups.

Both expose ``load_*(year, cache_dir=None)`` returning an
``OceanForcing`` NamedTuple of the seven OMIP-2 channels:

* ``u10``, ``v10``  -- 10 m wind components [m/s]
* ``T_air``         -- 2 m air temperature [K]
* ``q_air``         -- 2 m specific humidity [kg/kg]
* ``sw_down``       -- downward shortwave radiation [W/m^2]
* ``lw_down``       -- downward longwave radiation [W/m^2]
* ``precip``        -- precipitation flux [kg/m^2/s]
* ``runoff``        -- continental runoff flux [kg/m^2/s]

The loaders fall back to a deterministic synthetic-climatology
generator when the on-disk NetCDF / zarr cache is missing so the
matrix smoke tests do not require ~50 GB of JRA55-do data.
"""

from __future__ import annotations

from .jra55_do import OceanForcing, load_jra55_do, synthetic_ocean_forcing
from .core2 import core2_nyf_cache_dir, load_core2_nyf
from .woa import load_woa_sst, synthetic_woa_sst
from .woa_sss import load_woa_sss, synthetic_woa_sss
from .dai_trenberth import (
    RiverRunoffData,
    load_dai_trenberth,
    synthetic_dai_trenberth,
    project_runoff_to_grid,
)
from .sss_restoring import (
    SSSRestoringConfig,
    RegionMaskSpec,
    DEFAULT_OMIP2_REGIONS,
    build_region_masks,
    compute_sss_restoring_flux,
    interp_woa_sss_to_grid,
)

__all__ = [
    "OceanForcing",
    "load_jra55_do",
    "core2_nyf_cache_dir",
    "load_core2_nyf",
    "load_woa_sst",
    "load_woa_sss",
    "synthetic_ocean_forcing",
    "synthetic_woa_sst",
    "synthetic_woa_sss",
    "RiverRunoffData",
    "load_dai_trenberth",
    "synthetic_dai_trenberth",
    "project_runoff_to_grid",
    "SSSRestoringConfig",
    "RegionMaskSpec",
    "DEFAULT_OMIP2_REGIONS",
    "build_region_masks",
    "compute_sss_restoring_flux",
    "interp_woa_sss_to_grid",
]
