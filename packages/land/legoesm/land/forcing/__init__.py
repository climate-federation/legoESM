"""Land atmospheric-forcing readers (TRENDY / LMIP).

Currently: CRU-JRA (CLM datm format) for forced land-only runs.  See
``docs/land/lmip_s3_scope.md`` for the workplan.
"""

from legoesm.land.forcing.cru_jra import (
    CRUJRA_FREQ_HOURS,
    CRUJRA_NLAT,
    CRUJRA_NLON,
    LandForcing,
    LandForcingColumns,
    build_forcing_weights,
    disaggregate_forcing,
    forcing_to_atm_surface,
    load_cru_jra,
    read_crujra_year,
    regrid_forcing,
    stage_forcing,
    stage_forcing_years,
    synthetic_land_forcing,
)
from legoesm.land.forcing.solar import cos_solar_zenith

__all__ = [
    "CRUJRA_FREQ_HOURS",
    "CRUJRA_NLAT",
    "CRUJRA_NLON",
    "LandForcing",
    "LandForcingColumns",
    "build_forcing_weights",
    "cos_solar_zenith",
    "disaggregate_forcing",
    "forcing_to_atm_surface",
    "load_cru_jra",
    "read_crujra_year",
    "regrid_forcing",
    "stage_forcing",
    "stage_forcing_years",
    "synthetic_land_forcing",
]
