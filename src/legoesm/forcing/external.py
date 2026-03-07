"""External forcing interfaces for AMIP/CMIP-style experiments.

Provides scaffolded configuration and interpolation for time-varying
external forcings beyond SST/SIC:

- Greenhouse gas concentrations (CO2, CH4, N2O, CFCs)
- Ozone climatology / forcing
- Aerosol optical depth climatology
- Solar irradiance variations

Status
------
- GHG: config + constant-value mode active; time-varying from file is scaffolded
- Ozone: fully scaffolded (placeholder)
- Aerosol: fully scaffolded (placeholder)
- Solar: seasonal cycle already active in gray radiation; TSI variation scaffolded

These interfaces are designed so that physics parameterizations can query
them without knowing whether the forcing is constant, prescribed from
a file, or computed interactively.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


# ==============================================================================
# Greenhouse gases
# ==============================================================================

class GHGConfig(NamedTuple):
    """Greenhouse gas configuration.

    Currently used to inform radiation (e.g., scaling LW optical depth).
    The gray radiation does not use GHG concentrations directly, but this
    config provides the interface for future multi-band or correlated-k
    radiation schemes.

    Fields
    ------
    co2_ppmv : float
        CO2 concentration [ppmv]. Default: 348 (AMIP II reference ~1979-1996).
    ch4_ppbv : float
        CH4 concentration [ppbv].
    n2o_ppbv : float
        N2O concentration [ppbv].
    source : str
        "constant" (use values above) or "file" (load from NetCDF).
    path : str
        Path to time-varying GHG file (only used if source="file").
    """
    co2_ppmv: float = 348.0
    ch4_ppbv: float = 1650.0
    n2o_ppbv: float = 306.0
    source: str = "constant"
    path: str = ""


def get_ghg_at_time(config: GHGConfig, day: float) -> dict:
    """Return GHG concentrations at a given simulation day.

    Parameters
    ----------
    config : GHGConfig
    day : float
        Simulation day (unused for constant source).

    Returns
    -------
    dict with keys "co2_ppmv", "ch4_ppbv", "n2o_ppbv"
    """
    if config.source == "constant":
        return {
            "co2_ppmv": config.co2_ppmv,
            "ch4_ppbv": config.ch4_ppbv,
            "n2o_ppbv": config.n2o_ppbv,
        }
    elif config.source == "file":
        raise NotImplementedError(
            "Time-varying GHG from file is not yet implemented. "
            "Provide a GHG NetCDF file and implement interpolation in "
            "legoesm/forcing/external.py:get_ghg_at_time()."
        )
    else:
        raise ValueError(f"Unknown GHG source: {config.source!r}")


# ==============================================================================
# Ozone
# ==============================================================================

class OzoneConfig(NamedTuple):
    """Ozone forcing configuration.

    Placeholder for prescribed ozone climatology or forcing.
    When active, provides column ozone or 3D ozone mixing ratio
    for radiation calculations.

    Status: fully scaffolded. Not yet connected to radiation.

    Fields
    ------
    enabled : bool
        Whether ozone forcing is active.
    source : str
        "climatology" (monthly zonal-mean) or "file" (full 3D).
    path : str
        Path to ozone NetCDF file.
    """
    enabled: bool = False
    source: str = "climatology"
    path: str = ""


def get_ozone_at_time(config: OzoneConfig, day: float, grid=None, sigma=None):
    """Return ozone field at a given simulation day.

    Returns
    -------
    None if ozone is disabled, otherwise raises NotImplementedError.
    """
    if not config.enabled:
        return None
    raise NotImplementedError(
        "Ozone forcing is scaffolded but not yet implemented. "
        "Provide an ozone climatology NetCDF and implement interpolation in "
        "legoesm/forcing/external.py:get_ozone_at_time()."
    )


# ==============================================================================
# Aerosols
# ==============================================================================

class AerosolConfig(NamedTuple):
    """Aerosol forcing configuration.

    Placeholder for prescribed aerosol optical depth climatology.
    When active, modifies SW radiation via column AOD.

    Status: fully scaffolded. Not yet connected to radiation.

    Fields
    ------
    enabled : bool
        Whether aerosol forcing is active.
    source : str
        "climatology" or "file".
    path : str
        Path to aerosol NetCDF file.
    """
    enabled: bool = False
    source: str = "climatology"
    path: str = ""


def get_aerosol_at_time(config: AerosolConfig, day: float, grid=None):
    """Return aerosol optical depth at a given simulation day.

    Returns
    -------
    None if aerosol is disabled, otherwise raises NotImplementedError.
    """
    if not config.enabled:
        return None
    raise NotImplementedError(
        "Aerosol forcing is scaffolded but not yet implemented. "
        "Provide an aerosol climatology NetCDF and implement interpolation in "
        "legoesm/forcing/external.py:get_aerosol_at_time()."
    )


# ==============================================================================
# Solar irradiance
# ==============================================================================

class SolarConfig(NamedTuple):
    """Solar irradiance configuration.

    The seasonal cycle is already handled by the gray radiation's
    daily_mean_insolation(). This config adds support for TSI variations
    (e.g., solar cycle, volcanic dimming).

    Status: constant TSI active; time-varying TSI scaffolded.

    Fields
    ------
    S_0 : float
        Total solar irradiance [W/m2]. Default: 1360.
    source : str
        "constant" or "file" (time-varying TSI from NetCDF).
    path : str
        Path to TSI time series file.
    """
    S_0: float = 1360.0
    source: str = "constant"
    path: str = ""


def get_tsi_at_time(config: SolarConfig, day: float) -> float:
    """Return total solar irradiance at a given simulation day.

    Parameters
    ----------
    config : SolarConfig
    day : float

    Returns
    -------
    float : TSI [W/m2]
    """
    if config.source == "constant":
        return config.S_0
    elif config.source == "file":
        raise NotImplementedError(
            "Time-varying TSI from file is not yet implemented."
        )
    else:
        raise ValueError(f"Unknown solar source: {config.source!r}")


# ==============================================================================
# Combined external forcing config
# ==============================================================================

class ExternalForcingConfig(NamedTuple):
    """Combined configuration for all external forcings.

    Pass this to the AMIP experiment to enable/configure external forcings.
    All sub-configs default to inactive/constant, so the baseline AMIP
    experiment works without any external forcing files.
    """
    ghg: GHGConfig = GHGConfig()
    ozone: OzoneConfig = OzoneConfig()
    aerosol: AerosolConfig = AerosolConfig()
    solar: SolarConfig = SolarConfig()
