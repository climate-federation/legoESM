"""Analytical SST and SIC forcing for idealized experiments.

Provides a Qobs-like SST profile with seasonal cycle and derived sea-ice
concentration.  Used by both cubed-sphere and spectral AMIP scripts.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core.precision import get_policy
from legoesm.forcing.time_utils import day_to_calendar


# --- APE "Qobs" profile (Neale & Hoskins 2000, Atmos. Sci. Lett. 1, 101) ---
_QOBS_PEAK_C = 27.0
_QOBS_EDGE_DEG = 60.0


def qobs_sst_C(lat_deg) -> np.ndarray:
    """Aqua-Planet Experiment "Qobs" SST [degC]: 27 (1 - (sin^2 + sin^4)/2)
    of (3 lat / 2) inside |lat| < 60 deg, 0 beyond: the mean of the APE
    "Control" 27 (1 - sin^2) and "Flat" 27 (1 - sin^4) profiles.  Zonally
    symmetric."""
    lat = np.asarray(lat_deg, dtype=np.float64)
    s2 = np.sin(np.radians(1.5 * lat)) ** 2
    return np.where(np.abs(lat) < _QOBS_EDGE_DEG,
                    _QOBS_PEAK_C * (1.0 - 0.5 * (s2 + s2 * s2)), 0.0)


def analytical_sst_sic(
    lat_deg: np.ndarray,
    day: float,
    T_ice: float = constants.T_freeze_ocean,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute SST and SIC for a given day using an analytical seasonal cycle.

    Parameters
    ----------
    lat_deg : array
        Latitude in degrees, any shape.
    day : float
        Simulation day (uses day % 365 for seasonal cycle).
    T_ice : float
        SST floor / SIC ramp threshold [K] — typically the seawater
        freezing point, ``constants.T_freeze_ocean`` (271.35 K).
        Despite the legacy name this is NOT the ice surface
        temperature.

    Returns
    -------
    sst : jax.Array
        Sea surface temperature [K], same shape as *lat_deg*.
    sic : jax.Array
        Sea-ice concentration [0, 1], same shape as *lat_deg*.
    """
    day_of_year, _ = day_to_calendar(day)
    lat_shift = -5.0 * np.cos(2.0 * np.pi * day_of_year / 365.0)
    lat_eff = np.asarray(lat_deg) - lat_shift

    # Qobs-like SST profile: warm equator, cold poles
    sst = 27.0 * (1.0 - np.sin(np.radians(lat_eff)) ** 2) + constants.T_freeze
    seasonal_amp = 3.0 * np.cos(np.radians(lat_eff)) * np.cos(
        2.0 * np.pi * day_of_year / 365.0
    )
    sst = sst + seasonal_amp
    # Floor SST at the seawater freezing depression below T_freeze_ocean.
    # The 1.8 K offset is the freshwater/seawater freezing-point
    # difference (constants.T_freeze - constants.T_freeze_ocean = 1.8 K).
    sst_min = T_ice - (constants.T_freeze - constants.T_freeze_ocean)
    sst = np.maximum(sst, sst_min)

    # SIC: ramp from 0 to 1 as SST drops below T_ice
    sic = np.clip(((T_ice + 0.5) - sst) / 3.0, 0.0, 1.0)

    _dtype = get_policy().storage
    return jnp.array(sst, dtype=_dtype), jnp.array(sic, dtype=_dtype)
