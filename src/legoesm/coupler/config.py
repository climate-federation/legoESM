"""Coupler configuration."""

from __future__ import annotations

from typing import NamedTuple

import jax

from legoesm.surface_albedo import OceanAlbedoConfig


class TileConfig(NamedTuple):
    """Surface tile fraction configuration.

    f_land and f_lake are static masks from config/bathymetry, shape (6, n, n).
    Water fraction = 1 - f_land - f_lake is split into ocean and ice
    prognostically.
    """
    f_land: jax.Array      # Land fraction [0-1], shape (6, n, n)
    f_lake: jax.Array      # Lake fraction [0-1], shape (6, n, n)


class CouplerConfig(NamedTuple):
    """Configuration for the surface coupler."""
    coupling_dt: float = 3600.0       # Coupling interval [s]
    U_min: float = 1.0                # Minimum wind speed floor [m/s]
    ocean_albedo: float = 0.06        # Fallback constant ocean albedo
    ocean_emissivity: float = 0.97    # Default ocean emissivity
    ocean_z0: float = 1e-4            # Ocean roughness length [m]
    co2_ppmv_default: float = 400.0   # Default CO2 concentration
    blend_sharpness: float = 20.0     # Sigmoid sharpness for tile blending
    Cd_ocean: float = 1.5e-3          # Ocean drag coefficient
    Ch_ocean: float = 1.5e-3          # Ocean heat transfer coefficient
    bulk_scheme: str = "constant"     # "constant", "coare3", "large_yeager"
    z_ref: float = 10.0               # Wind reference height [m]
    # Air temperature / specific humidity reference heights. Default to
    # z_ref for legacy single-height callers (lake, idealized adapter,
    # AMIP-style runs that read from the lowest atm level). For OMIP /
    # JRA55-do, set both to 2.0 — the reanalysis delivers ``tas`` and
    # ``huss`` at 2 m while ``uas, vas`` are at 10 m.
    z_t_atm: float = 10.0             # Air-temperature reference height [m]
    z_q_atm: float = 10.0             # Specific-humidity reference height [m]
    bulk_n_iter: int = 5              # MOST iterations (coare3/large_yeager)
    # Zenith-dependent ocean albedo (Task 10)
    ocean_albedo_config: OceanAlbedoConfig = OceanAlbedoConfig()
