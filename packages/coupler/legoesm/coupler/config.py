"""Coupler configuration."""

from __future__ import annotations

from typing import NamedTuple

import jax

from legoesm.surface_albedo import OceanAlbedoConfig

__param_spec__ = {
    "CouplerConfig": {
        "scheme_key": "coupler.surface",
        "excluded": {
            "U_min": "numerics: minimum wind-speed floor for bulk fluxes [m/s]",
            "co2_ppmv_default": "forcing: default atmospheric CO2 [ppmv]",
            "coupling_dt": "numerics: coupling timestep [s]",
            "z_q_atm": "convention: humidity measurement reference height [m]",
            "z_ref": "convention: reference height [m]",
            "z_t_atm": "convention: temperature measurement reference height [m]",
        },
        "params": {
            "Cd_ocean": {
                "units": "1", "bounds": (5.0e-4, 5.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "ocean-atmosphere momentum drag coefficient", "shape": None,
            },
            "Ch_ocean": {
                "units": "1", "bounds": (5.0e-4, 5.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "ocean-atmosphere heat transfer coefficient", "shape": None,
            },
            "ocean_albedo": {
                "units": "1", "bounds": (0.03, 0.15), "tunable_tier": 1,
                "transform": "sigmoid", "category": "radiation",
                "reference": "open-ocean broadband albedo", "shape": None,
            },
            "ocean_emissivity": {
                "units": "1", "bounds": (0.9, 1.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "radiation",
                "reference": "open-ocean longwave emissivity", "shape": None,
            },
            "ocean_z0": {
                "units": "m", "bounds": (1.0e-5, 1.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "open-ocean aerodynamic roughness length", "shape": None,
            },
            "gustiness": {
                "units": "m s-1", "bounds": (0.0, 10.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "sub-grid convective gustiness floor (Wing 2018 RCEMIP1 / Beljaars 1995)", "shape": None,
            },
        },
    },
}


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
    U_min: float = 1.0                # Numerical wind speed floor [m/s]
    # Sub-grid convective gustiness floor [m/s] for the air-sea bulk fluxes:
    # |U|_eff = sqrt(|U|^2 + gustiness^2).  Boosts surface evaporation in
    # light-wind/convective regions (resolved grid-mean wind misses BL
    # gustiness); 5 m/s = Wing (2018) RCEMIP1, matching the SCM. Subsumes U_min.
    gustiness: float = 5.0
    ocean_albedo: float = 0.06        # Fallback constant ocean albedo
    ocean_emissivity: float = 0.97    # Default ocean emissivity
    ocean_z0: float = 1e-4            # Ocean roughness length [m]
    co2_ppmv_default: float = 400.0   # Default CO2 concentration
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
