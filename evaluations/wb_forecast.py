"""WeatherBench-2 forecast scorer for legoESM spectral dycores.

Stage 0 bridge: turn a rolled-out ``SpectralHydrostaticState`` into the WB2
headline-variable set on the model grid, so a trained dycore can be scored with
the (reused) ``evaluations.metrics`` RMSE/ACC/bias primitives.

Task 5 (this file, first piece): ``diagnose_headline_fields``.

Surface-variable caveat: the spectral DYNAMICAL state carries no skin
temperature, roughness length, or Obukhov length. Pressure-level fields
(z500, t850, q700, u/v at 850/700/500/250) and MSLP are diagnosed rigorously;
the surface fields are documented proxies (see ``diagnose_headline_fields``).
Total precipitation is not available from the dynamical state (it needs
physics-output accumulation) and is intentionally omitted here.
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants

from .headline_diagnostics import (
    geopotential_height_at,
    geopotential_on_levels,
    interp_to_pressure_level,
    mean_sea_level_pressure,
    wind_10m,
)

__all__ = ["diagnose_headline_fields", "HEADLINE_FIELD_KEYS"]

# --- WB2 headline pressure levels (Pa) ---
_Z500_PA = 50000.0
_T850_PA = 85000.0
_Q700_PA = 70000.0
_WIND_LEVELS_PA = (("850", 85000.0), ("700", 70000.0), ("500", 50000.0), ("250", 25000.0))

# --- surface-proxy constants ---
_Z0M_SURFACE_PROXY = 2e-4   # open-ocean momentum roughness [m] for the neutral 10 m reduction
_L_NEUTRAL = 1e12           # Obukhov length standing in for neutral stability [m]
_Z_10M = 10.0               # WMO anemometer reference height [m]

HEADLINE_FIELD_KEYS = (
    "z500", "t850", "q700",
    "u850", "v850", "u700", "v700", "u500", "v500", "u250", "v250",
    "mslp", "t2m", "u10", "v10", "wind_speed_10m",
)


def diagnose_headline_fields(state, grid, sigma_coord):
    """Map a rolled-out ``SpectralHydrostaticState`` to WB2 headline fields.

    Returns a ``dict[str, (n_lat, n_lon) array]`` keyed by
    :data:`HEADLINE_FIELD_KEYS`.

    Rigorous (pressure-level) fields: ``z500`` (geopotential height, m),
    ``t850`` (K), ``q700`` (kg/kg), ``u/v`` at 850/700/500/250 hPa (m/s),
    ``mslp`` (Pa).

    Surface proxies (the spectral state has no skin-T / roughness / Obukhov L):
      * ``t2m`` := lowest-level air temperature.
      * ``u10``/``v10``/``wind_speed_10m`` := the lowest-level wind reduced to
        10 m by a NEUTRAL log law, using the lowest-level height above ground
        from the geopotential and an open-terrain roughness. This is a
        documented global-neutral proxy; it does not resolve land/ocean
        roughness or stability, so surface-wind scores carry that caveat.
    """
    from legoesm.atmosphere.dynamics.spectral_pe import spectral_pe_to_grid

    fields = spectral_pe_to_grid(state, grid, sigma_coord)
    T, u, v = fields["T"], fields["u"], fields["v"]     # (n_lat, n_lon, nlev)
    p_s, phis = fields["p_s"], fields["phis"]           # (n_lat, n_lon)

    tracers = getattr(state, "tracers", None)
    if tracers is not None and "q_v" in tracers:
        q = tracers["q_v"].data                         # (n_lat, n_lon, nlev)
    else:
        q = jnp.zeros_like(T)

    p_model = sigma_coord.pressure_at_full(p_s)         # (n_lat, n_lon, nlev), Pa

    out = {}
    out["z500"] = geopotential_height_at(T, q, p_s, phis, sigma_coord, _Z500_PA)
    out["t850"] = interp_to_pressure_level(T, p_model, _T850_PA)
    out["q700"] = interp_to_pressure_level(q, p_model, _Q700_PA)
    for name, pa in _WIND_LEVELS_PA:
        out[f"u{name}"] = interp_to_pressure_level(u, p_model, pa)
        out[f"v{name}"] = interp_to_pressure_level(v, p_model, pa)
    out["mslp"] = mean_sea_level_pressure(p_s, T[..., -1], phis)

    # --- surface proxies ---
    out["t2m"] = T[..., -1]                             # lowest-level air T (proxy)
    phi = geopotential_on_levels(T, q, p_s, phis, sigma_coord)
    # lowest-level height above ground [m]; floor at the 10 m ref so the
    # log-law bracket z_ref <= z_low stays valid.
    z_low = jnp.maximum((phi[..., -1] - phis) / constants.g, _Z_10M)
    L = jnp.full_like(z_low, _L_NEUTRAL)
    z0m = jnp.full_like(z_low, _Z0M_SURFACE_PROXY)
    speed10 = wind_10m(u[..., -1], v[..., -1], z_low, L, z0m)
    speed_low = jnp.sqrt(u[..., -1] ** 2 + v[..., -1] ** 2 + 1e-12)
    scale = speed10 / speed_low                         # in [0, 1]
    out["u10"] = u[..., -1] * scale
    out["v10"] = v[..., -1] * scale
    out["wind_speed_10m"] = speed10
    return out
