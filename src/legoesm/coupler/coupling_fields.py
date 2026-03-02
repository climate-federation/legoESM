"""Fixed-slot coupling field containers.

All fields pre-allocated with fixed shape — no dicts, no Optional types.
These NamedTuples define the strict interface between atmosphere and surface.
"""

from __future__ import annotations

from typing import NamedTuple

import jax


class AtmToSurface(NamedTuple):
    """Atmosphere -> surface coupling fields. Shape (6, n, n)."""
    sw_down: jax.Array           # Downward shortwave [W/m2]
    lw_down: jax.Array           # Downward longwave [W/m2]
    precip_total: jax.Array      # Total precipitation rate [kg/m2/s]
    precip_snow: jax.Array       # Snow precipitation rate [kg/m2/s]
    T_lowest: jax.Array          # Lowest-level temperature [K]
    q_lowest: jax.Array          # Lowest-level specific humidity [kg/kg]
    u_lowest: jax.Array          # Lowest-level zonal wind [m/s]
    v_lowest: jax.Array          # Lowest-level meridional wind [m/s]
    p_lowest: jax.Array          # Lowest-level pressure [Pa]
    p_surface: jax.Array         # Surface pressure [Pa]
    rho_lowest: jax.Array        # Lowest-level air density [kg/m3]
    cos_zenith: jax.Array        # Cosine solar zenith angle
    co2_ppmv: jax.Array          # CO2 concentration [ppmv]
    has_radiation: jax.Array     # 1.0 = radiation fields valid, 0.0 = not
    has_precipitation: jax.Array # 1.0 = precip fields valid, 0.0 = not


class TileResponse(NamedTuple):
    """Per-tile surface response. Identical slots to SurfaceToAtm."""
    T_surface: jax.Array         # Surface skin temperature [K]
    albedo: jax.Array            # Surface albedo [0-1]
    emissivity: jax.Array        # Surface emissivity [0-1]
    z0: jax.Array                # Roughness length [m]
    q_surface: jax.Array         # Surface specific humidity [kg/kg]
    shflx: jax.Array             # Sensible heat flux [W/m2] (positive up)
    lhflx: jax.Array             # Latent heat flux [W/m2] (positive up)
    tau_x: jax.Array             # Zonal surface stress [Pa]
    tau_y: jax.Array             # Meridional surface stress [Pa]
    lw_up: jax.Array             # Upward longwave [W/m2]
    u_ocean_sfc: jax.Array       # Ocean surface zonal current [m/s]
    v_ocean_sfc: jax.Array       # Ocean surface meridional current [m/s]
    co2_flux: jax.Array          # CO2 flux [kg/m2/s] (positive up)


class SurfaceToAtm(NamedTuple):
    """Blended surface -> atmosphere coupling fields. Shape (6, n, n)."""
    T_surface: jax.Array
    albedo: jax.Array
    emissivity: jax.Array
    z0: jax.Array
    q_surface: jax.Array
    shflx: jax.Array
    lhflx: jax.Array
    tau_x: jax.Array
    tau_y: jax.Array
    lw_up: jax.Array
    u_ocean_sfc: jax.Array
    v_ocean_sfc: jax.Array
    co2_flux: jax.Array
