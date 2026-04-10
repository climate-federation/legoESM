"""Multi-layer soil heat diffusion (Task 8D).

Solves the 1D heat equation:
    C_eff(z) · ∂T/∂t = ∂/∂z [k_eff(z) · ∂T/∂z]

Thermal properties depend on soil moisture via the Johansen (1975) method.
Discretized with backward Euler and solved via Thomas algorithm.

References
----------
- Johansen (1975): Thermal conductivity of soils. PhD thesis.
- de Vries (1963): Thermal properties of soils.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm.land.soil_grid import SoilGrid
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig
from legoesm.land.tridiag import thomas_solve_batch


class SoilThermalConfig(NamedTuple):
    """Configuration for soil thermal properties."""
    C_soil: float = 2.0e6         # mineral soil heat capacity [J/m3/K]
    C_water: float = 4.18e6       # water heat capacity [J/m3/K]
    C_air: float = 1.25e3         # air heat capacity [J/m3/K]
    k_solid: float = 2.0          # mineral soil thermal conductivity [W/m/K]
    k_water: float = 0.57         # water thermal conductivity [W/m/K]
    rho_bulk: float = 1400.0      # bulk density [kg/m3]
    soil_texture: str = "loam"    # "sand" (coarse) or "loam" (fine)
    Q_geothermal: float = 0.05   # Geothermal heat flux at bottom [W/m2]


def compute_heat_capacity(
    theta: jnp.ndarray,
    hydro_config: SoilHydraulicsConfig,
    thermal_config: SoilThermalConfig,
) -> jnp.ndarray:
    """Compute effective volumetric heat capacity [J/m3/K].

    C_eff = (1 - θ_sat)·C_soil + θ·C_water + (θ_sat - θ)·C_air
    """
    theta_sat = hydro_config.theta_sat
    return ((1.0 - theta_sat) * thermal_config.C_soil
            + theta * thermal_config.C_water
            + (theta_sat - theta) * thermal_config.C_air)


def compute_thermal_conductivity(
    theta: jnp.ndarray,
    hydro_config: SoilHydraulicsConfig,
    thermal_config: SoilThermalConfig,
) -> jnp.ndarray:
    """Compute effective thermal conductivity [W/m/K].

    Uses Johansen (1975):
        k_eff = k_dry + (k_sat - k_dry) · K_e(Sr)

    where K_e is the Kersten number.
    """
    theta_sat = hydro_config.theta_sat
    theta_r = hydro_config.theta_r

    # Saturation ratio
    Sr = jnp.clip((theta - theta_r) / (theta_sat - theta_r + 1e-10), 0.01, 1.0)

    # Dry conductivity (de Vries 1963)
    rho_b = thermal_config.rho_bulk
    k_dry = (0.135 * rho_b + 64.7) / (2700.0 - 0.947 * rho_b)

    # Saturated conductivity (geometric mean)
    k_sat = (thermal_config.k_solid ** (1.0 - theta_sat)
             * thermal_config.k_water ** theta_sat)

    # Kersten number
    if thermal_config.soil_texture == "sand":
        K_e = 0.7 * jnp.log10(jnp.clip(Sr, 0.05, None)) + 1.0
    else:
        K_e = jnp.log10(jnp.clip(Sr, 0.1, None)) + 1.0
    K_e = jnp.clip(K_e, 0.0, 1.0)

    return k_dry + (k_sat - k_dry) * K_e


def solve_soil_thermal(
    T_soil: jnp.ndarray,
    theta: jnp.ndarray,
    grid: SoilGrid,
    hydro_config: SoilHydraulicsConfig,
    thermal_config: SoilThermalConfig,
    G_surface: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Solve soil heat diffusion for one time step (backward Euler).

    Parameters
    ----------
    T_soil : jnp.ndarray
        Soil temperature [K], shape (ncol, n_layers).
    theta : jnp.ndarray
        Volumetric water content [m3/m3], shape (ncol, n_layers).
    grid : SoilGrid
        Vertical soil grid.
    hydro_config : SoilHydraulicsConfig
        For theta_sat, theta_r.
    thermal_config : SoilThermalConfig
        Thermal property parameters.
    G_surface : jnp.ndarray
        Ground heat flux into top layer [W/m2], shape (ncol,).
        Positive = into soil.
    dt : float
        Time step [s].

    Returns
    -------
    T_new : jnp.ndarray
        Updated soil temperature [K], shape (ncol, n_layers).
    """
    ncol, nlayers = T_soil.shape
    dz = grid.dz                  # (nlayers,)
    dz_if = grid.dz_interface     # (nlayers-1,)

    # Compute thermal properties
    C_eff = compute_heat_capacity(theta, hydro_config, thermal_config)    # (ncol, nlayers)
    k_eff = compute_thermal_conductivity(theta, hydro_config, thermal_config)  # (ncol, nlayers)

    # Interface conductivity (harmonic mean for heat diffusion)
    k_half = 2.0 * k_eff[:, :-1] * k_eff[:, 1:] / (
        k_eff[:, :-1] + k_eff[:, 1:] + 1e-20
    )  # (ncol, nlayers-1)

    # Diffusion coefficient at interfaces
    coeff = k_half / dz_if  # (ncol, nlayers-1)

    # Build tridiagonal system for backward Euler:
    # C_eff * dz * (T_new - T_old) / dt = diffusion operator on T_new + source

    # Diagonal
    diag = C_eff * dz / dt
    diag = diag.at[:, 1:].add(coeff)
    diag = diag.at[:, :-1].add(coeff)

    # Sub-diagonal (lower)
    sub = -coeff  # (ncol, nlayers-1)

    # Super-diagonal (upper)
    sup = -coeff  # (ncol, nlayers-1)

    # RHS
    rhs = C_eff * dz * T_soil / dt

    # Top BC: ground heat flux
    rhs = rhs.at[:, 0].add(G_surface)

    # Bottom BC: geothermal heat flux (Neumann, positive into soil)
    rhs = rhs.at[:, -1].add(thermal_config.Q_geothermal)

    # Assemble full arrays
    a = jnp.zeros((ncol, nlayers))
    a = a.at[:, 1:].set(sub)
    c = jnp.zeros((ncol, nlayers))
    c = c.at[:, :-1].set(sup)

    T_new = thomas_solve_batch(a, diag, c, rhs)
    return T_new
