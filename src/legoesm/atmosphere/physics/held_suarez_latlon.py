"""Held-Suarez forcing for the lat-lon dynamical core.

Reuses the grid-agnostic equilibrium temperature from held_suarez.py.
Provides init and forcing functions with lat-lon shapes and dims.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState, HydrostaticTendencies
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.vertical import SigmaCoordinate, pressure_from_sigma
from legoesm.atmosphere.physics.held_suarez import (
    held_suarez_equilibrium_temperature,
    K_A, K_S, K_F, SIGMA_B,
)


def held_suarez_forcing_latlon(
    state: HydrostaticState,
    grid: LatLonGrid,
    sigma_coord: SigmaCoordinate,
) -> HydrostaticTendencies:
    """Compute Held-Suarez physics tendencies on a lat-lon grid.

    Parameters
    ----------
    state : HydrostaticState
        Current model state.
    grid : LatLonGrid
        Horizontal grid (provides latitude).
    sigma_coord : SigmaCoordinate
        Vertical coordinate.

    Returns
    -------
    HydrostaticTendencies : Physics forcing tendencies.
    """
    u = state.u.data       # (n_lat, n_lon, nlev)
    v = state.v.data
    T = state.T.data
    p_s = state.p_s.data   # (n_lat, n_lon)

    sigma_full = sigma_coord.sigma_full  # (nlev,)
    lat = grid.lat  # (n_lat,)

    # Pressure at full levels
    p_full = pressure_from_sigma(sigma_full, p_s)  # (n_lat, n_lon, nlev)

    # Equilibrium temperature
    # lat (n_lat,) -> broadcast to (n_lat, 1, 1) for (n_lat, n_lon, nlev)
    T_eq = held_suarez_equilibrium_temperature(
        lat[:, None, None], p_full
    )

    # Temperature relaxation coefficient k_T(σ, φ)
    sigma_factor = jnp.maximum(
        0.0, (sigma_full[None, None, :] - SIGMA_B) / (1.0 - SIGMA_B)
    )  # (1, 1, nlev)
    cos_lat_4 = jnp.cos(lat)**4  # (n_lat,)

    k_T = K_A + (K_S - K_A) * sigma_factor * cos_lat_4[:, None, None]

    # Newtonian relaxation
    dT_dt_phys = -k_T * (T - T_eq)

    # Rayleigh friction coefficient k_v(σ)
    k_v = K_F * jnp.maximum(
        0.0, (sigma_full[None, None, :] - SIGMA_B) / (1.0 - SIGMA_B)
    )

    du_dt_phys = -k_v * u
    dv_dt_phys = -k_v * v

    # Build tendency pytree
    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")

    return HydrostaticTendencies(
        du_dt=Field(data=du_dt_phys, name="du_dt_phys", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_dt_phys, name="dv_dt_phys", dims=dims_3d, units="m/s^2"),
        dT_dt=Field(data=dT_dt_phys, name="dT_dt_phys", dims=dims_3d, units="K/s"),
        dp_s_dt=Field(
            data=jnp.zeros_like(p_s), name="dp_s_dt_phys", dims=dims_2d, units="Pa/s"
        ),
        dphis_dt=Field(
            data=jnp.zeros_like(p_s), name="dphis_dt_phys", dims=dims_2d, units="m^2/s^3"
        ),
    )


def held_suarez_init_latlon(
    grid: LatLonGrid,
    sigma_coord: SigmaCoordinate,
    T_init: float = 300.0,
    p_s_init: float = 1.0e5,
    perturbation_amplitude: float = 1.0,
    seed: int = 42,
) -> HydrostaticState:
    """Create initial conditions for the Held-Suarez test on a lat-lon grid.

    Isothermal atmosphere at rest with a small random temperature
    perturbation at the lowest level to break symmetry.

    Parameters
    ----------
    grid : LatLonGrid
        Horizontal grid.
    sigma_coord : SigmaCoordinate
        Vertical coordinate.
    T_init : float
        Initial temperature [K].
    p_s_init : float
        Initial surface pressure [Pa].
    perturbation_amplitude : float
        Amplitude of temperature perturbation [K].
    seed : int
        Random seed for perturbation.

    Returns
    -------
    HydrostaticState : Initial state.
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = sigma_coord.n_levels
    shape_3d = (n_lat, n_lon, nlev)
    shape_2d = (n_lat, n_lon)
    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")

    # Uniform temperature
    T_data = jnp.ones(shape_3d) * T_init

    # Small perturbation at lowest level
    key = jax.random.PRNGKey(seed)
    perturbation = jax.random.normal(key, shape_2d) * perturbation_amplitude
    T_data = T_data.at[:, :, -1].add(perturbation)

    return HydrostaticState(
        u=Field(data=jnp.zeros(shape_3d), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros(shape_3d), name="v", dims=dims_3d, units="m/s"),
        T=Field(data=T_data, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=jnp.ones(shape_2d) * p_s_init, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=jnp.zeros(shape_2d), name="phis", dims=dims_2d, units="m^2/s^2"),
    )
