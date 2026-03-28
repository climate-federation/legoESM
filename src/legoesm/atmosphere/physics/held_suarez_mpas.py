"""Held-Suarez forcing for the MPAS Voronoi mesh dynamical core.

Mirrors the cubed-sphere Held-Suarez (held_suarez.py) but uses MPAS
state/tendency types with edge-normal velocity on a C-grid.

References
----------
- Held, I. M., & Suarez, M. J. (1994). A Proposal for the
  Intercomparison of the Dynamical Cores of Atmospheric General
  Circulation Models. Bull. Amer. Meteor. Soc., 75(10), 1825-1830.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import MPASHydrostaticState, MPASHydrostaticTendencies
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.grids.vertical import (
    SigmaCoordinate,
    HybridSigmaPressureCoordinate,
    pressure_from_sigma,
    pressure_from_hybrid,
)
from legoesm import constants
from legoesm.atmosphere.physics.held_suarez import (
    K_A,
    K_S,
    K_F,
    SIGMA_B,
    held_suarez_equilibrium_temperature,
)


def held_suarez_forcing_mpas(
    state: MPASHydrostaticState,
    mesh: VoronoiMesh,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
) -> MPASHydrostaticTendencies:
    """Compute Held-Suarez physics tendencies on an MPAS mesh.

    Parameters
    ----------
    state : MPASHydrostaticState
        u (nEdges, nlev), T (nCells, nlev), p_s (nCells,).
    mesh : VoronoiMesh
        Provides latCell, latEdge.
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
        Vertical coordinate.

    Returns
    -------
    MPASHydrostaticTendencies
    """
    u = state.u.data       # (nEdges, nlev)
    T = state.T.data       # (nCells, nlev)
    p_s = state.p_s.data   # (nCells,)

    nlev = T.shape[-1]
    lat_cell = mesh.latCell  # (nCells,)
    lat_edge = mesh.latEdge  # (nEdges,)

    # Pressure at cell centers
    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)
    if _hybrid:
        p_full = pressure_from_hybrid(sigma_coord, p_s)  # (nCells, nlev)
        # Effective sigma for BL parameterization
        sigma_eff = p_full / jnp.maximum(p_s[:, None], 1.0)
    else:
        sigma_full = sigma_coord.sigma_full  # (nlev,)
        p_full = pressure_from_sigma(sigma_full, p_s)    # (nCells, nlev)
        sigma_eff = jnp.broadcast_to(sigma_full[None, :], T.shape)

    # Equilibrium temperature (nCells, nlev)
    T_eq = held_suarez_equilibrium_temperature(lat_cell[:, None], p_full)

    # Temperature relaxation coefficient k_T(σ, φ) at cell centers
    sigma_factor = jnp.maximum(0.0, (sigma_eff - SIGMA_B) / (1.0 - SIGMA_B))
    cos_lat_4 = jnp.cos(lat_cell) ** 4  # (nCells,)
    k_T = K_A + (K_S - K_A) * sigma_factor * cos_lat_4[:, None]

    # Newtonian relaxation
    dT_dt = -k_T * (T - T_eq)  # (nCells, nlev)

    # Rayleigh friction coefficient k_v(σ) at edge locations
    # Use edge-adjacent cell pressures averaged
    c0 = mesh.cellsOnEdge[0]  # (nEdges,)
    c1 = mesh.cellsOnEdge[1]  # (nEdges,)
    if _hybrid:
        p_edge = 0.5 * (p_full[c0] + p_full[c1])        # (nEdges, nlev)
        ps_edge = 0.5 * (p_s[c0] + p_s[c1])             # (nEdges,)
        sigma_edge = p_edge / ps_edge[:, None]
    else:
        sigma_edge = jnp.broadcast_to(sigma_full[None, :], u.shape)

    sigma_factor_edge = jnp.maximum(
        0.0, (sigma_edge - SIGMA_B) / (1.0 - SIGMA_B)
    )
    k_v = K_F * sigma_factor_edge  # (nEdges, nlev)

    # Rayleigh friction on normal velocity
    du_dt = -k_v * u  # (nEdges, nlev)

    # Construct tendencies
    dims_edge = ("nEdges", "level")
    dims_cell = ("nCells", "level")
    dims_cell_2d = ("nCells",)

    return MPASHydrostaticTendencies(
        du_dt=Field(data=du_dt, name="du_dt_phys", dims=dims_edge, units="m/s^2"),
        dT_dt=Field(data=dT_dt, name="dT_dt_phys", dims=dims_cell, units="K/s"),
        dp_s_dt=Field(
            data=jnp.zeros_like(p_s), name="dp_s_dt_phys",
            dims=dims_cell_2d, units="Pa/s",
        ),
        dphis_dt=Field(
            data=jnp.zeros_like(p_s), name="dphis_dt_phys",
            dims=dims_cell_2d, units="m^2/s^3",
        ),
    )


def held_suarez_init_mpas(
    mesh: VoronoiMesh,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    T_init: float = 300.0,
    p_s_init: float = 1.0e5,
    perturbation_amplitude: float = 1.0,
    seed: int = 42,
    phis: jnp.ndarray | None = None,
) -> MPASHydrostaticState:
    """Create isothermal rest-state initial conditions for Held-Suarez on MPAS.

    Parameters
    ----------
    mesh : VoronoiMesh
        MPAS Voronoi mesh.
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
        Vertical coordinate.
    T_init : float
        Initial temperature [K].
    p_s_init : float
        Initial surface pressure [Pa].
    perturbation_amplitude : float
        Amplitude of random T perturbation at lowest level [K].
    seed : int
        Random seed.
    phis : jnp.ndarray or None
        Surface geopotential [m^2/s^2], shape (nCells,). If None, flat
        terrain is used. When provided, surface pressure is reduced
        hydrostatically: p_s = p_s_init * exp(-phis / (R_d * T_init)).

    Returns
    -------
    MPASHydrostaticState
    """
    nCells = mesh.nCells
    nEdges = mesh.nEdges
    nlev = sigma_coord.n_levels

    # Surface geopotential
    if phis is None:
        phis_data = jnp.zeros((nCells,))
    else:
        phis_data = phis

    # Surface pressure (hydrostatic adjustment for topography)
    p_s_data = p_s_init * jnp.exp(-phis_data / (constants.R_d * T_init))

    # Temperature: uniform with small perturbation at lowest level
    T_data = jnp.full((nCells, nlev), T_init)
    key = jax.random.PRNGKey(seed)
    perturbation = jax.random.normal(key, (nCells,)) * perturbation_amplitude
    T_data = T_data.at[:, -1].add(perturbation)

    # Velocity: at rest
    u_data = jnp.zeros((nEdges, nlev))

    return MPASHydrostaticState(
        u=Field(data=u_data, name="u", dims=("nEdges", "level"), units="m/s"),
        T=Field(data=T_data, name="T", dims=("nCells", "level"), units="K"),
        p_s=Field(data=p_s_data, name="p_s", dims=("nCells",), units="Pa"),
        phis=Field(data=phis_data, name="phis", dims=("nCells",), units="m^2/s^2"),
    )


def baroclinic_wave_init_mpas(
    mesh: VoronoiMesh,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    perturbed: bool = True,
) -> MPASHydrostaticState:
    """Initialize Jablonowski-Williamson baroclinic wave on MPAS mesh.

    Uses the same DCMIP 2016 analytic solution as the cubed-sphere and
    spectral initializations (see :mod:`baroclinic_wave`), ensuring
    identical balanced states across all grid types.  The height-based
    formulation is inverted to sigma levels via bisection.

    Parameters
    ----------
    mesh : VoronoiMesh
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    perturbed : bool
        If True, add the exponential perturbation to trigger instability.

    Returns
    -------
    MPASHydrostaticState
    """
    from legoesm.atmosphere.physics.baroclinic_wave import (
        P0,
        evaluate_pressure_temperature,
        find_z_for_pressure,
        compute_zonal_wind,
        exponential_perturbation,
    )

    nCells = mesh.nCells
    nEdges = mesh.nEdges
    nlev = sigma_coord.n_levels
    sigma_full = sigma_coord.sigma_full  # (nlev,)

    lat_c = mesh.latCell   # (nCells,)
    lat_e = mesh.latEdge   # (nEdges,)
    lon_e = mesh.lonEdge   # (nEdges,)
    cos_angle = jnp.cos(mesh.angleEdge)  # (nEdges,)

    # Surface pressure: constant (no topography)
    p_s_data = jnp.full((nCells,), P0)

    # Compute T at cell centers and u at edges, level by level
    T_data = jnp.zeros((nCells, nlev))
    u_data = jnp.zeros((nEdges, nlev))

    for k in range(nlev):
        sig_k = float(sigma_full[k])

        # --- Temperature at cell centers ---
        p_target_c = jnp.full((nCells,), sig_k * P0)
        z_c = find_z_for_pressure(p_target_c, lat_c)
        _, T_k = evaluate_pressure_temperature(z_c, lat_c)
        T_data = T_data.at[:, k].set(T_k)

        # --- Zonal wind at edge midpoints ---
        p_target_e = jnp.full((nEdges,), sig_k * P0)
        z_e = find_z_for_pressure(p_target_e, lat_e)
        _, T_e = evaluate_pressure_temperature(z_e, lat_e)
        u_zonal = compute_zonal_wind(z_e, lat_e, T_e)

        # Add perturbation (eastward direction, projected to edge normal)
        if perturbed:
            u_zonal = u_zonal + exponential_perturbation(lat_e, lon_e, z_e)

        # Project zonal wind to edge normal: u_n = u_east * cos(angle)
        # (v_north = 0 for this test case)
        u_data = u_data.at[:, k].set(u_zonal * cos_angle)

    # Surface geopotential: flat
    phis_data = jnp.zeros((nCells,))

    return MPASHydrostaticState(
        u=Field(data=u_data, name="u", dims=("nEdges", "level"), units="m/s"),
        T=Field(data=T_data, name="T", dims=("nCells", "level"), units="K"),
        p_s=Field(data=p_s_data, name="p_s", dims=("nCells",), units="Pa"),
        phis=Field(data=phis_data, name="phis", dims=("nCells",), units="m^2/s^2"),
    )
