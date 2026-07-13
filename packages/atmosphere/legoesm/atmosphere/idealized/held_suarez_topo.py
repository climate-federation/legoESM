"""Held-Suarez with idealized analytical topography.

Wraps the existing :mod:`legoesm.atmosphere.held_suarez` forcing with a
DCMIP-§2-0-0-style ridged cosine-bell mountain.  The relaxation /
friction kernels are unchanged — only the surface geopotential and
surface pressure differ from the flat-Earth baseline.  This is the
``HS-Topo`` entry in :doc:`/docs/validation/dycore_validation_catalog`.

The forcing functions are aliased directly from
:mod:`legoesm.atmosphere.held_suarez` (no behavioural change):

- ``held_suarez_topo_forcing``        ↔ :func:`held_suarez_forcing`
- ``held_suarez_topo_forcing_latlon`` ↔ :func:`held_suarez_forcing_latlon`
- ``held_suarez_topo_forcing_mpas``   ↔ :func:`held_suarez_forcing_mpas`
- ``held_suarez_topo_forcing_spectral`` ↔ :func:`held_suarez_forcing_spectral`

The init helpers below add a non-zero ``phis`` field; HS forcing then
sees a non-uniform pressure field through the standard
``pressure_from_hybrid`` / ``pressure_from_sigma`` paths.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.held_suarez import (
    held_suarez_forcing,
    held_suarez_forcing_latlon,
    held_suarez_forcing_mpas,
    held_suarez_forcing_spectral,
    held_suarez_init,
    held_suarez_init_latlon,
    held_suarez_init_mpas,
)
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.grids.vertical import (
    HybridSigmaPressureCoordinate,
    SigmaCoordinate,
)

from legoesm.atmosphere.idealized.topography import dcmip_2_0_0_mountain


# Re-export forcing functions under the topo-aware naming convention so
# the matrix-script runner can dispatch by name without conditionals.
held_suarez_topo_forcing = held_suarez_forcing
held_suarez_topo_forcing_latlon = held_suarez_forcing_latlon
held_suarez_topo_forcing_mpas = held_suarez_forcing_mpas
held_suarez_topo_forcing_spectral = held_suarez_forcing_spectral


# ---------------------------------------------------------------------------
# Default topography
# ---------------------------------------------------------------------------

DEFAULT_H_0 = 2000.0  # Mountain peak [m]


def _phis_cube(grid, h_0: float) -> jnp.ndarray:
    z_s = dcmip_2_0_0_mountain(
        grid.lon, grid.lat, float(grid.radius), h_0=h_0,
    )
    return constants.g * z_s


def _phis_latlon(grid, h_0: float) -> jnp.ndarray:
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    lat_2d = jnp.asarray(grid.lat)[:, None]
    lon_2d = jnp.asarray(grid.lon)[None, :]
    lat_b = jnp.broadcast_to(lat_2d, (n_lat, n_lon))
    lon_b = jnp.broadcast_to(lon_2d, (n_lat, n_lon))
    z_s = dcmip_2_0_0_mountain(lon_b, lat_b, float(grid.radius), h_0=h_0)
    return constants.g * z_s


def _phis_mpas(mesh, h_0: float) -> jnp.ndarray:
    z_s = dcmip_2_0_0_mountain(
        mesh.lonCell, mesh.latCell, float(mesh.radius), h_0=h_0,
    )
    return constants.g * z_s


def _phis_gaussian(grid, h_0: float) -> jnp.ndarray:
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    lat_2d = jnp.broadcast_to(grid.lat[:, None], (n_lat, n_lon))
    lon_2d = jnp.broadcast_to(grid.lon2d, (n_lat, n_lon))
    z_s = dcmip_2_0_0_mountain(lon_2d, lat_2d, float(grid.radius), h_0=h_0)
    return constants.g * z_s


# ---------------------------------------------------------------------------
# Init wrappers — add the mountain phis to the existing rest-state IC
# ---------------------------------------------------------------------------


def held_suarez_topo_init(
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    h_0: float = DEFAULT_H_0,
    T_init: float = 300.0,
    p_s_init: float = constants.p_ref,
    perturbation_amplitude: float = 1.0,
    seed: int = 42,
) -> HydrostaticState:
    """Held-Suarez init with DCMIP §2-0-0 mountain on cubed-sphere."""
    phis = _phis_cube(grid, h_0)
    return held_suarez_init(
        grid, sigma_coord,
        T_init=T_init, p_s_init=p_s_init,
        perturbation_amplitude=perturbation_amplitude, seed=seed,
        phis=phis,
    )


def held_suarez_topo_init_latlon(
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    h_0: float = DEFAULT_H_0,
    T_init: float = 300.0,
    p_s_init: float = constants.p_ref,
    perturbation_amplitude: float = 1.0,
    seed: int = 42,
) -> HydrostaticState:
    """Held-Suarez init with DCMIP §2-0-0 mountain on lat-lon."""
    phis = _phis_latlon(grid, h_0)
    return held_suarez_init_latlon(
        grid, sigma_coord,
        T_init=T_init, p_s_init=p_s_init,
        perturbation_amplitude=perturbation_amplitude, seed=seed,
        phis=phis,
    )


def held_suarez_topo_init_mpas(
    mesh,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    h_0: float = DEFAULT_H_0,
    T_init: float = 300.0,
    p_s_init: float = constants.p_ref,
    perturbation_amplitude: float = 1.0,
    seed: int = 42,
):
    """Held-Suarez init with DCMIP §2-0-0 mountain on MPAS Voronoi."""
    phis = _phis_mpas(mesh, h_0)
    return held_suarez_init_mpas(
        mesh, sigma_coord,
        T_init=T_init, p_s_init=p_s_init,
        perturbation_amplitude=perturbation_amplitude, seed=seed,
        phis=phis,
    )


def held_suarez_topo_init_spectral(
    grid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    *,
    h_0: float = DEFAULT_H_0,
    T_init: float = 300.0,
    p_s_init: float = constants.p_ref,
    perturbation_amplitude: float = 1.0,
    seed: int = 42,
):
    """Held-Suarez init with DCMIP §2-0-0 mountain in spectral space.

    The spectral PE has its own canonical isothermal-rest helper
    (``isothermal_rest_state_spectral``). To inject topography we
    construct the equivalent grid-space rest state with non-zero phis
    via lat-lon broadcasting and transform to spectral coefficients
    using the same convention as
    :func:`tests.test_cases.baroclinic_wave.baroclinic_wave_init_spectral`.
    """
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralHydrostaticState
    from legoesm.grids.gaussian import sh_analysis, sh_analysis_3d

    n_lat = grid.n_lat
    n_lon = grid.n_lon
    n_sh = grid.n_sh
    nlev = sigma_coord.n_levels

    phis_grid = _phis_gaussian(grid, h_0)  # (n_lat, n_lon)
    p_s_grid = (
        p_s_init * jnp.exp(-phis_grid / (constants.R_d * T_init))
    )

    # Uniform isothermal temperature, optionally perturbed at the surface
    T_grid = jnp.full((n_lat, n_lon, nlev), T_init, dtype=jnp.float64)
    if perturbation_amplitude > 0.0:
        import jax
        key = jax.random.PRNGKey(seed)
        pert = (
            jax.random.normal(key, (n_lat, n_lon), dtype=jnp.float64)
            * perturbation_amplitude
        )
        T_grid = T_grid.at[:, :, -1].add(pert)

    T_hat = sh_analysis_3d(grid, T_grid)
    lnps_hat = sh_analysis(grid, jnp.log(p_s_grid))
    phis_hat = sh_analysis(grid, phis_grid)

    vor_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
    div_hat = jnp.zeros_like(vor_hat)

    dims_3d = ("spectral", "level")
    dims_2d = ("spectral",)
    return SpectralHydrostaticState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims_3d, units="1/s"),
        T_hat=Field(data=T_hat, name="T_hat", dims=dims_3d, units="K"),
        lnps_hat=Field(data=lnps_hat, name="lnps_hat", dims=dims_2d, units=""),
        phis_hat=Field(
            data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2",
        ),
    )
