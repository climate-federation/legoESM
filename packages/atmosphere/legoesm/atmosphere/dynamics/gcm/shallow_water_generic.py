"""Shallow-water equations written once over the grid operator contract.

Vector-invariant form, shared by every grid::

    dh/dt   = mass tendency (default -div(h_edge * u))
    dvel/dt = -grad(KE + g (h + h_s)) + vorticity term + dissipation

Divergence and gradient are the B2 operator contract
(:mod:`legoesm.grids.operator_protocol`) implemented by the neighbour-sum
adapters (:func:`~legoesm.grids.operator_adapters.latlon_neighbour_operators`,
:func:`~legoesm.grids.operator_adapters.mpas_neighbour_operators`): the same
:mod:`legoesm.core.neighbour_ops` code on every grid (structured offsets on
lat-lon, index tables on MPAS).  Lat-lon grids the neighbour mesh does not cover
(regional, tripolar, rich metrics, MPI/SPMD) get the free-function adapter
instead (:func:`~legoesm.grids.operator_adapters.latlon_sw_grid_operators`).
The grid-specific numerics stay plug-ins
(:class:`SWOperators`): mass-flux reconstruction (lat-lon PPM computes its own
flux divergence), kinetic energy, the Coriolis / PV-flux term, viscosity and the
velocity boundary condition.

This is the tendency of :class:`.shallow_water_latlon_cgrid.CGridLatLonShallowWaterModel`,
:class:`.shallow_water_mpas.MPASShallowWaterModel` and the RBF-FD collocated
model (:mod:`.shallow_water_rbffd`; Cartesian 3-vector velocity, the radial
component removed by the ``boundary`` plug-in).  Known limits: the
lat-lon and MPAS vorticity terms still compute their own curl (the contract's
``vorticity`` is not routed into them).
"""

from __future__ import annotations

import math

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from legoesm.grids.operator_adapters import (
    latlon_sw_grid_operators,
    mpas_neighbour_operators,
    rbffd_operators,
)

Edge = tuple[jax.Array, ...]


class SWOperators(NamedTuple):
    """Grid operator contract plus the grid plug-ins around it."""

    grid_ops: Any
    mass_tendency: Callable[[jax.Array, Edge], jax.Array]
    kinetic_energy: Callable[[Edge], jax.Array]
    vorticity_term: Callable[[Edge, jax.Array, Edge], Edge]  # (vel, h, dvel) -> dvel
    dissipation: Callable[[Edge, Edge], Edge]                # (vel, dvel) -> dvel
    boundary: Callable[[Edge], Edge]


def shallow_water_tendencies(
    h: jax.Array, vel: Edge, h_s: jax.Array, ops: SWOperators, g: float,
) -> tuple[jax.Array, Edge]:
    """``(dh/dt, dvel/dt)`` for any grid; ``vel`` is a tuple of edge families."""
    dh_dt = ops.mass_tendency(h, vel)
    bernoulli = g * (h + h_s) + ops.kinetic_energy(vel)
    grad = ops.grid_ops.gradient(bernoulli)
    # Component-velocity grids return (gx, gy); edge-normal grids one array.
    grad = grad if isinstance(grad, tuple) else (grad,)
    dvel = tuple(-gb for gb in grad)
    dvel = ops.vorticity_term(vel, h, dvel)
    dvel = ops.dissipation(vel, dvel)
    return dh_dt, ops.boundary(dvel)


# ============================================================================
# Lat-lon C-grid plug-ins
# ============================================================================

def cell_winds_cgrid(u: jax.Array, v: jax.Array) -> tuple[jax.Array, jax.Array]:
    """C-grid face velocities averaged to cell centres ``(u_c, v_c)``."""
    return 0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1, :] + v[1:, :])


def kinetic_energy_cgrid(u: jax.Array, v: jax.Array) -> jax.Array:
    """Cell-centre ``0.5 * (u_c**2 + v_c**2)`` from C-grid face velocities."""
    u_c, v_c = cell_winds_cgrid(u, v)
    return 0.5 * (u_c**2 + v_c**2)


def latlon_sw_operators(grid: Any, config: Any, dt: Any = None) -> SWOperators:
    """Lat-lon C-grid plug-ins (PPM or centered mass flux, absolute-vorticity
    Coriolis, optional del-2 / del-4 viscosity, polar wall on v)."""
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        absolute_vorticity_coriolis,
        nu_del4_row_profiles,
    )
    from legoesm.core.operators_fv_latlon import cgrid_fv_flux_divergence_latlon
    from legoesm.grids.operators_latlon_cgrid import (
        interp_cell_to_uface,
        interp_cell_to_vface,
        vector_laplacian_cgrid,
    )

    if config.nu_del4 > 0.0 and dt is None:
        raise ValueError(
            "cgrid_latlon_sw_tendencies: nu_del4 > 0 requires the "
            "dt argument (per-row pole stability cap needs it).")
    grid_ops = latlon_sw_grid_operators(grid)

    if config.use_ppm_transport:
        def mass_tendency(h, vel):
            return cgrid_fv_flux_divergence_latlon(h, vel[0], vel[1], grid)
    else:
        # 2nd-order averaging of h to faces, legacy grid=None (pole-inert) form.
        def mass_tendency(h, vel):
            return -grid_ops.divergence(interp_cell_to_uface(h) * vel[0],
                                        interp_cell_to_vface(h) * vel[1])

    def vorticity_term(vel, h, dvel):
        cor_u, cor_v = absolute_vorticity_coriolis(vel[0], vel[1], grid)
        return (dvel[0] + cor_u, dvel[1] + cor_v)

    def dissipation(vel, dvel):
        du, dv = dvel
        # +A_h * lap(u): the Laplacian of a local extremum is negative -> damping.
        if config.A_h > 0.0:
            lap_u, lap_v = vector_laplacian_cgrid(vel[0], vel[1], grid)
            du = du + config.A_h * lap_u
            dv = dv + config.A_h * lap_v
        # -nu4 * lap(lap(u)): lap^2 -> +k^4 for a Fourier mode, so "-" decays.
        if config.nu_del4 > 0.0:
            lap_u, lap_v = vector_laplacian_cgrid(vel[0], vel[1], grid)
            lap2_u, lap2_v = vector_laplacian_cgrid(lap_u, lap_v, grid)
            nu_u, nu_v = nu_del4_row_profiles(
                grid, config.nu_del4, config.nu_del4_cfl_frac, dt)
            du = du - nu_u * lap2_u
            dv = dv - nu_v * lap2_v
        return (du, dv)

    def boundary(dvel):
        # Wall BC: zero meridional tendency on the polar v faces, so RK stages
        # never see polar v feeding divergence/Coriolis.
        return (dvel[0], jnp.pad(dvel[1][1:-1, :], ((1, 1), (0, 0))))

    return SWOperators(grid_ops, mass_tendency,
                       lambda vel: kinetic_energy_cgrid(vel[0], vel[1]),
                       vorticity_term, dissipation, boundary)


def cgrid_latlon_sw_tendencies(state: Any, grid: Any, config: Any = None, dt: Any = None):
    """Lat-lon C-grid shallow-water tendencies ``(dh, du, dv)`` at
    (cell centres, lon faces, lat faces); ``config`` defaults to
    ``CGridLatLonShallowWaterConfig()``."""
    if config is None:
        from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
            CGridLatLonShallowWaterConfig,
        )
        config = CGridLatLonShallowWaterConfig()
    h, u, v, h_s = state
    ops = latlon_sw_operators(grid, config, dt)
    dh, (du, dv) = shallow_water_tendencies(h, (u, v), h_s, ops, config.g)
    return dh, du, dv


# ============================================================================
# MPAS / TRiSK plug-ins
# ============================================================================

def mpas_sw_operators(mesh: Any, config: Any, dt: float = 0.0) -> SWOperators:
    """MPAS / TRiSK plug-ins (thickness flux, PV flux, APVM, del-2 / del-4)."""
    from legoesm.core.operators_voronoi import (
        apvm_correction,
        kinetic_energy_cell,
        potential_vorticity_vertex,
        pv_flux_energy_conserving,
        pv_flux_enstrophy_conserving,
        thickness_flux,
        vector_laplacian_del2,
        vector_laplacian_del4,
    )

    if config.pv_scheme == "energy":
        pv_flux = pv_flux_energy_conserving
    elif config.pv_scheme == "enstrophy":
        pv_flux = pv_flux_enstrophy_conserving
    else:
        raise ValueError(
            f"Unknown pv_scheme {config.pv_scheme!r}; "
            "expected one of: 'energy', 'enstrophy'.")
    grid_ops = mpas_neighbour_operators(mesh)

    def mass_tendency(h, vel):
        return -grid_ops.divergence(
            thickness_flux(h, vel[0], mesh, order=config.thickness_order))

    def vorticity_term(vel, h, dvel):
        q = potential_vorticity_vertex(vel[0], h, mesh.fVertex, mesh)
        # Python check: dt must be concrete here, and dt == 0 skips APVM.
        if config.apvm_scale > 0 and dt > 0:
            q = apvm_correction(q, vel[0], mesh, config.apvm_scale * dt)
        return (dvel[0] + pv_flux(vel[0], h, q, mesh),)

    def dissipation(vel, dvel):
        du = dvel[0]
        if config.nu_del2 > 0:
            du = du + config.nu_del2 * vector_laplacian_del2(vel[0], mesh)
        if config.nu_del4 > 0:
            du = du + config.nu_del4 * vector_laplacian_del4(vel[0], mesh)
        return (du,)

    return SWOperators(grid_ops, mass_tendency,
                       lambda vel: kinetic_energy_cell(vel[0], mesh),
                       vorticity_term, dissipation, lambda dvel: dvel)


def mpas_shallow_water_tendencies(state: Any, mesh: Any, config: Any = None, dt: float = 0.0):
    """MPAS shallow-water tendencies; ``dt`` feeds APVM; ``config`` defaults to
    ``MPASShallowWaterConfig()``."""
    from legoesm.core.state import MPASShallowWaterTendencies

    if config is None:
        from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import MPASShallowWaterConfig
        config = MPASShallowWaterConfig()
    ops = mpas_sw_operators(mesh, config, dt)
    dh, (du,) = shallow_water_tendencies(
        state.h.data, (state.u.data,), state.h_s.data, ops, config.g)
    return MPASShallowWaterTendencies(
        dh_dt=state.h.replace(data=dh, name="dh_dt", units="m/s"),
        du_dt=state.u.replace(data=du, name="du_dt", units="m/s^2"),
    )


# ============================================================================
# RBF-FD collocated plug-ins (Cartesian 3-vector velocity)
# ============================================================================

def rbffd_grid_scale_length2(mesh: Any) -> float:
    """``(h_mean / pi)**2`` with ``h_mean = sqrt(4 pi a^2 / n_nodes)``: inverse
    squared wavenumber of the grid-scale mode."""
    n_nodes = mesh.normal[0].shape[0]
    h_mean = mesh.radius * math.sqrt(4.0 * math.pi / n_nodes)
    return (h_mean / math.pi) ** 2


def rbffd_sw_operators(mesh: Any, config: Any) -> SWOperators:
    """RBF-FD plug-ins: flux-form mass tendency, vector-invariant Coriolis +
    relative vorticity, ``Lap**k`` hyperviscosity on ``h`` and each Cartesian
    velocity component, and projection of the tendency onto the tangent plane.

    Mass conserving to round-off: the mass equation uses the conservatively
    corrected divergence and Laplacian (``sum(area * dh/dt) = 0`` for tangent
    velocity), with ``area`` the quadrature weights the mesh was built with.
    See the mesh docstring for the known accuracy limit of that divergence.
    """
    grid_ops = rbffd_operators(mesh)
    nx, ny, nz = mesh.normal
    f = 2.0 * config.omega * nz
    k = config.hyperviscosity_order
    length2 = rbffd_grid_scale_length2(mesh)
    sign = (-1.0) ** (k + 1)

    def hyper(phi, lap=grid_ops.laplacian):
        # (-1)^(k+1) gamma Lap^k, gamma = length2^k / tau: the grid-scale mode
        # e-folds in tau.  Scaled per stage so every stage is O(phi) (fp32 range).
        # KNOWN: on a smooth l=4 mode (642-10242 nodes, k=4) the stacked
        # Laplacians amplify the grid-scale part of their error, so the
        # pointwise result is 41-371x the tiny (~1e-8/tau) intended rate; the
        # area-weighted rate matches to 1-4 %, and that noise is itself damped.
        for _ in range(k):
            phi = length2 * lap(phi)
        return sign * phi / config.hyperviscosity_tau_s

    def mass_tendency(h, vel):
        return (-grid_ops.divergence(*(h * u for u in vel))
                + hyper(h, grid_ops.laplacian_mass))

    def vorticity_term(vel, h, dvel):
        ux, uy, uz = vel
        q = grid_ops.vorticity(ux, uy, uz) + f
        n_cross_u = (ny * uz - nz * uy, nz * ux - nx * uz, nx * uy - ny * ux)
        return tuple(dv - q * c for dv, c in zip(dvel, n_cross_u))

    def dissipation(vel, dvel):
        return tuple(dv + hyper(u) for dv, u in zip(dvel, vel))

    def boundary(dvel):
        radial = nx * dvel[0] + ny * dvel[1] + nz * dvel[2]
        return tuple(dv - radial * n for dv, n in zip(dvel, (nx, ny, nz)))

    return SWOperators(grid_ops, mass_tendency,
                       lambda vel: 0.5 * sum(u * u for u in vel),
                       vorticity_term, dissipation, boundary)
