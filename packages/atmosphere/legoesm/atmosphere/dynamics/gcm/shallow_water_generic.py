"""Shallow-water equations written once over the neighbour-offset layer.

Vector-invariant form, shared by every grid::

    dh/dt   = mass tendency (default -div(h_edge * u))
    dvel/dt = -grad(KE + g (h + h_s)) + vorticity term + dissipation

Divergence and gradient come from :mod:`legoesm.core.neighbour_ops`, the same
code on every grid (structured offsets on lat-lon, index tables on MPAS).  The
grid-specific numerics stay plug-ins (:class:`SWOperators`): mass-flux
reconstruction (lat-lon PPM computes its own flux divergence), kinetic energy,
the Coriolis / PV-flux term, viscosity and the velocity boundary condition.

Pilot status: reproduces :mod:`.shallow_water_latlon_cgrid` (to round-off) and
:mod:`.shallow_water_mpas` (bit-for-bit) tendencies, proven by
``test_shallow_water_generic_equivalence``.  Not wired into a model or the dycore
factory; the drop-in tendency functions below share the existing signatures.
Known limits: the lat-lon and MPAS vorticity terms still compute their own curl
(the generic :func:`~legoesm.core.neighbour_ops.curl` is equivalence-tested but
not routed into them); lat-lon scope is a regular global grid on one device.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from legoesm.core.neighbour_ops import NeighbourMesh, divergence, gradient

Edge = tuple[jax.Array, ...]


class SWOperators(NamedTuple):
    """Grid plug-ins around the shared operators."""

    mesh: NeighbourMesh
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
    dvel = tuple(-gb for gb in gradient(ops.mesh, bernoulli))
    dvel = ops.vorticity_term(vel, h, dvel)
    dvel = ops.dissipation(vel, dvel)
    return dh_dt, ops.boundary(dvel)


# ============================================================================
# Lat-lon C-grid plug-ins
# ============================================================================

def latlon_sw_operators(grid: Any, config: Any, dt: Any = None) -> SWOperators:
    """Plug-ins reproducing :func:`.shallow_water_latlon_cgrid.cgrid_latlon_sw_tendencies`."""
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        absolute_vorticity_coriolis,
        kinetic_energy_cgrid,
        nu_del4_row_profiles,
    )
    from legoesm.core.operators_fv_latlon import cgrid_fv_flux_divergence_latlon
    from legoesm.grids.neighbour_connectivity import latlon_neighbour_mesh
    from legoesm.grids.operators_latlon_cgrid import (
        interp_cell_to_uface,
        interp_cell_to_vface,
        vector_laplacian_cgrid,
    )

    if config.nu_del4 > 0.0 and dt is None:
        raise ValueError(
            "latlon_sw_operators: nu_del4 > 0 requires dt "
            "(per-row pole stability cap needs it).")
    mesh = latlon_neighbour_mesh(grid)

    if config.use_ppm_transport:
        def mass_tendency(h, vel):
            return cgrid_fv_flux_divergence_latlon(h, vel[0], vel[1], grid)
    else:
        def mass_tendency(h, vel):
            flux = (interp_cell_to_uface(h) * vel[0],
                    interp_cell_to_vface(h) * vel[1])
            return -divergence(mesh, flux)

    def vorticity_term(vel, h, dvel):
        cor_u, cor_v = absolute_vorticity_coriolis(vel[0], vel[1], grid)
        return (dvel[0] + cor_u, dvel[1] + cor_v)

    def dissipation(vel, dvel):
        du, dv = dvel
        if config.A_h > 0.0:
            lap_u, lap_v = vector_laplacian_cgrid(vel[0], vel[1], grid)
            du = du + config.A_h * lap_u
            dv = dv + config.A_h * lap_v
        if config.nu_del4 > 0.0:
            lap_u, lap_v = vector_laplacian_cgrid(vel[0], vel[1], grid)
            lap2_u, lap2_v = vector_laplacian_cgrid(lap_u, lap_v, grid)
            nu_u, nu_v = nu_del4_row_profiles(
                grid, config.nu_del4, config.nu_del4_cfl_frac, dt)
            du = du - nu_u * lap2_u
            dv = dv - nu_v * lap2_v
        return (du, dv)

    def boundary(dvel):
        # Wall BC: zero meridional tendency on the polar v faces.
        return (dvel[0], jnp.pad(dvel[1][1:-1, :], ((1, 1), (0, 0))))

    return SWOperators(mesh, mass_tendency,
                       lambda vel: kinetic_energy_cgrid(vel[0], vel[1]),
                       vorticity_term, dissipation, boundary)


def latlon_sw_tendencies_generic(state: Any, grid: Any, config: Any, dt: Any = None):
    """Drop-in for ``cgrid_latlon_sw_tendencies``: returns ``(dh, du, dv)``."""
    h, u, v, h_s = state
    ops = latlon_sw_operators(grid, config, dt)
    dh, (du, dv) = shallow_water_tendencies(h, (u, v), h_s, ops, config.g)
    return dh, du, dv


# ============================================================================
# MPAS / TRiSK plug-ins
# ============================================================================

def mpas_sw_operators(mesh: Any, config: Any, dt: float = 0.0) -> SWOperators:
    """Plug-ins reproducing :func:`.shallow_water_mpas.mpas_shallow_water_tendencies`."""
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
    from legoesm.grids.neighbour_connectivity import mpas_neighbour_mesh

    if config.pv_scheme == "energy":
        pv_flux = pv_flux_energy_conserving
    elif config.pv_scheme == "enstrophy":
        pv_flux = pv_flux_enstrophy_conserving
    else:
        raise ValueError(
            f"Unknown pv_scheme {config.pv_scheme!r}; "
            "expected one of: 'energy', 'enstrophy'.")
    nmesh = mpas_neighbour_mesh(mesh)

    def mass_tendency(h, vel):
        return -divergence(nmesh, (thickness_flux(h, vel[0], mesh,
                                                  order=config.thickness_order),))

    def vorticity_term(vel, h, dvel):
        q = potential_vorticity_vertex(vel[0], h, mesh.fVertex, mesh)
        # Same Python check as the reference: dt must be concrete here, and
        # dt == 0 skips APVM (the reference's behaviour, kept for equivalence).
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

    return SWOperators(nmesh, mass_tendency,
                       lambda vel: kinetic_energy_cell(vel[0], mesh),
                       vorticity_term, dissipation, lambda dvel: dvel)


def mpas_sw_tendencies_generic(state: Any, mesh: Any, config: Any, dt: float = 0.0):
    """Drop-in for ``mpas_shallow_water_tendencies``."""
    from legoesm.core.state import MPASShallowWaterTendencies

    ops = mpas_sw_operators(mesh, config, dt)
    dh, (du,) = shallow_water_tendencies(
        state.h.data, (state.u.data,), state.h_s.data, ops, config.g)
    return MPASShallowWaterTendencies(
        dh_dt=state.h.replace(data=dh, name="dh_dt", units="m/s"),
        du_dt=state.u.replace(data=du, name="du_dt", units="m/s^2"),
    )
