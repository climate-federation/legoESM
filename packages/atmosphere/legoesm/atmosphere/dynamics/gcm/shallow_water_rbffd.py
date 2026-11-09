"""Shallow water on the sphere with collocated RBF-FD operators.

Nodes are any point set on the sphere (here Voronoi cell centres); the velocity
is the Cartesian 3-vector ``(ux, uy, uz)`` tangent to the sphere.  The
tendency is the shared core :func:`.shallow_water_generic.shallow_water_tendencies`
with the RBF-FD plug-ins; time stepping goes through
:func:`legoesm.timestepping.dispatch.dispatch_integrator`, so any registered
integrator is selectable via ``config.integrator``.

KNOWN DEFECT (Rule 0): the collocated divergence is not in flux form, so the
global mass ``sum(area * h)`` drifts; it is measured by
:func:`total_mass` and reported, not fixed.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.shallow_water_generic import (
    rbffd_sw_operators,
    shallow_water_tendencies,
)
from legoesm.core.williamson_sw_analytic import (
    W2_GH0,
    W5_H0_M,
    W5_UBAR_MS,
    solid_body_geopotential,
    solid_body_rotation_speed,
    solid_body_winds,
    williamson_5_mountain_height,
)
from legoesm.grids.neighbour_connectivity import rbffd_sphere_mesh
from legoesm.timestepping.dispatch import dispatch_integrator


class RBFFDShallowWaterState(NamedTuple):
    """Fluid depth ``h`` [m] and Cartesian velocity ``(ux, uy, uz)`` [m/s]."""

    h: jax.Array
    ux: jax.Array
    uy: jax.Array
    uz: jax.Array


class RBFFDShallowWaterConfig(NamedTuple):
    """RBF-FD shallow-water configuration (``integrator`` has no default)."""

    integrator: str
    g: float = constants.g
    omega: float = constants.Omega
    stencil_size: int = 31
    phs_degree: int = 7
    poly_degree: int = 4
    hyperviscosity_order: int = 4
    hyperviscosity_tau_s: float = 21600.0


def voronoi_nodes(vmesh: Any):
    """``(xyz, lon, lat, area)`` in float64 from a Voronoi mesh's cell centres
    (lon/lat recomputed from the float64 positions)."""
    xyz = jnp.stack([jnp.asarray(vmesh.xCell, jnp.float64),
                     jnp.asarray(vmesh.yCell, jnp.float64),
                     jnp.asarray(vmesh.zCell, jnp.float64)], axis=-1)
    r = jnp.linalg.norm(xyz, axis=-1)
    lat = jnp.arcsin(xyz[:, 2] / r)
    lon = jnp.arctan2(xyz[:, 1], xyz[:, 0])
    return xyz, lon, lat, jnp.asarray(vmesh.areaCell, jnp.float64)


def build_mesh(xyz: Any, radius: float, config: RBFFDShallowWaterConfig):
    """RBF-FD mesh from node positions ``xyz (n_nodes, 3)``."""
    return rbffd_sphere_mesh(xyz, radius, n=config.stencil_size,
                             phs_degree=config.phs_degree,
                             poly_degree=config.poly_degree)


def tendency(state: RBFFDShallowWaterState, mesh: Any, h_s: jax.Array,
             config: RBFFDShallowWaterConfig) -> RBFFDShallowWaterState:
    """State-shaped tendency from the shared shallow-water core."""
    ops = rbffd_sw_operators(mesh, config)
    dh, (dux, duy, duz) = shallow_water_tendencies(
        state.h, (state.ux, state.uy, state.uz), h_s, ops, config.g)
    return RBFFDShallowWaterState(dh, dux, duy, duz)


def step(state: RBFFDShallowWaterState, mesh: Any, h_s: jax.Array,
         config: RBFFDShallowWaterConfig, dt: float) -> RBFFDShallowWaterState:
    """One step of ``config.integrator`` (via ``dispatch_integrator``)."""
    return dispatch_integrator(state, lambda s: tendency(s, mesh, h_s, config),
                               dt, config.integrator)


def total_mass(h: jax.Array, area: jax.Array) -> jax.Array:
    """``sum(area * h)`` [m^3] with node areas ``area`` (e.g. Voronoi cells)."""
    return jnp.sum(area * h)


def total_energy(state: RBFFDShallowWaterState, h_s: jax.Array, area: jax.Array,
                 g: float) -> jax.Array:
    """``sum(area * (h |u|^2 / 2 + g h (h/2 + h_s)))`` [m^5/s^2]."""
    ke = 0.5 * (state.ux**2 + state.uy**2 + state.uz**2)
    return jnp.sum(area * (state.h * ke + g * state.h * (0.5 * state.h + h_s)))


def east_north_to_cartesian(u_east, v_north, lon, lat):
    """Cartesian ``(ux, uy, uz)`` of a tangent vector given by its east and
    north components at ``(lon, lat)``."""
    sl, cl = jnp.sin(lon), jnp.cos(lon)
    sp, cp = jnp.sin(lat), jnp.cos(lat)
    return (-sl * u_east - sp * cl * v_north,
            cl * u_east - sp * sl * v_north,
            cp * v_north)


def williamson2_state(lon, lat, radius: float, g: float, omega: float):
    """Williamson TC2 (steady solid-body flow, rotation angle 0, matching the
    Coriolis axis): ``(state, h_s = 0)``."""
    u0 = solid_body_rotation_speed(radius)
    h = solid_body_geopotential(lon, lat, radius=radius, omega=omega, u0=u0,
                                gh0=W2_GH0, xp=jnp) / g
    ue, vn = solid_body_winds(lon, lat, u0=u0, xp=jnp)
    return RBFFDShallowWaterState(h, *east_north_to_cartesian(ue, vn, lon, lat)), \
        jnp.zeros_like(h)


def williamson5_state(lon, lat, radius: float, g: float, omega: float):
    """Williamson TC5 (zonal flow over an isolated mountain): ``(state, h_s)``."""
    h_s = williamson_5_mountain_height(lon, lat, xp=jnp)
    h = solid_body_geopotential(lon, lat, radius=radius, omega=omega,
                                u0=W5_UBAR_MS, gh0=W5_H0_M * g, xp=jnp) / g - h_s
    ue, vn = solid_body_winds(lon, lat, u0=W5_UBAR_MS, xp=jnp)
    return RBFFDShallowWaterState(h, *east_north_to_cartesian(ue, vn, lon, lat)), h_s
