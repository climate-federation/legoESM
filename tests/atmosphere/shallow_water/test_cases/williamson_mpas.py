"""Williamson et al. (1992) test cases for the MPAS shallow water solver.

Provides initial conditions on Voronoi meshes where velocities are
projected onto edge normals (normal component staggering).

References
----------
- Williamson, D. L., et al. (1992). A standard test set for numerical
  approximations to the shallow water equations in spherical geometry.
  J. Comput. Phys., 102(1), 211-224.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import MPASShallowWaterState
from legoesm.grids.voronoi import VoronoiMesh
from legoesm import constants


def _project_velocity_to_edges(u_east, v_north, mesh):
    """Project (u_east, v_north) at cell centers onto edge normals.

    For each edge, the normal velocity is:
        u_n(e) = u_east(e_mid) * cos(angleEdge) + v_north(e_mid) * sin(angleEdge)

    We interpolate cell-center fields to edge midpoints via simple averaging.
    """
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]

    u_e = 0.5 * (u_east[c1] + u_east[c2])
    v_e = 0.5 * (v_north[c1] + v_north[c2])

    # Project onto edge normal direction
    u_normal = u_e * jnp.cos(mesh.angleEdge) + v_e * jnp.sin(mesh.angleEdge)
    return u_normal


def williamson_test2_mpas(mesh: VoronoiMesh) -> MPASShallowWaterState:
    """Williamson Test Case 2: steady-state geostrophic flow.

    Solid body rotation in exact geostrophic balance.
    The analytic solution equals the initial condition for all time.

    Parameters
    ----------
    mesh : VoronoiMesh

    Returns
    -------
    MPASShallowWaterState
    """
    R = mesh.radius
    Omega = constants.Omega
    g = constants.g

    u_0 = 2.0 * jnp.pi * R / (12.0 * 86400.0)  # ~38.6 m/s
    gh_0 = 2.94e4
    h_0 = gh_0 / g

    lat = mesh.latCell
    h_data = h_0 - (R * Omega * u_0 + u_0**2 / 2.0) * jnp.sin(lat)**2 / g
    h_s_data = jnp.zeros_like(h_data)

    # Velocity: solid body rotation u = u_0 * cos(lat), v = 0
    u_east_cell = u_0 * jnp.cos(lat)
    v_north_cell = jnp.zeros_like(lat)

    u_edge = _project_velocity_to_edges(u_east_cell, v_north_cell, mesh)

    return MPASShallowWaterState(
        h=Field(data=h_data, name="h", dims=("nCells",), units="m"),
        u=Field(data=u_edge, name="u", dims=("nEdges",), units="m/s",
                staggering="edge"),
        h_s=Field(data=h_s_data, name="h_s", dims=("nCells",), units="m"),
    )


def williamson_test5_mpas(mesh: VoronoiMesh) -> MPASShallowWaterState:
    """Williamson Test Case 5: zonal flow over an isolated mountain.

    Parameters
    ----------
    mesh : VoronoiMesh

    Returns
    -------
    MPASShallowWaterState
    """
    R = mesh.radius
    Omega = constants.Omega
    g = constants.g

    u_0 = 20.0  # m/s
    gh_0 = 5960.0 * g

    lat = mesh.latCell
    lon = mesh.lonCell

    h_data = (gh_0 - (R * Omega * u_0 + u_0**2 / 2.0) * jnp.sin(lat)**2) / g

    # Mountain topography
    lon_c = 3.0 * jnp.pi / 2.0
    lat_c = jnp.pi / 6.0
    R_m = jnp.pi / 9.0
    h_s0 = 2000.0

    r = jnp.arccos(jnp.clip(
        jnp.sin(lat_c) * jnp.sin(lat) +
        jnp.cos(lat_c) * jnp.cos(lat) * jnp.cos(lon - lon_c),
        -1.0, 1.0))
    h_s_data = jnp.where(r < R_m, h_s0 * (1.0 - r / R_m), 0.0)

    u_east_cell = u_0 * jnp.cos(lat)
    v_north_cell = jnp.zeros_like(lat)
    u_edge = _project_velocity_to_edges(u_east_cell, v_north_cell, mesh)

    return MPASShallowWaterState(
        h=Field(data=h_data, name="h", dims=("nCells",), units="m"),
        u=Field(data=u_edge, name="u", dims=("nEdges",), units="m/s",
                staggering="edge"),
        h_s=Field(data=h_s_data, name="h_s", dims=("nCells",), units="m"),
    )


def williamson_test6_mpas(mesh: VoronoiMesh) -> MPASShallowWaterState:
    """Williamson Test Case 6: Rossby-Haurwitz wave 4.

    Parameters
    ----------
    mesh : VoronoiMesh

    Returns
    -------
    MPASShallowWaterState
    """
    R = mesh.radius
    Omega = constants.Omega
    g = constants.g

    # Parameters
    K = 7.848e-6    # angular frequency
    R_val = 4       # wave number
    h_0 = 8000.0    # mean depth [m]

    lat = mesh.latCell
    lon = mesh.lonCell

    cos_lat = jnp.cos(lat)
    sin_lat = jnp.sin(lat)

    # Rossby-Haurwitz wave 4 initial conditions
    A = 0.5 * K * (2.0 * Omega + K) * cos_lat**2 + \
        0.25 * K**2 * cos_lat**(2 * R_val) * \
        ((R_val + 1) * cos_lat**2 +
         (2 * R_val**2 - R_val - 2) -
         2.0 * R_val**2 / (cos_lat**2 + 1e-30))

    B = (2.0 * (Omega + K) * K * cos_lat**(R_val - 1) *
         ((R_val**2 + 2 * R_val + 2) -
          (R_val + 1)**2 * cos_lat**2)) / \
        ((R_val + 1) * (R_val + 2) + 1e-30)

    C = 0.25 * K**2 * cos_lat**(2 * R_val) * \
        ((R_val + 1) * cos_lat**2 - (R_val + 2))

    h_data = h_0 + (R**2 / g) * (A + B * jnp.cos(R_val * lon) +
                                   C * jnp.cos(2 * R_val * lon))

    # Velocity field
    u_east_cell = R * cos_lat * K + \
        R * K * cos_lat**(R_val - 1) * \
        (R_val * sin_lat**2 - cos_lat**2) * jnp.cos(R_val * lon)
    v_north_cell = -R * K * R_val * cos_lat**(R_val - 1) * \
        sin_lat * jnp.sin(R_val * lon)

    u_edge = _project_velocity_to_edges(u_east_cell, v_north_cell, mesh)
    h_s_data = jnp.zeros_like(h_data)

    return MPASShallowWaterState(
        h=Field(data=h_data, name="h", dims=("nCells",), units="m"),
        u=Field(data=u_edge, name="u", dims=("nEdges",), units="m/s",
                staggering="edge"),
        h_s=Field(data=h_s_data, name="h_s", dims=("nCells",), units="m"),
    )


def compute_error_norms_mpas(h, h_ref, mesh):
    """Compute L1, L2, Linf error norms for height field.

    Parameters
    ----------
    h : jax.Array, shape (nCells,)
    h_ref : jax.Array, shape (nCells,)
    mesh : VoronoiMesh

    Returns
    -------
    dict : {l1, l2, linf} normalized error norms.
    """
    err = h - h_ref
    area = mesh.areaCell

    l1 = jnp.sum(jnp.abs(err) * area) / jnp.sum(jnp.abs(h_ref) * area)
    l2 = jnp.sqrt(jnp.sum(err**2 * area) / jnp.sum(h_ref**2 * area))
    linf = jnp.max(jnp.abs(err)) / jnp.max(jnp.abs(h_ref))

    return {"l1": float(l1), "l2": float(l2), "linf": float(linf)}
