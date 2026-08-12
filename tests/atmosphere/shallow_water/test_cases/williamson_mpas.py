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
from legoesm.core.precision import get_policy
from legoesm.core.williamson_sw_analytic import (
    RH4_MEAN_DEPTH_M,
    W2_GH0,
    W5_H0_M,
    W5_UBAR_MS,
    rossby_haurwitz_4_geopotential,
    rossby_haurwitz_4_winds,
    solid_body_geopotential,
    solid_body_rotation_speed,
    solid_body_winds,
    williamson_5_mountain_height,
)
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
    g = constants.g

    u_0 = solid_body_rotation_speed(R)          # ~38.6 m/s
    lat = mesh.latCell
    lon = mesh.lonCell

    # Analytic fields: ONE shared definition (williamson_sw_analytic).
    h_data = solid_body_geopotential(lon, lat, radius=R,
                                     omega=constants.Omega, u0=u_0,
                                     gh0=W2_GH0, xp=jnp) / g
    h_s_data = jnp.zeros_like(h_data)
    u_east_cell, v_north_cell = solid_body_winds(lon, lat, u0=u_0, xp=jnp)

    u_edge = _project_velocity_to_edges(u_east_cell, v_north_cell, mesh)

    # Thread the precision-policy storage dtype (see williamson_test5_mpas).
    _dtype = get_policy().storage
    return MPASShallowWaterState(
        h=Field(data=h_data.astype(_dtype), name="h", dims=("nCells",),
                units="m"),
        u=Field(data=u_edge.astype(_dtype), name="u", dims=("nEdges",),
                units="m/s", staggering="edge"),
        h_s=Field(data=h_s_data.astype(_dtype), name="h_s", dims=("nCells",),
                  units="m"),
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
    g = constants.g

    u_0 = W5_UBAR_MS
    lat = mesh.latCell
    lon = mesh.lonCell

    # ★ CHANGED: this used a GREAT-CIRCLE radius.  Williamson et al.
    # (1992) case 5 and test_cases.F90:1185 both specify the clipped
    # PLANAR (lon, lat) radius -- the cubed-sphere sibling already used
    # it, so W5 was not the same mountain across grids.
    h_s_data = williamson_5_mountain_height(lon, lat, xp=jnp)

    # h is fluid DEPTH: the solver computes B = KE + g*(h + h_s).
    h_free = solid_body_geopotential(lon, lat, radius=R,
                                     omega=constants.Omega, u0=u_0,
                                     gh0=W5_H0_M * g, xp=jnp) / g
    h_data = h_free - h_s_data

    u_east_cell, v_north_cell = solid_body_winds(lon, lat, u0=u_0, xp=jnp)
    u_edge = _project_velocity_to_edges(u_east_cell, v_north_cell, mesh)

    # Thread the precision-policy storage dtype through the state (the mesh
    # coords are float32, so the arithmetic above yields float32 regardless of
    # policy). Mirrors held_suarez_init_mpas: under PrecisionPolicy.fp64() the
    # state must be fp64 so it agrees with the fp64 mass-fixer tendency inside a
    # scan carry; under the default fp32 policy this is a no-op.
    _dtype = get_policy().storage
    return MPASShallowWaterState(
        h=Field(data=h_data.astype(_dtype), name="h", dims=("nCells",),
                units="m"),
        u=Field(data=u_edge.astype(_dtype), name="u", dims=("nEdges",),
                units="m/s", staggering="edge"),
        h_s=Field(data=h_s_data.astype(_dtype), name="h_s", dims=("nCells",),
                  units="m"),
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
    g = constants.g
    h_0 = RH4_MEAN_DEPTH_M    # mean depth [m]

    lat = mesh.latCell
    lon = mesh.lonCell

    # Rossby-Haurwitz wave 4 initial conditions -- ONE shared analytic
    # definition (legoesm.core.williamson_sw_analytic), which carries
    # B ~ cos^R per Williamson Eq. 145 and the FV3 duo oracle; this file
    # previously had its own copy with cos^(R-1).
    h_data = rossby_haurwitz_4_geopotential(
        lon, lat, radius=R, omega=constants.Omega, gh0=h_0 * g,
        xp=jnp) / g
    u_east_cell, v_north_cell = rossby_haurwitz_4_winds(
        lon, lat, radius=R, xp=jnp)

    u_edge = _project_velocity_to_edges(u_east_cell, v_north_cell, mesh)
    h_s_data = jnp.zeros_like(h_data)

    # Thread the precision-policy storage dtype (see williamson_test5_mpas).
    _dtype = get_policy().storage
    return MPASShallowWaterState(
        h=Field(data=h_data.astype(_dtype), name="h", dims=("nCells",),
                units="m"),
        u=Field(data=u_edge.astype(_dtype), name="u", dims=("nEdges",),
                units="m/s", staggering="edge"),
        h_s=Field(data=h_s_data.astype(_dtype), name="h_s", dims=("nCells",),
                  units="m"),
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
