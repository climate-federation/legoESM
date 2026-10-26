"""Cosine bell advection test case from Putman & Lin (2007), Section 4.1.

Solid body rotation of a cosine bell on the sphere with prescribed winds.
This is a pure transport test: winds are analytically prescribed and do
not evolve.  After 12 days (one full revolution) the bell returns to its
initial position, so the analytic solution equals the initial condition.

The flow angle beta = pi/4 directs the bell over the corners of the
cubed-sphere, which is the most challenging orientation.

References
----------
- Putman, W. M. & Lin, S.-J. (2007). Finite-volume transport on various
  cubed-sphere grids. J. Comput. Phys., 227(1), 55-78.
- Williamson, D. L. et al. (1992). A standard test set for numerical
  approximations to the shallow water equations in spherical geometry.
  J. Comput. Phys., 102(1), 211-224. (Test Case 1)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.field import Field


# ---------------------------------------------------------------------------
# Physical parameters (PL07 Section 4.1)
# ---------------------------------------------------------------------------

_H0 = 1000.0          # Peak height of cosine bell [m]
_PERIOD = 12 * 86400.0  # Full revolution period [s]


def _rotate_lonlat(lon, lat, angle, beta):
    """Rotate (lon, lat) backwards by *angle* around the PL07 rotation axis.

    The rotation axis lies in the XZ plane at colatitude beta from Z:
        n = (-sin(beta), 0, cos(beta))

    Used to compute the exact solution at arbitrary time t by evaluating
    the initial condition at the reverse-rotated coordinates.
    """
    cos_a = jnp.cos(-angle)
    sin_a = jnp.sin(-angle)
    nx, ny, nz = -jnp.sin(beta), 0.0, jnp.cos(beta)

    # (lon, lat) → Cartesian
    x = jnp.cos(lat) * jnp.cos(lon)
    y = jnp.cos(lat) * jnp.sin(lon)
    z = jnp.sin(lat)

    # Rodrigues rotation
    dot = nx * x + nz * z  # ny = 0
    cx = ny * z - nz * y
    cy = nz * x - nx * z
    cz = nx * y - ny * x

    xr = x * cos_a + cx * sin_a + nx * dot * (1 - cos_a)
    yr = y * cos_a + cy * sin_a + ny * dot * (1 - cos_a)
    zr = z * cos_a + cz * sin_a + nz * dot * (1 - cos_a)

    # Cartesian → (lon, lat)
    lon_r = jnp.arctan2(yr, xr)
    lat_r = jnp.arcsin(jnp.clip(zr, -1.0, 1.0))
    return lon_r, lat_r


def cosine_bell_exact(lon, lat, radius, t, beta=jnp.pi / 4):
    """Exact cosine bell solution at time *t* [s].

    Evaluates the initial cosine bell at the reverse-rotated coordinates.
    """
    u0 = 2.0 * jnp.pi * radius / _PERIOD
    angle = u0 * t / radius   # angular displacement [rad]
    lon_r, lat_r = _rotate_lonlat(lon, lat, angle, beta)
    return _cosine_bell(lon_r, lat_r, radius)


def _cosine_bell(lon, lat, radius):
    """Cosine bell height field (Eq. 25 of PL07).

    Parameters
    ----------
    lon, lat : array-like  [radians]
    radius : float  [m]

    Returns
    -------
    h : array, same shape as lon — height field [m].
    """
    R0 = radius / 3.0
    lon_c = 3.0 * jnp.pi / 2.0   # 270 E
    lat_c = 0.0                   # equator

    # Great-circle distance from bell centre
    r = radius * jnp.arccos(jnp.clip(
        jnp.sin(lat_c) * jnp.sin(lat)
        + jnp.cos(lat_c) * jnp.cos(lat) * jnp.cos(lon - lon_c),
        -1.0, 1.0,
    ))
    return jnp.where(r < R0, (_H0 / 2.0) * (1.0 + jnp.cos(jnp.pi * r / R0)),
                     0.0)


def _rotation_winds_geo(lon, lat, radius, beta):
    """Solid body rotation wind in geographic (east, north) components.

    Eqs. 29-30 of PL07.

    Parameters
    ----------
    lon, lat : array-like  [radians]
    radius : float  [m]
    beta : float  [radians] — angle of rotation axis from polar axis.

    Returns
    -------
    u_east, v_north : arrays — wind components [m/s].
    """
    u0 = 2.0 * jnp.pi * radius / _PERIOD
    cos_b = jnp.cos(beta)
    sin_b = jnp.sin(beta)
    u_east = u0 * (jnp.cos(lat) * cos_b + jnp.sin(lat) * jnp.cos(lon) * sin_b)
    v_north = -u0 * jnp.sin(lon) * sin_b
    return u_east, v_north


def rotation_streamfunction(lon, lat, radius, beta):
    """Velocity streamfunction [m^2/s] of the solid-body rotation wind.

    ``(u_east, v_north) = (-1/R dpsi/dlat, 1/(R cos lat) dpsi/dlon)`` is the
    curl of this ``psi``; it matches :func:`_rotation_winds_geo` exactly.  This
    is the FV3 ``test_cases.F90`` wind_field=0 streamfunction (with
    ``Ubar = u0``) and is used to build a discretely divergence-free transport
    mass flux on the cube (issue 504; see
    ``fv_tp_2d.streamfunction_mass_fluxes``).
    """
    u0 = 2.0 * jnp.pi * radius / _PERIOD
    return -u0 * radius * (jnp.sin(lat) * jnp.cos(beta)
                           - jnp.cos(lon) * jnp.cos(lat) * jnp.sin(beta))


# =========================================================================
# Cubed-sphere (FV3 C-D grid)
# =========================================================================

def cosine_bell_cubesphere(grid, cdgrid, beta=jnp.pi / 4):
    """Cosine bell initial condition for the FV3 edge-midpoint C-D grid.

    Returns
    -------
    FV3EdgeShallowWaterState
    """
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterState)

    R = grid.radius

    # Height field at cell centres
    h = _cosine_bell(grid.lon, grid.lat, R)
    h_s = jnp.zeros_like(h)

    # Winds at edge midpoints (D-grid stagger)
    u_e_x, v_n_x = _rotation_winds_geo(
        cdgrid.lon_edge_x, cdgrid.lat_edge_x, R, beta)
    u_d = (cdgrid.cos_angle_edge_x * u_e_x
           + cdgrid.sin_angle_edge_x * v_n_x)

    u_e_y, v_n_y = _rotation_winds_geo(
        cdgrid.lon_edge_y, cdgrid.lat_edge_y, R, beta)
    v_d = (-cdgrid.sin_angle_edge_y * u_e_y
           + cdgrid.cos_angle_edge_y * v_n_y)

    return FV3EdgeShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)


# =========================================================================
# Lat-lon
# =========================================================================

def cosine_bell_latlon(grid, beta=jnp.pi / 4):
    """Cosine bell initial condition on a lat-lon grid.

    Returns
    -------
    ShallowWaterState
    """
    from legoesm.core.state import ShallowWaterState

    R = grid.radius
    lat = grid.lat2d
    lon = grid.lon2d

    h = _cosine_bell(lon, lat, R)
    h_s = jnp.zeros_like(h)
    u_east, v_north = _rotation_winds_geo(lon, lat, R, beta)

    dims = ("lat", "lon")
    return ShallowWaterState(
        h=Field(data=h, name="h", dims=dims, units="m"),
        u=Field(data=u_east, name="u", dims=dims, units="m/s"),
        v=Field(data=v_north, name="v", dims=dims, units="m/s"),
        h_s=Field(data=h_s, name="h_s", dims=dims, units="m"),
    )


# =========================================================================
# MPAS (Voronoi)
# =========================================================================

def cosine_bell_mpas(mesh, beta=jnp.pi / 4):
    """Cosine bell initial condition on an MPAS Voronoi mesh.

    Returns
    -------
    MPASShallowWaterState
    """
    from legoesm.core.state import MPASShallowWaterState

    R = mesh.radius

    h = _cosine_bell(mesh.lonCell, mesh.latCell, R)
    h_s = jnp.zeros_like(h)

    # Cell-centre winds → project onto edge normals
    u_east_cell, v_north_cell = _rotation_winds_geo(
        mesh.lonCell, mesh.latCell, R, beta)
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    u_e = 0.5 * (u_east_cell[c1] + u_east_cell[c2])
    v_e = 0.5 * (v_north_cell[c1] + v_north_cell[c2])
    u_normal = u_e * jnp.cos(mesh.angleEdge) + v_e * jnp.sin(mesh.angleEdge)

    return MPASShallowWaterState(
        h=Field(data=h, name="h", dims=("nCells",), units="m"),
        u=Field(data=u_normal, name="u", dims=("nEdges",), units="m/s",
                staggering="edge"),
        h_s=Field(data=h_s, name="h_s", dims=("nCells",), units="m"),
    )


# =========================================================================
# Spectral (Gaussian grid)
# =========================================================================

def cosine_bell_spectral(grid, beta=jnp.pi / 4):
    """Cosine bell initial condition in spectral space.

    Returns
    -------
    SpectralSWState
    """
    from legoesm.atmosphere.dynamics.gcm.spectral_sw import SpectralSWState
    from legoesm.grids.gaussian import sh_analysis

    R = grid.radius
    g = constants.g

    lat2d = grid.lat2d
    lon2d = grid.lon2d

    h = _cosine_bell(lon2d, lat2d, R)
    phi = g * h   # geopotential
    phis = jnp.zeros_like(phi)

    # Vorticity and divergence from solid body rotation at angle beta
    # For arbitrary beta: zeta = (2*u0/R) * (sin(lat)*cos(beta)
    #                             - cos(lat)*cos(lon)*sin(beta)) ... complicated.
    # Easiest: compute u, v on grid, then get spectral vor/div via analysis.
    u_east, v_north = _rotation_winds_geo(lon2d, lat2d, R, beta)

    # Spectral vorticity = curl(V), divergence = div(V)
    # Use spectral analysis of u*cos(lat), v*cos(lat) and the
    # spherical harmonic derivative operators.
    cos_lat = jnp.cos(lat2d)
    u_cos = u_east * cos_lat
    v_cos = v_north * cos_lat

    from legoesm.grids.gaussian import (
        sh_analysis_oc2, sh_analysis_dmu)

    _cdt = jnp.complex64 if grid.Pnm.dtype == jnp.float32 else jnp.complex128
    im_over_a = (1j * grid.ms / R).astype(_cdt)
    one_over_a = 1.0 / R

    # vor_hat = -(im/a * sh_oc2(V) + (1/a) * sh_dmu(U))  [sign: curl]
    # Actually: curl = (im/a)*sh_oc2(v_cos) + (1/a)*sh_dmu(u_cos)
    # div  = (im/a)*sh_oc2(u_cos) - (1/a)*sh_dmu(v_cos)
    # But for the SW equations, vor_hat stores RELATIVE vorticity.
    # For solid body rotation: relative vorticity = curl(V) - 0 = curl(V)
    vor_hat = (im_over_a * sh_analysis_oc2(grid, v_cos)
               + one_over_a * sh_analysis_dmu(grid, u_cos))
    div_hat = (im_over_a * sh_analysis_oc2(grid, u_cos)
               - one_over_a * sh_analysis_dmu(grid, v_cos))

    phi_hat = sh_analysis(grid, phi)
    phis_hat = sh_analysis(grid, phis)

    dims = ("spectral",)
    return SpectralSWState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims, units="1/s"),
        phi_hat=Field(data=phi_hat, name="phi_hat", dims=dims, units="m^2/s^2"),
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims, units="m^2/s^2"),
    )


# =========================================================================
# Error norms
# =========================================================================

def cosine_bell_error_norms(h_final, h_initial, area):
    """Compute L1, L2, Linf error norms (PL07 convention).

    Uses the initial condition as the exact reference (12-day revolution).

    Parameters
    ----------
    h_final : jax.Array — final height field
    h_initial : jax.Array — initial height field (= exact solution)
    area : jax.Array — cell areas [m^2]

    Returns
    -------
    dict : {l1, l2, linf} normalised error norms.
    """
    err = h_final - h_initial
    ref = h_initial

    l1 = jnp.sum(jnp.abs(err) * area) / jnp.sum(jnp.abs(ref) * area)
    l2 = jnp.sqrt(jnp.sum(err ** 2 * area) / jnp.sum(ref ** 2 * area))
    linf = jnp.max(jnp.abs(err)) / jnp.max(jnp.abs(ref))

    return {"l1": float(l1), "l2": float(l2), "linf": float(linf)}
