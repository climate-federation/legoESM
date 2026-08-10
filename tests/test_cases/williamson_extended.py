"""Williamson (1992) shallow-water tests not yet wired into the matrix.

Adds W6 (Rossby-Haurwitz wave-4) for cubed-sphere, lat-lon, MPAS, and
spectral grids. W3 (steady-state with compact support) and W4 (forced
flow with translating low) are deferred to M1.b — registered as TODO in
``docs/validation/dycore_validation_catalog.md``.

References
----------
Williamson, D. L., Drake, J. B., Hack, J. J., Jakob, R., & Swarztrauber,
P. N. (1992). A standard test set for numerical approximations to the
shallow water equations in spherical geometry. *J. Comput. Phys.*, 102,
211-224. (Test cases 6.)
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.state import (
    MPASShallowWaterState,
    ShallowWaterState,
)
from legoesm.core.williamson_sw_analytic import (
    RH4_MEAN_DEPTH_M,
    rossby_haurwitz_4_geopotential,
    rossby_haurwitz_4_winds,
)


# ---------------------------------------------------------------------------
# Williamson Test 6 — Rossby-Haurwitz wave 4
# ---------------------------------------------------------------------------

_W6_H0 = RH4_MEAN_DEPTH_M       # mean depth [m], Williamson 1992


def _w6_height(
    lon: jnp.ndarray,
    lat: jnp.ndarray,
    radius: float,
) -> jnp.ndarray:
    """Analytic Rossby-Haurwitz height field h(lon, lat) [m].

    Thin adapter over the one shared analytic definition in
    :mod:`legoesm.core.williamson_sw_analytic`, which returns the
    geopotential ``g*h``; this file's callers want a depth, so divide.

    NOTE: the shared version has ``B ~ cos^R``, where the copy this
    replaced had ``cos^(R-1)``.  Williamson Eq. 145 and the FV3 duo
    oracle (``test_cases.F90:1226``) both say ``cos^R``, so W6 initial
    heights move slightly at every latitude off the equator.
    """
    return rossby_haurwitz_4_geopotential(
        lon, lat, radius=radius, omega=constants.Omega,
        gh0=_W6_H0 * constants.g, xp=jnp) / constants.g


def _w6_winds_geo(
    lon: jnp.ndarray,
    lat: jnp.ndarray,
    radius: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Analytic Rossby-Haurwitz wind field (u_east, v_north) [m/s].

    Williamson 1992 Eq. 146-147, via the shared definition."""
    return rossby_haurwitz_4_winds(lon, lat, radius=radius, xp=jnp)


# ---------------------------------------------------------------------------
# Cubed-sphere W6
# ---------------------------------------------------------------------------


def williamson_test6(grid) -> ShallowWaterState:
    """W6 Rossby-Haurwitz wave-4 on cubed-sphere C-grid."""
    R = grid.radius
    lat = grid.lat
    lon = grid.lon

    h_data = _w6_height(lon, lat, R)
    u_east, v_north = _w6_winds_geo(lon, lat, R)

    # Rotate geographic (east, north) to grid-aligned (x, y)
    cos_a = jnp.cos(grid.angle)
    sin_a = jnp.sin(grid.angle)
    u_grid = cos_a * u_east + sin_a * v_north
    v_grid = -sin_a * u_east + cos_a * v_north

    h_s_data = jnp.zeros_like(h_data)
    dims = ("face", "x", "y")

    return ShallowWaterState(
        h=Field(data=h_data, name="h", dims=dims, units="m"),
        u=Field(data=u_grid, name="u", dims=dims, units="m/s"),
        v=Field(data=v_grid, name="v", dims=dims, units="m/s"),
        h_s=Field(data=h_s_data, name="h_s", dims=dims, units="m"),
    )


# ---------------------------------------------------------------------------
# Lat-lon W6
# ---------------------------------------------------------------------------


def williamson_test6_latlon(grid) -> ShallowWaterState:
    """W6 Rossby-Haurwitz wave-4 on lat-lon grid."""
    R = grid.radius
    lat = grid.lat2d
    lon = grid.lon2d

    h_data = _w6_height(lon, lat, R)
    u_data, v_data = _w6_winds_geo(lon, lat, R)
    h_s_data = jnp.zeros_like(h_data)

    dims = ("lat", "lon")
    return ShallowWaterState(
        h=Field(data=h_data, name="h", dims=dims, units="m"),
        u=Field(data=u_data, name="u", dims=dims, units="m/s"),
        v=Field(data=v_data, name="v", dims=dims, units="m/s"),
        h_s=Field(data=h_s_data, name="h_s", dims=dims, units="m"),
    )


# ---------------------------------------------------------------------------
# MPAS Voronoi mesh W6 — re-export the existing implementation
# ---------------------------------------------------------------------------


def williamson_test6_mpas(mesh) -> MPASShallowWaterState:
    """W6 Rossby-Haurwitz wave-4 on an MPAS Voronoi mesh.

    Thin wrapper around the existing
    ``tests.atmosphere.shallow_water.test_cases.williamson_mpas``
    implementation so that the matrix script can route through one
    canonical entry-point per grid type.
    """
    from tests.atmosphere.shallow_water.test_cases.williamson_mpas import (
        williamson_test6_mpas as _impl,
    )
    return _impl(mesh)


# ---------------------------------------------------------------------------
# Spectral (Gaussian) W6
# ---------------------------------------------------------------------------


def williamson_test6_spectral(grid):
    """W6 Rossby-Haurwitz wave-4 in spectral space."""
    from legoesm.atmosphere.dynamics.gcm.spectral_sw import SpectralSWState
    from legoesm.grids.gaussian import (
        sh_analysis,
        sh_analysis_dmu,
        sh_analysis_oc2,
    )

    R = grid.radius
    g = constants.g

    lat2d = grid.lat2d
    lon2d = grid.lon2d

    h = _w6_height(lon2d, lat2d, R)
    phi = g * h
    phis = jnp.zeros_like(phi)

    u_east, v_north = _w6_winds_geo(lon2d, lat2d, R)

    cos_lat = jnp.cos(lat2d)
    u_cos = u_east * cos_lat
    v_cos = v_north * cos_lat

    im_over_a = 1j * grid.ms.astype(jnp.float64) / R
    one_over_a = 1.0 / R

    vor_hat = (
        im_over_a * sh_analysis_oc2(grid, v_cos)
        + one_over_a * sh_analysis_dmu(grid, u_cos)
    )
    div_hat = (
        im_over_a * sh_analysis_oc2(grid, u_cos)
        - one_over_a * sh_analysis_dmu(grid, v_cos)
    )

    phi_hat = sh_analysis(grid, phi)
    phis_hat = sh_analysis(grid, phis)

    dims = ("spectral",)
    return SpectralSWState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims, units="1/s"),
        phi_hat=Field(data=phi_hat, name="phi_hat", dims=dims, units="m^2/s^2"),
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims, units="m^2/s^2"),
    )
