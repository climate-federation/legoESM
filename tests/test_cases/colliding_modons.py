"""Colliding modons — a nonlinear shallow-water test for global dycores.

Lin et al. (2017), "Colliding Modons: A Nonlinear Test for the Evaluation
of Global Dynamical Cores", J. Adv. Model. Earth Syst., 9,
doi:10.1002/2017MS000965.

Two zonal Gaussian wind bursts on a **non-rotating** (f = 0) sphere with a
constant fluid depth.  The shear flanks of each burst roll up into a
counter-rotating vortex dipole (a modon).  The westerly modon (centre
lon = pi/2) and the easterly modon (centre lon = 3*pi/2, sign flipped)
propagate towards each other, collide, exchange partner vortices and
depart, returning near their initial positions after ~100 days — a
stringent test of nonlinear vorticity dynamics free of any analytic
steady state to hide errors behind.

Faithful port of the FV3 ``test_cases.F90`` case-8 initial condition.  The
FV3 code projects a purely *eastward* velocity ``utmp = Umax*exp(-(r/r0)^2)``
(``r`` = great-circle distance to the modon centre) onto the D-grid edge
tangents, i.e. the physical IC velocity is a zonal Gaussian jet; the dipole
structure is emergent, not prescribed.

Parameters (FV3 ``soliton_*`` namelist defaults):
    Umax = 50 m/s, size r0 = 750 km, Nsolitons = 2,
    gh0 = 5e3 * g  ->  h0 = 5000 m, centres at the equator.

IMPORTANT: the planet must be non-rotating.  Build the grid with
``omega=0.0`` (e.g. ``create_cubed_sphere(n, omega=0.0)``); these IC
functions only set the prognostic fields, not the Coriolis parameter.
"""

from __future__ import annotations

import jax.numpy as jnp
from legoesm.core.field import Field
from legoesm.core.state import MPASShallowWaterState, ShallowWaterState
from legoesm.grids.cubed_sphere import great_circle_distance

from legoesm import constants

# --- FV3 case-8 soliton defaults (test_cases.F90) ---
_MODON_UMAX = 50.0         # peak jet speed Umax [m/s]
_MODON_SIZE = 750.0e3      # Gaussian size r0 [m]
_MODON_H0 = 5000.0         # constant fluid depth h0 = gh0/g [m]  (gh0 = 5e3*g)
_MODON_LON1 = 0.5 * jnp.pi   # westerly burst centre longitude [rad]
_MODON_LON2 = 1.5 * jnp.pi   # easterly burst centre longitude [rad]
_MODON_LAT0 = 0.0            # both centres on the equator [rad]


def _modon_winds_geo(
    lon: jnp.ndarray,
    lat: jnp.ndarray,
    radius: float,
    *,
    u_max: float = _MODON_UMAX,
    r0: float = _MODON_SIZE,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Geographic (u_east, v_north) [m/s] for the two-soliton IC.

    Two zonal Gaussian jets: ``+u_max`` centred at ``lon = pi/2`` and
    ``-u_max`` centred at ``lon = 3*pi/2`` (FV3 ``v(i,j) -= ...`` for the
    second soliton).  ``v_north = 0`` everywhere — the velocity is purely
    eastward; haversine ``great_circle_distance`` handles longitude
    periodicity so the centres need no wrapping.
    """
    r1 = great_circle_distance(lon, lat, _MODON_LON1, _MODON_LAT0, radius)
    r2 = great_circle_distance(lon, lat, _MODON_LON2, _MODON_LAT0, radius)
    # FV3 test_cases.F90 case-8 convention: burst #1 at lon = pi/2 is
    # WESTERLY (``u += utmp`` at p0 = (pi/2, 0)), burst #2 at 3*pi/2 is
    # easterly (``u -= utmp``), so the modons collide at lon = 180 as in
    # Lin et al. (2017).  Inverting the signs (commit ba28d8b2b) is a
    # global 180-degree longitude mirror — identical dynamics, but every
    # position disagrees with the paper's figures, this docstring, and
    # the sign-pinning IC tests (test_colliding_modons_ic.py), which
    # that commit left red.  Keep FV3's orientation.
    u_east = (
        u_max * jnp.exp(-(r1 / r0) ** 2)
        - u_max * jnp.exp(-(r2 / r0) ** 2)
    )
    v_north = jnp.zeros_like(u_east)
    return u_east, v_north


def _modon_height(lon: jnp.ndarray, lat: jnp.ndarray) -> jnp.ndarray:
    """Constant fluid depth ``h0`` (flat free surface, flat terrain)."""
    return jnp.full_like(lat, _MODON_H0)


# ---------------------------------------------------------------------------
# Cubed-sphere
# ---------------------------------------------------------------------------


def colliding_modons(grid) -> ShallowWaterState:
    """Colliding-modons IC on the cubed-sphere C-grid (cell centres)."""
    R = grid.radius
    lat = grid.lat
    lon = grid.lon

    h_data = _modon_height(lon, lat)
    u_east, v_north = _modon_winds_geo(lon, lat, R)

    cos_a = jnp.cos(grid.angle)
    sin_a = jnp.sin(grid.angle)
    u_grid = cos_a * u_east + sin_a * v_north
    v_grid = -sin_a * u_east + cos_a * v_north

    dims = ("face", "x", "y")
    return ShallowWaterState(
        h=Field(data=h_data, name="h", dims=dims, units="m"),
        u=Field(data=u_grid, name="u", dims=dims, units="m/s"),
        v=Field(data=v_grid, name="v", dims=dims, units="m/s"),
        h_s=Field(data=jnp.zeros_like(h_data), name="h_s", dims=dims, units="m"),
    )


# ---------------------------------------------------------------------------
# Cubed-sphere FV3 edge-midpoint C-D grid (FV3EdgeShallowWaterModel)
# ---------------------------------------------------------------------------


def colliding_modons_cdgrid(grid, cdgrid):
    """Colliding-modons IC on the FV3 edge-midpoint C-D grid.

    Builds the ``FV3EdgeShallowWaterState`` consumed by
    ``FV3EdgeShallowWaterModel``: the shared zonal-Gaussian wind kernel
    ``_modon_winds_geo`` is evaluated at the two D-grid edge-midpoint staggers
    and projected onto the edge directions via the stored
    ``cos/sin_angle_edge_*`` metric rotation (``v_north = 0``, so only the
    eastward component survives — the same rotation the Williamson cube ICs
    use; no geometry re-derivation).  Returns ``(state, cdgrid_nonrot)`` where
    ``cdgrid_nonrot`` has the corner Coriolis field zeroed (FV3 case-8
    ``f0 = fC = 0``); assign it to ``model.cdgrid`` before stepping so the run
    is non-rotating.
    """
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterState,
    )

    R = grid.radius
    h = _modon_height(grid.lon, grid.lat)
    h_s = jnp.zeros_like(h)

    u_e_x, _ = _modon_winds_geo(cdgrid.lon_edge_x, cdgrid.lat_edge_x, R)
    u_d = cdgrid.cos_angle_edge_x * u_e_x
    u_e_y, _ = _modon_winds_geo(cdgrid.lon_edge_y, cdgrid.lat_edge_y, R)
    v_d = -cdgrid.sin_angle_edge_y * u_e_y

    state = FV3EdgeShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)
    cdgrid_nonrot = cdgrid._replace(f_corner=jnp.zeros_like(cdgrid.f_corner))
    return state, cdgrid_nonrot


# ---------------------------------------------------------------------------
# Lat-lon
# ---------------------------------------------------------------------------


def colliding_modons_latlon(grid) -> ShallowWaterState:
    """Colliding-modons IC on the lat-lon grid."""
    R = grid.radius
    lat = grid.lat2d
    lon = grid.lon2d

    h_data = _modon_height(lon, lat)
    u_data, v_data = _modon_winds_geo(lon, lat, R)

    dims = ("lat", "lon")
    return ShallowWaterState(
        h=Field(data=h_data, name="h", dims=dims, units="m"),
        u=Field(data=u_data, name="u", dims=dims, units="m/s"),
        v=Field(data=v_data, name="v", dims=dims, units="m/s"),
        h_s=Field(data=jnp.zeros_like(h_data), name="h_s", dims=dims, units="m"),
    )


# ---------------------------------------------------------------------------
# MPAS Voronoi mesh
# ---------------------------------------------------------------------------


def colliding_modons_mpas(mesh) -> MPASShallowWaterState:
    """Colliding-modons IC on an MPAS Voronoi mesh.

    Projects the geographic (u_east, v_north) onto edge normals via the
    mesh's edge unit vectors, mirroring the MPAS Williamson ICs.
    """
    from tests.atmosphere.shallow_water.test_cases.williamson_mpas import (
        _project_velocity_to_edges,
    )

    R = mesh.radius
    lat_c = mesh.latCell
    lon_c = mesh.lonCell

    h_data = _modon_height(lon_c, lat_c)

    # Winds at cell centres, then projected/averaged onto edge normals.
    u_east, v_north = _modon_winds_geo(lon_c, lat_c, R)
    u_edge = _project_velocity_to_edges(u_east, v_north, mesh)

    return MPASShallowWaterState(
        h=Field(data=h_data, name="h", dims=("nCells",), units="m"),
        u=Field(data=u_edge, name="u", dims=("nEdges",), units="m/s",
                staggering="edge"),
        h_s=Field(data=jnp.zeros_like(h_data), name="h_s", dims=("nCells",),
                  units="m"),
    )


# ---------------------------------------------------------------------------
# Spectral (Gaussian) grid
# ---------------------------------------------------------------------------


def colliding_modons_spectral(grid):
    """Colliding-modons IC in spectral space (Gaussian grid)."""
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

    phi = g * _modon_height(lon2d, lat2d)
    phis = jnp.zeros_like(phi)

    u_east, v_north = _modon_winds_geo(lon2d, lat2d, R)
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
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims,
                       units="m^2/s^2"),
    )
