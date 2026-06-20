"""Direct unit tests for the colliding-modons shallow-water IC (#521).

Guards the faithful FV3 case-8 initial condition: two zonal Gaussian wind
bursts of peak speed ``Umax`` and size ``r0`` on a non-rotating sphere with
constant fluid depth ``h0``.  Tests the grid-agnostic wind/height kernels
plus the cubed-sphere and lat-lon assemblers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from tests.test_cases.colliding_modons import (
    _MODON_H0,
    _MODON_LAT0,
    _MODON_LON1,
    _MODON_LON2,
    _MODON_SIZE,
    _MODON_UMAX,
    _modon_winds_geo,
    colliding_modons,
    colliding_modons_latlon,
)

_R = constants.R_earth


def test_winds_peak_at_centres_and_meridional_zero():
    """At each modon centre the eastward wind reaches +/-Umax; v_north=0."""
    lon = jnp.array([_MODON_LON1, _MODON_LON2, 0.0])
    lat = jnp.array([_MODON_LAT0, _MODON_LAT0, _MODON_LAT0])
    u_east, v_north = _modon_winds_geo(lon, lat, _R)

    # Centre 1: +Umax (minus negligible tail of the far burst, r~half-globe).
    assert u_east[0] > 0.99 * _MODON_UMAX
    assert abs(u_east[0] - _MODON_UMAX) < 1e-3
    # Centre 2: -Umax.
    assert u_east[1] < -0.99 * _MODON_UMAX
    # Purely zonal IC.
    assert np.allclose(np.asarray(v_north), 0.0)


def test_winds_antisymmetric_between_bursts():
    """The two bursts are sign-flipped copies pi apart in longitude, so the
    field is antisymmetric under lon -> lon + pi at the equator."""
    lon = jnp.linspace(-jnp.pi, jnp.pi, 37)
    lat = jnp.zeros_like(lon)
    u_a, _ = _modon_winds_geo(lon, lat, _R)
    u_b, _ = _modon_winds_geo(lon + jnp.pi, lat, _R)
    # fp32-safe: residual is ~3e-5 in default fp32 (trig/exp roundoff); a real
    # sign-flip regression is O(Umax)=O(50), so 1e-3 still catches it.
    assert np.allclose(np.asarray(u_a), -np.asarray(u_b), atol=1e-3)


def test_gaussian_decay_at_one_r0():
    """One e-folding (great-circle r = r0) away from a centre the burst
    contribution decays to exp(-1) of its peak."""
    # Move north from centre 1 by exactly r0 along the meridian.
    dlat = _MODON_SIZE / _R
    u_east, _ = _modon_winds_geo(
        jnp.array([_MODON_LON1]), jnp.array([_MODON_LAT0 + dlat]), _R)
    # Far from centre 2, so its tail is negligible -> ~ Umax*exp(-1).
    assert abs(float(u_east[0]) - _MODON_UMAX * np.exp(-1.0)) < 1e-2


def test_cube_ic_nonrotating_constant_depth():
    """Cubed-sphere assembler: constant depth, finite winds, and the
    non-rotating grid (omega=0) actually has f == 0 everywhere."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere

    grid = create_cubed_sphere(12, omega=0.0)
    assert np.allclose(np.asarray(grid.f), 0.0), "modons require f=0"

    state = colliding_modons(grid)
    assert np.allclose(np.asarray(state.h.data), _MODON_H0)
    assert np.allclose(np.asarray(state.h_s.data), 0.0)
    assert np.all(np.isfinite(np.asarray(state.u.data)))
    assert np.all(np.isfinite(np.asarray(state.v.data)))
    # Speed magnitude is bounded by Umax (rotation preserves magnitude).
    speed = np.sqrt(np.asarray(state.u.data) ** 2 + np.asarray(state.v.data) ** 2)
    assert speed.max() <= _MODON_UMAX + 1e-6
    assert speed.max() > 0.5 * _MODON_UMAX  # the jets are present


def test_latlon_ic_constant_depth_and_zonal():
    """Lat-lon assembler: constant depth and purely zonal IC (v == 0)."""
    from legoesm.grids.latlon import create_latlon_grid

    grid = create_latlon_grid(32, 64)
    state = colliding_modons_latlon(grid)
    assert np.allclose(np.asarray(state.h.data), _MODON_H0)
    assert np.allclose(np.asarray(state.v.data), 0.0)
    assert np.all(np.isfinite(np.asarray(state.u.data)))
    assert np.abs(np.asarray(state.u.data)).max() <= _MODON_UMAX + 1e-6
