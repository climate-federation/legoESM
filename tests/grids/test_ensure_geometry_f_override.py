"""Regression guard: ``ensure_geometry()`` must preserve ``grid.f``.

When ``LatLonCGridOceanModel.__init__`` converts a ``LatLonGrid`` to
a ``LatLonCGridGeometry`` via ``ensure_geometry()``, it must propagate
the input grid's ``f`` array.  Without that propagation, anyone who
sets ``grid._replace(f=…)`` to build an f-plane / β-plane / custom-
rotation idealised configuration silently loses the override, and
the model is built with the *default* ``constants.Omega × sin(lat)``
Coriolis instead.

The bug has been introduced and re-introduced twice on this codebase
(commit eb657b57 originally; reverted/lost in 57c01a23).  This test
exists so that any future revert is loud, not silent.

Tests three patterns idealised studies actually use:
  1. Pure f-plane (``f = 0`` for gravity-wave isolation)
  2. f-plane at a chosen latitude (``f = const ≠ 0`` for inertial
     oscillation, Eady, …)
  3. β-plane (``f = f0 + β·y`` for Rossby waves, gyre studies)

For each pattern, the test:
  - constructs a ``LatLonGrid``,
  - overrides ``grid.f``,
  - runs ``ensure_geometry``,
  - asserts the geometry's ``f_T`` matches the override,
  - asserts ``f_u`` and ``f_v`` are the correct face-averages of
    ``f_T`` (same stencil the rest of the dycore uses).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid, ensure_geometry


N_LAT = 4
N_LON = 32


def _expected_f_u(f_T: jnp.ndarray) -> jnp.ndarray:
    """Face-average of f_T to u-points using the same stencil ensure_geometry uses."""
    f_u_interior = 0.5 * (jnp.roll(f_T, 1, axis=1) + f_T)
    return jnp.concatenate([f_u_interior, f_u_interior[:, 0:1]], axis=1)


def _expected_f_v(f_T: jnp.ndarray) -> jnp.ndarray:
    """Face-average of f_T to v-points using the same stencil ensure_geometry uses."""
    f_v_interior = 0.5 * (f_T[:-1] + f_T[1:])
    return jnp.concatenate([f_T[0:1], f_v_interior, f_T[-1:]], axis=0)


def _check_geometry_uses_grid_f(grid, expected_f_T: jnp.ndarray) -> None:
    geom = ensure_geometry(grid)
    np.testing.assert_allclose(
        np.asarray(geom.f_T),
        np.asarray(expected_f_T),
        rtol=0.0, atol=1e-12,
        err_msg="ensure_geometry must propagate grid.f to geom.f_T",
    )
    np.testing.assert_allclose(
        np.asarray(geom.f_u),
        np.asarray(_expected_f_u(expected_f_T)),
        rtol=0.0, atol=1e-12,
        err_msg="ensure_geometry must average grid.f to f_u using the dycore stencil",
    )
    np.testing.assert_allclose(
        np.asarray(geom.f_v),
        np.asarray(_expected_f_v(expected_f_T)),
        rtol=0.0, atol=1e-12,
        err_msg="ensure_geometry must average grid.f to f_v using the dycore stencil",
    )


def test_fplane_f_zero_preserved():
    """``grid._replace(f = 0)`` must produce a geometry with f = 0 everywhere."""
    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
    grid = grid._replace(f=jnp.zeros_like(grid.f))
    _check_geometry_uses_grid_f(grid, jnp.zeros((N_LAT, N_LON), dtype=grid.f.dtype))


def test_fplane_f_constant_preserved():
    """``grid._replace(f = f0)`` must produce a geometry with f = f0 everywhere."""
    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
    f0 = 1.0e-4
    custom_f = jnp.full_like(grid.f, f0)
    grid = grid._replace(f=custom_f)
    _check_geometry_uses_grid_f(grid, custom_f)


def test_beta_plane_f_y_linear_preserved():
    """``grid._replace(f = f0 + β·y)`` (β-plane) must be preserved cell-by-cell."""
    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
    f0 = 1.0e-4
    beta = 2.0e-11
    # y-coordinate at cell centres (just a representative axis — the
    # specific physical interpretation doesn't matter for this test).
    y = jnp.arange(N_LAT, dtype=grid.f.dtype) - N_LAT / 2.0
    custom_f = (f0 + beta * y)[:, None] * jnp.ones((1, N_LON), dtype=grid.f.dtype)
    grid = grid._replace(f=custom_f)
    _check_geometry_uses_grid_f(grid, custom_f)


def test_default_f_unchanged_when_grid_f_matches_omega_sin_lat():
    """When grid.f is the default ``2Ω sin(lat)`` (i.e. no override),
    ensure_geometry's output must equal grid.f to machine precision —
    no double-application of the omega computation."""
    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
    # grid.f was set by create_latlon_grid via 2*omega*sin(lat).
    expected_f_T = grid.f
    _check_geometry_uses_grid_f(grid, expected_f_T)
