"""#1418: the vertex Neumann fill must see its true west neighbour at column 0.

`neumann_fill_vertex` rolled the ALREADY-WRAPPED (n_lat+1, n_lon+1) array, so
column 0 got the duplicate of itself as its west neighbour. The adjoint is the
sharpest probe: an all-land vertex column fills from E and W, so d(fill)/dq_W
and d(fill)/dq_E must be SYMMETRIC — at the seam they were not.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.dynamics.latlon_cgrid_operators import neumann_fill_vertex

jax.config.update("jax_enable_x64", True)

N_LAT, N_LON = 8, 16


def _mask_with_land_column(col: int) -> jnp.ndarray:
    m = np.ones((N_LAT + 1, N_LON + 1))
    m[:, col] = 0.0
    if col == 0:
        m[:, -1] = 0.0  # the wrap duplicate of column 0
    return jnp.asarray(m)


def _fill_at(col: int, j: int, q: jnp.ndarray, mask: jnp.ndarray):
    return neumann_fill_vertex(q, mask, n_passes=1)[j, col]


@pytest.mark.parametrize("land_col", [7, 0])
def test_fill_gradient_is_symmetric_in_east_and_west(land_col):
    """Interior and SEAM land columns must both average E and W equally."""
    mask = _mask_with_land_column(land_col)
    q0 = jnp.asarray(np.arange((N_LAT + 1) * (N_LON + 1), dtype=float).reshape(
        N_LAT + 1, N_LON + 1))
    j = 4
    g = jax.grad(lambda q: _fill_at(land_col, j, q, mask))(q0)
    west = (land_col - 1) % N_LON
    east = (land_col + 1) % N_LON
    assert float(g[j, west]) == pytest.approx(0.5), (
        f"west sensitivity at land column {land_col}: got {float(g[j, west])}, "
        f"expected 0.5 — column 0 receiving ITSELF as its west neighbour is "
        f"the #1418 defect.")
    assert float(g[j, east]) == pytest.approx(0.5)


def test_seam_column_is_not_self_referential():
    """d(fill)/d(itself) must be zero at a land vertex: it is filled, not kept."""
    mask = _mask_with_land_column(0)
    q0 = jnp.asarray(np.random.default_rng(0).normal(
        size=(N_LAT + 1, N_LON + 1)))
    g = jax.grad(lambda q: _fill_at(0, 4, q, mask))(q0)
    assert float(g[4, 0]) == pytest.approx(0.0, abs=1e-12)
    assert float(g[4, -1]) == pytest.approx(0.0, abs=1e-12)


def test_result_is_zonally_shift_equivariant():
    """Rolling the ocean by one column must roll the filled field by one.

    A seam-only index error breaks this; a correct periodic operator cannot.
    """
    rng = np.random.default_rng(1)
    m = np.ones((N_LAT + 1, N_LON + 1))
    m[:, 5] = 0.0
    q = rng.normal(size=(N_LAT + 1, N_LON + 1))
    q[:, -1] = q[:, 0]

    def roll_core(a, k):
        core = np.roll(a[:, :-1], k, axis=1)
        return np.concatenate([core, core[:, :1]], axis=1)

    a = np.asarray(neumann_fill_vertex(jnp.asarray(q), jnp.asarray(m), 1))
    b = np.asarray(neumann_fill_vertex(jnp.asarray(roll_core(q, 3)),
                                       jnp.asarray(roll_core(m, 3)), 1))
    np.testing.assert_allclose(roll_core(a, 3), b, atol=1e-12)
