"""Equivalence tests: halo-aware ops match originals under jnp.pad(wrap)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.les import compressible_euler_plane as cep
from legoesm.atmosphere.dynamics.les import plane_operators as ops
from legoesm.atmosphere.dynamics.les import plane_operators_halo as ops_h
from legoesm.grids.plane import create_plane_grid

jax.config.update("jax_enable_x64", True)


def _pad(arr_zyx, halo=1):
    """Wrap-pad the last two axes (ny, nx). Matches single-rank
    exchange_halo_plane_yxz behavior."""
    pad_widths = [(0, 0)] * (arr_zyx.ndim - 2) + [(halo, halo)] * 2
    return jnp.pad(arr_zyx, pad_widths, mode="wrap")


@pytest.fixture
def grid():
    return create_plane_grid(
        nx=8, ny=6, nlev=4, dx=1000.0, dy=2000.0, dtype=jnp.float64,
    )


@pytest.fixture
def random_scalar(grid):
    rng = np.random.default_rng(0)
    return jnp.asarray(
        rng.standard_normal((grid.nlev, grid.ny, grid.nx)),
    )


@pytest.fixture
def random_vector(grid):
    rng = np.random.default_rng(1)
    u = jnp.asarray(rng.standard_normal((grid.nlev, grid.ny, grid.nx)))
    v = jnp.asarray(rng.standard_normal((grid.nlev, grid.ny, grid.nx)))
    return u, v


def _eq(out_halo, out_orig):
    np.testing.assert_allclose(
        np.asarray(out_halo), np.asarray(out_orig),
        rtol=1.0e-14, atol=1.0e-14,
    )


def _vlast_ref(fn, *arrs_zyx, grid):
    """Non-halo reference via the (ny, nx, nlev) vlast op.

    The base zyx interp helpers moved to ``compressible_euler_plane`` in
    vertical-last (ny, nx, nlev) layout. Bridge the test's zyx
    (nlev, ny, nx) arrays through it: zyx -> yxz -> op -> zyx.
    """
    yxz = [jnp.moveaxis(a, 0, -1) for a in arrs_zyx]
    return jnp.moveaxis(fn(*yxz, grid), -1, 0)


def test_grad_x_equiv(grid, random_scalar):
    _eq(
        ops_h.grad_x_3d_halo(_pad(random_scalar), grid),
        ops.grad_x_3d(random_scalar, grid),
    )


def test_grad_y_equiv(grid, random_scalar):
    _eq(
        ops_h.grad_y_3d_halo(_pad(random_scalar), grid),
        ops.grad_y_3d(random_scalar, grid),
    )


def test_divergence_equiv(grid, random_vector):
    u, v = random_vector
    _eq(
        ops_h.divergence_3d_halo(_pad(u), _pad(v), grid),
        ops.divergence_3d(u, v, grid),
    )


def test_laplacian_equiv(grid, random_scalar):
    _eq(
        ops_h.laplacian_3d_halo(_pad(random_scalar), grid),
        ops.laplacian_3d(random_scalar, grid),
    )


def test_interp_cell_to_xface_equiv(grid, random_scalar):
    _eq(
        ops_h.interp_cell_to_xface_halo(_pad(random_scalar), grid),
        _vlast_ref(cep.interp_cell_to_xface_vlast, random_scalar, grid=grid),
    )


def test_interp_cell_to_yface_equiv(grid, random_scalar):
    _eq(
        ops_h.interp_cell_to_yface_halo(_pad(random_scalar), grid),
        _vlast_ref(cep.interp_cell_to_yface_vlast, random_scalar, grid=grid),
    )


def test_interp_xface_to_cell_equiv(grid, random_scalar):
    _eq(
        ops_h.interp_xface_to_cell_halo(_pad(random_scalar), grid),
        _vlast_ref(cep.interp_xface_to_cell_vlast, random_scalar, grid=grid),
    )


def test_interp_yface_to_cell_equiv(grid, random_scalar):
    _eq(
        ops_h.interp_yface_to_cell_halo(_pad(random_scalar), grid),
        _vlast_ref(cep.interp_yface_to_cell_vlast, random_scalar, grid=grid),
    )


def test_interp_yface_to_xface_equiv(grid, random_vector):
    _, v = random_vector
    _eq(
        ops_h.interp_yface_to_xface_halo(_pad(v), grid),
        _vlast_ref(cep.interp_yface_to_xface_vlast, v, grid=grid),
    )


def test_interp_xface_to_yface_equiv(grid, random_vector):
    u, _ = random_vector
    _eq(
        ops_h.interp_xface_to_yface_halo(_pad(u), grid),
        _vlast_ref(cep.interp_xface_to_yface_vlast, u, grid=grid),
    )


def test_packed_exchange_halo_plane_yxz_matches_unpacked():
    """packed_exchange returns same as per-field exchange (single-rank)."""
    from legoesm.parallel.plane_mpi import (
        exchange_halo_plane_yxz, make_plane_pencil_layout,
        packed_exchange_halo_plane_yxz,
    )
    layout = make_plane_pencil_layout(
        rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1,
        ny_global=8, nx_global=6,
    )
    rng = np.random.default_rng(42)
    a = jnp.asarray(rng.standard_normal((8, 6, 4)))
    b = jnp.asarray(rng.standard_normal((8, 6, 4)))
    c = jnp.asarray(rng.standard_normal((8, 6, 4)))
    packed = packed_exchange_halo_plane_yxz(a, b, c, layout=layout)
    indiv = [
        exchange_halo_plane_yxz(x, layout) for x in (a, b, c)
    ]
    for p, i in zip(packed, indiv):
        np.testing.assert_array_equal(np.asarray(p), np.asarray(i))


def _pad_yxz(arr_yxz, halo=1):
    """Wrap-pad the FIRST two axes for (ny, nx, *) layout."""
    pad_widths = [(halo, halo)] * 2 + [(0, 0)] * (arr_yxz.ndim - 2)
    return jnp.pad(arr_yxz, pad_widths, mode="wrap")


def test_upwind_advection_x_halo_equiv():
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        _upwind_advection_x,
    )
    grid = create_plane_grid(
        nx=8, ny=6, nlev=4, dx=1000.0, dy=2000.0, dtype=jnp.float64,
    )
    rng = np.random.default_rng(3)
    f = jnp.asarray(rng.standard_normal((6, 8, 4)))
    u = jnp.asarray(rng.standard_normal((6, 8, 4)))
    expected = _upwind_advection_x(f, u, grid.dx)
    actual = ops_h.upwind_advection_x_halo(_pad_yxz(f), _pad_yxz(u), grid.dx)
    _eq(actual, expected)


def test_upwind_advection_y_halo_equiv():
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        _upwind_advection_y,
    )
    grid = create_plane_grid(
        nx=8, ny=6, nlev=4, dx=1000.0, dy=2000.0, dtype=jnp.float64,
    )
    rng = np.random.default_rng(4)
    f = jnp.asarray(rng.standard_normal((6, 8, 4)))
    v = jnp.asarray(rng.standard_normal((6, 8, 4)))
    expected = _upwind_advection_y(f, v, grid.dy)
    actual = ops_h.upwind_advection_y_halo(_pad_yxz(f), _pad_yxz(v), grid.dy)
    _eq(actual, expected)


def test_variable_K_diffusion_vlast_halo_equiv():
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        _variable_K_diffusion_vlast,
    )
    grid = create_plane_grid(
        nx=8, ny=6, nlev=4, dx=1000.0, dy=2000.0, dtype=jnp.float64,
    )
    rng = np.random.default_rng(5)
    f = jnp.asarray(rng.standard_normal((6, 8, 4)))
    K = jnp.asarray(np.abs(rng.standard_normal((6, 8, 4))))
    expected = _variable_K_diffusion_vlast(f, K, grid)
    actual = ops_h.variable_K_diffusion_vlast_halo(
        _pad_yxz(f), _pad_yxz(K), grid,
    )
    _eq(actual, expected)


def test_packed_exchange_halo_plane_rejects_shape_mismatch():
    from legoesm.parallel.plane_mpi import (
        make_plane_pencil_layout, packed_exchange_halo_plane_yxz,
    )
    layout = make_plane_pencil_layout(
        rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1,
        ny_global=8, nx_global=6,
    )
    a = jnp.zeros((8, 6, 4))
    b = jnp.zeros((8, 6, 3))   # different trailing
    with pytest.raises(ValueError, match="same shape"):
        packed_exchange_halo_plane_yxz(a, b, layout=layout)
