"""Neighbour-offset layer: div/grad/curl vs the existing per-grid operators.

Lat-lon C-grid must agree with ``operators_latlon_cgrid`` to round-off
(sum order differs); MPAS must be bit-identical to ``operators_voronoi``
(same product and reduction order).  Plus discrete mass closure and scope
guards.  fp64 throughout.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core import neighbour_ops as nops
from legoesm.core.operators_voronoi import curl_vertex, divergence_cell, gradient_edge
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.neighbour_connectivity import (
    latlon_neighbour_mesh,
    mpas_neighbour_mesh,
)
from legoesm.grids.operators_latlon_cgrid import (
    curl_vertex_cgrid,
    divergence_cgrid,
    gradient_x_cgrid,
    gradient_y_cgrid,
)
from legoesm.grids.voronoi import create_voronoi_mesh

LATLON_RTOL = 1e-13


@pytest.fixture(scope="module", autouse=True)
def _x64():
    prev_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)
    jax.config.update("jax_enable_x64", prev_x64)


@pytest.fixture(scope="module", params=[(16, 32), (46, 90)], ids=["16x32", "46x90"])
def grid(request, _x64):
    n_lat, n_lon = request.param
    return create_latlon_grid(n_lat, n_lon)


@pytest.fixture(scope="module")
def mesh(_x64):
    return create_voronoi_mesh(3, lloyd_iterations=50)


def _rel(new, old):
    scale = float(jnp.max(jnp.abs(old)))
    assert scale > 0
    return float(jnp.max(jnp.abs(new - old))) / scale


def _latlon_fields(grid, seed=0):
    """Random admissible C-grid fields: duplicate periodic u column, v=0 at poles."""
    k = jax.random.split(jax.random.PRNGKey(seed), 3)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    h = 1e3 + jax.random.normal(k[0], (n_lat, n_lon))
    u = jax.random.normal(k[1], (n_lat, n_lon))
    u = jnp.concatenate([u, u[:, :1]], axis=1)
    v = jax.random.normal(k[2], (n_lat + 1, n_lon)).at[0].set(0.0).at[-1].set(0.0)
    return h, u, v


def _mpas_fields(mesh, seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 2)
    phi = 1e3 + jax.random.normal(k[0], mesh.areaCell.shape)
    u = jax.random.normal(k[1], mesh.dvEdge.shape)
    return phi, u


# ---------------------------------------------------------------- lat-lon

def test_latlon_divergence_matches(grid):
    h, u, v = _latlon_fields(grid)
    nm = latlon_neighbour_mesh(grid)
    assert _rel(nops.divergence(nm, (u, v)), divergence_cgrid(u, v, grid)) <= LATLON_RTOL


def test_latlon_gradient_matches(grid):
    h, _, _ = _latlon_fields(grid)
    nm = latlon_neighbour_mesh(grid)
    gx, gy = nops.gradient(nm, h)
    assert _rel(gx, gradient_x_cgrid(h, grid)) <= LATLON_RTOL
    assert _rel(gy, gradient_y_cgrid(h, grid)) <= LATLON_RTOL
    assert jnp.all(gy[0] == 0.0) and jnp.all(gy[-1] == 0.0)


def test_latlon_curl_matches(grid):
    _, u, v = _latlon_fields(grid)
    nm = latlon_neighbour_mesh(grid)
    new = nops.curl(nm, (u, v))
    assert _rel(new, curl_vertex_cgrid(u, v, grid)) <= LATLON_RTOL
    assert jnp.all(new[0] == 0.0) and jnp.all(new[-1] == 0.0)


def test_latlon_divergence_closes_mass(grid):
    """sum(div(F) * area) is zero to round-off: the face fluxes telescope."""
    _, u, v = _latlon_fields(grid, seed=1)
    nm = latlon_neighbour_mesh(grid)
    terms = nops.divergence(nm, (u, v)) * grid.area
    scale = float(jnp.sum(jnp.abs(terms)))
    assert abs(float(jnp.sum(terms))) <= 1e-13 * scale


def test_latlon_jit_matches_eager(grid):
    h, u, v = _latlon_fields(grid)
    nm = latlon_neighbour_mesh(grid)
    for op, x in ((nops.divergence, (u, v)), (nops.curl, (u, v))):
        assert _rel(jax.jit(lambda a: op(nm, a))(x), op(nm, x)) <= LATLON_RTOL


@pytest.mark.parametrize("module, name, value, message", [
    ("legoesm.grids.operators_latlon_cgrid", "is_tripolar", True, "tripolar"),
    ("legoesm.grids.operators_latlon_cgrid", "fold_is_local", True, "north fold"),
    ("legoesm.grids.operators_latlon_cgrid", "north_fold_mask", object(), "north fold"),
    ("legoesm.grids.halo_latlon", "get_meridionally_flat", True, "meridionally flat"),
    ("legoesm.grids.halo_latlon", "get_meridionally_periodic", True, "meridionally periodic"),
    ("legoesm.grids.halo", "get_halo_backend", "mpi", "non-local halo"),
    ("legoesm.parallel.latlon_spmd", "spmd_pole_end_masks", (1, 1), "SPMD"),
])
def test_latlon_rejects_unsupported_scope(grid, monkeypatch, module, name, value, message):
    import importlib
    monkeypatch.setattr(importlib.import_module(module), name,
                        lambda *a, **k: value)
    with pytest.raises(ValueError, match=message):
        latlon_neighbour_mesh(grid)


def test_latlon_rejects_regional_and_rich_grids(grid):
    with pytest.raises(ValueError, match="not pole-to-pole"):
        latlon_neighbour_mesh(grid._replace(dlat=grid.dlat * 0.5))

    class Rich(type(grid)):
        dx_v = None
    with pytest.raises(ValueError, match="stored grid metric"):
        latlon_neighbour_mesh(Rich(*grid))


# ---------------------------------------------------------------- MPAS

def test_mpas_operators_bit_identical(mesh):
    phi, u = _mpas_fields(mesh)
    nm = mpas_neighbour_mesh(mesh)
    pairs = [
        (nops.divergence(nm, u), divergence_cell(u, mesh)),
        (nops.gradient(nm, phi)[0], gradient_edge(phi, mesh)),
        (nops.curl(nm, u), curl_vertex(u, mesh)),
    ]
    for new, old in pairs:
        np.testing.assert_array_equal(np.asarray(new), np.asarray(old))


def test_mpas_operators_bit_identical_under_jit(mesh):
    phi, u = _mpas_fields(mesh)
    nm = mpas_neighbour_mesh(mesh)
    np.testing.assert_array_equal(
        np.asarray(jax.jit(lambda a: nops.divergence(nm, a))(u)),
        np.asarray(jax.jit(lambda a: divergence_cell(a, mesh))(u)))
    np.testing.assert_array_equal(
        np.asarray(jax.jit(lambda a: nops.curl(nm, a))(u)),
        np.asarray(jax.jit(lambda a: curl_vertex(a, mesh))(u)))


def test_mpas_divergence_closes_mass(mesh):
    _, u = _mpas_fields(mesh, seed=1)
    nm = mpas_neighbour_mesh(mesh)
    terms = nops.divergence(nm, u) * mesh.areaCell
    scale = float(jnp.sum(jnp.abs(terms)))
    assert abs(float(jnp.sum(terms))) <= 1e-13 * scale


def test_operators_check_grads(grid, mesh):
    from jax.test_util import check_grads
    h, u, v = _latlon_fields(grid)
    nm = latlon_neighbour_mesh(grid)
    check_grads(lambda a, b: nops.divergence(nm, (a, b)), (u, v), order=2)
    check_grads(lambda a: nops.gradient(nm, a), (h,), order=2)
    check_grads(lambda a, b: nops.curl(nm, (a, b)), (u, v), order=2)
    phi, ue = _mpas_fields(mesh)
    mm = mpas_neighbour_mesh(mesh)
    check_grads(lambda a: nops.divergence(mm, a), (ue,), order=2)
    check_grads(lambda a: nops.gradient(mm, a), (phi,), order=2)
    check_grads(lambda a: nops.curl(mm, a), (ue,), order=2)
