"""NEMO rotation-divergence viscosity on the Voronoi mesh + its coastal factor."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.operators_voronoi import (
    nemo_ldf_lap_e3_voronoi_3d,
    nemo_vertex_thickness_3d,
    nemo_vertex_shlat_factor_3d,
    vector_laplacian_del2_3d,
)
from legoesm.grids.voronoi import create_voronoi_mesh

NK = 2


@pytest.fixture(scope="module")
def mesh():
    return create_voronoi_mesh(subdivision_level=2)


def _active(mesh, land=()):
    a = np.ones((mesh.areaCell.shape[0], NK), bool)
    for c in land:
        a[c] = False
    return jnp.asarray(a)


def _edge_mask(mesh, act):
    a = np.asarray(act, float)
    return jnp.asarray(a[np.asarray(mesh.cellsOnEdge[0])] * a[np.asarray(mesh.cellsOnEdge[1])])


def _u(mesh, seed):
    return jnp.asarray(np.random.default_rng(seed).standard_normal((mesh.dcEdge.shape[0], NK)))


def test_constant_coefficients_reduce_to_vector_laplacian(mesh):
    act = _active(mesh)
    u = _u(mesh, 0)
    nC, nV, nE = mesh.areaCell.shape[0], mesh.areaTriangle.shape[0], mesh.dcEdge.shape[0]
    A = 1.7e4
    t = nemo_ldf_lap_e3_voronoi_3d(u, mesh, jnp.full((nC, NK), A), jnp.full((nV, NK), A),
                                   jnp.full((nC, NK), 40.0), jnp.full((nE, NK), 40.0),
                                   jnp.full((nV, NK), 40.0), jnp.ones((nE, NK)))
    np.testing.assert_allclose(np.asarray(t), A * np.asarray(vector_laplacian_del2_3d(u, mesh)),
                               rtol=1e-10, atol=1e-20)


def test_shlat_factor_interior_one_coast_geometric(mesh):
    land = 5
    act = _active(mesh, [land])
    f2 = np.asarray(nemo_vertex_shlat_factor_3d(act, mesh, 2.0))[:, 0]
    f0 = np.asarray(nemo_vertex_shlat_factor_3d(act, mesh, 0.0))[:, 0]
    cov = np.asarray(mesh.cellsOnVertex)
    touch = np.any(cov == land, axis=0)
    assert np.all(f2[~touch] == 1.0) and np.all(f0[~touch] == 1.0)
    ka = np.asarray(mesh.kiteAreasOnVertex); at = np.asarray(mesh.areaTriangle)
    for v in np.nonzero(touch)[0]:
        wet = cov[:, v] != land
        np.testing.assert_allclose(f2[v], at[v] / ka[wet, v].sum(), rtol=1e-12)
        assert 1.3 < f2[v] < 1.7 and f0[v] == 0.0


def _call(mesh, act, shlat, h_vertex, seed=3):
    nC, nE = mesh.areaCell.shape[0], mesh.dcEdge.shape[0]
    em = _edge_mask(mesh, act)
    fac = nemo_vertex_shlat_factor_3d(act, mesh, shlat)
    u = _u(mesh, seed) * em
    t = nemo_ldf_lap_e3_voronoi_3d(u, mesh, 1e4 * act.astype(float), 1e4 * fac,
                                   jnp.where(act, 40.0, 0.0), 40.0 * em, h_vertex, em)
    return u, t


def test_zero_coastal_thickness_would_disable_noslip(mesh):
    act = _active(mesh, [5, 6, 30])
    nV = mesh.areaTriangle.shape[0]
    cov = np.asarray(mesh.cellsOnVertex)
    a = np.asarray(act)
    hv_minrule = jnp.asarray(np.min(np.where(a[cov], 40.0, 0.0), axis=0))   # 0 at coast
    hv_filled = jnp.full((nV, NK), 40.0)
    _, bug = _call(mesh, act, 2.0, hv_minrule)
    _, free = _call(mesh, act, 0.0, hv_minrule)
    _, ok = _call(mesh, act, 2.0, hv_filled)
    np.testing.assert_allclose(np.asarray(bug), np.asarray(free), atol=1e-18)
    assert np.max(np.abs(np.asarray(ok) - np.asarray(free))) > 0.01 * np.max(np.abs(np.asarray(free)))


@pytest.mark.parametrize("shlat", [0.0, 2.0])
def test_dissipates_kinetic_energy(mesh, shlat):
    act = _active(mesh, [5, 6, 30, 31, 77])
    u, t = _call(mesh, act, shlat, jnp.full((mesh.areaTriangle.shape[0], NK), 40.0), seed=11)
    w = np.asarray(mesh.dcEdge * mesh.dvEdge)[:, None] * 40.0
    assert np.sum(w * np.asarray(u) * np.asarray(t)) < 0.0


def test_vertex_thickness_fills_land_with_reference(mesh):
    act = _active(mesh, [5])
    h = jnp.where(act, 50.0, 0.0)
    hv = np.asarray(nemo_vertex_thickness_3d(h, act, jnp.array([30.0, 60.0]), mesh))
    touch = np.any(np.asarray(mesh.cellsOnVertex) == 5, axis=0)
    assert np.all(hv[~touch] == 50.0)
    assert np.all(hv[touch, 0] == 30.0) and np.all(hv[touch, 1] == 50.0)   # min(50, dz_ref)
