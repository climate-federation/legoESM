"""Unit tests for MPAS partial-cell helpers (P2 of the MPAS
realistic-geometry plan; see
``docs/ocean_experiments/realistic_geometry_mpas_plan.md``).

Covers ``compute_edge_mask``, ``compute_vertex_mask``,
``compute_max_level_edge_bot/top``, ``min_cell_to_edge``,
``donor_cell_to_edge``, ``kite_area_vertex_thickness``,
``min_cell_to_vertex``, and ``vertex_thickness_hybrid`` from
``src/legoesm/ocean/dynamics/mpas_partial_cell_helpers.py``.

Each helper is tested for: shape correctness, the algebraic
invariant that defines it, and behavior on the wet-only,
all-dry, and partially-dry edge/vertex cases.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")
jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.mpas_partial_cell_helpers import (
    BIG_H,
    compute_edge_mask,
    compute_vertex_mask,
    compute_max_level_edge_bot,
    compute_max_level_edge_top,
    min_cell_to_edge,
    donor_cell_to_edge,
    kite_area_vertex_thickness,
    min_cell_to_vertex,
    partial_cell_pgf_correction_edge,
    vertex_thickness_hybrid,
)


@pytest.fixture(scope="module")
def mesh():
    return create_voronoi_mesh(subdivision_level=2)


# ----- Masks --------------------------------------------------------------


def test_compute_edge_mask_all_ocean(mesh):
    """All-ocean → edge mask all 1."""
    land_mask = jnp.ones(mesh.nCells, dtype=jnp.float64)
    em = compute_edge_mask(land_mask, mesh)
    assert em.shape == (mesh.nEdges,)
    np.testing.assert_array_equal(np.asarray(em), 1.0)


def test_compute_edge_mask_step_pattern(mesh):
    """Half-ocean / half-land split: edges crossing the divide are
    dry; edges in the all-ocean half are wet."""
    lat = np.asarray(mesh.latCell)
    land_mask = jnp.asarray(np.where(lat < 0.0, 1.0, 0.0))
    em = compute_edge_mask(land_mask, mesh)
    em_np = np.asarray(em)
    # Construct the expected mask directly.
    c1 = np.asarray(mesh.cellsOnEdge[0])
    c2 = np.asarray(mesh.cellsOnEdge[1])
    expected = (lat[c1] < 0.0).astype(np.float64) * (lat[c2] < 0.0).astype(np.float64)
    np.testing.assert_array_equal(em_np, expected)


def test_compute_vertex_mask_all_ocean(mesh):
    """All-ocean → vertex mask all 1."""
    land_mask = jnp.ones(mesh.nCells, dtype=jnp.float64)
    vm = compute_vertex_mask(land_mask, mesh)
    assert vm.shape == (mesh.nVertices,)
    np.testing.assert_array_equal(np.asarray(vm), 1.0)


def test_compute_vertex_mask_partial_dry(mesh):
    """If even one of the (up-to-3) adjacent cells is dry, vertex
    mask is 0."""
    nCells = mesh.nCells
    # Make exactly cell 0 dry.
    lm = np.ones(nCells, dtype=np.float64)
    lm[0] = 0.0
    vm = np.asarray(compute_vertex_mask(jnp.asarray(lm), mesh))
    cov = np.asarray(mesh.cellsOnVertex)  # (vertexDegree, nVertices)
    has_cell_0 = np.any(cov == 0, axis=0)
    # Vertices touching cell 0 must be dry.
    assert vm[has_cell_0].sum() == 0.0
    # Other vertices stay wet.
    assert vm[~has_cell_0].mean() == 1.0


# ----- Per-edge bottom-level indices --------------------------------------


def test_max_level_edge_bot_min_of_endpoints(mesh):
    """Bot index = min(bot[c1], bot[c2])."""
    rng = np.random.default_rng(0)
    bot = jnp.asarray(rng.integers(-1, 10, size=mesh.nCells), dtype=jnp.int32)
    out = compute_max_level_edge_bot(bot, mesh)
    c1 = np.asarray(mesh.cellsOnEdge[0])
    c2 = np.asarray(mesh.cellsOnEdge[1])
    expected = np.minimum(np.asarray(bot)[c1], np.asarray(bot)[c2])
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_max_level_edge_top_max_of_endpoints(mesh):
    rng = np.random.default_rng(1)
    bot = jnp.asarray(rng.integers(-1, 10, size=mesh.nCells), dtype=jnp.int32)
    out = compute_max_level_edge_top(bot, mesh)
    c1 = np.asarray(mesh.cellsOnEdge[0])
    c2 = np.asarray(mesh.cellsOnEdge[1])
    expected = np.maximum(np.asarray(bot)[c1], np.asarray(bot)[c2])
    np.testing.assert_array_equal(np.asarray(out), expected)


# ----- Edge thickness -----------------------------------------------------


def test_min_cell_to_edge_uniform_recovers_h(mesh):
    """Uniform h → edge thickness = h."""
    nlev = 5
    h = jnp.full((mesh.nCells, nlev), 100.0, dtype=jnp.float64)
    h_e = min_cell_to_edge(h, mesh)
    assert h_e.shape == (mesh.nEdges, nlev)
    np.testing.assert_array_equal(np.asarray(h_e), 100.0)


def test_min_cell_to_edge_satisfies_min_inequality(mesh):
    """h_e <= min(h[c1], h[c2]) — and equals it on a 2-cell edge."""
    rng = np.random.default_rng(2)
    nlev = 4
    h = jnp.asarray(
        rng.uniform(10.0, 1000.0, size=(mesh.nCells, nlev)),
        dtype=jnp.float64,
    )
    h_e = np.asarray(min_cell_to_edge(h, mesh))
    c1 = np.asarray(mesh.cellsOnEdge[0])
    c2 = np.asarray(mesh.cellsOnEdge[1])
    expected = np.minimum(np.asarray(h)[c1], np.asarray(h)[c2])
    np.testing.assert_array_equal(h_e, expected)


def test_donor_cell_to_edge_picks_upstream(mesh):
    """u >= 0 → donor = c1; u < 0 → donor = c2."""
    rng = np.random.default_rng(3)
    nlev = 3
    h = jnp.asarray(
        rng.uniform(10.0, 1000.0, size=(mesh.nCells, nlev)),
        dtype=jnp.float64,
    )
    u = jnp.asarray(
        rng.standard_normal((mesh.nEdges, nlev)),
        dtype=jnp.float64,
    )
    h_e = np.asarray(donor_cell_to_edge(h, u, mesh))
    c1 = np.asarray(mesh.cellsOnEdge[0])
    c2 = np.asarray(mesh.cellsOnEdge[1])
    h_np = np.asarray(h)
    u_np = np.asarray(u)
    expected = np.where(u_np >= 0.0, h_np[c1], h_np[c2])
    np.testing.assert_array_equal(h_e, expected)


def test_donor_cell_differentiable_in_h(mesh):
    """``donor_cell_to_edge`` is differentiable through h (through the
    ``jnp.where`` branches), critical for AD through the continuity
    equation.

    Use a u-field with both signs so every cell contributes as a
    donor on at least one of its incident edges (a cell that is
    ``c2`` on all its incident edges only contributes when u<0).
    """
    nlev = 2
    h = jnp.full((mesh.nCells, nlev), 100.0, dtype=jnp.float64)
    rng = np.random.default_rng(99)
    u = jnp.asarray(
        rng.standard_normal((mesh.nEdges, nlev)), dtype=jnp.float64,
    )

    def loss(h_in):
        return jnp.sum(donor_cell_to_edge(h_in, u, mesh))

    g = jax.grad(loss)(h)
    g_np = np.asarray(g)
    assert g_np.shape == h.shape
    assert np.all(np.isfinite(g_np))
    # The gradient at each cell equals the number of incident edges
    # where the cell is the donor (c1 with u>=0, or c2 with u<0).
    # On a connected Voronoi mesh every cell has incident edges and
    # for random u with both signs, almost surely each cell is the
    # donor at least once.  Check the global sum equals nEdges*nlev
    # (every edge picks exactly one donor cell per level).
    expected_total = mesh.nEdges * nlev
    np.testing.assert_allclose(g_np.sum(), expected_total, rtol=1.0e-12)
    # And that no cell has a *negative* contribution (donor weights
    # are all 1.0 when picked, 0 when not).
    assert np.all(g_np >= 0.0)


# ----- Vertex thickness ---------------------------------------------------


def test_kite_vertex_uniform_recovers_h(mesh):
    """Uniform-wet h → kite vertex thickness = h (renormalization
    cancels)."""
    nlev = 3
    h = jnp.full((mesh.nCells, nlev), 250.0, dtype=jnp.float64)
    h_v = kite_area_vertex_thickness(h, mesh)
    assert h_v.shape == (mesh.nVertices, nlev)
    np.testing.assert_allclose(np.asarray(h_v), 250.0, rtol=1.0e-12)


def test_kite_vertex_dry_neighbor_renormalizes(mesh):
    """One dry cell among 3 → vertex thickness uses only the wet
    cells, weighted by their kite areas (not the full triangle)."""
    nlev = 1
    nCells = mesh.nCells
    # Build h with one dry cell (cell 0) and uniform 100 elsewhere.
    h_np = np.full((nCells, nlev), 100.0)
    h_np[0, 0] = 0.0
    h = jnp.asarray(h_np)
    h_v = np.asarray(kite_area_vertex_thickness(h, mesh))
    cov = np.asarray(mesh.cellsOnVertex)  # (vertexDegree, nVertices)
    # On vertices touching cell 0: at least one wet cell remains, so h_v=100.
    has_cell_0 = np.any(cov == 0, axis=0)
    np.testing.assert_allclose(h_v[has_cell_0, 0], 100.0, rtol=1.0e-12)
    # Other vertices unchanged at 100.
    np.testing.assert_allclose(h_v[~has_cell_0, 0], 100.0, rtol=1.0e-12)


def test_kite_vertex_all_dry_returns_zero(mesh):
    """If all cells around a vertex are dry, h_v=0."""
    nlev = 1
    nCells = mesh.nCells
    h = jnp.zeros((nCells, nlev), dtype=jnp.float64)
    h_v = np.asarray(kite_area_vertex_thickness(h, mesh))
    np.testing.assert_array_equal(h_v, 0.0)


def test_min_cell_to_vertex_uniform_recovers_h(mesh):
    nlev = 2
    h = jnp.full((mesh.nCells, nlev), 75.0, dtype=jnp.float64)
    h_v = min_cell_to_vertex(h, mesh)
    assert h_v.shape == (mesh.nVertices, nlev)
    np.testing.assert_array_equal(np.asarray(h_v), 75.0)


def test_min_cell_to_vertex_picks_thinnest_wet(mesh):
    """``min_cell_to_vertex`` returns the thinnest wet cell at each
    vertex, ignoring dry (h=0) entries."""
    nlev = 1
    nCells = mesh.nCells
    rng = np.random.default_rng(4)
    h_np = rng.uniform(10.0, 1000.0, size=(nCells, nlev))
    h = jnp.asarray(h_np)
    h_v = np.asarray(min_cell_to_vertex(h, mesh))
    cov = np.asarray(mesh.cellsOnVertex)  # (vertexDegree, nVertices)
    # For each vertex, the result should equal the min over wet
    # adjacent cells.  All cells are wet here (h>0).
    cov_safe = np.maximum(cov, 0)
    valid = cov >= 0
    h_gathered = h_np[cov_safe, 0]  # (vertexDegree, nVertices)
    h_for_min = np.where(valid, h_gathered, BIG_H)
    expected = np.min(h_for_min, axis=0)
    # Vertices where all cov < 0 (shouldn't happen on full Voronoi):
    # our helper returns 0; expected returns BIG_H. Mask these out.
    any_valid = np.any(valid, axis=0)
    np.testing.assert_allclose(
        h_v[any_valid, 0], expected[any_valid], rtol=1.0e-12,
    )


def test_min_cell_to_vertex_ignores_dry(mesh):
    """A dry cell among 3 should not pull the min down to 0."""
    nlev = 1
    nCells = mesh.nCells
    h_np = np.full((nCells, nlev), 200.0)
    h_np[0, 0] = 0.0  # dry
    h = jnp.asarray(h_np)
    h_v = np.asarray(min_cell_to_vertex(h, mesh))
    cov = np.asarray(mesh.cellsOnVertex)
    has_cell_0 = np.any(cov == 0, axis=0)
    # Vertices touching cell 0 still see a wet 200; the BIG_H sentinel
    # is correctly excluded by ``min``.
    np.testing.assert_allclose(h_v[has_cell_0, 0], 200.0, rtol=1.0e-12)


def test_hybrid_vertex_uses_kite_in_uniform_interior(mesh):
    """When all wet cells at a vertex have the same h, hybrid =
    kite mean (h_min == h_max → use_min False)."""
    nlev = 1
    h = jnp.full((mesh.nCells, nlev), 500.0, dtype=jnp.float64)
    h_v = np.asarray(vertex_thickness_hybrid(h, mesh, alpha=0.5))
    np.testing.assert_allclose(h_v, 500.0, rtol=1.0e-12)


def test_hybrid_vertex_uses_min_at_steep_step(mesh):
    """When one wet cell is much thinner than the others, hybrid
    falls back to min (h_min < alpha * h_max)."""
    nlev = 1
    nCells = mesh.nCells
    # Pick a vertex that has 3 valid adjacent cells; set one of them
    # to a thin layer, others to 1000.
    cov = np.asarray(mesh.cellsOnVertex)
    # Find the first vertex with exactly 3 valid cells.
    valid_count = np.sum(cov >= 0, axis=0)
    v_idx = int(np.argmax(valid_count == 3))
    cells_v = [int(cov[k, v_idx]) for k in range(3)]
    h_np = np.full((nCells, nlev), 1000.0)
    h_np[cells_v[0], 0] = 100.0  # thin: 1/10 of others
    h = jnp.asarray(h_np)
    h_v = np.asarray(vertex_thickness_hybrid(h, mesh, alpha=0.5))
    # At that vertex: h_min = 100 < 0.5 * 1000 = 500 → use min → h_v = 100.
    assert h_v[v_idx, 0] == pytest.approx(100.0, rel=1.0e-12)


def test_ac_pgf_correction_zero_on_full_cells(mesh):
    """When centroids align across every edge (full-cell columns),
    the Adcroft-Campin correction is identically zero — the legacy
    z-star path is bit-exact unaffected by enabling
    ``pgf_scheme="adcroft"``.
    """
    nlev = 5
    nCells = mesh.nCells
    # Centroid the same on every cell — identical depths → zero excess
    # at every edge.
    centroid = jnp.broadcast_to(
        jnp.linspace(10.0, 5000.0, nlev)[None, :], (nCells, nlev),
    ).astype(jnp.float64)
    rho_prime = jnp.full(
        (nCells, nlev), 0.5, dtype=jnp.float64,
    )  # nonzero rho_prime to rule out trivial passing
    correction = partial_cell_pgf_correction_edge(
        centroid, rho_prime, mesh, g=constants.g, rho_0=constants.rho_ocean,
    )
    assert correction.shape == (mesh.nEdges, nlev)
    np.testing.assert_array_equal(np.asarray(correction), 0.0)


def test_ac_pgf_correction_nonzero_on_step(mesh):
    """A centroid-depth step across an edge produces a finite, nonzero
    correction proportional to ``g * rho_prime * delta_centroid /
    (dcEdge * rho_0)``."""
    nlev = 1
    nCells = mesh.nCells
    lat = np.asarray(mesh.latCell)
    # Centroid 100 m on south, 200 m on north (a 100 m step).
    centroid_np = np.where(lat < 0.0, 100.0, 200.0)[:, None]
    centroid = jnp.asarray(centroid_np, dtype=jnp.float64)
    rho_prime = jnp.full((nCells, nlev), 1.0, dtype=jnp.float64)

    g, rho_0 = constants.g, constants.rho_ocean
    correction = np.asarray(
        partial_cell_pgf_correction_edge(
            centroid, rho_prime, mesh, g, rho_0,
        )
    )

    c1 = np.asarray(mesh.cellsOnEdge[0])
    c2 = np.asarray(mesh.cellsOnEdge[1])
    centroid_c1 = centroid_np[c1, 0]
    centroid_c2 = centroid_np[c2, 0]
    face_ref = np.minimum(centroid_c1, centroid_c2)
    excess_c1 = centroid_c1 - face_ref
    excess_c2 = centroid_c2 - face_ref
    expected = (
        -g * (excess_c2 - excess_c1)
        / np.asarray(mesh.dcEdge)
        / rho_0
    )
    np.testing.assert_allclose(correction[:, 0], expected, rtol=1.0e-12)

    # On any edge crossing the step (one cell south, one north),
    # the correction is nonzero.
    crossing = (lat[c1] < 0.0) ^ (lat[c2] < 0.0)
    assert correction[crossing, 0].any(), (
        "AC correction should be nonzero on at least one step-crossing edge"
    )


def test_ac_pgf_correction_dcEdge_inverse_scaling(mesh):
    """The correction scales as 1/dcEdge — doubling dcEdge halves the
    correction (sanity check on the divisor placement)."""
    nlev = 1
    nCells = mesh.nCells
    rng = np.random.default_rng(0)
    centroid = jnp.asarray(
        rng.uniform(50.0, 4000.0, size=(nCells, nlev)), dtype=jnp.float64,
    )
    rho_prime = jnp.full((nCells, nlev), 0.7, dtype=jnp.float64)
    g, rho_0 = constants.g, constants.rho_ocean

    c1_orig = partial_cell_pgf_correction_edge(
        centroid, rho_prime, mesh, g, rho_0,
    )
    # Build a doctored mesh with 2x dcEdge.
    mesh_2x = mesh._replace(dcEdge=mesh.dcEdge * 2.0)
    c1_2x = partial_cell_pgf_correction_edge(
        centroid, rho_prime, mesh_2x, g, rho_0,
    )
    np.testing.assert_allclose(
        np.asarray(c1_2x), np.asarray(c1_orig) * 0.5, rtol=1.0e-12,
    )


def test_hybrid_vertex_alpha_threshold(mesh):
    """alpha=0.99 means even small variations trigger min;
    alpha=0.01 means only very strong contrasts do."""
    nlev = 1
    nCells = mesh.nCells
    # Pick a 3-cell vertex.
    cov = np.asarray(mesh.cellsOnVertex)
    v_idx = int(np.argmax(np.sum(cov >= 0, axis=0) == 3))
    cells_v = [int(cov[k, v_idx]) for k in range(3)]
    h_np = np.full((nCells, nlev), 1000.0)
    h_np[cells_v[0], 0] = 800.0  # 0.8x of others
    h = jnp.asarray(h_np)

    h_v_strict = np.asarray(vertex_thickness_hybrid(h, mesh, alpha=0.9))
    # 800 < 0.9 * 1000 = 900 → use min → 800
    assert h_v_strict[v_idx, 0] == pytest.approx(800.0, rel=1.0e-12)

    h_v_loose = np.asarray(vertex_thickness_hybrid(h, mesh, alpha=0.5))
    # 800 >= 0.5 * 1000 = 500 → use kite mean (between 800 and 1000).
    assert 800.0 < h_v_loose[v_idx, 0] < 1000.0
