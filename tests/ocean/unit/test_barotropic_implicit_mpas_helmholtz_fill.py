"""Non-vacuous value-preservation test for the fill-free Helmholtz apply.

``_helmholtz_apply_mpas`` no longer calls ``fill_land_cells_mpas``. The claim,
proven and asserted here, is EXACT value preservation: the two-cell edge
gradient can only see a filled value through a land endpoint, and
``edge_mask = mask[c1]*mask[c2]`` zeroes exactly those edges. The fixture below
has real coastline structure (land cells with wet neighbours) and positively
checks that the fill WOULD have changed the intermediate field, so the
exact-equality assertion cannot pass by accident on a no-op fixture.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.core.operators_voronoi import divergence_cell, gradient_edge
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.barotropic_implicit_mpas import _helmholtz_apply_mpas
from legoesm.ocean.dynamics.mpas_fill import fill_land_cells_mpas


def _reference_apply_with_fill(eta_in, H_e, coeff, mesh, mask, edge_mask):
    """Frozen copy of the operator as it stood BEFORE the fill was removed."""
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    H_e_face = H_e * edge_mask
    eta_m = eta_in * mask
    eta_filled = fill_land_cells_mpas(eta_m, mask, c1, c2)
    grad = gradient_edge(eta_filled, mesh)
    flux = H_e_face * grad
    div_grad = divergence_cell(flux, mesh) * mask
    return (eta_m - coeff * div_grad) * mask


def _fixture():
    rng = np.random.default_rng(42)
    mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
    n_cells = int(mesh.nCells)
    n_edges = int(mesh.nEdges)

    # A coastline, not a checkerboard: a contiguous-ish fifth of the cells is
    # land, so many land cells keep wet neighbours and the fill has work to do.
    mask_np = np.ones(n_cells)
    mask_np[np.linspace(0, n_cells - 1, max(2, n_cells // 5)).astype(int)] = 0.0
    mask = jnp.asarray(mask_np)

    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    edge_mask = mask[c1] * mask[c2]
    eta_in = jnp.asarray(rng.standard_normal(n_cells))
    H_e = jnp.asarray(50.0 + 10.0 * rng.standard_normal(n_edges))
    coeff = jnp.asarray(0.01 * np.abs(rng.standard_normal(n_cells)) + 1e-3)
    return mesh, mask, edge_mask, eta_in, H_e, coeff, c1, c2


def test_fixture_actually_exercises_the_fill():
    """Guards the test below: on a fixture where the fill is a no-op, exact
    equality would hold trivially and prove nothing."""
    mesh, mask, _, eta_in, _, _, c1, c2 = _fixture()
    eta_m = eta_in * mask
    eta_filled = fill_land_cells_mpas(eta_m, mask, c1, c2)
    assert not np.array_equal(np.asarray(eta_filled), np.asarray(eta_m)), (
        "fixture degenerate: the fill changed nothing, so the "
        "value-preservation assertion would not be exercised")
    assert 0.0 < float(mask.min()) + 1.0 <= 2.0  # mask really is mixed
    assert float(mask.min()) == 0.0 and float(mask.max()) == 1.0


def test_helmholtz_apply_fill_removal_is_value_preserving():
    mesh, mask, edge_mask, eta_in, H_e, coeff, _, _ = _fixture()
    got = _helmholtz_apply_mpas(eta_in, H_e, coeff, mesh, mask, edge_mask)
    want = _reference_apply_with_fill(eta_in, H_e, coeff, mesh, mask, edge_mask)
    # Exact, not to a tolerance: the claim is that the fill cannot reach the
    # output at all, so any difference at any bit refutes it.
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_adjoint_contract_after_fill_removal():
    """Pin what the REVERSE mode does, since removing the fill changes it.

    Forward values are EXACTLY identical (max abs difference 0.0). The adjoint
    is not quite: on the maximum-contrast fixture below (eta = 1 on wet, 0 on
    land, the worst case for this term) the cell-mask cotangent differs by one
    ULP on 17 of 162 entries (4.4e-16 against a peak of 2.0), and the edge-mask
    cotangent was EXACTLY 0.0 with the fill and is at most 6.6e-13 without it.
    Both are round-off, not a mechanism, but they are a real change to the
    adjoint contract and are pinned here so a future edit that makes either
    term actually matter cannot pass silently.
    """
    mesh = create_voronoi_mesh(subdivision_level=2, lloyd_iterations=5)
    n_cells, n_edges = int(mesh.nCells), int(mesh.nEdges)
    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    m = np.ones(n_cells)
    m[np.linspace(0, n_cells - 1, n_cells // 5).astype(int)] = 0.0
    mask = jnp.asarray(m)
    edge_mask = mask[c1] * mask[c2]
    eta = jnp.asarray(m.copy())          # maximum wet/land contrast
    H_e = jnp.ones(n_edges) * 50.0
    coeff = jnp.ones(n_cells) * 0.05
    cot = jnp.ones(n_cells)

    def _pull(fn):
        out, vjp = jax.vjp(
            lambda mk, em: fn(eta, H_e, coeff, mesh, mk, em), mask, edge_mask)
        dmask, dedge = vjp(cot)
        return (np.asarray(out), np.asarray(dmask), np.asarray(dedge))

    out_new, dmask_new, dedge_new = _pull(_helmholtz_apply_mpas)
    out_old, dmask_old, dedge_old = _pull(_reference_apply_with_fill)

    # Forward: exact. This is the whole justification for the removal.
    np.testing.assert_array_equal(out_new, out_old)

    # Adjoint: round-off only.
    dmask_rel = np.max(np.abs(dmask_new - dmask_old)) / np.max(np.abs(dmask_old))
    assert dmask_rel < 1e-14, (
        f"cell-mask cotangent moved by {dmask_rel:.3e} relative; it was one "
        f"ULP when the fill was removed and must stay at round-off")
    dedge_abs = np.max(np.abs(dedge_new - dedge_old))
    assert dedge_abs < 1e-11, (
        f"edge-mask cotangent residue {dedge_abs:.3e}; it was exactly zero "
        f"with the fill and 6.6e-13 without it, and must stay negligible")
