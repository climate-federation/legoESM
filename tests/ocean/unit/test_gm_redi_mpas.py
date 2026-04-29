"""GM/Redi MPAS — Phase 1 (slopes) numerical tests + dispatch guards.

Phase 1 (``compute_isopycnal_slopes_mpas``) is implemented and has its
own correctness suite below.  The remaining entry points
(``gm_redi_tracer_tendency_centered_mpas``, ``..._triads_mpas``, and
the top-level ``gm_redi_tracer_tendency_mpas``) are still skeletons —
they are guarded by the dispatch test that asserts they raise with a
plan-doc pointer.  As each phase lands, replace the corresponding
parametrize entry with a real numerical test.

See docs/ocean_experiments/gm_redi_mpas_plan.md for the phased plan.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
from legoesm.ocean.physics.lateral_mixing.gm_redi_mpas import (
    _voronoi_neumann_fill,
    compute_isopycnal_slopes_mpas,
    gm_redi_tracer_tendency_centered_mpas,
    gm_redi_tracer_tendency_mpas,
    gm_redi_tracer_tendency_triads_mpas,
)
from legoesm.ocean.vertical import create_ocean_z_star


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def mesh():
    """Level-2 icosahedral Voronoi mesh (162 cells, 480 edges)."""
    return create_voronoi_mesh(subdivision_level=2)


@pytest.fixture(scope="module")
def z_coord():
    """5-level z-star with H=500 m for shallow Phase-1 tests."""
    return create_ocean_z_star(
        n_levels=5, H_max=500.0, dz_surface=20.0, dz_deep=200.0
    )


@pytest.fixture(scope="module")
def cfg():
    return GMRediConfig(kappa_GM=1e3, kappa_Redi=1e3, S_max=1e-2)


# ============================================================================
# Helpers
# ============================================================================

def _all_ocean_mask(mesh) -> jnp.ndarray:
    return jnp.ones((mesh.nCells,), dtype=jnp.float64)


def _unit_jacobian(mesh) -> jnp.ndarray:
    return jnp.ones((mesh.nCells,), dtype=jnp.float64)


def _z_centers(z_coord) -> jnp.ndarray:
    """Cell-centre z [m], negative downward.  Shape (nlev,)."""
    # z_full_ref is at full levels (cell centres), starting from z = -dz/2
    # at the surface and going downward.  Construct directly from dz_ref.
    dz = np.asarray(z_coord.dz_ref)
    # mid-point of layer k: -(sum dz[0..k-1] + dz[k]/2)
    edges = np.concatenate([[0.0], -np.cumsum(dz)])
    centers = 0.5 * (edges[:-1] + edges[1:])
    return jnp.asarray(centers, dtype=jnp.float64)


# ============================================================================
# Phase 1 — slope correctness
# ============================================================================

def test_slopes_constant_rho_are_zero(mesh, z_coord, cfg):
    """ρ uniform in space ⇒ S_n = 0 (machine precision)."""
    nlev = z_coord.dz_ref.shape[0]
    rho = jnp.full((mesh.nCells, nlev), 1027.5, dtype=jnp.float64)
    mask = _all_ocean_mask(mesh)
    jac = _unit_jacobian(mesh)

    S_n, taper = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)

    assert S_n.shape == (mesh.nEdges, nlev - 1)
    assert taper.shape == (mesh.nEdges, nlev - 1)
    np.testing.assert_allclose(np.asarray(S_n), 0.0, atol=1e-14)
    # Constant ρ → zero slope → taper saturates to 1.
    np.testing.assert_allclose(np.asarray(taper), 1.0, atol=1e-12)


def test_slopes_pure_vertical_stratification_are_zero(mesh, z_coord, cfg):
    """ρ = ρ₀ + α·(-z) (depth-only) ⇒ no horizontal gradient ⇒ S_n = 0."""
    z = _z_centers(z_coord)                          # (nlev,)
    nlev = z.shape[0]
    alpha = 1e-3                                     # kg/m^4
    rho_1d = 1027.5 + alpha * (-z)                   # (nlev,) increases with depth
    rho = jnp.broadcast_to(rho_1d[None, :], (mesh.nCells, nlev))
    mask = _all_ocean_mask(mesh)
    jac = _unit_jacobian(mesh)

    S_n, taper = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)

    np.testing.assert_allclose(np.asarray(S_n), 0.0, atol=1e-14)


def test_slopes_meridional_gradient_matches_analytic(mesh, z_coord, cfg):
    """ρ = ρ₀ + β·sin(latCell) + α·(-z) ⇒ analytic S_n on each edge.

    Construction:
      ∂ρ/∂y = β · cos(lat) / R_earth                  (per metre, towards north)
      ∂ρ/∂z = -α                                       (z increasing upward)
      Slope along the edge normal:
        S_n = -∂ρ/∂n / ∂ρ/∂z
            = -(∂ρ/∂y · n_y) / (-α)
            =  (β · cos(lat_e) / R_earth · n_y) / α

      where n_y = sin(angleEdge) is the y-component of the edge unit
      normal (angleEdge is measured from east).  We compare against
      this analytic value on edges that are far from the discretisation
      boundary (i.e. interior of the level-2 mesh, no land).
    """
    z = _z_centers(z_coord)
    nlev = z.shape[0]
    beta  = 0.5      # kg/m^3 per unit sin(lat)  (huge to dominate float noise)
    alpha = 1e-3     # kg/m^4
    rho_y = beta * jnp.sin(mesh.latCell)             # (nCells,)
    rho_z = alpha * (-z)                             # (nlev,)
    rho = 1027.5 + rho_y[:, None] + rho_z[None, :]   # (nCells, nlev)

    mask = _all_ocean_mask(mesh)
    jac = _unit_jacobian(mesh)

    S_n, taper = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)

    # Analytic slope at each edge centre, projected onto edge normal.
    n_y = jnp.sin(mesh.angleEdge)                    # (nEdges,)
    S_n_analytic = (beta * jnp.cos(mesh.latEdge) * n_y / constants.R_earth) / alpha
    # Same value at every interface (vertical gradient is constant).
    S_n_analytic = jnp.broadcast_to(
        S_n_analytic[:, None], (mesh.nEdges, nlev - 1)
    )

    # Allow slope-of-slope discretisation error because the edge-normal
    # gradient samples ρ at neighbouring cells whose (lat, sin(lat))
    # values differ by O(dcEdge / R_earth).  The level-2 mesh has
    # dcEdge ~ R_earth/8 so we expect ~10% error on β · sin(lat) — but
    # the slope direction projection should match to that precision.
    err = np.abs(np.asarray(S_n) - np.asarray(S_n_analytic))
    rel = err / np.maximum(np.abs(np.asarray(S_n_analytic)), 1e-30)
    # Median relative error should be small even if a few coarse-mesh
    # outliers hit ~30%.
    assert np.median(rel) < 0.15, (
        f"median rel error too high: {np.median(rel):.3g}"
    )

    # Taper should be ≈ 1 for these small slopes (S_max = 1e-2 in cfg,
    # |S_n| analytic ~ β/(α·R) · 0.5 = 5e-7 · 0.5 ~ 4e-8 ≪ S_max).
    assert float(jnp.min(taper)) > 0.99


def test_slopes_respect_land_mask_after_neumann_fill(mesh, z_coord, cfg):
    """Edges on the land/ocean coastline see a finite slope only because
    the Neumann fill propagates ocean values into land.  Test that:

    - With Neumann fill, a single land cell embedded in an otherwise
      uniform ρ field produces no spurious slope on its incident edges
      (the fill puts the ocean ρ back into the land cell).
    """
    nlev = z_coord.dz_ref.shape[0]
    rho_uniform = 1027.5
    rho = jnp.full((mesh.nCells, nlev), rho_uniform, dtype=jnp.float64)
    # Land cell with sentinel value that would blow up the gradient.
    rho = rho.at[0, :].set(0.0)
    mask = jnp.ones((mesh.nCells,), dtype=jnp.float64).at[0].set(0.0)
    jac = _unit_jacobian(mesh)

    S_n, _ = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)

    # The Neumann fill drops 1027.5 back into cell 0; uniform ρ
    # everywhere ⇒ zero slopes everywhere, including land-adjacent edges.
    np.testing.assert_allclose(np.asarray(S_n), 0.0, atol=1e-12)


def test_neumann_fill_propagates_ocean_into_isolated_land(mesh):
    """Neumann fill turns a single land cell's sentinel into the ocean
    neighbour mean within one pass."""
    f = jnp.full((mesh.nCells,), 1.0, dtype=jnp.float64).at[0].set(0.0)
    mask = jnp.ones((mesh.nCells,), dtype=jnp.float64).at[0].set(0.0)
    filled = _voronoi_neumann_fill(f, mask, mesh)
    # All ocean neighbours have value 1.0, so the land cell should be 1.0.
    assert float(filled[0]) == pytest.approx(1.0, abs=1e-12)
    # Other cells unchanged.
    np.testing.assert_allclose(np.asarray(filled[1:]), 1.0, atol=0.0)


# ============================================================================
# Dispatch guards for phases not yet implemented
# ============================================================================

@pytest.mark.parametrize("fn,name", [
    (gm_redi_tracer_tendency_centered_mpas, "gm_redi_tracer_tendency_centered_mpas"),
    (gm_redi_tracer_tendency_triads_mpas,   "gm_redi_tracer_tendency_triads_mpas"),
    (gm_redi_tracer_tendency_mpas,          "gm_redi_tracer_tendency_mpas"),
])
def test_unimplemented_phases_raise_with_plan_pointer(fn, name):
    """Phases 2/4/5 are not implemented yet — they must raise a clear
    NotImplementedError that names the function and points at the plan."""
    import inspect
    sig = inspect.signature(fn)
    n_pos = sum(
        1 for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )
    with pytest.raises(NotImplementedError) as exc_info:
        fn(*[None] * n_pos)
    msg = str(exc_info.value)
    assert name in msg, f"error must name the called function: got {msg!r}"
    assert "gm_redi_mpas_plan.md" in msg, (
        "error must point at the implementation plan doc")
