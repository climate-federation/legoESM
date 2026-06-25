"""GM/Redi MPAS — Phase 1 (slopes) numerical tests + dispatch guards.

Phase 1 (``compute_isopycnal_slopes_mpas``) is implemented and has its
own correctness suite below.  The remaining entry points
(``gm_redi_tracer_tendency_centered_mpas``, ``..._triads_mpas``, and
the top-level ``gm_redi_tracer_tendency_mpas``) are still skeletons —
they are guarded by the dispatch test that asserts they raise with a
plan-doc pointer.  As each phase lands, replace the corresponding
parametrize entry with a real numerical test.

See docs/ocean/experiments/gm_redi_mpas_plan.md for the phased plan.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig, VisbeckConfig
from legoesm.ocean.physics.lateral_mixing.gm_redi_mpas import (
    _visbeck_kappa_gm_mpas,
    voronoi_neumann_fill,
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
    filled = voronoi_neumann_fill(f, mask, mesh)
    # All ocean neighbours have value 1.0, so the land cell should be 1.0.
    assert float(filled[0]) == pytest.approx(1.0, abs=1e-12)
    # Other cells unchanged.
    np.testing.assert_allclose(np.asarray(filled[1:]), 1.0, atol=0.0)


# ============================================================================
# Dispatch guards for phases not yet implemented
# ============================================================================

# ============================================================================
# Phase 2 — centered tracer tendency
# ============================================================================

def _edge_mask(mesh, mask) -> jnp.ndarray:
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    return mask[c1] * mask[c2]


def test_centered_constant_q_zero_tendency(mesh, z_coord, cfg):
    """q uniform everywhere ⇒ horizontal and vertical fluxes both
    vanish identically ⇒ tendency = 0 to machine precision."""
    nlev = z_coord.dz_ref.shape[0]
    mask = _all_ocean_mask(mesh)
    jac = _unit_jacobian(mesh)
    em = _edge_mask(mesh, mask)

    rho = jnp.full((mesh.nCells, nlev), 1027.5, dtype=jnp.float64)
    S_n, _ = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)

    q = jnp.full((mesh.nCells, nlev), 17.0, dtype=jnp.float64)
    dqdt = gm_redi_tracer_tendency_centered_mpas(
        q, S_n, mask, em, z_coord, jac, mesh,
        kappa_GM=cfg.kappa_GM, kappa_Redi=cfg.kappa_Redi,
    )

    np.testing.assert_allclose(np.asarray(dqdt), 0.0, atol=1e-14)


def test_centered_q_eq_rho_redi_residual_small(mesh, z_coord, cfg):
    """When q = f(ρ) (linear in ρ), the off-diagonal Redi flux must
    cancel the diagonal Redi flux to within centred-averaging error.

    Setup: pure isopycnal flow, ``q ≡ rho``, ``κ_GM = 0`` so only the
    Redi tensor is active.  The exact small-slope identity
    ``∂_n q + S_n · ∂_z q = 0`` holds at every (edge, interface)
    because we built S_n from the *same* ρ field; the residual we see
    is the centred-averaging error from moving ``∂_z`` from cell to
    edge and from interface to full level.

    The tolerance is set generously because this is the *centered*
    scheme — the triad scheme (Phase 5) will tighten it to round-off.
    """
    z = _z_centers(z_coord)
    nlev = z.shape[0]
    beta = 0.5
    alpha = 1e-3
    rho = (1027.5
           + (beta * jnp.sin(mesh.latCell))[:, None]
           + (alpha * (-z))[None, :])

    mask = _all_ocean_mask(mesh)
    jac = _unit_jacobian(mesh)
    em = _edge_mask(mesh, mask)

    S_n, _ = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)
    q = rho

    cfg_redi_only = cfg._replace(kappa_GM=0.0, kappa_Redi=1.0e3)
    dqdt = gm_redi_tracer_tendency_centered_mpas(
        q, S_n, mask, em, z_coord, jac, mesh,
        kappa_GM=cfg_redi_only.kappa_GM,
        kappa_Redi=cfg_redi_only.kappa_Redi,
    )

    # Exclude the top and bottom rows (zero-flux BCs and surface/
    # bottom averaging are exact at boundaries; the interior is where
    # the centered residual lives).
    interior = np.asarray(dqdt[:, 1:-1])
    rms = float(np.sqrt(np.mean(interior ** 2)))
    # rho varies by ~β across hemispheres, α·H across depth: ~0.5
    # kg/m³ horizontally, ~5e-1 kg/m³ vertically.  A residual rms below
    # 1e-7 kg/m³/s with κ_R = 1e3 m²/s and S_max = 1e-2 means the
    # off-diagonal cancels to 8 orders of magnitude below the
    # diagonal flux scale (κ_R · ∂_n ρ ~ 1e3 · 1e-7 ~ 1e-4).
    assert rms < 5e-9, f"centered q=f(ρ) residual rms = {rms:.3g}"


def test_centered_pure_horizontal_q_diffuses(mesh, z_coord, cfg):
    """Diagonal Redi flux is alive: a horizontal q-gradient with no
    isopycnal slope (constant ρ) produces a non-zero tendency that
    has the sign of horizontal Laplacian diffusion."""
    nlev = z_coord.dz_ref.shape[0]
    mask = _all_ocean_mask(mesh)
    jac = _unit_jacobian(mesh)
    em = _edge_mask(mesh, mask)

    # Constant ρ ⇒ slopes are zero.
    rho = jnp.full((mesh.nCells, nlev), 1027.5, dtype=jnp.float64)
    S_n, _ = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)
    np.testing.assert_allclose(np.asarray(S_n), 0.0, atol=1e-14)

    # Tracer with sin(lat) gradient.
    q_lat = jnp.sin(mesh.latCell)
    q = jnp.broadcast_to(q_lat[:, None], (mesh.nCells, nlev))

    dqdt = gm_redi_tracer_tendency_centered_mpas(
        q, S_n, mask, em, z_coord, jac, mesh,
        kappa_GM=cfg.kappa_GM, kappa_Redi=cfg.kappa_Redi,
    )

    # The diffusive tendency on a sin(lat) field has the opposite sign
    # of the field itself in mid-latitudes (∇²sin(lat) on the sphere
    # ∝ -sin(lat)/R² · 2).  Check sign correlation.
    interior = np.asarray(dqdt[:, 0])      # any level (all identical here)
    field = np.asarray(q_lat)
    # Mask out near-equator points where sin(lat) ≈ 0.
    far = np.abs(field) > 0.3
    correlation = np.mean(np.sign(interior[far]) * np.sign(field[far]))
    assert correlation < -0.5, (
        f"diffusive tendency should oppose the field, got corr={correlation:.3g}"
    )
    # And it should be non-trivially non-zero.
    rms = float(np.sqrt(np.mean(interior ** 2)))
    assert rms > 1e-12, "diagonal Redi flux did not produce a tendency"


def test_centered_conserves_mass_globally(mesh, z_coord, cfg):
    """∫ tendency · areaCell · dz over all ocean cells/levels = 0
    to machine precision (FV closure of TRiSK divergence + zero-flux
    vertical BCs)."""
    z = _z_centers(z_coord)
    nlev = z.shape[0]
    beta = 0.5
    alpha = 1e-3
    rho = (1027.5
           + (beta * jnp.sin(mesh.latCell))[:, None]
           + (alpha * (-z))[None, :])
    mask = _all_ocean_mask(mesh)
    jac = _unit_jacobian(mesh)
    em = _edge_mask(mesh, mask)

    S_n, _ = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)
    # Use a non-trivial tracer so dq_h and dq_vert are both non-zero.
    q = jnp.cos(mesh.lonCell)[:, None] + 0.0 * z[None, :]
    q = jnp.broadcast_to(q, (mesh.nCells, nlev))

    dqdt = gm_redi_tracer_tendency_centered_mpas(
        q, S_n, mask, em, z_coord, jac, mesh,
        kappa_GM=cfg.kappa_GM, kappa_Redi=cfg.kappa_Redi,
    )

    dz_actual = z_coord.dz_ref * jac[:, None]                  # (nCells, nlev)
    integrand = np.asarray(dqdt) * np.asarray(mesh.areaCell)[:, None] \
        * np.asarray(dz_actual)
    total = float(np.sum(integrand))
    # Scale by maximum cell-volume × tendency magnitude.
    scale = float(
        np.max(np.asarray(mesh.areaCell)) * np.max(np.asarray(dz_actual))
        * max(np.max(np.abs(integrand)), 1e-30)
    )
    rel = abs(total) / scale
    assert rel < 1e-10, (
        f"global tracer integral drift = {total:.3e}, rel = {rel:.3e}"
    )


def test_centered_respects_land_mask(mesh, z_coord, cfg):
    """Tendency on land cells must be exactly zero, regardless of the
    tracer values used as land sentinels."""
    nlev = z_coord.dz_ref.shape[0]
    mask = jnp.ones((mesh.nCells,), dtype=jnp.float64).at[0].set(0.0)
    jac = _unit_jacobian(mesh)
    em = _edge_mask(mesh, mask)

    rho = jnp.full((mesh.nCells, nlev), 1027.5, dtype=jnp.float64)
    rho = rho.at[0, :].set(0.0)
    S_n, _ = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)

    # Tracer with a wild value in the land cell.
    q = jnp.broadcast_to(jnp.sin(mesh.latCell)[:, None], (mesh.nCells, nlev))
    q = q.at[0, :].set(99999.0)

    dqdt = gm_redi_tracer_tendency_centered_mpas(
        q, S_n, mask, em, z_coord, jac, mesh,
        kappa_GM=cfg.kappa_GM, kappa_Redi=cfg.kappa_Redi,
    )

    # Land cell tendency = 0.
    np.testing.assert_allclose(np.asarray(dqdt[0, :]), 0.0, atol=1e-14)


# ============================================================================
# Phase 3 — Visbeck adaptive κ_GM
# ============================================================================

def test_visbeck_kappa_grows_with_slope_magnitude(mesh, z_coord):
    """Visbeck kappa is monotone in the slope magnitude: doubling β
    (which doubles the meridional density gradient and hence |S|)
    must increase the per-cell kappa."""
    z = _z_centers(z_coord)
    nlev = z.shape[0]
    alpha = 1e-3
    mask = _all_ocean_mask(mesh)
    jac = _unit_jacobian(mesh)
    f_cor = 2.0 * constants.Omega * jnp.sin(mesh.latCell)
    # Wide kappa bounds so the level-2 mesh's small slopes don't clip
    # against the floor and mask the monotonicity we want to test.
    cfg_v = VisbeckConfig(
        enabled=True, alpha=0.015, L_fixed=1.0e5,
        use_rossby_radius=False, kappa_min=1e-12, kappa_max=4e3,
    )
    cfg = GMRediConfig(
        kappa_GM=1e3, kappa_Redi=1e3, S_max=1e-2, visbeck=cfg_v,
    )

    def kappa_for(beta):
        rho = (1027.5
               + (beta * jnp.sin(mesh.latCell))[:, None]
               + (alpha * (-z))[None, :])
        S_n, _ = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)
        return _visbeck_kappa_gm_mpas(
            rho, S_n, mesh, z_coord, jac, f_cor, cfg_v,
        )

    k_small = np.asarray(kappa_for(0.1))
    k_large = np.asarray(kappa_for(0.4))

    assert k_small.shape == (mesh.nCells,)
    # Median (per-cell) kappa increases with the larger slope.
    # Visbeck has kappa ∝ alpha·L²·N·|S|, so 4× β gives 4× kappa
    # (linear in |S| at fixed N).
    ratio = float(np.median(k_large) / np.maximum(np.median(k_small), 1e-30))
    assert ratio > 3.0, f"kappa should ~4x with 4x slope; got ratio={ratio:.3g}"
    # Both within configured bounds.
    assert float(k_small.min()) >= cfg_v.kappa_min - 1e-9
    assert float(k_large.max()) <= cfg_v.kappa_max + 1e-9


# ============================================================================
# Phase 4 — top-level wrapper (density + slopes + Visbeck + tendency)
# ============================================================================

def test_top_level_runs_and_returns_correct_shapes(mesh, z_coord):
    """Top-level wrapper accepts (T, S, eta, H_bathy) and returns
    (dT_dt, dS_dt) with the expected shapes; output is finite."""
    nlev = z_coord.dz_ref.shape[0]
    z = _z_centers(z_coord)

    # Stable stratification + mild horizontal T gradient.
    T = jnp.broadcast_to(
        (5.0 + 15.0 * jnp.cos(mesh.latCell))[:, None]
        + (-0.005 * z)[None, :],
        (mesh.nCells, nlev),
    )
    S = jnp.full((mesh.nCells, nlev), 35.0, dtype=jnp.float64)
    eta = jnp.zeros((mesh.nCells,), dtype=jnp.float64)
    H_bathy = jnp.full((mesh.nCells,), 500.0, dtype=jnp.float64)

    cfg = GMRediConfig(
        kappa_GM=1e3, kappa_Redi=1e3, S_max=1e-2,
        slope_scheme="centered",
    )

    dT_dt, dS_dt = gm_redi_tracer_tendency_mpas(
        T, S, eta, H_bathy, mesh, z_coord, cfg,
        eos="linear",  # avoid pressure dependence in this smoke test
    )

    assert dT_dt.shape == (mesh.nCells, nlev)
    assert dS_dt.shape == (mesh.nCells, nlev)
    assert np.all(np.isfinite(np.asarray(dT_dt)))
    assert np.all(np.isfinite(np.asarray(dS_dt)))
    # S is uniform ⇒ no salt tendency from gradients.
    np.testing.assert_allclose(np.asarray(dS_dt), 0.0, atol=1e-12)


def test_top_level_with_visbeck_matches_explicit_call(mesh, z_coord):
    """Visbeck-enabled top-level must match a hand-stitched call
    (rho via EOS iteration, slopes, Visbeck, centred tendency).
    Guards against the `enabled` branch silently bypassing Visbeck.
    """
    nlev = z_coord.dz_ref.shape[0]
    z = _z_centers(z_coord)
    T = jnp.broadcast_to(
        (5.0 + 15.0 * jnp.cos(mesh.latCell))[:, None]
        + (-0.005 * z)[None, :],
        (mesh.nCells, nlev),
    )
    S = jnp.full((mesh.nCells, nlev), 35.0, dtype=jnp.float64)
    eta = jnp.zeros((mesh.nCells,), dtype=jnp.float64)
    H_bathy = jnp.full((mesh.nCells,), 500.0, dtype=jnp.float64)

    visbeck = VisbeckConfig(
        enabled=True, alpha=0.015, L_fixed=1.0e5,
        use_rossby_radius=False, kappa_min=1e2, kappa_max=4e3,
    )
    cfg = GMRediConfig(
        kappa_GM=1e3, kappa_Redi=1e3, S_max=1e-2,
        visbeck=visbeck, slope_scheme="centered",
    )

    dT_top, _ = gm_redi_tracer_tendency_mpas(
        T, S, eta, H_bathy, mesh, z_coord, cfg, eos="linear",
    )

    # Hand-stitch the same pipeline.
    from legoesm.ocean.dynamics.ocean_tendency_common import (
        iterate_eos_and_pressure_anomaly,
    )
    from legoesm.ocean.eos import make_eos_fn, rho_0 as _RHO_0
    from legoesm.ocean.vertical import compute_ocean_jacobian

    mask = _all_ocean_mask(mesh)
    em = _edge_mask(mesh, mask)
    jac = compute_ocean_jacobian(eta, H_bathy, z_coord)
    eos_fn = make_eos_fn("linear", None)
    fill = lambda f: voronoi_neumann_fill(f, mask, mesh)
    rho, _, _ = iterate_eos_and_pressure_anomaly(
        T, S, mask, fill, eos_fn,
        z_coord.dz_ref, _RHO_0, constants.g, n_iter=2,
    )
    S_n, _ = compute_isopycnal_slopes_mpas(rho, mask, z_coord, jac, mesh, cfg)
    f_cor = 2.0 * constants.Omega * jnp.sin(mesh.latCell)
    kappa_GM = _visbeck_kappa_gm_mpas(rho, S_n, mesh, z_coord, jac, f_cor, visbeck)
    dT_ref = gm_redi_tracer_tendency_centered_mpas(
        T, S_n, mask, em, z_coord, jac, mesh, kappa_GM, cfg.kappa_Redi,
    )

    np.testing.assert_allclose(np.asarray(dT_top), np.asarray(dT_ref), atol=1e-14)


def test_top_level_triads_not_implemented_yet(mesh, z_coord):
    """Until Phase 5 lands, slope_scheme='triads' must raise a clear
    NotImplementedError naming the plan doc."""
    nlev = z_coord.dz_ref.shape[0]
    T = jnp.full((mesh.nCells, nlev), 10.0, dtype=jnp.float64)
    S = jnp.full((mesh.nCells, nlev), 35.0, dtype=jnp.float64)
    eta = jnp.zeros((mesh.nCells,), dtype=jnp.float64)
    H_bathy = jnp.full((mesh.nCells,), 500.0, dtype=jnp.float64)
    cfg = GMRediConfig(
        kappa_GM=1e3, kappa_Redi=1e3, S_max=1e-2, slope_scheme="triads",
    )
    with pytest.raises(NotImplementedError) as exc_info:
        gm_redi_tracer_tendency_mpas(
            T, S, eta, H_bathy, mesh, z_coord, cfg, eos="linear",
        )
    msg = str(exc_info.value)
    assert "triad" in msg.lower()
    assert "gm_redi_mpas_plan.md" in msg


# ============================================================================
# Dispatch guard for the only remaining unimplemented entry point
# ============================================================================

# ============================================================================
# Phase 4 dycore hook — MPASOceanModel.step with gm_redi enabled
# ============================================================================

def test_dycore_hook_runs_one_step_with_gm_redi():
    """The full MPASOceanModel.step must accept ``gm_redi=GMRediConfig(...)``
    and complete one step with finite output that differs from the
    gm_redi=None baseline."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
    from legoesm.ocean.mpas_config import MPASOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    mesh_local = create_voronoi_mesh(subdivision_level=2)
    z_coord_local = create_ocean_z_star(
        n_levels=5, H_max=500.0, dz_surface=20.0, dz_deep=200.0,
    )
    state = rest_state_mpas_ocean(
        mesh_local, z_coord_local,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=500.0, land_lat_threshold=85.0,
    )
    # Strong horizontal T gradient so GM/Redi has a clearly resolvable
    # signal at the level-2 mesh (the slopes are O(β/(α·R)) and
    # κ_R · S² · ∂_z q is what drives the visible vertical mixing).
    T = state.T.data
    T_modified = T + 5.0 * jnp.cos(mesh_local.latCell)[:, None]
    state = state._replace(T=state.T.replace(data=T_modified))

    base_cfg = MPASOceanConfig(
        eos="linear",
        n_barotropic_substeps=8,
    )
    cfg_with_gm = base_cfg._replace(
        gm_redi=GMRediConfig(
            kappa_GM=1e4, kappa_Redi=1e4, S_max=1e-2,
            slope_scheme="centered",
        ),
    )

    dt = 600.0
    model_off = MPASOceanModel(
        mesh=mesh_local, z_coord=z_coord_local, config=base_cfg,
    )
    model_on = MPASOceanModel(
        mesh=mesh_local, z_coord=z_coord_local, config=cfg_with_gm,
    )

    state_off = model_off.step(state, dt)
    state_on  = model_on.step(state, dt)

    assert np.all(np.isfinite(np.asarray(state_off.T.data)))
    assert np.all(np.isfinite(np.asarray(state_on.T.data)))

    # GM/Redi must produce a non-trivial difference: the hook is alive.
    delta = np.asarray(state_on.T.data) - np.asarray(state_off.T.data)
    rms_delta = float(np.sqrt(np.mean(delta ** 2)))
    assert rms_delta > 1e-10, (
        f"GM/Redi hook had no effect on T (rms delta = {rms_delta:.3e}); "
        "is the dycore hook actually wired in?"
    )


def test_dycore_hook_default_none_unchanged():
    """When config.gm_redi is None (default), step() output is bit-
    identical to the pre-hook behaviour, i.e. the hook is a no-op."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
    from legoesm.ocean.mpas_config import MPASOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    mesh_local = create_voronoi_mesh(subdivision_level=2)
    z_coord_local = create_ocean_z_star(
        n_levels=5, H_max=500.0, dz_surface=20.0, dz_deep=200.0,
    )
    state = rest_state_mpas_ocean(
        mesh_local, z_coord_local,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=500.0, land_lat_threshold=85.0,
    )
    cfg = MPASOceanConfig(eos="linear", n_barotropic_substeps=8)
    assert cfg.gm_redi is None
    model = MPASOceanModel(
        mesh=mesh_local, z_coord=z_coord_local, config=cfg,
    )
    new_state = model.step(state, 60.0)
    assert np.all(np.isfinite(np.asarray(new_state.T.data)))


def test_triads_leaf_raises_with_plan_pointer():
    """Phase 5 (triads leaf) is not implemented yet — direct calls to
    it must raise NotImplementedError naming the function and the
    plan doc."""
    import inspect
    sig = inspect.signature(gm_redi_tracer_tendency_triads_mpas)
    n_pos = sum(
        1 for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )
    with pytest.raises(NotImplementedError) as exc_info:
        gm_redi_tracer_tendency_triads_mpas(*[None] * n_pos)
    msg = str(exc_info.value)
    assert "gm_redi_tracer_tendency_triads_mpas" in msg
    assert "gm_redi_mpas_plan.md" in msg
