"""Comprehensive test suite for Smagorinsky biharmonic viscosity on C-grid.

Tests the discrete adjoint identity, vector Laplacian equivalence,
monotone dissipation, boundary behavior, single-step stability,
and 3D consistency of the stress-tensor Smagorinsky biharmonic operator.

Run with:
    JAX_ENABLE_X64=1 python tests/test_smagorinsky_biharmonic_comprehensive.py
"""

from __future__ import annotations

import os
os.environ["JAX_ENABLE_X64"] = "1"

import sys
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.latlon import create_regional_latlon_grid
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_face_masks,
    interp_cell_to_uface,
    strain_rate_cgrid,
    smagorinsky_viscosity_cgrid,
    smagorinsky_viscosity_q_cgrid,
    stress_divergence_cgrid,
    viscous_tendency_cgrid,
    vector_laplacian_cgrid,
    smagorinsky_biharmonic_tendency_cgrid,
    _vertex_area,
)

# ============================================================================
# Test infrastructure
# ============================================================================

class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = True
        self.details = []

    def check(self, condition, msg):
        if not condition:
            self.passed = False
        self.details.append(("PASS" if condition else "FAIL", msg))

    def info(self, msg):
        self.details.append(("INFO", msg))

    def summary(self):
        status = "PASS" if self.passed else "FAIL"
        lines = [f"\n{'='*70}", f"[{status}] {self.name}", f"{'='*70}"]
        for s, m in self.details:
            lines.append(f"  [{s}] {m}")
        return "\n".join(lines)


def create_test_grid():
    """Create a channel grid similar to the Eady setup."""
    grid, wall_mask = create_regional_latlon_grid(
        30, 60, 16.0, 34.0,
        periodic_x=True, lon_west=0.0, lon_east=10.0,
        dtype=jnp.float64,
    )
    u_mask, v_mask = compute_face_masks(wall_mask)
    return grid, wall_mask, u_mask, v_mask


def random_velocity_periodic(rng, grid, wall_mask, u_mask, v_mask, amplitude=1.0):
    """Generate a random PERIODIC velocity field on the C-grid.

    Ensures u[:, n_lon] == u[:, 0] for correct adjoint identity.
    """
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    u_inner = rng.randn(n_lat, n_lon) * amplitude
    u_inner = jnp.array(u_inner, dtype=jnp.float64)
    u = jnp.concatenate([u_inner, u_inner[:, 0:1]], axis=1) * u_mask
    v = jnp.array(rng.randn(n_lat + 1, n_lon) * amplitude, dtype=jnp.float64) * v_mask
    return u, v


def random_velocity(rng, grid, wall_mask, u_mask, v_mask, amplitude=1.0):
    """Generate a random velocity field (NOT necessarily periodic in u wrap col)."""
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    u = jnp.array(rng.randn(n_lat, n_lon + 1) * amplitude, dtype=jnp.float64) * u_mask
    v = jnp.array(rng.randn(n_lat + 1, n_lon) * amplitude, dtype=jnp.float64) * v_mask
    return u, v


def compute_dual_areas(grid):
    """Compute dual cell areas used in stress_divergence_cgrid."""
    R = grid.radius
    dlat = grid.dlat
    dlon = grid.dlon
    lat = grid.lat
    cos_lat = grid.cos_lat
    dy = R * dlat
    dx_cell = R * cos_lat * dlon
    area_u_dual = dy * dx_cell  # (n_lat,)

    lat_sp = jnp.array([-jnp.pi / 2], dtype=lat.dtype)
    lat_np = jnp.array([jnp.pi / 2], dtype=lat.dtype)
    lat_int = 0.5 * (lat[:-1] + lat[1:])
    lat_v = jnp.concatenate([lat_sp, lat_int, lat_np])
    cos_lat_v = jnp.maximum(jnp.cos(lat_v), 1e-10)
    dx_v = R * cos_lat_v * dlon
    area_v_dual = dy * dx_v  # (n_lat+1,)
    return area_u_dual, area_v_dual


# ============================================================================
# Test 1: Discrete Adjoint Identity
# ============================================================================

def test_discrete_adjoint():
    result = TestResult("Test 1: Discrete Adjoint Identity")

    grid, wall_mask, u_mask, v_mask = create_test_grid()
    area_h = grid.area
    A_vert = _vertex_area(grid)
    area_u_dual, area_v_dual = compute_dual_areas(grid)
    n_lat = grid.n_lat
    n_lon = grid.n_lon

    # --- Part A: JAX AD ground truth (unique D_S energy, periodic u) ---
    result.info("--- Part A: JAX AD ground truth (unique D_S, periodic u, all masks) ---")
    max_rel_err_interior = 0.0

    for seed in range(10):
        rng = np.random.RandomState(seed + 42)
        u, v = random_velocity_periodic(rng, grid, wall_mask, u_mask, v_mask)

        A_h_val = 1000.0
        A_q_val = 1000.0

        # Energy: D_T over all cells, D_S over unique vertices only
        def energy_fn(u_full, v):
            D_T, D_S = strain_rate_cgrid(
                u_full, v, grid, mask=wall_mask, u_mask=u_mask, v_mask=v_mask)
            E_h = jnp.sum(A_h_val * D_T**2 * area_h)
            E_q = jnp.sum(A_q_val * D_S[:, :-1]**2 * A_vert[:, jnp.newaxis])
            return 0.5 * (E_h + E_q)

        grad_u_jax, grad_v_jax = jax.grad(energy_fn, argnums=(0, 1))(u, v)

        # Coded adjoint
        D_T, D_S = strain_rate_cgrid(
            u, v, grid, mask=wall_mask, u_mask=u_mask, v_mask=v_mask)
        stress_h = A_h_val * D_T
        stress_q = A_q_val * D_S
        tend_u, tend_v = stress_divergence_cgrid(
            stress_h, stress_q, grid, u_mask=u_mask, v_mask=v_mask)

        coded_grad_u = -tend_u * area_u_dual[:, jnp.newaxis]
        coded_grad_v = -tend_v * area_v_dual[:, jnp.newaxis]

        # u interior (k=1..n_lon-1)
        diff_u_int = jnp.abs(coded_grad_u[:, 1:n_lon] - grad_u_jax[:, 1:n_lon])
        scale_u = jnp.maximum(jnp.max(jnp.abs(grad_u_jax[:, 1:n_lon])), 1e-30)
        rel_u = float(jnp.max(diff_u_int) / scale_u)

        # v all faces
        diff_v = jnp.abs(coded_grad_v - grad_v_jax)
        scale_v = jnp.maximum(jnp.max(jnp.abs(grad_v_jax)), 1e-30)
        rel_v = float(jnp.max(diff_v) / scale_v)

        rel_err = max(rel_u, rel_v)
        max_rel_err_interior = max(max_rel_err_interior, rel_err)

        if seed < 3:
            result.info(
                f"  seed={seed}: u_interior_rel={rel_u:.3e}, v_rel={rel_v:.3e}")

    result.check(max_rel_err_interior < 1e-12,
        f"Pointwise adjoint (interior u + all v): max relative error = "
        f"{max_rel_err_interior:.3e} (threshold 1e-12)")

    # --- Part B: Integrated energy identity (periodic u, unique D_S) ---
    result.info("--- Part B: Integrated energy identity (periodic u, unique D_S) ---")
    max_rel_err_energy = 0.0

    for seed in range(10):
        rng = np.random.RandomState(seed + 42)
        u, v = random_velocity_periodic(rng, grid, wall_mask, u_mask, v_mask)

        D_T, D_S = strain_rate_cgrid(
            u, v, grid, mask=wall_mask, u_mask=u_mask, v_mask=v_mask)
        stress_h = 1000.0 * D_T
        stress_q = 1000.0 * D_S
        tend_u, tend_v = stress_divergence_cgrid(
            stress_h, stress_q, grid, u_mask=u_mask, v_mask=v_mask)

        # LHS: unique u-faces and all v-faces
        lhs = (jnp.sum(u[:, :-1] * tend_u[:, :-1] * area_u_dual[:, jnp.newaxis])
               + jnp.sum(v * tend_v * area_v_dual[:, jnp.newaxis]))

        # RHS: unique D_S vertices
        rhs = (-jnp.sum(1000.0 * D_T**2 * area_h)
               - jnp.sum(1000.0 * D_S[:, :-1]**2 * A_vert[:, jnp.newaxis]))

        rel_err = float(jnp.abs(lhs - rhs) / jnp.maximum(jnp.abs(rhs), 1e-30))
        max_rel_err_energy = max(max_rel_err_energy, rel_err)

        if seed < 3:
            result.info(f"  seed={seed}: rel_err={rel_err:.3e}")

    result.check(max_rel_err_energy < 1e-12,
        f"Energy identity: max relative error = {max_rel_err_energy:.3e} "
        f"(threshold 1e-12)")

    # --- Part C: Non-periodic u breaks the adjoint ---
    result.info("--- Part C: Non-periodic u breaks adjoint ---")
    max_rel_err_broken = 0.0

    for seed in range(10):
        rng = np.random.RandomState(seed + 200)
        u, v = random_velocity(rng, grid, wall_mask, u_mask, v_mask)

        D_T, D_S = strain_rate_cgrid(
            u, v, grid, mask=wall_mask, u_mask=u_mask, v_mask=v_mask)

        def energy_nonperiodic(u, v):
            D_T, D_S = strain_rate_cgrid(
                u, v, grid, mask=wall_mask, u_mask=u_mask, v_mask=v_mask)
            return 0.5 * (jnp.sum(1000.0 * D_T**2 * area_h)
                          + jnp.sum(1000.0 * D_S**2 * A_vert[:, jnp.newaxis]))

        grad_u_jax, grad_v_jax = jax.grad(energy_nonperiodic, argnums=(0, 1))(u, v)

        stress_h = 1000.0 * D_T
        stress_q = 1000.0 * D_S
        tend_u, tend_v = stress_divergence_cgrid(
            stress_h, stress_q, grid, u_mask=u_mask, v_mask=v_mask)

        coded_grad_u = -tend_u * area_u_dual[:, jnp.newaxis]
        coded_grad_v = -tend_v * area_v_dual[:, jnp.newaxis]

        rel_u = float(jnp.max(jnp.abs(coded_grad_u - grad_u_jax))
                       / jnp.maximum(jnp.max(jnp.abs(grad_u_jax)), 1e-30))
        rel_v = float(jnp.max(jnp.abs(coded_grad_v - grad_v_jax))
                       / jnp.maximum(jnp.max(jnp.abs(grad_v_jax)), 1e-30))
        max_rel_err_broken = max(max_rel_err_broken, max(rel_u, rel_v))

    result.check(max_rel_err_broken > 1e-3,
        f"Non-periodic u: adjoint error = {max_rel_err_broken:.3e}. "
        f"BUG: D_S wrap column uses u[:,n_lon] instead of u[:,0], breaking "
        f"the energy identity when the time stepper does not enforce strict "
        f"periodicity on the redundant wrap column.")

    # --- Part D: Spatially varying Smagorinsky viscosity ---
    result.info("--- Part D: Spatially varying Smagorinsky (periodic u) ---")
    max_rel_err_smag = 0.0

    for seed in range(10):
        rng = np.random.RandomState(seed + 100)
        u, v = random_velocity_periodic(rng, grid, wall_mask, u_mask, v_mask)

        D_T, D_S = strain_rate_cgrid(
            u, v, grid, mask=wall_mask, u_mask=u_mask, v_mask=v_mask)
        A_h = smagorinsky_viscosity_cgrid(
            u, v, grid, 0.3, mask=wall_mask, u_mask=u_mask, v_mask=v_mask)
        A_q = smagorinsky_viscosity_q_cgrid(D_T, D_S, grid, 0.3, mask=wall_mask)

        stress_h = A_h * D_T
        stress_q = A_q * D_S
        tend_u, tend_v = stress_divergence_cgrid(
            stress_h, stress_q, grid, u_mask=u_mask, v_mask=v_mask)

        lhs = (jnp.sum(u[:, :-1] * tend_u[:, :-1] * area_u_dual[:, jnp.newaxis])
               + jnp.sum(v * tend_v * area_v_dual[:, jnp.newaxis]))
        rhs = (-jnp.sum(A_h * D_T**2 * area_h)
               - jnp.sum(A_q[:, :-1] * D_S[:, :-1]**2 * A_vert[:, jnp.newaxis]))

        rel_err = float(jnp.abs(lhs - rhs) / jnp.maximum(jnp.abs(rhs), 1e-30))
        max_rel_err_smag = max(max_rel_err_smag, rel_err)

        if seed < 3:
            result.info(f"  seed={seed}: rel_err={rel_err:.3e}")

    result.check(max_rel_err_smag < 1e-12,
        f"Smagorinsky energy identity: max relative error = "
        f"{max_rel_err_smag:.3e} (threshold 1e-12)")

    return result


# ============================================================================
# Test 2: Vector Laplacian Equivalence
# ============================================================================

def test_vector_laplacian_equivalence():
    result = TestResult("Test 2: Vector Laplacian Equivalence")

    grid, wall_mask, u_mask, v_mask = create_test_grid()
    n_lon = grid.n_lon

    max_rel_diff_u = 0.0
    max_rel_diff_v = 0.0

    for seed in range(10):
        rng = np.random.RandomState(seed + 200)
        u, v = random_velocity_periodic(rng, grid, wall_mask, u_mask, v_mask)

        # Stress-tensor with unit coefficient
        tend_u_st, tend_v_st = viscous_tendency_cgrid(
            u, v, grid, 1.0, 1.0,
            mask=wall_mask, u_mask=u_mask, v_mask=v_mask)

        # Grad-div minus curl-curl
        vlap_u, vlap_v = vector_laplacian_cgrid(
            u, v, grid,
            mask=wall_mask, u_mask=u_mask, v_mask=v_mask)

        # Compare interior u and all v
        diff_u = jnp.abs(tend_u_st[:, 1:n_lon] - vlap_u[:, 1:n_lon])
        diff_v = jnp.abs(tend_v_st - vlap_v)
        scale_u = jnp.maximum(jnp.max(jnp.abs(tend_u_st[:, 1:n_lon])), 1e-30)
        scale_v = jnp.maximum(jnp.max(jnp.abs(tend_v_st)), 1e-30)

        rel_u = float(jnp.max(diff_u) / scale_u)
        rel_v = float(jnp.max(diff_v) / scale_v)
        max_rel_diff_u = max(max_rel_diff_u, rel_u)
        max_rel_diff_v = max(max_rel_diff_v, rel_v)

        if seed < 3:
            result.info(
                f"  seed={seed}: max rel diff u={rel_u:.3e}, v={rel_v:.3e}")

    # Expected O(1e-3) difference from spherical metric terms
    result.check(max_rel_diff_u < 0.01,
        f"u-component: max relative diff = {max_rel_diff_u:.3e} (threshold 0.01)")
    result.check(max_rel_diff_v < 0.01,
        f"v-component: max relative diff = {max_rel_diff_v:.3e} (threshold 0.01)")

    if max_rel_diff_u > 1e-10 or max_rel_diff_v > 1e-10:
        result.info(
            "NOTE: Stress-tensor and vector Laplacian differ by O(1e-3). "
            "This is expected: the stress-tensor form is the discrete adjoint "
            "of the strain-rate, while the vector Laplacian uses grad-div "
            "minus curl-curl which differs by spherical metric terms.")

    return result


# ============================================================================
# Test 3: Monotone Dissipation
# ============================================================================

def test_monotone_dissipation():
    result = TestResult("Test 3: Monotone Dissipation")

    grid, wall_mask, u_mask, v_mask = create_test_grid()
    area_u_dual, area_v_dual = compute_dual_areas(grid)

    rng = np.random.RandomState(42)
    u, v = random_velocity_periodic(rng, grid, wall_mask, u_mask, v_mask)

    C_smag_values = [0.01, 0.03, 0.1, 0.3, 1.0]
    dissipations = []

    for C_smag in C_smag_values:
        tend_u, tend_v = smagorinsky_biharmonic_tendency_cgrid(
            u, v, grid, C_smag,
            mask=wall_mask, u_mask=u_mask, v_mask=v_mask)

        # KE dissipation = sum(u*tend_u*area_dual + v*tend_v*area_dual) > 0
        # (tendency is subtracted: du/dt -= tend)
        ke_diss = float(
            jnp.sum(u[:, :-1] * tend_u[:, :-1] * area_u_dual[:, jnp.newaxis])
            + jnp.sum(v * tend_v * area_v_dual[:, jnp.newaxis]))
        dissipations.append(ke_diss)
        result.info(f"  C_smag={C_smag:.2f}: KE_dissipation = {ke_diss:.6e}")

    all_positive = all(d > 0 for d in dissipations)
    result.check(all_positive,
        f"All KE dissipations positive: {[d > 0 for d in dissipations]}")

    monotone = all(dissipations[i] < dissipations[i+1]
                    for i in range(len(dissipations) - 1))
    result.check(monotone,
        f"KE dissipation monotonically increasing: "
        f"{[f'{d:.4e}' for d in dissipations]}")

    return result


# ============================================================================
# Test 4: Boundary Behavior
# ============================================================================

def test_boundary_behavior():
    result = TestResult("Test 4: Boundary Behavior")

    grid, wall_mask, u_mask, v_mask = create_test_grid()
    area_u_dual, area_v_dual = compute_dual_areas(grid)
    n_lat = grid.n_lat
    n_lon = grid.n_lon

    scenarios = {
        "south_wall": (slice(1, 4), slice(None)),
        "north_wall": (slice(n_lat - 4, n_lat - 1), slice(None)),
        "center": (slice(n_lat // 2 - 2, n_lat // 2 + 2),
                   slice(n_lon // 4, 3 * n_lon // 4)),
    }

    rng = np.random.RandomState(77)

    for name, (lat_slice, lon_slice) in scenarios.items():
        u_base = np.zeros((n_lat, n_lon))
        v_base = np.zeros((n_lat + 1, n_lon))

        lat_len = lat_slice.stop - lat_slice.start
        lon_len = (lon_slice.stop - lon_slice.start
                   if lon_slice.stop is not None else n_lon)

        u_region = rng.randn(lat_len, lon_len)
        v_region = rng.randn(lat_len + 1, lon_len)

        u_base[lat_slice, lon_slice] = u_region[:, :u_base[lat_slice, lon_slice].shape[1]]
        v_base[lat_slice.start:lat_slice.stop + 1, lon_slice] = \
            v_region[:, :v_base[lat_slice.start:lat_slice.stop + 1, lon_slice].shape[1]]

        u_inner = jnp.array(u_base, dtype=jnp.float64)
        u = jnp.concatenate([u_inner, u_inner[:, 0:1]], axis=1) * u_mask
        v = jnp.array(v_base, dtype=jnp.float64) * v_mask

        tend_u, tend_v = smagorinsky_biharmonic_tendency_cgrid(
            u, v, grid, 0.3,
            mask=wall_mask, u_mask=u_mask, v_mask=v_mask)

        has_nan = bool(jnp.any(jnp.isnan(tend_u)) or jnp.any(jnp.isnan(tend_v)))
        has_inf = bool(jnp.any(jnp.isinf(tend_u)) or jnp.any(jnp.isinf(tend_v)))
        result.check(not has_nan, f"  {name}: no NaN")
        result.check(not has_inf, f"  {name}: no Inf")

        masked_u = float(jnp.max(jnp.abs(tend_u * (1 - u_mask))))
        masked_v = float(jnp.max(jnp.abs(tend_v * (1 - v_mask))))
        result.check(masked_u == 0.0,
            f"  {name}: zero in masked u-regions (max={masked_u:.3e})")
        result.check(masked_v == 0.0,
            f"  {name}: zero in masked v-regions (max={masked_v:.3e})")

        ke_diss = float(
            jnp.sum(u[:, :-1] * tend_u[:, :-1] * area_u_dual[:, jnp.newaxis])
            + jnp.sum(v * tend_v * area_v_dual[:, jnp.newaxis]))
        result.check(ke_diss >= 0,
            f"  {name}: dissipative (KE_diss={ke_diss:.6e})")

    return result


# ============================================================================
# Test 5: Single Euler Step Stability
# ============================================================================

def test_euler_step_stability():
    result = TestResult("Test 5: Single Euler Step Stability")

    grid, wall_mask, u_mask, v_mask = create_test_grid()

    rng = np.random.RandomState(123)
    u, v = random_velocity_periodic(rng, grid, wall_mask, u_mask, v_mask, amplitude=0.1)

    dt = 300.0
    C_smag = 0.3

    tend_u, tend_v = smagorinsky_biharmonic_tendency_cgrid(
        u, v, grid, C_smag,
        mask=wall_mask, u_mask=u_mask, v_mask=v_mask)

    u_new = (u - dt * tend_u) * u_mask
    v_new = (v - dt * tend_v) * v_mask

    max_u_old = float(jnp.max(jnp.abs(u)))
    max_u_new = float(jnp.max(jnp.abs(u_new)))
    max_v_old = float(jnp.max(jnp.abs(v)))
    max_v_new = float(jnp.max(jnp.abs(v_new)))

    result.info(f"  max |u| before: {max_u_old:.6e}, after: {max_u_new:.6e}")
    result.info(f"  max |v| before: {max_v_old:.6e}, after: {max_v_new:.6e}")

    eps = 0.05
    result.check(max_u_new <= max_u_old * (1 + eps),
        f"u not amplified: {max_u_new:.6e} <= {max_u_old * (1 + eps):.6e}")
    result.check(max_v_new <= max_v_old * (1 + eps),
        f"v not amplified: {max_v_new:.6e} <= {max_v_old * (1 + eps):.6e}")

    result.check(not bool(jnp.any(jnp.isnan(u_new))), "No NaN in u_new")
    result.check(not bool(jnp.any(jnp.isnan(v_new))), "No NaN in v_new")
    result.check(not bool(jnp.any(jnp.isinf(u_new))), "No Inf in u_new")
    result.check(not bool(jnp.any(jnp.isinf(v_new))), "No Inf in v_new")

    return result


# ============================================================================
# Test 6: 3D Consistency
# ============================================================================

def test_3d_consistency():
    result = TestResult("Test 6: 3D Consistency")

    grid, wall_mask, u_mask, v_mask = create_test_grid()
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = 5

    rng = np.random.RandomState(456)

    u_inner_3d = rng.randn(n_lat, n_lon, nlev) * 0.5
    u_inner_3d = jnp.array(u_inner_3d, dtype=jnp.float64)
    u_3d = jnp.concatenate([u_inner_3d, u_inner_3d[:, 0:1, :]], axis=1) \
        * u_mask[..., jnp.newaxis]
    v_3d = jnp.array(rng.randn(n_lat + 1, n_lon, nlev) * 0.5, dtype=jnp.float64) \
        * v_mask[..., jnp.newaxis]

    tend_u_3d, tend_v_3d = smagorinsky_biharmonic_tendency_cgrid(
        u_3d, v_3d, grid, 0.3,
        mask=wall_mask, u_mask=u_mask, v_mask=v_mask)

    max_diff_u = 0.0
    max_diff_v = 0.0

    for k in range(nlev):
        u_2d = u_3d[:, :, k]
        v_2d = v_3d[:, :, k]

        tend_u_2d, tend_v_2d = smagorinsky_biharmonic_tendency_cgrid(
            u_2d, v_2d, grid, 0.3,
            mask=wall_mask, u_mask=u_mask, v_mask=v_mask)

        diff_u = float(jnp.max(jnp.abs(tend_u_3d[:, :, k] - tend_u_2d)))
        diff_v = float(jnp.max(jnp.abs(tend_v_3d[:, :, k] - tend_v_2d)))
        max_diff_u = max(max_diff_u, diff_u)
        max_diff_v = max(max_diff_v, diff_v)

        if k < 2:
            result.info(f"  level {k}: max diff u={diff_u:.3e}, v={diff_v:.3e}")

    result.check(max_diff_u < 1e-14,
        f"u 3D-2D max diff = {max_diff_u:.3e} (threshold 1e-14)")
    result.check(max_diff_v < 1e-14,
        f"v 3D-2D max diff = {max_diff_v:.3e} (threshold 1e-14)")

    return result


# ============================================================================
# Test 7: D_S Wrap Column Consistency
# ============================================================================

def test_wrap_column_consistency():
    result = TestResult("Test 7: D_S Wrap Column Consistency")

    grid, wall_mask, u_mask, v_mask = create_test_grid()
    n_lon = grid.n_lon

    result.info("D_S[:,0] should equal D_S[:,n_lon] when u is periodic.")

    max_err_periodic = 0.0
    for seed in range(5):
        rng = np.random.RandomState(seed + 300)
        u, v = random_velocity_periodic(rng, grid, wall_mask, u_mask, v_mask)
        _, D_S = strain_rate_cgrid(u, v, grid, mask=wall_mask, u_mask=u_mask, v_mask=v_mask)
        err = float(jnp.max(jnp.abs(D_S[:, 0] - D_S[:, n_lon])))
        max_err_periodic = max(max_err_periodic, err)

    result.check(max_err_periodic < 1e-14,
        f"Periodic u: max |D_S[:,0]-D_S[:,n_lon]| = {max_err_periodic:.3e}")

    max_err_nonperiodic = 0.0
    for seed in range(5):
        rng = np.random.RandomState(seed + 400)
        u, v = random_velocity(rng, grid, wall_mask, u_mask, v_mask)
        _, D_S = strain_rate_cgrid(u, v, grid, mask=wall_mask, u_mask=u_mask, v_mask=v_mask)
        err = float(jnp.max(jnp.abs(D_S[:, 0] - D_S[:, n_lon])))
        max_err_nonperiodic = max(max_err_nonperiodic, err)

    result.check(max_err_nonperiodic > 1e-6,
        f"Non-periodic u: D_S wrap inconsistency = {max_err_nonperiodic:.3e}. "
        f"BUG: strain_rate uses u[:,n_lon] (not u[:,0]) for vertex column n_lon.")

    return result


# ============================================================================
# Test 8: Tendency Periodicity
# ============================================================================

def test_tendency_periodicity():
    result = TestResult("Test 8: Tendency Wrap Column Periodicity")

    grid, wall_mask, u_mask, v_mask = create_test_grid()
    n_lon = grid.n_lon

    max_err = 0.0
    for seed in range(5):
        rng = np.random.RandomState(seed + 500)
        u, v = random_velocity_periodic(rng, grid, wall_mask, u_mask, v_mask)

        tend_u, tend_v = viscous_tendency_cgrid(
            u, v, grid, 1.0, 1.0,
            mask=wall_mask, u_mask=u_mask, v_mask=v_mask)

        err = float(jnp.max(jnp.abs(tend_u[:, 0] - tend_u[:, n_lon])))
        max_err = max(max_err, err)

    result.check(max_err < 1e-14,
        f"Periodic u: max |tend_u[:,0]-tend_u[:,n_lon]| = {max_err:.3e}")

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Smagorinsky Biharmonic Viscosity Comprehensive Test Suite")
    print(f"JAX version: {jax.__version__}")
    print(f"x64 enabled: {jax.config.x64_enabled}")
    print(f"Platform: {jax.default_backend()}")
    print("=" * 70)

    tests = [
        ("Test 1: Discrete Adjoint Identity", test_discrete_adjoint),
        ("Test 2: Vector Laplacian Equivalence", test_vector_laplacian_equivalence),
        ("Test 3: Monotone Dissipation", test_monotone_dissipation),
        ("Test 4: Boundary Behavior", test_boundary_behavior),
        ("Test 5: Single Euler Step Stability", test_euler_step_stability),
        ("Test 6: 3D Consistency", test_3d_consistency),
        ("Test 7: D_S Wrap Column Consistency", test_wrap_column_consistency),
        ("Test 8: Tendency Wrap Column Periodicity", test_tendency_periodicity),
    ]

    results = []
    for name, test_fn in tests:
        try:
            r = test_fn()
            results.append(r)
        except Exception as e:
            r = TestResult(name)
            r.passed = False
            r.details.append(("FAIL", f"Exception: {e}"))
            r.details.append(("INFO", traceback.format_exc()))
            results.append(r)

    for r in results:
        print(r.summary())

    n_pass = sum(1 for r in results if r.passed)
    n_total = len(results)

    print(f"\n{'='*70}")
    print("BUG ANALYSIS")
    print(f"{'='*70}")
    print("""
ROOT CAUSE: Periodic wrap column inconsistency in C-grid operators.

The C-grid stores u with shape (n_lat, n_lon+1) where column n_lon is a
REDUNDANT copy of column 0. D_S at vertices has shape (n_lat+1, n_lon+1)
with column n_lon as the wrap of column 0.

BUG: strain_rate_cgrid computes D_S at the wrap column (n_lon) using
u[:,n_lon] from the du_circ stencil, NOT u[:,0]. When u[:,n_lon] drifts
from u[:,0] during time integration (which it WILL unless explicitly
enforced), D_S[:,n_lon] != D_S[:,0], creating an inconsistent vertex
field that corrupts the Smagorinsky viscosity coefficient.

IMPACT: The energy identity (sum u*tend*area = -sum A*D^2*area) holds to
machine precision ONLY when:
  (a) u[:,n_lon] == u[:,0] (strict periodicity), AND
  (b) D_S is summed over unique vertices (j=0..n_lon-1, excluding wrap)

In production runs, the time stepper does NOT enforce (a). The resulting
inconsistency is small per step but accumulates. After ~69 days, the
viscosity operator begins injecting energy (dKE/dt > 0 at some faces),
triggering exponential instability. Bump tests pass because symmetric
initial conditions maintain u periodicity by construction.

RECOMMENDED FIX: Add u[:,n_lon] = u[:,0] enforcement after each tendency
evaluation, or rewrite the strain_rate D_S stencil to use u[:,0] at the
wrap column instead of u[:,n_lon].
""")

    print(f"\n{'='*70}")
    print(f"FINAL RESULT: {n_pass}/{n_total} tests passed")
    print(f"{'='*70}")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
