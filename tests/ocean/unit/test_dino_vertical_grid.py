"""Unit tests for the DINO 36-level z* grid (Phase 2E, Appendix C eq C3).

Cross-checks against:
- Paper text: K=36 levels, dz_min=10m at top, total depth 4000m.
- Paper Fig C2: bottom layer thickness ~600m, surface-to-bottom centers
  range from ~5m down to ~3700m.
- NEMO source convention (vopikamm/DINO@v0.2.0 MY_SRC/zgr_lib.F90):
  formula evaluated at integer k=1..jpk for interfaces with K_formula =
  n_levels + 1.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.ocean.experiments.dino import (
    DINOConfig,
    create_dino_z_star,
)


def test_z_star_has_36_layers_and_37_interfaces():
    z = create_dino_z_star()
    assert z.n_levels == 36
    assert z.dz_ref.shape == (36,)
    assert z.z_full_ref.shape == (36,)
    assert z.z_half_ref.shape == (37,)
    assert z.dz_half_ref.shape == (35,)


def test_surface_interface_is_zero():
    z = create_dino_z_star()
    assert float(z.z_half_ref[0]) == 0.0


def test_bottom_interface_is_minus_H():
    z = create_dino_z_star()
    assert float(z.z_half_ref[-1]) == pytest.approx(-4000.0)


def test_total_depth_equals_H_deep():
    z = create_dino_z_star()
    assert float(jnp.sum(z.dz_ref)) == pytest.approx(4000.0, abs=1e-6)


def test_top_layer_thickness_close_to_dz_min():
    z = create_dino_z_star()
    # The dz_min constraint in eq C3 is on the *derivative* dz/dk at k=1
    # (the surface interface), not the integrated top-layer thickness.
    # The integrated thickness dz_ref[0] = z(k=2) - z(k=1) is slightly
    # larger than dz_min because dz/dk increases through the top layer.
    # For DINO defaults this gives ~10.12 m, well within 1% of 10 m.
    assert float(z.dz_ref[0]) == pytest.approx(10.0, rel=0.02)


def test_layer_thickness_grows_monotonically_with_depth():
    z = create_dino_z_star()
    dz = z.dz_ref
    # Strictly increasing surface-to-bottom (smooth tanh stretching).
    diffs = dz[1:] - dz[:-1]
    assert bool(jnp.all(diffs > 0))


def test_bottom_layer_thickness_in_expected_range():
    """Bottom-layer thickness from eq C3 with default DINO parameters
    (K=36, kth=35, acr=10.5, dz_min=10, H=4000) is ≈ 453 m. The 600 m
    figure on the right axis of Fig C2 is the axis upper limit, not the
    value at the deepest layer.
    """
    z = create_dino_z_star()
    dz_bottom = float(z.dz_ref[-1])
    assert 400.0 < dz_bottom < 500.0


def test_full_levels_strictly_below_surface_above_bottom():
    z = create_dino_z_star()
    z_full = z.z_full_ref
    assert float(z_full[0]) < 0.0
    assert float(z_full[-1]) > -4000.0
    # Strictly monotonic (deeper as k increases)
    assert bool(jnp.all(z_full[1:] < z_full[:-1]))


def test_half_levels_strictly_decreasing():
    z = create_dino_z_star()
    z_half = z.z_half_ref
    diffs = z_half[1:] - z_half[:-1]
    assert bool(jnp.all(diffs < 0))


def test_full_level_consistency_with_interfaces():
    z = create_dino_z_star()
    # Cell centers should be midpoints of adjacent interfaces
    midpoints = 0.5 * (z.z_half_ref[:-1] + z.z_half_ref[1:])
    assert bool(jnp.allclose(z.z_full_ref, midpoints))


def test_dz_consistency_with_interfaces():
    z = create_dino_z_star()
    dz_from_interfaces = z.z_half_ref[:-1] - z.z_half_ref[1:]
    assert bool(jnp.allclose(z.dz_ref, dz_from_interfaces))


def test_stretching_coefficients_satisfy_constraints():
    """Verify the (a₀, a₁, a₂) constraints algebraically (via the
    promoted general helper in legoesm.ocean.vertical)."""
    from legoesm.ocean.vertical import (
        _levy_depth_at_k, _levy_stretching_coefficients,
    )
    cfg = DINOConfig()
    K_formula = cfg.n_levels + 1
    a0, a1, a2 = _levy_stretching_coefficients(
        K_formula=K_formula, H=cfg.H_deep, dz_min=cfg.dz_min,
        k_th=float(cfg.k_th), a_cr=cfg.a_cr,
    )
    # Constraint 1: z(k=1) = 0 (surface)
    z_at_1 = _levy_depth_at_k(1.0, a0, a1, a2, float(cfg.k_th), cfg.a_cr)
    assert z_at_1 == pytest.approx(0.0, abs=1e-9)

    # Constraint 2: z(k=K_formula) = H (bottom)
    z_at_K = _levy_depth_at_k(
        float(K_formula), a0, a1, a2, float(cfg.k_th), cfg.a_cr,
    )
    assert z_at_K == pytest.approx(cfg.H_deep, abs=1e-6)

    # Constraint 3: dz/dk at k=1 = dz_min (derivative)
    import math
    deriv_at_1 = a1 + a0 * math.tanh((1 - cfg.k_th) / cfg.a_cr)
    assert deriv_at_1 == pytest.approx(cfg.dz_min, abs=1e-9)


def test_overrideable_n_levels():
    """Sanity: changing n_levels gives a self-consistent grid."""
    cfg = DINOConfig(n_levels=20, k_th=19, a_cr=8.0)
    z = create_dino_z_star(cfg)
    assert z.n_levels == 20
    assert float(jnp.sum(z.dz_ref)) == pytest.approx(4000.0, abs=1e-6)
    assert float(z.z_half_ref[0]) == 0.0
    assert float(z.z_half_ref[-1]) == pytest.approx(-4000.0)


def test_overrideable_H_deep():
    cfg = DINOConfig(H_deep=3500.0)
    z = create_dino_z_star(cfg)
    assert z.H_max == pytest.approx(3500.0)
    assert float(jnp.sum(z.dz_ref)) == pytest.approx(3500.0, abs=1e-6)
