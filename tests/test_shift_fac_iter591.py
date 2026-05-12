"""FV3_3D iter 591: shift_fac longitude shift.

Faithful port of FV3 fv_grid_tools.F90:662-663: when not using
Schmidt/cube_transform and shift_fac > 1e-4, shift lon by -π/shift_fac.
FV3 default shift_fac=18 → shift west by 10°.

Tests
-----

1. ``test_shift_fac_default_zero_is_no_op``.
2. ``test_shift_fac_18_shifts_10deg_west``.
3. ``test_shift_fac_gated_by_schmidt`` — Schmidt active → no shift.
4. ``test_shift_fac_wraps_negative_to_positive``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere


def test_shift_fac_default_zero_is_no_op():
    """shift_fac=0 → no shift, matches pre-iter-591 grid."""
    n = 8
    g_default = create_cubed_sphere(n)
    g_explicit = create_cubed_sphere(n, shift_fac=0.0)
    diff = float(jnp.abs(g_default.lon - g_explicit.lon).max())
    assert diff < 1e-10, f"shift_fac=0 should be no-op; diff={diff}"


def test_shift_fac_18_shifts_10deg_west():
    """shift_fac=18 → lon shifted by -π/18 (10° west)."""
    n = 8
    g_base = create_cubed_sphere(n)
    g_shifted = create_cubed_sphere(n, shift_fac=18.0)
    lon_diff = np.asarray(g_shifted.lon) - np.asarray(g_base.lon)
    # Diff should be -π/18 mod 2π
    expected_shift = -jnp.pi / 18
    # Account for 2π wrap: shifted - base ∈ {-π/18, -π/18+2π}
    lon_diff_wrapped = np.where(
        lon_diff > jnp.pi, lon_diff - 2 * jnp.pi,
        np.where(lon_diff < -jnp.pi, lon_diff + 2 * jnp.pi, lon_diff),
    )
    assert np.allclose(lon_diff_wrapped, expected_shift, atol=1e-6), (
        f"shift_fac=18 should shift lon by -π/18; "
        f"actual diff range: [{lon_diff_wrapped.min():.4f}, "
        f"{lon_diff_wrapped.max():.4f}]"
    )


def test_shift_fac_gated_by_schmidt():
    """shift_fac is ignored when Schmidt transformation is active."""
    n = 8
    g_with_schmidt = create_cubed_sphere(
        n,
        stretch_fac=2.0,
        target_lon=0.5,
        target_lat=0.3,
        shift_fac=18.0,  # should be ignored
    )
    g_without_shift = create_cubed_sphere(
        n,
        stretch_fac=2.0,
        target_lon=0.5,
        target_lat=0.3,
        shift_fac=0.0,
    )
    diff = float(jnp.abs(g_with_schmidt.lon - g_without_shift.lon).max())
    assert diff < 1e-10, (
        f"shift_fac should be no-op under Schmidt; diff={diff}"
    )


def test_shift_fac_wraps_negative_to_positive():
    """lon < 0 after shift gets wrapped to lon + 2π."""
    n = 8
    g = create_cubed_sphere(n, shift_fac=18.0)
    # All lon values must be ≥ 0
    assert float(g.lon.min()) >= 0.0, f"lon should be ≥ 0 after wrap"
    # All lon values must be < 2π
    assert float(g.lon.max()) < 2 * jnp.pi + 1e-10
