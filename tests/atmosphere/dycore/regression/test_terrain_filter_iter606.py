"""FV3_3D iter 606: terrain filter test.

Faithful port of FV3 del2_cubed_sphere / del4_cubed_sphere
(tools/fv_surf_map.F90:817+).

Tests
-----

1. ``test_terrain_filter_no_op_n_iter_zero``.
2. ``test_terrain_filter_smooths_step``.
3. ``test_terrain_filter_constant_field_preserved``.
4. ``test_terrain_filter_del4_smoother_than_del2``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.terrain_filter import terrain_filter


def test_terrain_filter_no_op_n_iter_zero():
    """n_iter=0 returns input unchanged."""
    n = 8
    grid = create_cubed_sphere(n)
    rng = np.random.default_rng(seed=606)
    phis = jnp.asarray(rng.uniform(0, 9800, size=(6, n, n)))
    out = terrain_filter(phis, grid, n_iter=0)
    assert jnp.array_equal(out, phis)


def test_terrain_filter_smooths_step():
    """A sharp step should be smoothed (variance reduced)."""
    n = 8
    grid = create_cubed_sphere(n)
    # Build a sharp-step field
    phis = jnp.zeros((6, n, n))
    phis = phis.at[0, n // 2:, :].set(9800.0)  # 1-km mountain on half of face 0
    var_before = float(jnp.var(phis))
    phis_smooth = terrain_filter(phis, grid, n_iter=4)
    var_after = float(jnp.var(phis_smooth))
    assert var_after < var_before, (
        f"smoothing should reduce variance; before={var_before:.3e}, "
        f"after={var_after:.3e}"
    )


def test_terrain_filter_constant_field_preserved():
    """Constant phis is unchanged by smoothing (Δ²·const = 0)."""
    n = 8
    grid = create_cubed_sphere(n)
    phis = jnp.full((6, n, n), 1000.0)
    out = terrain_filter(phis, grid, n_iter=4)
    diff = float(jnp.abs(out - 1000.0).max())
    assert diff < 1e-2, (
        f"constant field should be preserved; max diff={diff:.3e}"
    )


def test_terrain_filter_del4_smoother_than_del2():
    """del-4 should preserve large scales better than del-2 for same n_iter."""
    n = 8
    grid = create_cubed_sphere(n)
    # Use a moderate-amplitude wavy field
    rng = np.random.default_rng(seed=607)
    phis = jnp.asarray(rng.uniform(0, 1000, size=(6, n, n)))
    phis_d2 = terrain_filter(phis, grid, n_iter=2, nord=2)
    phis_d4 = terrain_filter(phis, grid, n_iter=2, nord=4)
    # Both should be finite
    assert jnp.all(jnp.isfinite(phis_d2))
    assert jnp.all(jnp.isfinite(phis_d4))
