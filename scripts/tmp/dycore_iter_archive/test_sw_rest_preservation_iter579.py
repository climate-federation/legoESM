"""FV3_3D iter 579: SW rest preservation — mirror iter-572.

iter-572 showed NH from truly zero IC → exactly zero output.
Does SW dycore similarly preserve rest? (constant h, zero u/v)

Tests
-----

1. ``test_sw_constant_h_zero_uv_preserved``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import make_clipped_step


def test_sw_constant_h_zero_uv_preserved(capsys):
    """SW from rest (constant h, zero u/v, no terrain)."""
    n = 12
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    # Constant h = 1000m, zero winds, flat surface
    h = 1000.0 * jnp.ones((6, n, n), dtype=jnp.float64)
    u_d = jnp.zeros((6, n, n + 1), dtype=jnp.float64)
    v_d = jnp.zeros((6, n + 1, n), dtype=jnp.float64)
    h_s = jnp.zeros((6, n, n), dtype=jnp.float64)
    state = FV3EdgeShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=0.0,
        boundary_fix=True,
        damp_v=0.0,
        nord_v=2,
        apply_fortran_xppm_boundary=True,
    )
    m = FV3EdgeShallowWaterModel(grid, cfg)
    step = make_clipped_step(m, state, dt=300.0, slack=0.5)
    s = state
    for _ in range(10):
        s = step(s, 300.0)
    h_final = np.asarray(s.h)
    u_final = np.asarray(s.u_d)
    v_final = np.asarray(s.v_d)
    with capsys.disabled():
        print(
            f"\n[iter-579 SW rest preservation @ N=12, 10 steps]"
        )
        print(f"  h drift:    max|h - 1000| = {np.abs(h_final - 1000.0).max():.3e}")
        print(f"  u_d drift:  max|u_d| = {np.abs(u_final).max():.3e}")
        print(f"  v_d drift:  max|v_d| = {np.abs(v_final).max():.3e}")
    # Tolerances — SW dycore has internal projections that may
    # introduce tiny float roundoff (1e-13 level on rest state)
    assert np.abs(h_final - 1000.0).max() < 1e-6, (
        f"h drifted: {np.abs(h_final - 1000.0).max()}"
    )
    assert np.abs(u_final).max() < 1e-6, (
        f"u_d drifted: {np.abs(u_final).max()}"
    )
    assert np.abs(v_final).max() < 1e-6
