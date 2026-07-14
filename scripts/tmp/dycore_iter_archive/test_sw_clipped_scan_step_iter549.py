"""FV3_3D iter 549: SW make_clipped_scan_step validation.

iter-544 NH scan_step + iter-547 PE scan_step → iter-549
adds SW for full dycore coverage (NH/PE/SW).

Tests
-----

1. ``test_sw_scan_step_runs`` — scan_step on SW dycore
   produces finite output.
2. ``test_sw_scan_step_matches_loop`` — bit-for-bit match
   with Python loop.
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
from legoesm.grids.halo import make_clipped_step, make_clipped_scan_step
try:
    from tests.atmosphere.shallow_water.test_cases.williamson import (
        williamson_test2,
    )
    HAVE_W2 = True
except Exception:
    HAVE_W2 = False


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def _build_sw_state(n=12):
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data.astype(jnp.float64),
        u_d=u_d.astype(jnp.float64),
        v_d=v_d.astype(jnp.float64),
        h_s=sw.h_s.data.astype(jnp.float64),
    )
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.04,
        nord_v=2,
        apply_fortran_xppm_boundary=True,
    )
    return grid, cdgrid, cfg, state


@pytest.mark.skipif(not HAVE_W2, reason="williamson_test2 not importable")
def test_sw_scan_step_runs():
    grid, cdgrid, cfg, state = _build_sw_state(n=12)
    m = FV3EdgeShallowWaterModel(grid, cfg)
    scan_step = make_clipped_scan_step(
        m, state, dt=300.0, n_steps=5, slack=0.5,
    )
    final = scan_step(state)
    assert jnp.all(jnp.isfinite(final.h))
    assert jnp.all(jnp.isfinite(final.u_d))
    assert jnp.all(jnp.isfinite(final.v_d))


@pytest.mark.skipif(not HAVE_W2, reason="williamson_test2 not importable")
def test_sw_scan_step_matches_loop():
    # Python loop
    grid_loop, cdgrid_loop, cfg_loop, state_loop = _build_sw_state(n=12)
    m_loop = FV3EdgeShallowWaterModel(grid_loop, cfg_loop)
    step_loop = make_clipped_step(m_loop, state_loop, dt=300.0, slack=0.5)
    s_loop = state_loop
    for _ in range(3):
        s_loop = step_loop(s_loop, 300.0)

    # scan
    grid_scan, cdgrid_scan, cfg_scan, state_scan = _build_sw_state(n=12)
    m_scan = FV3EdgeShallowWaterModel(grid_scan, cfg_scan)
    scan_step = make_clipped_scan_step(
        m_scan, state_scan, dt=300.0, n_steps=3, slack=0.5,
    )
    s_scan = scan_step(state_scan)

    # Compare h, u_d, v_d
    for fld in ("h", "u_d", "v_d"):
        a = getattr(s_loop, fld)
        b = getattr(s_scan, fld)
        diff = float(jnp.abs(a - b).max())
        assert diff < 1e-6, (
            f"SW scan vs loop diverges on {fld}: max|diff|={diff}"
        )
