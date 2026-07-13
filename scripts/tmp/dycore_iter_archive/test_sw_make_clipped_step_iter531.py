"""FV3_3D iter 531: does ``make_clipped_step`` work on SW
(FV3EdgeShallowWaterModel)?

iter-526 validates the helper on NH; iter-529 on PE.  SW has
its own ``step(state, dt)`` signature on
``FV3EdgeShallowWaterModel`` — generic enough that the helper
should "just work".

Tests
-----

1. ``test_sw_make_clipped_step_runs`` — basic invocation +
   short Williamson 2 step run.
2. ``test_sw_make_clipped_step_finite`` — h/u_d/v_d stay
   finite after 10 steps.
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
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
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
def test_sw_make_clipped_step_runs():
    grid, cdgrid, cfg, state = _build_sw_state(n=12)
    m = FV3EdgeShallowWaterModel(grid, cfg)
    step = make_clipped_step(m, state, dt=300.0, slack=0.5)
    assert callable(step)
    new_state = step(state, 300.0)
    assert new_state.h.shape == state.h.shape
    assert jnp.all(jnp.isfinite(new_state.h))


@pytest.mark.skipif(not HAVE_W2, reason="williamson_test2 not importable")
def test_sw_make_clipped_step_finite():
    grid, cdgrid, cfg, state = _build_sw_state(n=12)
    m = FV3EdgeShallowWaterModel(grid, cfg)
    step = make_clipped_step(m, state, dt=300.0, slack=0.5)
    s = state
    for _ in range(10):
        s = step(s, 300.0)
    assert jnp.all(jnp.isfinite(s.h))
    assert jnp.all(jnp.isfinite(s.u_d))
    assert jnp.all(jnp.isfinite(s.v_d))
