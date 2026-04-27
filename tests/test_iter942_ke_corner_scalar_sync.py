"""Iter-942 sentinel: pin the FB chain step-survival improvement
from adding `ke_corner` scalar sync after the d_sw5 KE-add.

Cumulative FB chain step survival on W2 C36 dt=300 s, default
damping (target 1-day = 288 steps):

| iter      | survived | improvement |
|-----------|----------|-------------|
| baseline  | 41       | —           |
| iter-941  | 63       | +54 %       |
| iter-942  | 209      | +410 %      |

iter-942 added a scalar `synchronize_corner_scalar(ke_corner, n)`
call AFTER the d_sw5 corner-divergence KE-add and BEFORE the d_sw6
KE-gradient (`fv3_sw_core.py:_d_sw_native` step 5 → step 6).  This
makes the four faces meeting at each cube vertex agree on the
final `ke_corner` value (combined initial KE + d_sw5 ke_damping)
that feeds the gradient stencil.

Without the sync, each face computes its own ke_corner at the
cube vertex.  The d_sw6 KE-gradient `(ke_corner[i] - ke_corner[i+1])
/ dx_u` reads adjacent corners, so cube-vertex inconsistency
propagates into the wind update via `u_d_new = u_d + ke_diff / dx_u`.

Still doesn't reach 1-day stability — there's at least one more
contributor to the FB chain growth in the d_sw6 path or downstream.
But iter-942 is a major partial fix (5× the iter-940 baseline).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import warnings

import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    FV3EdgeShallowWaterState,
    FV3FBShallowWaterModel,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
)


def _fb_chain_survival(N: int, dt: float, max_steps: int) -> int:
    grid = create_cubed_sphere(N)
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
        d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
        damp_v=0.06, nord_v=2,
    )
    model = FV3FBShallowWaterModel(grid, cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(max_steps):
            state = model.step(state, dt)
            if not bool(np.all(np.isfinite(np.asarray(state.h)))):
                return k
    return max_steps


def test_iter942_fb_chain_c36_survives_at_least_180_steps():
    """Post-iter-942 FB chain at C36 dt=300 s must survive >= 180 steps
    (iter-942 baseline measurement: 209).  Tolerance ±29 steps to
    allow numerical drift; firing this gate signals the iter-942
    `ke_corner` scalar sync has been disturbed or another structural
    bug has emerged.

    Pre-iter-942 baseline: 63 steps (iter-941 alone) and 41 steps
    (iter-940 baseline).
    """
    n = _fb_chain_survival(36, 300.0, max_steps=220)
    assert n >= 180, (
        f"FB chain at C36 dt=300 s survived only {n} steps; iter-942 "
        f"baseline expected >= 180 steps (post-iter-942 measurement: 209)."
    )
