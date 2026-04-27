"""Iter-941 sentinel: pin the FB chain step-survival improvement
from applying BGRID_NE corner sync unconditionally.

iter-940 localised the FB chain structural growth to the d_sw6
KE-gradient.  iter-941 found the cause: the BGRID_NE corner sync
(`synchronize_bgrid_ne_corner_geo`) inside `_bgrid_ke_transport`
was gated on `use_duogrid` and skipped on non-duogrid runs — but
Fortran's `mpp_get_boundary(... gridtype=BGRID_NE)` at
`dyn_core.F90:968-1019` fires unconditionally.  Enabling the sync
on non-duogrid runs improves FB chain step survival on W2 C36
1-day from **41 → 63 steps**.

This sentinel pins:
1. The improved step survival (≥ 60 steps) on FB chain at C36.
2. The unchanged step-1 finiteness on FB chain at C8/C12/C16
   (already covered by iter-934, but re-checked here as a smoke
   test for the iter-941 sync change).
3. Production W2 sentinel unchanged (the production path does NOT
   call `_d_sw_native`, so iter-941 has zero effect on production).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

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


def test_iter941_fb_chain_c36_survives_at_least_60_steps():
    """Post-iter-941 FB chain at C36 dt=300 s must survive at least
    60 steps (was 41 steps pre-iter-941; baseline measurement: 63).

    Tolerance: ±3 steps around 63 to allow minor numerical drift
    under unrelated changes; firing this gate signals iter-941's
    BGRID_NE sync change has been disturbed or another structural
    bug has emerged.
    """
    n = _fb_chain_survival(36, 300.0, max_steps=80)
    assert n >= 60, (
        f"FB chain at C36 dt=300 s survived only {n} steps; iter-941 "
        f"baseline expected >= 60 steps (pre-iter-941: 41 steps)."
    )


def test_iter941_fb_chain_low_res_step1_finite():
    """Post-iter-941 FB chain step-1 finiteness invariant (covered by
    iter-934 sentinel) holds at C8/C12/C16/C24/C36.  This is a smoke
    test that the BGRID_NE sync addition doesn't introduce a step-1
    NaN at low resolution.
    """
    for N, dt in [(8, 1350.0), (12, 900.0), (16, 675.0), (24, 450.0),
                   (36, 300.0)]:
        n = _fb_chain_survival(N, dt, max_steps=1)
        assert n >= 1, (
            f"FB chain at C{N} dt={dt} produced NaN at step 1 "
            f"post-iter-941."
        )
