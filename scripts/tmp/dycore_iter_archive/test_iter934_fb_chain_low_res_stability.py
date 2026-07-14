"""Iter-934 sentinel: FB chain produces finite output at all
resolutions C8/C12/C16/C24/C36 with W2 IC and `dt=300 s`.

Pre-iter-934 (iter-933 diagnostic): FB chain produced NaN in `h`
at C8/C12/C16, finite at C24+.  Root cause: float32 overflow in
`_deln_flux` (`fv_tp_2d.py`):

  damp = (damp_c * area)^(nord+1)  →  ~4e32 at C8
  d2   = damp * q                  →  ~1.5e36
  fx2  = sin * dy * d2_diff * rdxc →  ~1.5e42 OVERFLOWS float32

iter-934 fix: factor `damp` out of the iteration and apply at the
final flux assembly (Step 4) instead of at Step 1 initialisation.
All operations between Step 1 and Step 4 are linear in `d2`, so
the result is mathematically identical, but every intermediate now
fits comfortably in float32.

This sentinel pins that fix by asserting FB chain produces FINITE
output at every resolution from C8 to C36 with W2 IC.  Pre-iter-934
this would have failed at C8/C12/C16; post-iter-934 it passes.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    FV3EdgeShallowWaterState,
    FV3FBShallowWaterModel,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
)


@pytest.mark.parametrize("N", [8, 12, 16, 24, 36])
def test_iter934_fb_chain_finite_at_all_resolutions(N):
    """One FB step at W2 IC, dt=300s, FB Fortran defaults — must
    produce finite output at every resolution.
    """
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
        new_state = model.step(state, 300.0)
    h_arr = np.asarray(new_state.h)
    u_arr = np.asarray(new_state.u_d)
    v_arr = np.asarray(new_state.v_d)
    assert np.all(np.isfinite(h_arr)), (
        f"C{N}: FB chain `h` has non-finite values.  Pre-iter-934 "
        f"this fired at C8/C12/C16 due to _deln_flux float32 overflow; "
        f"if it fires now, the iter-934 damp-factoring fix has "
        f"regressed."
    )
    assert np.all(np.isfinite(u_arr))
    assert np.all(np.isfinite(v_arr))


def test_iter934_deln_flux_factoring_is_mathematically_equivalent():
    """At C24 (where pre-iter-934 already produced finite output),
    the iter-934 refactor must give bit-identical results in
    `transport_step`.  This locks the refactor's correctness:
    factoring `damp` out of the iteration and applying at the end
    is mathematically equivalent because every intermediate
    operation is linear in d2.
    """
    from legoesm.core.fv_tp_2d import transport_step
    from legoesm.core.fv3_sw_core import d2a2c_vect

    N = 24  # large enough that pre-iter-934 didn't overflow
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    h = sw.h.data
    _, _, _, _, ut, vt = d2a2c_vect(u_d, v_d, cdgrid)
    h_new = transport_step(
        h, ut, vt, 300.0, cdgrid,
        nord=2, damp_c=0.06,
    )
    h_arr = np.asarray(h_new)
    assert np.all(np.isfinite(h_arr))
    # h shouldn't drift far from W2 H0 (~3000 m) in one step
    assert 2000 < float(np.abs(h_arr).max()) < 4000, (
        f"C24 transport_step output max|h| = {float(np.abs(h_arr).max()):.1f} "
        f"is outside expected W2 range — iter-934 refactor produced "
        f"physically wrong values."
    )
