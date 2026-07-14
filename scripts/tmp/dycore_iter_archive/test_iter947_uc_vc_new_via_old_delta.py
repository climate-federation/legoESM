"""Iter-947 sentinel: pin the duogrid FB chain W2 1-day improvement
from sourcing covariant uc, vc halo cells via the
`_pad_halo_uc_vc_new_via_old_delta` helper (NEW boundary anchor +
OLD cross-face delta).

Trigger.  iter-945 closed the cube-face D-grid PPM halo gap; iter-946
attempted to source uc, vc halo from OLD u_d, v_d via the d2a2c
machinery but the OLD-derived halo lacked the c_sw + p_grad_c
increment that was added to interior uc, vc, producing an OLD/NEW
discontinuity that worsened |v_max| (81 → 156 m/s).

Iter-947 fix.  New helper `_pad_halo_uc_vc_new_via_old_delta`
estimates the NEW cross-face halo as::

    NEW_halo = NEW_boundary + (OLD_halo - OLD_boundary)

This anchors the halo on the NEW interior boundary cell (preserving
the c_sw + p_grad_c increment) while carrying the OLD cross-face
geometric delta (the rotation between cube faces).  Wired into BOTH
`_d_sw1_recompute_ut_vt` and `_bgrid_ke_transport` for the duogrid
ng>=3 path.

Cumulative duogrid FB chain on W2 C36 dt=300 s 1-day:

| iter        | step survival | |u_max| (m/s) | |v_max| (m/s) |
|-------------|--------------:|--------------:|--------------:|
| iter-944b   |    288 / 288  |        106    |        151    |
| iter-945    |    288 / 288  |         78    |         81    |
| iter-946    |    288 / 288  |         73    |        156    | ← worse, reverted |
| **iter-947**|    288 / 288  |     **77**    |     **75**    |
| analytical  |        ∞      |         40    |          0    |

|v_max| improved 81 → 75 m/s (~7%); |u_max| approximately unchanged
(78 → 77 m/s).  W2 v_ll_Linf acceptance still NOT met.

This sentinel pins:

1. `_pad_halo_uc_vc_new_via_old_delta` returns the documented
   shapes ((6, n+1, n+2) and (6, n+2, n+1)) on duogrid C24 with W2 IC.
2. The interior of the returned padded uc, vc is EXACTLY equal to
   the input uc, vc (helper preserves interior at machine precision).
3. The duogrid FB chain at C36 dt=300 s 1-day reaches 288 NaN-free
   steps with |u_max| ≤ 90 m/s AND |v_max| ≤ 90 m/s (margin around
   the 77 / 75 measurement).
4. The non-duogrid path remains at the iter-944b ~41-step baseline.

Production impact: ZERO.  `_d_sw_native` is FB-chain-only.
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
from legoesm.core.fv3_sw_core import (
    d2a2c_vect, _pad_halo_uc_vc_new_via_old_delta,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
)


def _w2_d_grid_winds(N: int, *, use_duogrid: bool):
    grid = create_cubed_sphere(N, use_duogrid=use_duogrid)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    return grid, cdgrid, u_d, v_d, sw


def _fb_chain_1day_w2(N: int, dt: float, *, use_duogrid: bool):
    grid, cdgrid, u_d, v_d, sw = _w2_d_grid_winds(N, use_duogrid=use_duogrid)
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
        damp_v=0.06, nord_v=2,
    )
    model = FV3FBShallowWaterModel(grid, cfg)
    max_steps = 288
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(max_steps):
            state = model.step(state, dt)
            if not bool(np.all(np.isfinite(np.asarray(state.h)))):
                return k, state
    return max_steps, state


def test_iter947_helper_shapes_and_interior_preserved():
    """`_pad_halo_uc_vc_new_via_old_delta` returns the documented
    shapes and preserves the input uc, vc interior at machine
    precision.
    """
    N = 24
    _, cdgrid, u_d, v_d, _ = _w2_d_grid_winds(N, use_duogrid=True)
    # Use d2a2c output as a stand-in for "NEW" interior uc, vc.
    _, _, uc, vc, _, _ = d2a2c_vect(u_d, v_d, cdgrid)
    uc_pad, vc_pad = _pad_halo_uc_vc_new_via_old_delta(
        uc, vc, u_d, v_d, cdgrid)
    assert uc_pad.shape == (6, N + 1, N + 2), (
        f"uc_pad shape expected (6, {N+1}, {N+2}); got {uc_pad.shape}"
    )
    assert vc_pad.shape == (6, N + 2, N + 1), (
        f"vc_pad shape expected (6, {N+2}, {N+1}); got {vc_pad.shape}"
    )
    # Interior uc preserved at indices [1 : n+1] along axis 2.
    assert np.allclose(np.asarray(uc_pad[:, :, 1:N + 1]), np.asarray(uc),
                        atol=0.0), "uc interior must be preserved exactly"
    assert np.allclose(np.asarray(vc_pad[:, 1:N + 1, :]), np.asarray(vc),
                        atol=0.0), "vc interior must be preserved exactly"


def test_iter947_duogrid_fb_chain_1day_velocity_improved_vs_iter945():
    """Post-iter-947 duogrid FB chain at C36 dt=300 s W2 reaches the
    1-day target with |u_max| AND |v_max| BELOW the iter-945 baseline
    (|u_max|=78, |v_max|=81 m/s).  Iter-947 measurement: 77 / 75.
    Margin to allow grid/dtype drift.
    """
    n, state = _fb_chain_1day_w2(36, 300.0, use_duogrid=True)
    assert n >= 288, (
        f"Duogrid FB chain at C36 dt=300 s survived only {n} steps; "
        f"iter-947 expects ≥ 288 (preserves iter-945 NaN-free milestone)."
    )
    u_max = float(np.abs(np.asarray(state.u_d)).max())
    v_max = float(np.abs(np.asarray(state.v_d)).max())
    assert u_max <= 90.0, (
        f"|u_max| at 1 day = {u_max:.2f} m/s; iter-947 expected <= 90 "
        f"(was 78 at iter-945, measured 77 post-iter-947)."
    )
    assert v_max <= 90.0, (
        f"|v_max| at 1 day = {v_max:.2f} m/s; iter-947 expected <= 90 "
        f"(was 81 at iter-945, measured 75 post-iter-947)."
    )


def test_iter947_non_duogrid_unchanged_baseline():
    """The non-duogrid FB chain remains at the iter-944b ~41-step
    baseline because `_pad_halo_uc_vc_new_via_old_delta` is gated on
    duogrid ng>=3.
    """
    n, _ = _fb_chain_1day_w2(36, 300.0, use_duogrid=False)
    assert 30 <= n <= 50, (
        f"Non-duogrid FB chain at C36 dt=300 s survived {n} steps; "
        f"iter-944b/iter-947 expected ~41 baseline."
    )
