"""Iter-941 sentinel (REWRITTEN at iter-944b): pin the Fortran-faithful
BGRID_NE corner sync gate decision in `_bgrid_ke_transport`.

iter-941 (original): made `synchronize_bgrid_ne_corner_geo` UNCONDITIONAL,
based on the erroneous belief that Fortran's
`mpp_get_boundary(... gridtype=BGRID_NE)` "fires regardless of duogrid".

iter-944b (Fortran-fidelity audit): the reference source
`atmos_cubed_sphere-symmetryclean/model/dyn_core.F90:968-1011` shows the
call is explicitly inside an `if (duogrid)` block.  Non-duogrid Fortran
runs do NOT fire BGRID_NE here; they rely on the standard halo from
`mpp_update_domains(uc, vc, gridtype=CGRID_NE)` (dyn_core.F90:633/689/
702/1291) propagating through the (ubb, vbbtemp) construction in d_sw3.

This sentinel pins:

1. **Duogrid path**: BGRID_NE sync fires; FB chain at C36 dt=300 s
   reaches the 1-day target (288 steps NaN-free).
2. **Non-duogrid path**: BGRID_NE sync does NOT fire; FB chain at C36
   dt=300 s reverts to the unsynced ~41-step baseline (Fortran-faithful
   behaviour without the standard mpp halo, which our Python halo
   approximation does not provide for the in-d_sw3 corner stencils).

A regression that flips either gate (uncondition the sync, or skip it
under duogrid) will fire the corresponding sub-test.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import warnings

import jax.numpy as jnp
import numpy as np

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


def _fb_chain_survival(N: int, dt: float, max_steps: int,
                       *, use_duogrid: bool) -> int:
    grid = create_cubed_sphere(N, use_duogrid=use_duogrid)
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


def test_iter941_duogrid_bgrid_ne_sync_fires_fb_chain_1_day():
    """Duogrid path with Fortran-faithful BGRID_NE sync: FB chain at
    C36 dt=300 s must reach 288 steps (1-day NaN-free).

    iter-944b measurement: 288/288 finite, |u_max|=106 m/s,
    |v_max|=151 m/s.  This pins the Fortran-faithful duogrid path.
    """
    n = _fb_chain_survival(36, 300.0, max_steps=288, use_duogrid=True)
    assert n >= 288, (
        f"Duogrid FB chain at C36 dt=300 s survived only {n} steps; "
        f"iter-944b baseline expected 288 (1-day NaN-free).  Either "
        f"the BGRID_NE sync gate has been disabled or another "
        f"structural regression has appeared."
    )


def test_iter941_non_duogrid_bgrid_ne_sync_skipped_baseline_41():
    """Non-duogrid path: the BGRID_NE sync is gated OFF (Fortran-
    faithful), so FB chain at C36 dt=300 s NaN's at the iter-940
    baseline of ~41 steps.

    A regression that re-enables the sync unconditionally would lift
    this far above 41 (iter-941 original measured 63 with the sync;
    iter-942 added another sync and got 209).  Pinning the unsynced
    baseline guards against accidentally re-introducing Python-only
    smoothing that violates the Fortran-fidelity contract.
    """
    n = _fb_chain_survival(36, 300.0, max_steps=80, use_duogrid=False)
    assert 30 <= n <= 50, (
        f"Non-duogrid FB chain at C36 dt=300 s survived {n} steps; "
        f"iter-944b expected ~41 (no syncs fire — Fortran-faithful).  "
        f"A value above 50 likely means the BGRID_NE / CGRID_NE / "
        f"ke_corner syncs were re-enabled unconditionally.  A value "
        f"below 30 likely means another structural regression."
    )


def test_iter941_fb_chain_low_res_step1_finite_both_grids():
    """FB chain step-1 finiteness invariant (covered by iter-934
    sentinel) holds at C8/C12/C16/C24/C36 on BOTH duogrid and
    non-duogrid paths.  This is a smoke test that neither gate
    introduces a step-1 NaN at low resolution.
    """
    for use_dg in (False, True):
        for N, dt in [(8, 1350.0), (12, 900.0), (16, 675.0),
                       (24, 450.0), (36, 300.0)]:
            n = _fb_chain_survival(N, dt, max_steps=1, use_duogrid=use_dg)
            assert n >= 1, (
                f"FB chain at C{N} dt={dt} use_duogrid={use_dg} "
                f"produced NaN at step 1 post-iter-944b."
            )
