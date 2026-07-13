"""Iter-944 sentinel (REWRITTEN at iter-945): pin the FB chain 1-day
NaN-free milestone, now correctly gated on the duogrid path.

iter-944b (Fortran-fidelity audit) reverted the Python-only
`synchronize_cgrid_fluxes` syncs of (ut, vt) and (fx_vort, fy_vort)
that iter-944 had added — Fortran's `mpp_get_boundary` blocks for
those quantities are explicitly commented out (dyn_core.F90:1124-1207
and the standalone (ut, vt) sync is not present in Fortran at all).
Post-iter-944b the **non-duogrid** FB chain reverts to the unsynced
~41-step baseline.  The 1-day NaN-free milestone is preserved on the
**duogrid** path because the BGRID_NE corner sync (gated on duogrid)
plus the iter-945 D-grid PPM cross-face halo (`_pad_halo_dgrid_for_ppm`)
keep the structural growth mode at bay.

Cumulative FB chain step survival on W2 C36 dt=300 s (target 1-day =
288 steps), default damping (d4_bg=0.16, nord=1, damp_v=0.06, nord_v=2):

| iter           | non-duogrid | duogrid |
|----------------|-------------|---------|
| baseline       |     41      |   —     |
| iter-941..944  |    288      |   —     |
| iter-944b      |     41      |  288    |
| iter-945       |     41      |  288    | (cross-face PPM halo) |

Caveat: 288-step "1-day NaN-free" is NUMERICAL stability, not W2
acceptance.  Iter-944b duogrid baseline: |u_max|=106 m/s, |v_max|=151
m/s.  Iter-945 duogrid: |u_max|≈78 m/s, |v_max|≈81 m/s — closer to
the analytical (40 m/s, 0 m/s) but W2 v_ll_Linf acceptance still
unmet.  Further fidelity work continues in iter-946+.

Production impact: ZERO.  `_d_sw_native` is FB-chain-only; production
`fv3_sw_tendencies` (`FV3EdgeShallowWaterModel` default) does not
invoke this code path.  Production W2 sentinel (iter-921) and
rest-state sentinel (iter-925) bit-identical post-iter-944.
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
                       *, use_duogrid: bool = True) -> int:
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


def test_iter944_fb_chain_c36_reaches_1_day_nan_free_duogrid():
    """Post-iter-944b duogrid FB chain at C36 dt=300 s must reach the
    1-day target (288 steps) NaN-free.  iter-944b reverted iter-944's
    Python-only `synchronize_cgrid_fluxes` syncs and gated the BGRID_NE
    sync on duogrid; the 1-day NaN-free milestone is preserved on
    duogrid.  iter-945's D-grid PPM cross-face halo
    (`_pad_halo_dgrid_for_ppm`) keeps the milestone while improving
    W2 fidelity.

    A regression below 288 on duogrid signals either a removal of
    the BGRID_NE corner sync, a regression in the iter-945 PPM halo,
    or another structural-growth NaN class.
    """
    n = _fb_chain_survival(36, 300.0, max_steps=288, use_duogrid=True)
    assert n >= 288, (
        f"Duogrid FB chain at C36 dt=300 s survived only {n} steps; "
        f"iter-944b/iter-945 expected >= 288 (1-day NaN-free target)."
    )


def test_iter944_fb_chain_c36_non_duogrid_baseline_41():
    """Post-iter-944b non-duogrid FB chain reverts to the unsynced
    ~41-step baseline (Fortran-faithful with no `mpp_get_boundary`
    syncs firing).  A value above ~50 likely means a Python-only
    sync hack has been re-introduced; below ~30 likely means a new
    structural regression.
    """
    n = _fb_chain_survival(36, 300.0, max_steps=80, use_duogrid=False)
    assert 30 <= n <= 50, (
        f"Non-duogrid FB chain at C36 dt=300 s survived {n} steps; "
        f"iter-944b expected ~41 (no syncs fire)."
    )
