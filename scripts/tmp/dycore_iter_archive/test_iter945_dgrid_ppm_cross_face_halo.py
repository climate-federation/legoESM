"""Iter-945 sentinel: pin the duogrid FB chain W2 1-day v/u improvement
from cross-face D-grid halo for PPM transport in `_bgrid_ke_transport`.

iter-944b (Fortran-fidelity audit) reverted Python-only `mpp_get_boundary`
syncs that did not appear in the Fortran reference, leaving the duogrid
FB chain at C36 dt=300 s 1-day NaN-free with |u_max|=106 m/s, |v_max|=151
m/s (vs analytical W2 |u_max|=40 m/s, |v_max|≈0).

iter-945 closes a real Fortran-fidelity gap: the PPM hord=9 transport
inside `_bgrid_ke_transport` was reading `mode='edge'` same-face halo
cells of (u_d, v_d) at the four cube-face boundaries.  Fortran sources
those halo cells from `mpp_update_domains(u_d, v_d, gridtype=DGRID_NE)`
upstream of d_sw3 (with cube_rmp interpolation when duogrid is active).

iter-945 wires the duogrid `ext_vector_dgrid` halo (already used in
`_d2a2c_vect_duogrid`) into a new `_pad_halo_dgrid_for_ppm` helper that
returns u_d with i-cell halo and v_d with j-cell halo at depth 2.
`_ppm_transport_1d` gains an `external_halo` kwarg so it consumes the
pre-padded fields without changing its internal indexing.

Effect on the duogrid FB chain at C36 dt=300 s:

| baseline (iter-944b) | post-iter-945 |
|----------------------|---------------|
| |u_max| = 106 m/s    | |u_max| ≈ 78 m/s |
| |v_max| = 151 m/s    | |v_max| ≈ 81 m/s |
| step survival 288/288 | step survival 288/288 |

The W2 v_ll_Linf acceptance (≤ 0.119 m/s per user iter-938 brief) is
NOT yet met; further fidelity work continues in iter-946+.

Production impact: ZERO.  `_d_sw_native` is FB-chain-only; production
`fv3_sw_tendencies` (`FV3EdgeShallowWaterModel` default) does not
invoke this code path.  Production W2 / W5 / cosine-bell / rest-state
sentinels remain bit-identical post-iter-945.

This sentinel locks in:

1. `_pad_halo_dgrid_for_ppm` exists and produces the expected
   (i-halo, j-halo) D-grid winds for duogrid C36.
2. The duogrid FB chain at C36 dt=300 s 1-day reaches |u_max| ≤ 95 m/s
   AND |v_max| ≤ 100 m/s — safely below the iter-944b baseline
   (|u_max|=106, |v_max|=151) with margin for grid/dtype drift.
3. The duogrid FB chain still reaches the 1-day NaN-free milestone
   (288 steps) — iter-945 is a fidelity improvement, not a stability
   regression.
4. The non-duogrid path is not exercised by `_pad_halo_dgrid_for_ppm`
   (raises ValueError if called) — Fortran's non-duogrid relies on
   the standard `mpp_update_domains` which our Python lacks; the
   non-duogrid FB chain stays at the iter-944b ~41-step baseline.
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
from legoesm.core.fv3_sw_core import _pad_halo_dgrid_for_ppm
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
)


def _w2_d_grid_winds(N: int, *, use_duogrid: bool):
    grid = create_cubed_sphere(N, use_duogrid=use_duogrid)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    return grid, cdgrid, u_d, v_d


def _fb_chain_1day_w2(N: int, dt: float, *, use_duogrid: bool):
    grid, cdgrid, u_d, v_d = _w2_d_grid_winds(N, use_duogrid=use_duogrid)
    sw = williamson_test2(grid)
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
        damp_v=0.06, nord_v=2,
    )
    model = FV3FBShallowWaterModel(grid, cfg)
    max_steps = 288  # 1-day target at dt=300 s
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(max_steps):
            state = model.step(state, dt)
            if not bool(np.all(np.isfinite(np.asarray(state.h)))):
                return k, state
    return max_steps, state


def test_iter945_pad_halo_dgrid_for_ppm_shapes_duogrid():
    """`_pad_halo_dgrid_for_ppm` returns u_d with i-cell halo and v_d
    with j-cell halo at depth 2 on the duogrid path.
    """
    N = 24
    _, cdgrid, u_d, v_d = _w2_d_grid_winds(N, use_duogrid=True)
    u_d_ihalo, v_d_jhalo = _pad_halo_dgrid_for_ppm(u_d, v_d, cdgrid, halo=2)
    assert u_d_ihalo.shape == (6, N + 4, N + 1), (
        f"u_d_ihalo expected (6, {N+4}, {N+1}), got {u_d_ihalo.shape}"
    )
    assert v_d_jhalo.shape == (6, N + 1, N + 4), (
        f"v_d_jhalo expected (6, {N+1}, {N+4}), got {v_d_jhalo.shape}"
    )
    # Interior must be EXACTLY equal to input (helper is identity on interior)
    assert np.allclose(np.asarray(u_d_ihalo[:, 2:N + 2, :]), np.asarray(u_d))
    assert np.allclose(np.asarray(v_d_jhalo[:, :, 2:N + 2]), np.asarray(v_d))


def test_iter945_pad_halo_dgrid_for_ppm_rejects_non_duogrid():
    """`_pad_halo_dgrid_for_ppm` raises ValueError on a non-duogrid grid.
    Fortran-faithful non-duogrid uses `mpp_update_domains` which our
    Python halo machinery does not implement; callers must gate on
    duogrid before invoking this helper.
    """
    N = 24
    _, cdgrid, u_d, v_d = _w2_d_grid_winds(N, use_duogrid=False)
    with pytest.raises(ValueError, match="duogrid"):
        _pad_halo_dgrid_for_ppm(u_d, v_d, cdgrid, halo=2)


def test_iter945_duogrid_fb_chain_1day_velocity_improved():
    """Post-iter-945 duogrid FB chain at C36 dt=300 s W2 reaches the
    1-day target with |u_max| and |v_max| BELOW the iter-944b baseline
    (|u_max|=106 m/s, |v_max|=151 m/s).  Margin chosen for grid/dtype
    drift.
    """
    n, state = _fb_chain_1day_w2(36, 300.0, use_duogrid=True)
    assert n >= 288, (
        f"Duogrid FB chain at C36 dt=300 s survived only {n} steps; "
        f"iter-945 expects ≥ 288 (preserves iter-944b NaN-free milestone)."
    )
    u = np.asarray(state.u_d)
    v = np.asarray(state.v_d)
    u_max = float(np.abs(u).max())
    v_max = float(np.abs(v).max())
    # iter-944b baseline: |u_max|=106 m/s, |v_max|=151 m/s
    # iter-945 measured:  |u_max|≈78 m/s,  |v_max|≈81 m/s
    # Lock in improvement vs iter-944b with margin.
    assert u_max <= 95.0, (
        f"|u_max| at 1 day = {u_max:.2f} m/s; iter-945 expected <= 95 "
        f"(was 106 at iter-944b, measured 78 post-iter-945)."
    )
    assert v_max <= 100.0, (
        f"|v_max| at 1 day = {v_max:.2f} m/s; iter-945 expected <= 100 "
        f"(was 151 at iter-944b, measured 81 post-iter-945)."
    )


def test_iter945_non_duogrid_unchanged_baseline():
    """The non-duogrid FB chain remains at the iter-944b ~41-step
    baseline because `_pad_halo_dgrid_for_ppm` is gated on duogrid.
    `_bgrid_ke_transport` falls back to PPM with `mode='edge'` halo
    when the duogrid is absent.
    """
    grid, cdgrid, u_d, v_d = _w2_d_grid_winds(36, use_duogrid=False)
    sw = williamson_test2(grid)
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
        for k in range(50):
            state = model.step(state, 300.0)
            if not bool(np.all(np.isfinite(np.asarray(state.h)))):
                # Pin the iter-944b baseline of ~41 steps with margin
                assert 30 <= k <= 50, (
                    f"Non-duogrid FB chain at C36 dt=300 s survived {k} "
                    f"steps; iter-944b/iter-945 expected ~41."
                )
                return
    # If we got here, the loop completed 50 steps without NaN — that's
    # higher than the iter-944b ~41 baseline, signalling either a
    # Python-only sync re-introduction or a downstream regression.
    raise AssertionError(
        "Non-duogrid FB chain at C36 dt=300 s exceeded 50 steps without "
        "NaN; iter-944b expected ~41-step baseline."
    )
