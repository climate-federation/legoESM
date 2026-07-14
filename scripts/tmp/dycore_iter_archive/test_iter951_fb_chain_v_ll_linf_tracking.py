"""Iter-951 sentinel: pin v_ll_Linf (the actual W2 acceptance metric)
on the duogrid FB chain at C36 dt=300 s.

Background.  The W2 acceptance gate per user iter-938 brief is
v_ll_Linf ≤ 0.119 m/s.  iter-944b through iter-950 tracked the
proxy metrics |u_max| and |v_max| (face-covariant), and iter-950
discovered these can move OPPOSITE to v_ll_Linf — extending the
PPM cross-face halo from h_dg=2 to h_dg=3 reduced |v_max| by 6%
but regressed v_ll_Linf by 4%.

Iter-951 adds a direct v_ll_Linf measurement to lock the actual
acceptance-gate metric.  The current measured value is 55.6 m/s on
the duogrid FB chain at C36 1-day — far from the 0.119 m/s
acceptance threshold but the improvement trajectory matters:

| iter        | |u_max| | |v_max| | v_ll_Linf |
|-------------|--------:|--------:|----------:|
| iter-944b   |    106  |    151  |     ~85   | (estimate from |v|×0.55 ratio)
| iter-945    |     78  |     81  |     ~57   |
| iter-947    |     77  |     75  |     55.6  |
| acceptance  |     -   |     -   |    0.119  |

Iter-951 pins v_ll_Linf <= 60.0 m/s on the duogrid FB chain at
C36 W2 1-day, with margin around 55.6 to allow grid/dtype drift
and small per-iter improvements that don't drop below this gate.

A regression that increases v_ll_Linf above 60 likely means a
new structural error has appeared.

Production impact: ZERO.  This is FB-chain-specific.
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
from legoesm.grids.regridding import (
    apply_cubedsphere_to_latlon,
    get_cubedsphere_to_latlon_weights,
)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
)
from tests.test_iter921_w2_v_vs_h_pareto_sentinel import (
    cell_centre_angles_from_4edge,
)


def _fb_chain_v_ll_linf_at_1_day(N: int, dt: float):
    grid = create_cubed_sphere(N, use_duogrid=True)
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
    n_steps = int(86400 / dt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_steps):
            state = model.step(state, dt)
    ca, sa = cell_centre_angles_from_4edge(cdgrid)
    u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                   + np.asarray(state.u_d)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                   + np.asarray(state.v_d)[:, 1:, :])
    v_north = np.asarray(sa) * u_cc + np.asarray(ca) * v_cc
    weights = get_cubedsphere_to_latlon_weights(N, n_lon=360, n_lat=181)
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    return float(np.abs(v_ll).max())


def test_iter951_fb_chain_duogrid_w2_v_ll_linf_below_60():
    """Post-iter-947 duogrid FB chain at C36 dt=300 s W2 1-day must
    have v_ll_Linf below 60 m/s.  Measured value at iter-947 baseline
    is 55.6 m/s; the 60 m/s gate provides margin for grid/dtype
    drift while catching regressions.

    A drop in this metric (towards the 0.119 m/s acceptance) is a
    real fidelity win.  Lift the gate downward when iter-952+ closes
    additional Fortran-fidelity gaps.
    """
    v_ll_linf = _fb_chain_v_ll_linf_at_1_day(36, 300.0)
    assert v_ll_linf <= 60.0, (
        f"FB chain duogrid C36 W2 1-day v_ll_Linf = {v_ll_linf:.4f} m/s "
        f"exceeds the iter-951 gate of 60.0 m/s.  iter-947 baseline "
        f"is 55.6 m/s; a value above 60 likely signals a new "
        f"structural Fortran-fidelity regression."
    )
