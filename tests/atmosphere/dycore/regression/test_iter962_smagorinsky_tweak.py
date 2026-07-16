"""Iter-962 sentinel: pin the iter-959 Smagorinsky calibration tweak.

iter-959 swept d_sw5 damping coefficients on the FB chain duogrid
C36 W2 1-day and found that enabling Smagorinsky adaptive damping
(`d2_bg=0.01, dddmp=0.05`) — left at 0.0 by the iter-934 default —
gives a 4% v_ll_Linf improvement (55.6 → 53.2 m/s) with NO
stability cost.

This is calibration tuning, not Fortran-fidelity per se: the
default `d2_bg=0, dddmp=0` is preserved (changing it would shift
many existing pinned sentinels), but callers who want better W2
fidelity on the FB chain can opt in.

Iter-962 locks in the iter-959 measurement so a future change to
d_sw5's adaptive Smagorinsky branch (`d_sw5_corner_divergence`
nord==0 path or the nord>=1 Smagorinsky inside _divergence_corner_duo)
that changes this output gets caught.
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
from tests.atmosphere.dycore.regression.test_iter921_w2_v_vs_h_pareto_sentinel import (
    cell_centre_angles_from_4edge,
)


def _fb_chain_w2_1day(N: int, dt: float, **cfg_kwargs):
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
        d4_bg=0.16, nord=1, damp_v=0.06, nord_v=2,
        **cfg_kwargs,
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


def test_iter962_smagorinsky_tuned_improves_w2_v_ll_linf():
    """Iter-959 finding: enabling Smagorinsky adaptive damping
    (d2_bg=0.01, dddmp=0.05) on the FB chain duogrid C36 W2 1-day
    improves v_ll_Linf from 55.6 to ~53.2 m/s (4% reduction).
    """
    v_default = _fb_chain_w2_1day(36, 300.0, d2_bg=0.0, dddmp=0.0)
    v_smag = _fb_chain_w2_1day(36, 300.0, d2_bg=0.01, dddmp=0.05)
    assert v_smag < v_default, (
        f"Smagorinsky tweak should improve v_ll_Linf: "
        f"default = {v_default:.4f}, smag = {v_smag:.4f}.  "
        f"Iter-959 measured 55.61 → 53.17 m/s."
    )
    # Pin the iter-959 measurement with margin.
    assert v_smag <= 55.0, (
        f"Smagorinsky-tuned v_ll_Linf = {v_smag:.4f} m/s; iter-959 "
        f"baseline is 53.17 m/s.  A value above 55.0 likely signals "
        f"a regression in the d_sw5 Smagorinsky branch."
    )


def test_iter963_high_smagorinsky_drives_w2_to_44_m_s():
    """Iter-963 finding: pushing Smagorinsky to (d2_bg=0.09, dddmp=0.45)
    drives v_ll_Linf from 55.6 to ~43.8 m/s on FB chain duogrid C36
    W2 1-day (21% reduction; |v_max| 75 → 55 m/s; h_err_max 18591
    → 12855).  The d_sw5 Smagorinsky branch has substantial unused
    headroom under the iter-934 default (0.0, 0.0).

    This pins the higher Smagorinsky measurement so a future change
    to the adaptive Smagorinsky branch that regresses this output
    is caught.
    """
    v_high = _fb_chain_w2_1day(36, 300.0, d2_bg=0.09, dddmp=0.45)
    # Iter-963 measured 43.77 m/s — pin <= 47 with margin.
    assert v_high <= 47.0, (
        f"High-Smagorinsky v_ll_Linf = {v_high:.4f} m/s; iter-963 "
        f"baseline is 43.77 m/s.  A value above 47.0 likely signals "
        f"a regression in the d_sw5 Smagorinsky branch."
    )
