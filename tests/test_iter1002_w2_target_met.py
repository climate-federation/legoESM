"""Iter-1002 sentinel: pin the W2 v_ll_Linf ≤ 0.119 m/s target met.

Discovered in iter-1001/1002 sweep that with the production
CDGrid path (Arakawa-Lamb + RK3) and config:
  - div_damp = 12 * _div_damp_cube(N)  (vs iter-893 baseline 8 *)
  - damp_v = 0.04                       (vs iter-893 baseline 0.06)
  - apply_fortran_xppm_boundary = True  (iter-888 Fortran fidelity)
  - boundary_fix = True
  - hyperdiff_coeff = 0.0
  - nord_v = 2

W2 C36 1-day → v_ll_Linf = 0.1154 m/s ≤ 0.119 m/s target.
h_err_max = 9.58 m (small, no artifacts).

This is a CALIBRATION refinement of the iter-893 production
baseline (which gave 0.132 m/s).  The PRODUCTION DEFAULT in
CDGridShallowWaterConfig stays at iter-893 baseline values to
preserve every other downstream sentinel; this test pins the
iter-1002 calibration as the demonstrated W2-target-met config
that future regressions on production CDGrid must not cross.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import warnings

import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
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
    _div_damp_cube,
)


def test_iter1002_w2_v_ll_linf_meets_target():
    """W2 C36 1-day v_ll_Linf ≤ 0.119 m/s with iter-1002 calibration."""
    N = 36
    DT = 300.0
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
        div_damp=12.0 * _div_damp_cube(N),
        boundary_fix=True,
        damp_v=0.04, nord_v=2,
        apply_fortran_xppm_boundary=True,
    )
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)
    n_steps = int(86400 / DT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_steps):
            state = model.step(state, DT)

    ca, sa = cell_centre_angles_from_4edge(cdgrid)
    u_arr = np.asarray(state.u_d)
    v_arr = np.asarray(state.v_d)
    u_cc = 0.5 * (u_arr[:, :, :-1] + u_arr[:, :, 1:])
    v_cc = 0.5 * (v_arr[:, :-1, :] + v_arr[:, 1:, :])
    v_north = np.asarray(sa) * u_cc + np.asarray(ca) * v_cc
    weights = get_cubedsphere_to_latlon_weights(N, n_lon=360, n_lat=181)
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    v_ll_Linf = float(np.abs(v_ll).max())
    h_err_max = float(np.max(np.abs(np.asarray(state.h) - sw.h.data)))

    assert v_ll_Linf <= 0.119, (
        f"W2 v_ll_Linf = {v_ll_Linf:.4f} m/s > 0.119 m/s target.  "
        f"Iter-1002 measured 0.1154; allowing a small upward drift "
        f"to 0.119 catches obvious regressions while permitting tiny "
        f"numerical noise.")
    assert h_err_max < 20.0, (
        f"W2 h_err_max = {h_err_max:.4f} m > 20.0 m soft bound.  "
        f"Iter-1002 measured ~9.58 m.")
