"""Iter-1032 sentinel: full W2 + W5 + cosine-bell verification on iter-1030.

The iter-1030 calibration (div=8, damp_v=0.030) is exposed as the
default of `iter1009_dual_target_config(N)`.  This sentinel runs
the FULL Williamson 1992 verification matrix on it:

  W2 1-day:    v_ll_Linf ≤ 0.119 m/s ✓
  W5 day-5:    h_min > 0, speed_max < 80 m/s ✓
  Cosine bell: mass_drift < 5e-7 ✓

If any of these fail, the iter-1009/1021/1030 calibration has
regressed and the user's "no artifacts" requirement is at risk.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
    FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
    iter1009_dual_target_config,
)
from legoesm.core.fv3_sw_core import d2a2c_vect
from legoesm.core.fv_tp_2d import transport_step
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.regridding import (
    apply_cubedsphere_to_latlon,
    get_cubedsphere_to_latlon_weights,
)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
    williamson_test5,
)
from tests.test_cases.cosine_bell import cosine_bell_cubesphere
from tests.atmosphere.dycore.regression.test_iter921_w2_v_vs_h_pareto_sentinel import (
    cell_centre_angles_from_4edge,
)


N = 36


@pytest.fixture(scope="module")
def cdgrid_setup():
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    cfg = iter1009_dual_target_config(N)
    return grid, cdgrid, cfg


def test_iter1032_W2_meets_target(cdgrid_setup):
    """W2 1-day with iter-1030 calibration must satisfy v_ll ≤ 0.119."""
    grid, cdgrid, cfg = cdgrid_setup
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)
    DT = 300.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(int(86400 / DT)):
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
    assert v_ll_Linf <= 0.119, (
        f"W2 v_ll_Linf = {v_ll_Linf:.4f} > 0.119 m/s.  "
        f"iter-1030 measured ~0.1138; iter-1032 catches drift."
    )


def test_iter1032_W5_day5_artifact_free(cdgrid_setup):
    """W5 day-5 with iter-1030 calibration must be artifact-free."""
    grid, cdgrid, cfg = cdgrid_setup
    sw = williamson_test5(grid)
    u_d = cdgrid.cos_angle_edge_x * (
        20.0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (
        20.0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)
    DT = 300.0
    DAYS = 5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(int(DAYS * 86400 / DT)):
            state = model.step(state, DT)
    h = np.asarray(state.h)
    u = np.asarray(state.u_d)
    v = np.asarray(state.v_d)
    assert np.isfinite(h).all() and np.isfinite(u).all(), (
        "W5 day-5 produced NaN")
    h_min = float(h.min())
    u_cc = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    v_cc = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    speed_max = float(np.sqrt(u_cc**2 + v_cc**2).max())
    assert h_min > 0.0, (
        f"W5 day-5 h_min = {h_min:.2f} m ≤ 0.  "
        f"iter-1030 measured ~3888 m.")
    assert speed_max < 80.0, (
        f"W5 day-5 speed_max = {speed_max:.2f} m/s ≥ 80.  "
        f"iter-1030 measured ~45.1 m/s.")


def test_iter1032_cosine_bell_mass_conserves(cdgrid_setup):
    """Cosine bell day-1 with iter-1030 calibration must conserve mass."""
    grid, cdgrid, cfg = cdgrid_setup
    BETA = jnp.pi / 4.0
    DT = 1440.0
    DAYS = 1.0
    NSTEPS = int(round(DAYS * 86400 / DT))

    state = cosine_bell_cubesphere(grid, cdgrid, BETA)
    _, _, _, _, ut, vt = d2a2c_vect(state.u_d, state.v_d, cdgrid)
    mass_init = float(jnp.sum(state.h * grid.area))
    h = state.h
    for _ in range(NSTEPS):
        h = transport_step(
            h, ut, vt, DT, cdgrid,
            mass_target=mass_init,
            apply_fortran_xppm_boundary=True,
        )
    mass_final = float(jnp.sum(h * np.asarray(grid.area)))
    # iter-164: migrated from inline ``abs(final - init) / init``
    # (note: lacked ``abs()`` on denom — would silently pass for
    # negative ``mass_init``).  Centralized helper uses
    # ``abs(baseline)`` and is NaN-aware.  Same migration as
    # iter-157 / iter-159 sweep.
    from legoesm.diagnostics import compute_relative_drift
    drift = compute_relative_drift([mass_init, mass_final])
    assert drift < 5e-7, (
        f"Cosine bell mass drift = {drift:.3e} > 5e-7.  "
        f"iter-1030 measured ~3.2e-7.")
