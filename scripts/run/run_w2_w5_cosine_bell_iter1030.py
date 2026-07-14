"""Run Williamson 2, 5, and cosine bell with iter-1030 calibration.

Quick smoke check that exercises the iter-1030 dual-target preset
across all three Williamson 1992 verification tests.  Output is
human-readable PASS/FAIL with measured metrics.

Equivalent to `pytest tests/test_iter1032_dual_target_full_matrix.py`
but runnable directly without pytest, useful for first-time setup
verification or post-edit smoke testing.

Usage:
    JAX_ENABLE_X64=1 python scripts/run_w2_w5_cosine_bell_iter1030.py
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Allow running from the repo root: prepend repo to sys.path so
# `tests/` is importable.
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import warnings

import jax.numpy as jnp
import numpy as np

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
from tests.test_iter921_w2_v_vs_h_pareto_sentinel import (
    cell_centre_angles_from_4edge,
)


def main():
    N = 36
    DT = 300.0
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    cfg = iter1009_dual_target_config(N)

    print("=" * 60)
    print("Iter-1030 dual-target verification on Williamson 1992 matrix")
    print("=" * 60)
    print()
    print(f"Resolution: C{N}, dt={DT}s")
    print(f"Calibration: div_damp_factor=8.0, damp_v=0.030, "
          f"apply_fortran_xppm_boundary=True")
    print()

    # ---------- W2 1-day ----------
    print("[1/3] Williamson Test 2 (1-day) ...")
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)
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
    h_err = float(np.max(np.abs(np.asarray(state.h) - sw.h.data)))
    w2_pass = v_ll_Linf <= 0.119
    print(f"      v_ll_Linf = {v_ll_Linf:.4f} m/s  "
          f"({'PASS ≤0.119' if w2_pass else 'FAIL >0.119'})")
    print(f"      h_err_max = {h_err:.2f} m")
    print()

    # ---------- W5 day-5 ----------
    print("[2/3] Williamson Test 5 (day-5) ...")
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(int(5 * 86400 / DT)):
            state = model.step(state, DT)
    h = np.asarray(state.h)
    u = np.asarray(state.u_d)
    v = np.asarray(state.v_d)
    u_cc = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    v_cc = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    speed = np.sqrt(u_cc**2 + v_cc**2)
    h_min = float(h.min())
    speed_max = float(speed.max())
    w5_pass = h_min > 0 and speed_max < 80
    print(f"      h_min     = {h_min:.1f} m  "
          f"({'PASS >0' if h_min > 0 else 'FAIL ≤0'})")
    print(f"      speed_max = {speed_max:.1f} m/s  "
          f"({'PASS <80' if speed_max < 80 else 'FAIL ≥80'})")
    print()

    # ---------- Cosine bell ----------
    print("[3/3] Cosine bell (1-day, mass conservation) ...")
    BETA = jnp.pi / 4.0
    DT_CB = 1440.0
    DAYS_CB = 1.0
    NSTEPS = int(round(DAYS_CB * 86400 / DT_CB))
    state = cosine_bell_cubesphere(grid, cdgrid, BETA)
    _, _, _, _, ut, vt = d2a2c_vect(state.u_d, state.v_d, cdgrid)
    mass_init = float(jnp.sum(state.h * grid.area))
    h = state.h
    for _ in range(NSTEPS):
        h = transport_step(
            h, ut, vt, DT_CB, cdgrid,
            mass_target=mass_init,
            apply_fortran_xppm_boundary=True,
        )
    mass_final = float(jnp.sum(h * np.asarray(grid.area)))
    # iter-93 audit followup: previously
    # ``abs(mass_final - mass_init) / mass_init`` would NaN if
    # mass_init = 0 (degenerate test case).  Migrate to the
    # shared helper for consistency with iter-88 conventions.
    from legoesm.diagnostics.conservation_drift import compute_relative_drift
    drift = compute_relative_drift([mass_init, mass_final])
    cb_pass = drift < 5e-7
    print(f"      mass_drift = {drift:.3e}  "
          f"({'PASS <5e-7' if cb_pass else 'FAIL ≥5e-7'})")
    print()

    print("=" * 60)
    if w2_pass and w5_pass and cb_pass:
        print("ALL 3 PASS — iter-1030 calibration verified.")
    else:
        print("ONE OR MORE FAILED — see above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
