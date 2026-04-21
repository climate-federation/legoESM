"""Iter-766b: measure W2 C36 1-day with the iter-766
`fortran_a2b_corner_avg` flag ON vs OFF (default).  Reproduces the
CANONICAL MATRIX MEASUREMENT PATH used by
`scripts/run_atmosphere_test_matrix.py --only sw --grid cubed_sphere
--quick` at the iter-761 tuned config — so that the OFF/ON numbers
this script reports are bit-identical to what the production matrix
would print.

Canonical matrix path (see run_atmosphere_test_matrix.py around the
`shallow_water/williamson2/cubed_sphere` case):
- Grid: `create_cubed_sphere(n)`
- IC: `williamson_test2(grid)` for h & h_s, then
      `u_d = cdgrid.cos_angle_edge_x * u0 * cos(cdgrid.lat_edge_x)`
      `v_d = -cdgrid.sin_angle_edge_y * u0 * cos(cdgrid.lat_edge_y)`
      where u0 = 2*pi*R / (12*86400).
- dt = 300s, n_steps = 86400/dt = 288 (1-day integration).
- Config: `CDGridShallowWaterConfig(hyperdiff_coeff=0.0,
      div_damp=8*_div_damp_cube(n), boundary_fix=True, damp_v=0.06,
      nord_v=2)`.
- v_north: via `cell_centre_angles_from_4edge(cdgrid)` then
      `v_north = sa_4edge * u_cc + ca_4edge * v_cc` using the
      edge-averaged (not cell-centre) grid angles.
- Regrid: `get_cubedsphere_to_latlon_weights(n, 360, 181)` then
      `apply_cubedsphere_to_latlon(v_north, weights)` — the exact
      iter-765f / iter-766 regression-sentinel helpers.

Iter-766 end-of-iter Codex stop-time flagged the previous version of
this script (pre-fix): it used `williamson2_initial_condition` from
the unit-test file (which returns D-grid corner-staggered winds, not
edge-midpoint), dt=60s (not 300s), and `grid.angle` for v_north
computation (cell-centre, not edge-averaged).  Those three
divergences meant the numbers the script printed were NOT the matrix
numbers.  This rewrite fixes all three so the diag truly reproduces
the canonical matrix path.

Expected numbers at iter-766 commit time:
- OFF (2-pt-avg, default): v_ll_Linf ≈ 1.59e-01 m/s, h_L2 ≈ 2.07e-04
- ON  (a2b 3-pt-avg):      v_ll_Linf ≈ 3.00e-01 m/s, h_L2 ≈ 4.50e-04

Ratio ON/OFF ~ 1.89× — the iter-766 falsification evidence.
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import cell_centre_angles_from_4edge
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from legoesm.grids.regridding import (
    get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2, williamson_test2_exact)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    """Match run_atmosphere_test_matrix._div_damp_cube (quadratic, not quartic)."""
    return ref_coeff * (ref_n / n) ** 2


# Canonical matrix settings.
n = 36
dt = 300.0
days = 1
n_steps = int(days * 86400 / dt)   # 288 steps → 1-day integration

grid = create_cubed_sphere(n)
# FV3EdgeShallowWaterModel creates its own cdgrid from the grid, and
# we'll read it back via `model.cdgrid` — mirrors the matrix script.

# Use grid.radius (NOT hardcoded 6.371e6) — iter-766b Codex 2nd-pass
# flagged the hardcoded value drifted from the canonical grid radius
# by ~229 m, which propagated a 3.6e-5 relative error into u0.
u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)

sw = williamson_test2(grid)
# Matrix reference is the exact solution at t=days (steady for W2,
# so `williamson_test2_exact = williamson_test2` numerically —
# but we call the matrix-identical function for source-of-truth
# parity with `scripts/run_atmosphere_test_matrix.py:1467`).
exact = williamson_test2_exact(grid, days * 86400.0)
weights = get_cubedsphere_to_latlon_weights(n, 360, 181)


def run_and_measure(fortran_a2b_corner_avg: bool):
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        fortran_a2b_corner_avg=fortran_a2b_corner_avg,
    )
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    # Canonical matrix IC: edge-midpoint D-grid winds.
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    # Matrix reference: `exact.h.data` (from williamson_test2_exact).
    h_exact = np.asarray(exact.h.data)
    area = np.asarray(grid.area)

    for _ in range(n_steps):
        state = model.step(state, dt)

    # Area-weighted + normalized norms (matrix convention — see
    # run_atmosphere_test_matrix.py around the
    # `test_num == 2 and tc.grid_type == "cubed_sphere"` block
    # that computes the h error from `state.h - exact.h.data`).
    # `state.h` is the jax.Array member of FV3EdgeShallowWaterState;
    # matrix uses `state.h if isinstance(state.h, jnp.ndarray) else
    # state.h.data` — for FV3EdgeShallowWaterState this resolves to
    # `state.h` directly, which np.asarray converts losslessly.
    h_final = np.asarray(state.h)
    h_err = h_final - h_exact
    h_err_l2 = float(np.sqrt(np.sum(h_err ** 2 * area)
                              / np.sum(h_exact ** 2 * area)))
    h_err_linf = float(np.max(np.abs(h_err)) / np.max(np.abs(h_exact)))

    # v_north via edge-averaged cell-centre angles (matrix convention).
    ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdgrid)
    u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                   + np.asarray(state.u_d)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                   + np.asarray(state.v_d)[:, 1:, :])
    v_north = np.asarray(sa_4edge) * u_cc + np.asarray(ca_4edge) * v_cc
    v_north_linf = float(np.max(np.abs(v_north)))
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    v_ll_linf = float(np.max(np.abs(v_ll)))
    return h_err_l2, h_err_linf, v_north_linf, v_ll_linf


for flag, label in [(False, "OFF (2-pt-avg, default)"),
                     (True,  "ON  (Fortran a2b 3-pt) ")]:
    l2, linf, vnlinf, vllinf = run_and_measure(flag)
    print(f"[{label}]  h_L2={l2:.3e}  h_Linf={linf:.3e}  "
          f"v_north_Linf={vnlinf:.3e}  v_ll_Linf={vllinf:.3e}")

print("\nMatrix-script canonical reference (iter-766 commit-time):")
print("  OFF: h_L2=2.07e-04  h_Linf=1.53e-03  v_ll_Linf=1.59e-01")
print("  ON : h_L2=4.50e-04  h_Linf=3.75e-03  v_ll_Linf=3.00e-01")
