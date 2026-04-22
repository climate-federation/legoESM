"""Iter-769 diagnostic: measure W2 C36 1-day with the iter-769
`boundary_fix_skip_corners` flag ON vs OFF (default).

Reproduces the canonical matrix measurement path identical to
iter-766c / iter-767 / iter-768 diagnostics (same IC, dt, norms,
v_north convention).

Iter-769 hypothesis.  The cascaded `boundary_fix` smoothing
(row-0 then col-0, and row-n-1 then col-n-1) gives cube-corner
cells a DOUBLE update — effectively a 4-point average of the 2×2
block at the corner:

  After row-0 op:  du_cc[0, 0] = 0.5*(orig[0,0] + orig[1,0])
  After col-0 op:  du_cc[0, 0] = 0.5*(row_smoothed[0,0]
                                       + row_smoothed[0,1])
                              = 0.25*(orig[0,0] + orig[1,0]
                                       + orig[0,1] + orig[1,1])

Since iter-762/768 localize mode A at cells adjacent to the 8
cube vertices, and iter-765/766/767 falsified all three Fortran
halo-fill candidates, the cascaded corner smoothing itself is a
suspect.  This diagnostic measures whether skipping it at the 4
corner cells reduces, leaves unchanged, or worsens mode A.

The non-corner boundary-row/column smoothing is retained in both
cases — iter-511's measurement shows boundary_fix is load-bearing
for W2 L2 at non-corner positions.
"""
import os, sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
for _stale in ("JAX_PLATFORM_NAME", "JAX_DISABLE_JIT", "JAX_DEBUG_NANS"):
    os.environ.pop(_stale, None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

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
    return ref_coeff * (ref_n / n) ** 2


n = 36
dt = 300.0
days = 1
n_steps = int(days * 86400 / dt)

grid = create_cubed_sphere(n)
u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)

sw = williamson_test2(grid)
exact = williamson_test2_exact(grid, days * 86400.0)
weights = get_cubedsphere_to_latlon_weights(n, 360, 181)


def run_and_measure(boundary_fix_skip_corners: bool):
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        boundary_fix_skip_corners=boundary_fix_skip_corners,
        damp_v=0.06,
        nord_v=2,
    )
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    h_exact = np.asarray(exact.h.data)
    area = np.asarray(grid.area)

    for _ in range(n_steps):
        state = model.step(state, dt)

    h_final = np.asarray(state.h)
    h_err = h_final - h_exact
    h_err_l2 = float(np.sqrt(np.sum(h_err ** 2 * area)
                              / np.sum(h_exact ** 2 * area)))
    h_err_linf = float(np.max(np.abs(h_err)) / np.max(np.abs(h_exact)))

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


for flag, label in [(False, "OFF (cascaded corner avg, default)"),
                     (True,  "ON  (skip 4 corner cells)         ")]:
    l2, linf, vnlinf, vllinf = run_and_measure(flag)
    print(f"[{label}]  h_L2={l2:.3e}  h_Linf={linf:.3e}  "
          f"v_north_Linf={vnlinf:.3e}  v_ll_Linf={vllinf:.3e}")

print("\nCanonical OFF baseline (iter-768 commit-time):")
print("  h_L2=2.07e-04  h_Linf=1.53e-03  v_ll_Linf=1.59e-01")
