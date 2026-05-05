"""Iter-767: measure W2 C36 1-day with the iter-767
`fortran_vector_corner_fill` flag ON vs OFF (default).  Reproduces
the canonical matrix measurement path (iter-766b/766c).

Fortran reference: `fill_corners_agrid_r8` in
`../atmos_cubed_sphere-symmetryclean/tools/fv_mp_mod.F90:1433-1457`
with mySign=-1 (VECTOR).  At the SW cube-vertex halo cell:

    x(0, 0) = -y(0, 1)
    y(0, 0) = -x(1, 0)

— direct cross-component swap with sign flip.  See iter-767 review-
doc entry for the SE/NE/NW analogues and index mappings.

Hypothesis (iter-767).  The A-grid vector cube-corner fill in the
current Python `pad_halo_vector` rotates through geographic
components and back, using `compute_padded_angle` at halo cells.
At the 3-face cube vertex the face-local grid angle is
discontinuous and the rotate-pad-rotate chain produces a cube-
corner halo value that is inconsistent with the neighbor faces'
values.  Fortran's direct swap (u ← ±v) sidesteps the angle
discontinuity entirely.  This might reduce mode A if that
inconsistency is the driver.

Expected outcome: either
- mode A reduced (W2 v_ll_Linf < 0.159 m/s) — then enable by
  default, add sentinel.
- mode A unchanged or worse — then falsify, add known-worse
  sentinel (iter-765/766 pattern).
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


def run_and_measure(fortran_vector_corner_fill: bool):
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        fortran_vector_corner_fill=fortran_vector_corner_fill,
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

    h_err = np.asarray(state.h) - h_exact
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


for flag, label in [(False, "OFF (rotate-pad-rotate, default)"),
                     (True,  "ON  (Fortran agrid vector fill) ")]:
    l2, linf, vnlinf, vllinf = run_and_measure(flag)
    print(f"[{label}]  h_L2={l2:.3e}  h_Linf={linf:.3e}  "
          f"v_north_Linf={vnlinf:.3e}  v_ll_Linf={vllinf:.3e}")

print("\nCanonical OFF baseline (iter-766 commit-time):")
print("  h_L2=2.07e-04  h_Linf=1.53e-03  v_ll_Linf=1.59e-01")
