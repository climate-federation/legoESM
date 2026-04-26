"""Iter-900 W2 measurement: does the Fortran-faithful LEFT-side override
improve, hurt, or leave unchanged the W2 v_ll_Linf at C36 1-day?

Compares two configurations:
  (A) iter-892 default (production matrix runner uses this since iter-893):
      `apply_fortran_xppm_boundary=True, fortran_faithful_ppm_left=False`.
      Reference baseline: v_ll_Linf ~= 0.132 m/s.
  (B) iter-900 candidate:
      `apply_fortran_xppm_boundary=True, fortran_faithful_ppm_left=True`.
      Replaces the LEFT-side overrides at q_face[2,3] with Fortran-faithful
      formulas at the corrected indices q_face[2,3,4], implementing
      al(0)/al(1)/al(2) per `tp_core.F90:359-362`.

iter-892's docstring index map (q[k]=q1(k-1)) was wrong; production
strip is q[k]=q1(k-2) (Hypothesis A, verified iter-899).  iter-892's
xt_L formula at q_face[2] therefore implements a non-Fortran value at
the al(0) slot, and iter-892's c3/c2/c1 mirror at q_face[3] implements
a non-Fortran value at the al(1) slot.

iter-900 candidate places Fortran's actual recipes at the correct slots:
  q_face[2] = al(0) = c1*q1(-2) + c2*q1(-1) + c3*q1(0).
  q_face[3] = al(1) = xt clipped using q1(-1..2).
  q_face[4] = al(2) = c3*q1(1) + c2*q1(2) + c1*q1(3) (NEW override).

If iter-900 (B) is at least as good as iter-892 (A) on W2, then a
default flip in iter-901+ is justified.  If worse, document the
empirical advantage of iter-892 (likely due to the xt-clipping at the
mismatched slot reducing overshoot at the smooth W2 cube vertex).

Output a clear comparison table.
"""
import os, sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)
from legoesm.grids.cubed_sphere_cdgrid import (
    cell_centre_angles_from_4edge)
from legoesm.grids.regridding import (
    get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon)


def run_w2(use_fortran_faithful_left: bool):
    n = 36
    dt = 300.0
    n_steps = int(86400 / dt)
    div_damp = 8.0 * 1.5e7 * (48.0 / n) ** 2
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    weights = get_cubedsphere_to_latlon_weights(n, n_lon=360, n_lat=181)

    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=div_damp,
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        apply_fortran_xppm_boundary=True,
        fortran_faithful_ppm_left=use_fortran_faithful_left,
    )
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    h_ic = np.asarray(sw.h.data)
    area = np.asarray(grid.area)
    mass_ic = float(np.sum(h_ic * area))

    for _ in range(n_steps):
        state = model.step(state, dt)

    h_final = np.asarray(state.h)
    mass_final = float(np.sum(h_final * area))
    mass_drift = abs(mass_final - mass_ic) / mass_ic

    h_err = h_final - h_ic
    h_l2 = float(np.sqrt(np.sum(h_err ** 2 * area) / np.sum(h_ic ** 2 * area)))
    h_linf = float(np.max(np.abs(h_err)))

    ca, sa = cell_centre_angles_from_4edge(cdgrid)
    u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                   + np.asarray(state.u_d)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                   + np.asarray(state.v_d)[:, 1:, :])
    v_north = np.asarray(sa) * u_cc + np.asarray(ca) * v_cc
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    v_ll_linf = float(np.max(np.abs(v_ll)))

    return {
        'mass_drift': mass_drift,
        'h_L2': h_l2,
        'h_Linf': h_linf,
        'v_ll_Linf': v_ll_linf,
    }


print("Iter-900 W2 measurement: Fortran-faithful LEFT-side overrides")
print("vs iter-892 default at C36, dt=300s, 1-day integration.")
print()

print(f"{'config':<35}  {'mass_drift':>12}  {'h_L2':>10}  "
      f"{'h_Linf':>10}  {'v_ll_Linf':>11}")
print("-" * 90)

iter892 = run_w2(use_fortran_faithful_left=False)
print(f"{'(A) iter-892 default':<35}  "
      f"{iter892['mass_drift']:>12.3e}  "
      f"{iter892['h_L2']:>10.3e}  "
      f"{iter892['h_Linf']:>10.3e}  "
      f"{iter892['v_ll_Linf']:>11.4e}")

iter900 = run_w2(use_fortran_faithful_left=True)
print(f"{'(B) iter-900 fortran-faithful left':<35}  "
      f"{iter900['mass_drift']:>12.3e}  "
      f"{iter900['h_L2']:>10.3e}  "
      f"{iter900['h_Linf']:>10.3e}  "
      f"{iter900['v_ll_Linf']:>11.4e}")
print()
print(f"v_ll_Linf delta (B - A): {iter900['v_ll_Linf'] - iter892['v_ll_Linf']:+.4e} m/s")
print(f"v_ll_Linf relative (B/A): {iter900['v_ll_Linf'] / iter892['v_ll_Linf']:.4f}")

if iter900['v_ll_Linf'] < iter892['v_ll_Linf']:
    print()
    print("VERDICT: iter-900 IMPROVES W2 - Fortran-faithful LEFT-side")
    print("         overrides reduce v_ll_Linf.  iter-901+ should default-flip.")
elif iter900['v_ll_Linf'] > iter892['v_ll_Linf'] * 1.02:
    print()
    print("VERDICT: iter-900 WORSENS W2 (>2% increase) - keep iter-892 as default,")
    print("         document iter-892 formulas as deliberate empirical choice")
    print("         despite the index-map shift.")
else:
    print()
    print("VERDICT: iter-900 NEUTRAL on W2 (<2% change) - either flip or not")
    print("         at iter-901's discretion; no W2 progress in iter-900.")
