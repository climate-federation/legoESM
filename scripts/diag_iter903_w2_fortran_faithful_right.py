"""Iter-903 W2 measurement: does the Fortran-faithful RIGHT-side
override improve, hurt, or leave unchanged the W2 v_ll_Linf at C36
1-day?

Symmetric counterpart to iter-900's LEFT-side measurement.  The
RIGHT-side override has TWO halo=2-feasible slots (al(npx-1)
fully-faithful, al(npx) partial-faithful with mode='edge' replica
for q1(npx+1)) plus REMOVAL of iter-892's misplaced q_face[n+1]
override.  Per iter-903 unit tests: the q_face[n+3] override is
limiter-dominated on smooth fields (saturated by overshoot
constraint to 3*q-2*q_R), so the effective change is concentrated
at q_face[n+2] and the q_face[n+1] revert.

Compares two configurations:
  (A) iter-892 default (production):
      `apply_fortran_xppm_boundary=True, fortran_faithful_ppm_right=False`
      Reference baseline: v_ll_Linf ~= 0.132 m/s.
  (B) iter-903 RIGHT-side fortran-faithful:
      `apply_fortran_xppm_boundary=True, fortran_faithful_ppm_right=True`
      LEFT side stays on iter-892 (no `_left=True`).

If (B) is BETTER on W2, iter-903 yields a small Fortran-fidelity
gain.  If WORSE, iter-892's RIGHT-side accidentally-good behavior
is also W2-load-bearing (analogous to iter-900's LEFT-side finding).
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
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)
from legoesm.grids.cubed_sphere_cdgrid import (
    cell_centre_angles_from_4edge)
from legoesm.grids.regridding import (
    get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def run_w2(use_fortran_faithful_right: bool):
    n = 36
    dt = 300.0
    n_steps = int(86400 / dt)
    div_damp = 8.0 * _div_damp_cube(n)
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
        fortran_faithful_ppm_right=use_fortran_faithful_right,
    )
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdg = model.cdgrid
    u_d = cdg.cos_angle_edge_x * (u0 * jnp.cos(cdg.lat_edge_x))
    v_d = -cdg.sin_angle_edge_y * (u0 * jnp.cos(cdg.lat_edge_y))
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

    ca, sa = cell_centre_angles_from_4edge(cdg)
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


def main():
    print("Iter-903 W2 measurement: Fortran-faithful RIGHT-side overrides")
    print("vs iter-892 default at C36, dt=300s, 1-day integration.")
    print()
    print(f"{'config':<35}  {'mass_drift':>12}  {'h_L2':>10}  "
          f"{'h_Linf':>10}  {'v_ll_Linf':>11}")
    print("-" * 90)

    iter892 = run_w2(use_fortran_faithful_right=False)
    print(f"{'(A) iter-892 default':<35}  "
          f"{iter892['mass_drift']:>12.3e}  "
          f"{iter892['h_L2']:>10.3e}  "
          f"{iter892['h_Linf']:>10.3e}  "
          f"{iter892['v_ll_Linf']:>11.4e}")

    iter903 = run_w2(use_fortran_faithful_right=True)
    print(f"{'(B) iter-903 fortran-faithful right':<35}  "
          f"{iter903['mass_drift']:>12.3e}  "
          f"{iter903['h_L2']:>10.3e}  "
          f"{iter903['h_Linf']:>10.3e}  "
          f"{iter903['v_ll_Linf']:>11.4e}")
    print()
    print(f"v_ll_Linf delta (B - A): "
          f"{iter903['v_ll_Linf'] - iter892['v_ll_Linf']:+.4e} m/s")
    print(f"v_ll_Linf relative (B/A): "
          f"{iter903['v_ll_Linf'] / iter892['v_ll_Linf']:.4f}")

    if iter903['v_ll_Linf'] < iter892['v_ll_Linf'] * 0.98:
        print()
        print("VERDICT: iter-903 IMPROVES W2 by >2% — Fortran-faithful")
        print("         RIGHT-side overrides reduce v_ll_Linf.")
        print("         iter-904+ should consider default-flip.")
    elif iter903['v_ll_Linf'] > iter892['v_ll_Linf'] * 1.02:
        print()
        print("VERDICT: iter-903 WORSENS W2 by >2% — like iter-900,")
        print("         iter-892's RIGHT-side accidentally-good")
        print("         formulas are W2-load-bearing.  Keep iter-892")
        print("         as default.")
    else:
        print()
        print("VERDICT: iter-903 NEUTRAL on W2 (<2% change) — likely")
        print("         because q_face[n+3] override is limiter-")
        print("         dominated, leaving only q_face[n+2] and the")
        print("         q_face[n+1] revert as effective changes.")


if __name__ == "__main__":
    main()
