"""Iter-910 diagnostic: W2 v_ll_Linf resolution sweep.

Tests the CLAUDE.md memory claim:
  "0.159 m/s LEGACY W2 residual is STRUCTURAL cancellation failure.
   Resolution-invariant, knob-invariant. Architectural fix required."

iter-908b/iter-909 confirmed knob-invariant within div_damp tuning.
iter-910 tests resolution-invariance by sweeping {C16, C24, C36, C48}.

If v_ll_Linf is FLAT or grows with resolution: structural — no
discretization-driven refinement is possible, and iter-911+ MUST
pursue an architectural change (option 3 d_sw5 port or FB pivot).

If v_ll_Linf scales like 1/n^p (some p>0): refinable — the post-
iter-893 0.132 m/s at C36 is a discretization residual that decreases
at higher resolution.  iter-911+ could focus on dt scaling or
filtering.

dt is held at the canonical 300s (W2 production dt), but smaller dt
may be needed for higher-resolution stability.  Use dt = 300 *
(36 / n) to maintain a constant CFL number.
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
from legoesm.grids.cubed_sphere_cdgrid import (
    cell_centre_angles_from_4edge)
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)
from legoesm.grids.regridding import (
    get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def run_w2(n: int):
    # CFL-preserving dt: dt scales with grid spacing.
    dt = 300.0 * (36 / n)
    n_steps = int(86400 / dt)
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    weights = get_cubedsphere_to_latlon_weights(n, n_lon=360, n_lat=181)

    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        apply_fortran_xppm_boundary=True,
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

    for _ in range(n_steps):
        state = model.step(state, dt)

    h_final = np.asarray(state.h)
    if not np.all(np.isfinite(h_final)):
        return {'dt': dt, 'n_steps': n_steps,
                'h_L2': float('nan'), 'v_ll_Linf': float('nan')}
    h_l2 = float(np.sqrt(np.sum((h_final - h_ic) ** 2 * area)
                          / np.sum(h_ic ** 2 * area)))

    ca, sa = cell_centre_angles_from_4edge(cdg)
    u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                   + np.asarray(state.u_d)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                   + np.asarray(state.v_d)[:, 1:, :])
    v_north = np.asarray(sa) * u_cc + np.asarray(ca) * v_cc
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    v_ll_linf = float(np.max(np.abs(v_ll)))
    return {'dt': dt, 'n_steps': n_steps, 'h_L2': h_l2, 'v_ll_Linf': v_ll_linf}


def main():
    print("Iter-910 W2 resolution sweep.")
    print("CFL-preserving dt = 300 * (36/n) seconds.")
    print("All configs use iter-892/iter-893 production matrix (8x div_damp,")
    print("apply_fortran_xppm_boundary=True, boundary_fix=True, damp_v=0.06,")
    print("nord_v=2, dddmp=0.2).")
    print()
    print(f"{'n':>4}  {'dt':>6}  {'n_steps':>7}  {'h_L2':>10}  "
          f"{'v_ll_Linf':>11}  {'rel C36':>8}")
    print("-" * 65)
    Ns = [16, 24, 36, 48]
    results = {}
    for n in Ns:
        r = run_w2(n)
        results[n] = r
        rel = ""
        if 36 in results and np.isfinite(r['v_ll_Linf']) and results[36]['v_ll_Linf'] > 0:
            rel = f"{r['v_ll_Linf'] / results[36]['v_ll_Linf']:.2f}x"
        print(f"{n:>4}  {r['dt']:>6.1f}  {r['n_steps']:>7}  "
              f"{r['h_L2']:>10.3e}  {r['v_ll_Linf']:>11.4e}  {rel:>8}")
    print()

    # Convergence-rate analysis.
    finite_ns = [n for n in Ns if np.isfinite(results[n]['v_ll_Linf'])]
    if len(finite_ns) < 2:
        print("VERDICT: insufficient finite results to determine resolution scaling.")
        return

    print("Convergence rate analysis (assumes v_ll_Linf ~ 1/n^p):")
    print(f"  C16 -> C36 ratio: {results[16]['v_ll_Linf']/results[36]['v_ll_Linf']:.2f}")
    print(f"  C24 -> C36 ratio: {results[24]['v_ll_Linf']/results[36]['v_ll_Linf']:.2f}")
    print(f"  C36 -> C48 ratio: {results[36]['v_ll_Linf']/results[48]['v_ll_Linf']:.2f}")
    if 48 in finite_ns and 16 in finite_ns:
        # log(v_n / v_m) / log(m / n) = order p (if v ~ 1/n^p).
        p = (np.log(results[16]['v_ll_Linf'] / results[48]['v_ll_Linf'])
             / np.log(48 / 16))
        print(f"  estimated order p (16→48): {p:.2f}")
        print()
        if abs(p) < 0.3:
            print("VERDICT: nearly RESOLUTION-INVARIANT (|p| < 0.3).  Confirms")
            print("         CLAUDE.md memory: the W2 residual is STRUCTURAL.")
            print("         No discretization-driven refinement possible.")
            print("         iter-911+ MUST pursue architectural change:")
            print("           - Option (3): d_sw5 holistic port.")
            print("           - Or: FB chain stabilization (orthogonal track).")
        elif p > 0.5:
            print(f"VERDICT: REFINABLE (order ~{p:.1f}).  Higher resolution")
            print(f"         reduces v_ll_Linf.  iter-911+ could focus on")
            print(f"         keeping the SAME architecture but running at higher")
            print(f"         production resolution (C48+).  But likely still")
            print(f"         beneficial to pursue d_sw5 port for absolute reduction.")
        else:
            print(f"VERDICT: weakly resolution-dependent (p ~ {p:.2f}).  Some")
            print(f"         refinement possible at higher resolution but")
            print(f"         architectural fix likely needed for substantial")
            print(f"         improvement.")


if __name__ == "__main__":
    main()
