"""Iter-909 W2 measurement: does softening div_damp's adaptive_coeff
ONLY at face-boundary cells reduce W2 v_ll_Linf below the iter-892
0.132 m/s baseline?

Per iter-908b's recommendation (option 2), iter-909 implements a
per-cell mask for the divergence-damping adaptive_coeff: factor=
`cube_edge_div_damp_factor` at boundary band cells (i in [0..band-1]
U [n-band..n-1] and j similarly), 1.0 at deep interior.

This W2 sweep tests:
  - factor ∈ {0.0, 0.25, 0.5, 0.75, 1.0}, band=2.
  - factor=1.0 should ≈ iter-892 baseline (no softening).
  - factor=0.0 disables div_damp at boundary cells entirely.

Acceptance per user's iter-904 spec:
  - improve v_ll_Linf >= 10 % AND worsen h_L2 < 5 %.
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


def run_w2(softer: bool, factor: float, band: int):
    n = 36
    dt = 300.0
    n_steps = int(86400 / dt)
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    weights = get_cubedsphere_to_latlon_weights(n, n_lon=360, n_lat=181)

    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),   # iter-892/iter-893 production
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        apply_fortran_xppm_boundary=True,
        cube_edge_softer_div_damp=softer,
        cube_edge_div_damp_factor=factor,
        cube_edge_div_damp_band=band,
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
        return {'h_L2': float('nan'), 'v_ll_Linf': float('nan')}
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
    return {'h_L2': h_l2, 'v_ll_Linf': v_ll_linf}


def main():
    print("Iter-909 W2 sweep: cube-edge-aware softer div_damp")
    print("at C36 dt=300s 1-day, band=2 cells per face boundary.")
    print()
    print(f"{'factor':>7}  {'h_L2':>10}  {'v_ll_Linf':>11}  {'%v vs 8x':>10}")
    print("-" * 55)

    # Baseline: iter-892 default (= factor=1.0 disabled).
    base = run_w2(softer=False, factor=1.0, band=2)
    print(f"{'OFF':>7}  {base['h_L2']:>10.3e}  "
          f"{base['v_ll_Linf']:>11.4e}  {'0.00%':>10} (production)")

    factors = [0.0, 0.25, 0.5, 0.75, 1.0]
    for f in factors:
        r = run_w2(softer=True, factor=f, band=2)
        v = r['v_ll_Linf']
        if np.isfinite(v) and base['v_ll_Linf'] > 0:
            pct = 100.0 * (v - base['v_ll_Linf']) / base['v_ll_Linf']
            print(f"{f:>7.2f}  {r['h_L2']:>10.3e}  "
                  f"{v:>11.4e}  {pct:>+9.2f}%")
        else:
            print(f"{f:>7.2f}  NaN         NaN         (blowup)")

    print()
    print("Hypothesis: factor < 1 may improve W2 if the boundary-cell")
    print("contribution of div_damp is more 'erroneous' than 'corrective'.")
    print("Per iter-908b's interpretation, this is uncertain — both")
    print("hypotheses are consistent with the iter-908b 1-D sweep monotone.")


if __name__ == "__main__":
    main()
