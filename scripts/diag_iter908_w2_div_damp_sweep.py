"""Iter-908 diagnostic: sweep `div_damp` coefficient at C36 W2 1-day
to identify the post-iter-893 sweet spot.

Per iter-907's ablation finding, div_damp dominates the W2 D-grid
hot-spot magnitude (29× more than boundary_fix).  iter-907 noted
that the 8× iter-761 tuning was a HISTORICAL improvement (0.303 →
0.159 m/s, before iter-893) but is now the dominant residual
contributor at the post-iter-893 baseline (0.132 m/s).

iter-908 sweeps div_damp ∈ {1×, 2×, 4×, 6×, 8×} the iter-761 reference
`_div_damp_cube(n)` (= 1.5e7 * (48/n)²) and measures W2 v_ll_Linf,
h_L2, and stability indicators.  Acceptance: a coefficient that
improves v_ll_Linf below 0.132 m/s without breaking conservation
or resurrecting catastrophic instability.
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


def _div_damp_cube_ref(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def run_w2(div_damp_mult: float):
    n = 36
    dt = 300.0
    n_steps = int(86400 / dt)
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    weights = get_cubedsphere_to_latlon_weights(n, n_lon=360, n_lat=181)

    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=div_damp_mult * _div_damp_cube_ref(n),
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
    mass_ic = float(np.sum(h_ic * area))

    for _ in range(n_steps):
        state = model.step(state, dt)

    h_final = np.asarray(state.h)
    if not np.all(np.isfinite(h_final)):
        return {'mass_drift': float('nan'), 'h_L2': float('nan'),
                'h_Linf': float('nan'), 'v_ll_Linf': float('nan'),
                'h_min': float('nan'), 'h_max': float('nan')}
    mass_final = float(np.sum(h_final * area))
    mass_drift = abs(mass_final - mass_ic) / mass_ic

    h_err = h_final - h_ic
    h_l2 = float(np.sqrt(np.sum(h_err ** 2 * area) / np.sum(h_ic ** 2 * area)))
    h_linf = float(np.max(np.abs(h_err)))
    h_min = float(np.min(h_final))
    h_max = float(np.max(h_final))

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
        'h_min': h_min,
        'h_max': h_max,
    }


def main():
    print("Iter-908 W2 div_damp sweep at C36 dt=300s 1-day.")
    print(f"Reference _div_damp_cube(36) = {_div_damp_cube_ref(36):.3e}")
    print(f"All configs use apply_fortran_xppm_boundary=True, boundary_fix=True,")
    print(f"damp_v=0.06, nord_v=2, dddmp_prod=0.2 (production matrix).")
    print()
    print(f"{'mult':>5}  {'div_damp':>12}  "
          f"{'mass_drift':>12}  {'h_L2':>10}  {'h_Linf':>10}  "
          f"{'v_ll_Linf':>11}  {'h_min':>11}  {'h_max':>11}")
    print("-" * 110)
    # Iter-908b (Codex iter-908 stop-time): extend sweep to higher
    # multipliers to test whether 10x's -1.43% is a local minimum or
    # the descent continues.  Without this extension, the iter-908
    # claim "8x is approximately optimal" overclaims the data — the
    # 10x point is the right edge of the 0.5x-10x range and could be
    # the START of further improvement.
    multipliers = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 24.0]
    results = {}
    for mult in multipliers:
        r = run_w2(mult)
        results[mult] = r
        marker = " <- iter-892/iter-893 production (8x)" if mult == 8.0 else ""
        print(f"{mult:>5.1f}  {mult * _div_damp_cube_ref(36):>12.3e}  "
              f"{r['mass_drift']:>12.3e}  {r['h_L2']:>10.3e}  "
              f"{r['h_Linf']:>10.3e}  {r['v_ll_Linf']:>11.4e}  "
              f"{r['h_min']:>11.4e}  {r['h_max']:>11.4e}{marker}")
    print()

    baseline = results[8.0]['v_ll_Linf']
    print(f"Baseline (8x): v_ll_Linf = {baseline:.4e}")
    print()
    print(f"v_ll_Linf relative to 8x baseline:")
    for mult in multipliers:
        v = results[mult]['v_ll_Linf']
        if np.isfinite(v):
            pct = 100.0 * (v - baseline) / baseline
            print(f"  {mult:>5.1f}x: {v:.4e}  ({pct:+.2f}%)")
        else:
            print(f"  {mult:>5.1f}x: NaN")
    print()

    # Identify best mult by v_ll_Linf, ignoring NaN.
    finite_results = {m: r for m, r in results.items()
                       if np.isfinite(r['v_ll_Linf'])}
    if not finite_results:
        print("VERDICT: all configs produced NaN — div_damp range may be wrong.")
        return
    best_mult = min(finite_results, key=lambda m: finite_results[m]['v_ll_Linf'])
    best_v = finite_results[best_mult]['v_ll_Linf']
    if best_mult == 8.0:
        print(f"VERDICT: 8x iter-761 tuning is the W2 v_ll_Linf optimum within")
        print(f"         the swept range — no improvement possible by simple")
        print(f"         coefficient adjustment.  iter-909+ should pursue option (2)")
        print(f"         (cube-edge-aware adaptive_coeff) or (3) (d_sw5 holistic")
        print(f"         port) instead.")
    elif best_v < baseline * 0.95:
        improvement_pct = 100.0 * (1.0 - best_v / baseline)
        print(f"VERDICT: BETTER W2 at div_damp = {best_mult}x "
              f"(v_ll_Linf = {best_v:.4e}, -{improvement_pct:.2f}% vs 8x).")
        print(f"         Recommend iter-909 commit production default flip to")
        print(f"         {best_mult}x AFTER broader (W5/cosine bell/ocean rest)")
        print(f"         confirmation.")
    else:
        print(f"VERDICT: Mild improvement at div_damp = {best_mult}x but below")
        print(f"         5% threshold.  Likely within numerical noise.  iter-909+")
        print(f"         should pursue option (2) cube-edge-aware adaptive_coeff.")


if __name__ == "__main__":
    main()
