"""Iter-822 diagnostic: W2 LEGACY v_ll_Linf across C24/C36/C48.

Tests whether the cube-vertex mode-A (0.159 m/s at C36) reduces
with grid refinement.  If v_ll_Linf decreases proportional to
dx^p for some p>0, the artifact is truncation-level (improves
naturally with resolution).  If resolution-invariant, it's
structural and requires an algorithmic fix.
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
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def _run_w2(n, days=1.0, dt=300.0):
    n_steps = int(round(days * 86400 / dt))
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06, nord_v=2)
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    h0 = state.h
    for _ in range(n_steps):
        state = model.step(state, dt)

    h_mean = float(jnp.mean(jnp.abs(h0)))
    err = state.h - h0
    L2 = float(jnp.sqrt(jnp.mean(err ** 2)) / h_mean)

    from legoesm.grids.cubed_sphere_cdgrid import (
        cell_centre_angles_from_4edge)
    from legoesm.grids.regridding import (
        get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon)
    ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdgrid)
    u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                   + np.asarray(state.u_d)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                   + np.asarray(state.v_d)[:, 1:, :])
    v_north = np.asarray(sa_4edge) * u_cc + np.asarray(ca_4edge) * v_cc
    weights = get_cubedsphere_to_latlon_weights(n, n_lon=360, n_lat=181)
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)

    return {
        'dx_km': (6371229.0 * (np.pi / 2) / n) / 1000.0,
        'L2': L2,
        'v_ll_linf': float(np.max(np.abs(v_ll))),
        'v_cc_linf': float(np.max(np.abs(v_north))),
    }


print(f"Iter-822 W2 LEGACY v_ll_Linf resolution scan (C24/C36/C48), 1 day")
print()
print(f"{'n':>4}  {'dx [km]':>9}  {'L2':>11}  {'v_ll_Linf':>11}  {'v_cc_Linf':>11}")
print("-" * 60)

results = {}
for n in (24, 36, 48):
    r = _run_w2(n)
    results[n] = r
    print(f"  {n:>2}  {r['dx_km']:>8.1f}  {r['L2']:>11.3e}  "
          f"{r['v_ll_linf']:>11.3e}  {r['v_cc_linf']:>11.3e}")

print()
print("Convergence analysis (observational only):")
ns = sorted(results.keys())
for i in range(len(ns) - 1):
    n1, n2 = ns[i], ns[i+1]
    r1 = results[n1]; r2 = results[n2]
    dx_ratio = r2['dx_km'] / r1['dx_km']
    for k in ('L2', 'v_ll_linf', 'v_cc_linf'):
        rate = r2[k] / r1[k]
        # order-of-convergence estimate: if rate = dx_ratio^p then p = log(rate) / log(dx_ratio)
        if 0 < rate < 1 and 0 < dx_ratio < 1:
            p = np.log(rate) / np.log(dx_ratio)
        else:
            p = float('nan')
        print(f"  C{n1}→C{n2}: {k} ratio {rate:.3f} (dx ratio {dx_ratio:.3f}) — apparent order p={p:.2f}")

print()
print("Interpretation cues (observational only):")
print("- Apparent p > 0: artifact DECREASES with resolution.")
print("- p ≈ 1: 1st-order convergence (truncation-level).")
print("- p ≈ 2: 2nd-order convergence.")
print("- p ≈ 0 or negative: resolution-invariant / structural.")
