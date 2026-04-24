"""Iter-825 diagnostic: combined Fortran corner-fill knob test
on W2 LEGACY post iter-808.

Iter-765/766/767 individually tested three Fortran-corner-fill
flags and found each worsened W2 on their own.  But iter-808's
sign-flip flux sync has since landed.  iter-825 re-tests the 3
flags AND their combinations under LEGACY to see if any
(possibly including a combination) now reduces v_ll_Linf below
0.159 m/s.

Knobs:
  fortran_dir_aware_corners (iter-765)
  fortran_a2b_corner_avg (iter-766)
  fortran_vector_corner_fill (iter-767)

Scope: observational only.  If any combination helps, it's a
candidate for production; if all still worsen, the W2 mode-A
cannot be addressed by these known Fortran-like corner fills.
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


def _run_w2(n, fa2b, fvcf, days=1.0, dt=300.0):
    # Note: fortran_dir_aware_corners is NOT a config field — it's an
    # internal parameter of _arakawa_lamb_gradient, not reachable from
    # the production harness.  Only fortran_a2b_corner_avg and
    # fortran_vector_corner_fill are config-configurable.
    n_steps = int(round(days * 86400 / dt))
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06, nord_v=2,
        fortran_a2b_corner_avg=fa2b,
        fortran_vector_corner_fill=fvcf,
    )
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
    try:
        for _ in range(n_steps):
            state = model.step(state, dt)
    except Exception as e:
        return {'error': str(e)[:100]}

    if not np.all(np.isfinite(np.asarray(state.h))):
        return {'error': 'NaN in h'}

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
        'L2': L2,
        'v_ll_linf': float(np.max(np.abs(v_ll))),
    }


n = 36

print(f"Iter-825 LEGACY W2 combined Fortran corner-fill knob test")
print(f"C36 1-day, iter-761 canonical config + knob toggles")
print()
print(f"{'a2b':>5}  {'vec_fill':>10}  {'L2':>11}  {'v_ll_Linf':>11}")
print("-" * 45)

import itertools
# All 4 combinations of 2 bool flags.
for fa2b, fvcf in itertools.product([False, True], repeat=2):
    r = _run_w2(n, fa2b, fvcf)
    label_b = "T" if fa2b else "F"
    label_c = "T" if fvcf else "F"
    if 'error' in r:
        print(f"   {label_b}    {label_c}       ERROR: {r['error']}")
    else:
        print(f"   {label_b}    {label_c}     {r['L2']:>11.3e}  {r['v_ll_linf']:>11.3e}")

print()
print("Interpretation cues (observational only):")
print("- Baseline (all F): v_ll_Linf = 0.159 m/s (iter-820).")
print("- If any combo gives v_ll_Linf < 0.159: candidate for production.")
print("- If all combos give v_ll_Linf >= 0.159: none of these knobs")
print("  reduce W2 mode-A under the iter-808-fixed codebase.")
