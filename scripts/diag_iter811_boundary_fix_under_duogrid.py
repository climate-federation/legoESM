"""Iter-811 diagnostic: test boundary_fix behaviour under DUOGRID
per Ralph loop critical constraint #2.

Ralph loop brief: "Legacy edge handling must be disabled in
duogrid mode via bounded_domain = .true.  Verify that legacy
edge paths are actually bypassed."

`boundary_fix` (operators_cdgrid.py:1725) is a Python-specific
legacy-edge smoothing that cascades row-0/col-0 averaging into
the cell-centre tendencies.  The code comment at line 1719-1724
describes it as a "NON-FV3 hack" needed because the A-L+RK3
production path is not FV3-faithful.

Under DUOGRID (post-iter-808), the halo now provides Fortran-
faithful cross-face data via extended grid + corner fill +
signed flux sync.  So `boundary_fix` should be UNNECESSARY —
and possibly counter-productive — in duogrid mode.

Test: run W2 at C36 under DUOGRID with and without boundary_fix.

Expected outcomes:
1. boundary_fix ON (current default): v_ll_Linf ≈ 1.2 m/s (iter-809 result).
2. boundary_fix OFF under DUOGRID: either BETTER (confirms it's
   an unnecessary hack) or WORSE (confirms it's load-bearing
   even with duogrid, so iter-808 doesn't fully replace it).
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


def _run_w2(n, use_duogrid, boundary_fix, days=1.0, dt=300.0):
    n_steps = int(round(days * 86400 / dt))
    grid = create_cubed_sphere(n=n, use_duogrid=use_duogrid)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=boundary_fix,
        damp_v=0.06,
        nord_v=2)
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
        'v_cc_linf': float(np.max(np.abs(v_north))),
    }


n = 36

print(f"Iter-811 boundary_fix under DUOGRID (W2 C36 1-day)")
print(f"Per Ralph loop constraint #2: legacy edge paths should be")
print(f"disabled in duogrid mode.  `boundary_fix` is a Python-")
print(f"specific legacy smoothing.")
print()
print(f"{'variant':>38}  {'L2':>11}  {'v_ll_Linf':>11}  {'v_cc_Linf':>11}")
print("-" * 78)

cases = [
    ('LEGACY, boundary_fix=True (baseline)', False, True),
    ('LEGACY, boundary_fix=False', False, False),
    ('DUOGRID, boundary_fix=True', True, True),
    ('DUOGRID, boundary_fix=False', True, False),
]

results = {}
for label, use_dg, bf in cases:
    r = _run_w2(n, use_dg, bf)
    results[label] = r
    if 'error' in r:
        print(f"{label:>38}  ERROR: {r['error']}")
    else:
        print(f"{label:>38}  {r['L2']:>11.3e}  {r['v_ll_linf']:>11.3e}  "
              f"{r['v_cc_linf']:>11.3e}")

print()
r_legacy_bf = results['LEGACY, boundary_fix=True (baseline)']
for label in (
    'LEGACY, boundary_fix=False',
    'DUOGRID, boundary_fix=True',
    'DUOGRID, boundary_fix=False',
):
    r = results[label]
    if 'error' in r:
        continue
    for k, ref in (('L2', r_legacy_bf['L2']),
                    ('v_ll_linf', r_legacy_bf['v_ll_linf']),
                    ('v_cc_linf', r_legacy_bf['v_cc_linf'])):
        ratio = r[k] / ref
        print(f"  {label} vs baseline: {k} = {ratio:.2f}x")

print()
print("Interpretation cues (observational only):")
print("- If DUOGRID+bf=False is BETTER than DUOGRID+bf=True:")
print("  boundary_fix is a legacy hack now unnecessary under duogrid.")
print("- If DUOGRID+bf=False is WORSE:")
print("  boundary_fix is load-bearing even with duogrid halos.")
print("- If LEGACY+bf=False blows up (2.4x per iter-511):")
print("  confirms legacy path still needs boundary_fix.")
