"""Iter-819 diagnostic: damp_v sweep on FB DUOGRID.

Per iter-818's iter-819+ candidate, iter-819 sweeps damp_v
(0, 0.001, 0.003, 0.01, 0.03, 0.06) with nord_v=2 on FB DUOGRID
C24 W2 24h to find the stable range.

iter-818 showed damp_v=0.06 + nord_v=2 crashes at 7.9h.
iter-816 showed damp_v=0 (nord_v=0 irrelevant) is stable to 24h
but with h_max=24040 (8× physical).

If a small damp_v > 0 gives finite 24h AND smaller h_max, the
FB chain has a viable damping sweet spot.
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
    CDGridShallowWaterConfig, FV3FBShallowWaterModel,
    FV3EdgeShallowWaterState)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def _run_fb(n, hours, dt, damp_v, nord_v):
    n_steps = int(round(hours * 3600 / dt))
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0, div_damp=0.0, boundary_fix=True,
        damp_v=damp_v, nord_v=nord_v, d4_bg=0.16, nord=1)
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    model = FV3FBShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    crashed_at_step = None
    for step in range(n_steps):
        state = model.step(state, dt)
        if not np.all(np.isfinite(np.asarray(state.h))):
            crashed_at_step = step
            break
    if crashed_at_step is not None:
        return {'crashed_at_step': crashed_at_step,
                'crashed_at_hour': crashed_at_step * dt / 3600.0}

    h_np = np.asarray(state.h)
    u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                   + np.asarray(state.u_d)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                   + np.asarray(state.v_d)[:, 1:, :])
    return {
        'h_max': float(np.max(np.abs(h_np))),
        'u_cc_linf': float(np.max(np.abs(u_cc))),
        'v_cc_linf': float(np.max(np.abs(v_cc))),
    }


n = 24
hours = 24.0
dt = 300.0
damp_v_values = [0.0, 0.001, 0.003, 0.01, 0.03, 0.06]

print(f"Iter-819 FB DUOGRID damp_v sweep (nord_v=2) W2 C{n} {hours}h")
print()
print(f"{'damp_v':>10}  {'status':>30}  {'h_max':>8}  {'u_cc_Linf':>11}  {'v_cc_Linf':>11}")
print("-" * 75)

for dv in damp_v_values:
    nv = 2 if dv > 0 else 0
    r = _run_fb(n, hours, dt, damp_v=dv, nord_v=nv)
    if 'crashed_at_step' in r:
        status = f"CRASH step {r['crashed_at_step']} ({r['crashed_at_hour']:.1f}h)"
        h_max = u_cc = v_cc = '—'
    else:
        status = 'ok (24h)'
        h_max = f"{r['h_max']:.0f}"
        u_cc = f"{r['u_cc_linf']:.3e}"
        v_cc = f"{r['v_cc_linf']:.3e}"
    print(f"  {dv:>8.3f}  {status:>30}  {h_max:>8}  {u_cc:>11}  {v_cc:>11}")

print()
print("Interpretation cues (observational only):")
print("- If small damp_v > 0 gives smaller h_max than baseline (0):")
print("  there's a stable sweet spot.  FB DUOGRID viable with tuning.")
print("- If all damp_v > 0 destabilise OR keep h_max >= 24040:")
print("  damp_v under FB is not the fix; need other approach.")
