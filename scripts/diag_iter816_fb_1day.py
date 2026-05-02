"""Iter-816 diagnostic: FB chain W2 at 1 day (C24) — does prior
85 m/s blowup still occur post iter-808?

iter-815b's scope-caveat correction flagged that the 6h snapshot
can't rule out later-horizon blowup.  iter-816 extends to 1 day
(288 steps at dt=300s) to directly test whether FB DUOGRID still
blows up at the 1-day mark that produced the prior 85 m/s report.

Sample every 1 hour to track how error evolves.  If v_cc_Linf
grows toward 85 m/s or higher at later horizons, the prior
report reproduces and the iter-815 "stable" framing was wrong.
If it stays finite (say < 1000 m/s), the FB chain no longer
hits that particular blowup at C24 1-day.
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


def _run_fb_sampled(n, use_duogrid, hours=24.0, dt=300.0,
                     sample_every_h=1.0):
    n_steps = int(round(hours * 3600 / dt))
    sample_stride = int(round(sample_every_h * 3600 / dt))
    grid = create_cubed_sphere(n=n, use_duogrid=use_duogrid)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0, div_damp=0.0, boundary_fix=True,
        damp_v=0.0, nord_v=0, d4_bg=0.16, nord=1)
    model = FV3FBShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    h0 = state.h
    mass_init = float(jnp.sum(state.h * grid.area))
    samples = []
    crashed_at = None
    try:
        for step in range(n_steps):
            state = model.step(state, dt)
            h_np = np.asarray(state.h)
            if not np.all(np.isfinite(h_np)):
                crashed_at = step
                break
            if (step + 1) % sample_stride == 0:
                u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                               + np.asarray(state.u_d)[:, :, 1:])
                v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                               + np.asarray(state.v_d)[:, 1:, :])
                mass = float(np.sum(h_np * np.asarray(grid.area)))
                samples.append({
                    'hour': (step + 1) * dt / 3600.0,
                    'h_max': float(np.max(np.abs(h_np))),
                    'u_cc_linf': float(np.max(np.abs(u_cc))),
                    'v_cc_linf': float(np.max(np.abs(v_cc))),
                    'mass_drift': (mass - mass_init) / mass_init,
                })
    except Exception as e:
        crashed_at = (str(e)[:100], step)

    return {'samples': samples, 'crashed_at': crashed_at}


n = 24
hours = 24.0

print(f"Iter-816 FB chain W2 C{n} {hours}h sampled every 1h")
print()

for label, use_dg in (('FB LEGACY', False), ('FB DUOGRID', True)):
    r = _run_fb_sampled(n, use_dg, hours=hours, sample_every_h=1.0)
    print(f"{label}:")
    if r['crashed_at'] is not None:
        print(f"  crashed at step {r['crashed_at']}")
    print(f"  {'hr':>4}  {'h_max':>10}  {'u_cc_Linf':>11}  "
          f"{'v_cc_Linf':>11}  {'mass drift':>11}")
    for s in r['samples']:
        print(f"  {s['hour']:>4.1f}  {s['h_max']:>10.3f}  "
              f"{s['u_cc_linf']:>11.3e}  {s['v_cc_linf']:>11.3e}  "
              f"{s['mass_drift']:>+11.3e}")
    print()

print("Interpretation cues (observational only):")
print("- If FB DUOGRID reaches hour 24 with v_cc_Linf < 100 m/s:")
print("  the prior '85 m/s at 1 day' blowup is NOT reproduced")
print("  on this C24 run (one specific IC/config combination).")
print("- If FB DUOGRID crashes before hour 24 or v_cc explodes:")
print("  the FB chain remains unstable at 1-day horizon and")
print("  iter-815's '6h stable' snapshot did not indicate a fix.")
