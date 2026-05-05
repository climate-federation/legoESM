"""Iter-818 diagnostic: knob-at-a-time FB DUOGRID damping test.

Per iter-817's iter-818+ candidate, iter-818 tests individual
damping knobs on FB DUOGRID C24 W2 to locate which knob triggers
the ~8h crash seen in iter-817 (iter-761 config: damp_v=0.06,
nord_v=2, d4_bg=0.16, nord=2).

Baseline (no damping, iter-816): finite at 24h, h_max=24040.
Iter-761-like damping: crash at 8h.

Knobs to test (one at a time, starting from baseline):
  A. damp_v=0.06 (vs 0) — vorticity damping on.
  B. nord_v=2 (vs 0) — higher vorticity damping order.
  C. nord=2 (vs 1) — higher del-4+ damping order.
  D. d4_bg=0.0 (vs 0.16) — turn OFF del-4 background.

Results locate the crash trigger.
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


def _run_fb(n, hours, dt, **cfg_kwargs):
    n_steps = int(round(hours * 3600 / dt))
    cfg = CDGridShallowWaterConfig(**cfg_kwargs)
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

baseline_kwargs = dict(
    hyperdiff_coeff=0.0, div_damp=0.0, boundary_fix=True,
    damp_v=0.0, nord_v=0, d4_bg=0.16, nord=1)

cases = [
    ('baseline (iter-816)', {}),
    ('+ damp_v=0.06', dict(damp_v=0.06)),
    ('+ nord_v=2', dict(nord_v=2)),
    ('+ nord=2', dict(nord=2)),
    ('- d4_bg (=0.0)', dict(d4_bg=0.0)),
    ('+ damp_v=0.06 + nord_v=2', dict(damp_v=0.06, nord_v=2)),
    ('+ nord=2 + damp_v=0.06 + nord_v=2', dict(nord=2, damp_v=0.06, nord_v=2)),
]

print(f"Iter-818 FB DUOGRID knob-at-a-time damping test W2 C{n} {hours}h")
print()
print(f"{'variant':>38}  {'status':>30}  {'h_max':>8}  {'u_cc':>8}  {'v_cc':>8}")
print("-" * 90)

for label, overrides in cases:
    cfg_kwargs = {**baseline_kwargs, **overrides}
    r = _run_fb(n, hours, dt, **cfg_kwargs)
    if 'crashed_at_step' in r:
        status = f"CRASH step {r['crashed_at_step']} ({r['crashed_at_hour']:.1f}h)"
        h_max = u_cc = v_cc = '—'
    else:
        status = 'ok (24h)'
        h_max = f"{r['h_max']:.0f}"
        u_cc = f"{r['u_cc_linf']:.2f}"
        v_cc = f"{r['v_cc_linf']:.2f}"
    print(f"{label:>38}  {status:>30}  {h_max:>8}  {u_cc:>8}  {v_cc:>8}")

print()
print("Interpretation cues (observational only):")
print("- The single-knob variant that crashes first identifies the")
print("  trigger.")
print("- If ALL single-knob variants run to 24h but iter-817's full")
print("  combo crashes, the interaction of multiple knobs is the cause.")
