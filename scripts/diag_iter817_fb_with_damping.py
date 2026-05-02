"""Iter-817 diagnostic: FB DUOGRID with damping vs without.

Per iter-816's iter-817+ candidate, iter-817 re-runs the C24 24h
FB DUOGRID test with iter-761-equivalent damping enabled
(div_damp, damp_v, d4_bg, nord).  Tests whether h_max growth
from 3074 to 24040 (iter-816 no-damping) is suppressed.

This clarifies whether the h-growth is:
(a) driven by the FB chain's discretisation itself, OR
(b) mostly a consequence of disabling damping in iter-816.
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


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def _run_fb(n, use_duogrid, with_damping, hours=24.0, dt=300.0):
    n_steps = int(round(hours * 3600 / dt))
    if with_damping:
        cfg = CDGridShallowWaterConfig(
            hyperdiff_coeff=0.0,
            div_damp=8.0 * _div_damp_cube(n),
            boundary_fix=True,
            damp_v=0.06, nord_v=2,
            d4_bg=0.16, nord=2,
        )
    else:
        cfg = CDGridShallowWaterConfig(
            hyperdiff_coeff=0.0, div_damp=0.0, boundary_fix=True,
            damp_v=0.0, nord_v=0,
            d4_bg=0.16, nord=1,
        )
    grid = create_cubed_sphere(n=n, use_duogrid=use_duogrid)
    model = FV3FBShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    h0_init = state.h
    mass_init = float(jnp.sum(state.h * grid.area))
    crashed_at = None
    try:
        for step in range(n_steps):
            state = model.step(state, dt)
            if not np.all(np.isfinite(np.asarray(state.h))):
                crashed_at = step
                break
    except Exception as e:
        crashed_at = ('exception', str(e)[:80], step)

    h_np = np.asarray(state.h)
    u_np = np.asarray(state.u_d)
    v_np = np.asarray(state.v_d)
    if crashed_at is not None:
        return {'crashed_at': crashed_at}

    u_cc = 0.5 * (u_np[:, :, :-1] + u_np[:, :, 1:])
    v_cc = 0.5 * (v_np[:, :-1, :] + v_np[:, 1:, :])
    mass_final = float(np.sum(h_np * np.asarray(grid.area)))

    h0_np = np.asarray(h0_init)
    h_mean = float(np.mean(np.abs(h0_np)))
    err = h_np - h0_np
    L2 = float(np.sqrt(np.mean(err ** 2)) / h_mean)
    return {
        'h_max_final': float(np.max(np.abs(h_np))),
        'h0_max': float(np.max(np.abs(h0_np))),
        'L2': L2,
        'u_cc_linf': float(np.max(np.abs(u_cc))),
        'v_cc_linf': float(np.max(np.abs(v_cc))),
        'mass_drift': (mass_final - mass_init) / mass_init,
    }


n = 24
hours = 24.0

print(f"Iter-817 FB DUOGRID W2 C{n} {hours}h: damping on vs off")
print()
print(f"{'variant':>35}  {'h0':>8}  {'h_max':>8}  {'L2':>8}  "
      f"{'u_cc':>8}  {'v_cc':>8}  {'mass':>11}")
print("-" * 96)

cases = [
    ('FB DUOGRID, no damping (iter-816)', True, False),
    ('FB DUOGRID, iter-761 damping', True, True),
    ('FB LEGACY, iter-761 damping', False, True),
]

for label, use_dg, with_damp in cases:
    r = _run_fb(n, use_dg, with_damp, hours)
    if 'crashed_at' in r:
        print(f"{label:>35}  CRASH at step {r['crashed_at']}")
    else:
        print(f"{label:>35}  {r['h0_max']:>8.0f}  {r['h_max_final']:>8.0f}  "
              f"{r['L2']:>8.3e}  {r['u_cc_linf']:>8.3e}  "
              f"{r['v_cc_linf']:>8.3e}  {r['mass_drift']:>+11.3e}")

print()
print("Interpretation cues (observational only):")
print("- If FB DUOGRID with damping has h_max near 2960 (physical):")
print("  iter-816's 8x h growth was mainly due to no-damping config,")
print("  NOT the FB chain discretisation.  Properly-configured FB")
print("  DUOGRID could be viable.")
print("- If FB DUOGRID still shows large h_max growth with damping:")
print("  FB chain has structural issues beyond damping.")
