"""Iter-831 diagnostic: FB chain phase ablation.

iter-816 showed FB DUOGRID at C24 24h has h_max = 24040 (8x
physical).  The FB chain has 3 phases:
  1. _c_sw: C-grid half-step (KE, vorticity, mass-transport dt/2).
  2. _p_grad_c: C-grid pressure gradient update (dt/2).
  3. _d_sw_native: D-grid full-step (PPM transport + wind update,
     d_sw1-6).

iter-831 ablates each by monkey-patching the phase to a no-op
and observes which one drives the h-growth.  If disabling _c_sw
gives h_max ≈ physical, it's the culprit.  If _d_sw_native, it's
the PPM transport chain.  Etc.

Scope: observational only.  Monkey-patching in-place; results only
indicate directional sensitivity, not a fix.
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

from legoesm.core import fv3_sw_core as fv3_mod
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3FBShallowWaterModel,
    FV3EdgeShallowWaterState)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


orig_c_sw = fv3_mod._c_sw
orig_p_grad_c = fv3_mod._p_grad_c
orig_d_sw_native = fv3_mod._d_sw_native


def _run_fb(n, disable_c_sw, disable_p_grad_c, disable_d_sw_native,
            hours=12.0, dt=300.0):
    n_steps = int(round(hours * 3600 / dt))

    # Set up monkey-patches.  Each no-op preserves required outputs.
    if disable_c_sw:
        # c_sw returns (h_star, uc_new, vc_new, ua, va).  Monkey-patch to
        # pass-through: h_star = h, uc_new = 0, vc_new = 0, ua = 0, va = 0.
        def _noop_c_sw(h, u_d, v_d, h_s, cdgrid, dt, g):
            sh_u = u_d.shape  # (6, n, n+1)
            sh_v = v_d.shape  # (6, n+1, n)
            sh_h = h.shape    # (6, n, n)
            sh_uc = (sh_u[0], sh_u[1] + 1, sh_u[2] - 1)  # (6, n+1, n)
            sh_vc = (sh_v[0], sh_v[1] - 1, sh_v[2] + 1)  # (6, n, n+1)
            zeros_h = jnp.zeros(sh_h)
            zeros_uc = jnp.zeros(sh_uc)
            zeros_vc = jnp.zeros(sh_vc)
            return h, zeros_uc, zeros_vc, zeros_h, zeros_h
        fv3_mod._c_sw = _noop_c_sw
    else:
        fv3_mod._c_sw = orig_c_sw
    if disable_p_grad_c:
        def _noop_p_grad_c(h_star, h_s, cdgrid, dt2, g):
            sh = (6, cdgrid.n + 1, cdgrid.n)  # uc shape
            return jnp.zeros(sh), jnp.zeros((6, cdgrid.n, cdgrid.n + 1))
        fv3_mod._p_grad_c = _noop_p_grad_c
    else:
        fv3_mod._p_grad_c = orig_p_grad_c
    if disable_d_sw_native:
        # d_sw returns (h_new, u_d_new, v_d_new).  Pass-through: no update.
        def _noop_d_sw_native(h, u_d, v_d, h_s, uc_new, vc_new, ua, va,
                              cdgrid, dt, g, **kwargs):
            return h, u_d, v_d
        fv3_mod._d_sw_native = _noop_d_sw_native
    else:
        fv3_mod._d_sw_native = orig_d_sw_native

    grid = create_cubed_sphere(n=n, use_duogrid=True)
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

    crashed_step = None
    for step in range(n_steps):
        state = model.step(state, dt)
        if not np.all(np.isfinite(np.asarray(state.h))):
            crashed_step = step
            break

    if crashed_step is not None:
        return {'crashed_at_step': crashed_step,
                'crashed_at_hour': crashed_step * dt / 3600.0}

    h_np = np.asarray(state.h)
    return {
        'h_max': float(np.max(np.abs(h_np))),
        'h_min': float(np.min(h_np)),
    }


n = 24
hours = 12.0

print(f"Iter-831 FB DUOGRID C{n} {hours}h phase ablation")
print()
print(f"{'c_sw':>7}  {'p_grad':>7}  {'d_sw':>7}  {'status':>25}  {'h_min':>9}  {'h_max':>9}")
print("-" * 75)

# Restore originals after the test
try:
    for dc, dp, dd in [
        (False, False, False),  # baseline (all on)
        (True,  False, False),  # disable c_sw only
        (False, True,  False),  # disable p_grad_c only
        (False, False, True),   # disable d_sw_native only
        (True,  True,  False),  # disable c_sw + p_grad_c
        (True,  False, True),   # disable c_sw + d_sw_native
        (False, True,  True),   # disable p_grad_c + d_sw_native (should be identity)
    ]:
        r = _run_fb(n, dc, dp, dd, hours=hours)
        if 'crashed_at_step' in r:
            status = f"CRASH {r['crashed_at_step']} step ({r['crashed_at_hour']:.1f}h)"
            h_min = h_max = '—'
        else:
            status = 'ok'
            h_min = f"{r['h_min']:.0f}"
            h_max = f"{r['h_max']:.0f}"
        print(f"   {'T' if dc else 'F':>4}    {'T' if dp else 'F':>4}    "
              f"{'T' if dd else 'F':>4}   {status:>25}  {h_min:>9}  {h_max:>9}")
finally:
    fv3_mod._c_sw = orig_c_sw
    fv3_mod._p_grad_c = orig_p_grad_c
    fv3_mod._d_sw_native = orig_d_sw_native

print()
print("Interpretation cues (observational only):")
print("- If disabling one phase gives h_max ≈ 2960 (physical):")
print("  that phase was driving the h-growth.")
print("- If all ablations still grow h_max, the growth is from")
print("  all 3 phases acting together.")
