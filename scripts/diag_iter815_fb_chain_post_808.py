"""Iter-815 diagnostic: test FV3 FB chain W2 under iter-808 sync fix.

`FV3FBShallowWaterModel` (src/legoesm/atmosphere/dynamics/shallow_
water_fv3_cdgrid.py:367) was marked "EXPERIMENTAL, NOT PRODUCTION-
READY. Known unstable (85 m/s v-wind after 1 day, 3% mass error)".

But the FB chain DOES call `synchronize_cgrid_fluxes` (fv3_sw_core.
py:1309-1312, gated on `use_duogrid`).  iter-808's sign-flip fix
might therefore also help the FB chain.

iter-815 tests the FB chain on W2 at C24 with and without duogrid
to see if iter-808 has changed the stability picture.
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


def _run_fb(n, use_duogrid, hours=6.0, dt=300.0):
    n_steps = int(round(hours * 3600 / dt))
    grid = create_cubed_sphere(n=n, use_duogrid=use_duogrid)
    # FB chain doesn't use div_damp/damp_v the same way as RK3.
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=0.0,
        boundary_fix=True,
        damp_v=0.0,
        nord_v=0,
        d4_bg=0.16,
        nord=1,
    )
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
    try:
        for step in range(n_steps):
            state = model.step(state, dt)
            if step % 10 == 0:
                h_max = float(jnp.max(jnp.abs(state.h)))
                if not np.isfinite(h_max) or h_max > 1e6:
                    return {'error': f'h blowup at step {step}: h_max={h_max}'}
    except Exception as e:
        return {'error': str(e)[:100]}

    h_mean = float(jnp.mean(jnp.abs(h0)))
    err = state.h - h0
    L2 = float(jnp.sqrt(jnp.mean(err ** 2)) / h_mean)
    mass_final = float(jnp.sum(state.h * grid.area))
    mass_drift = (mass_final - mass_init) / mass_init

    u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                   + np.asarray(state.u_d)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                   + np.asarray(state.v_d)[:, 1:, :])
    v_cc_linf = float(np.max(np.abs(v_cc)))
    u_cc_linf = float(np.max(np.abs(u_cc)))

    return {
        'L2': L2,
        'u_cc_linf': u_cc_linf,
        'v_cc_linf': v_cc_linf,
        'mass_drift': mass_drift,
    }


n = 24
hours = 6.0

print(f"Iter-815 FV3 FB chain W2 C{n} {hours}h (post iter-808)")
print(f"(Previous note: unstable with 85 m/s v-wind after 1 day,")
print(f" 3% mass error — that was before iter-808's sync fix)")
print()

print(f"{'variant':>30}  {'L2':>11}  {'u_cc_Linf':>11}  "
      f"{'v_cc_Linf':>11}  {'mass drift':>11}")
print("-" * 78)

for label, use_dg in (('FB LEGACY', False), ('FB DUOGRID', True)):
    r = _run_fb(n, use_dg, hours)
    if 'error' in r:
        print(f"{label:>30}  ERROR: {r['error']}")
    else:
        print(f"{label:>30}  {r['L2']:>11.3e}  {r['u_cc_linf']:>11.3e}  "
              f"{r['v_cc_linf']:>11.3e}  {r['mass_drift']:>+11.3e}")

print()
print("Interpretation cues (observational only):")
print("- If FB DUOGRID is now stable (v_cc_Linf < 10 m/s):")
print("  iter-808's sync fix has unblocked the FB chain.")
print("- If both FB runs error out or v_cc blows up:")
print("  FB chain has issues beyond the sync (e.g. p_grad_c,")
print("  d_sw1-6 chain, nord handling).")
