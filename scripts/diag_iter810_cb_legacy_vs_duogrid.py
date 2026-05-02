"""Iter-810 diagnostic: cosine bell LEGACY vs DUOGRID after iter-
808 sign-flip flux sync fix.

Per iter-809's iter-810+ candidate "Run cosine bell under DUOGRID
to measure impact", iter-810 compares cosine bell Linf and L2 at
C36 for 1 day under both paths.

Expected: DUOGRID should give comparable results to LEGACY (per
iter-809 W5 cross-validation showing within 1%).

Scope: observational only.  Cosine bell uses transport_step (not
fv3_sw_tendencies), which has its own cgrid_mass_flux_divergence
equivalent via `fv_tp_2d`.  This lets us check whether iter-808's
fix generalises to the transport-only test case.
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
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel)
from legoesm.core.fv3_sw_core import _d2a2c_vect
from legoesm.core.fv_tp_2d import transport_step
from tests.test_cases.cosine_bell import (
    cosine_bell_cubesphere, cosine_bell_exact)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def _run_cb(n, use_duogrid, dt=1440.0, days=1.0):
    n_steps = int(round(days * 86400 / dt))
    beta = jnp.pi / 4.0
    grid = create_cubed_sphere(n, use_duogrid=use_duogrid)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=_div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2)
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid
    state = cosine_bell_cubesphere(grid, cdgrid, beta)
    _ua, _va, _uc, _vc, ut, vt = _d2a2c_vect(state.u_d, state.v_d, cdgrid)
    mass_init = float(jnp.sum(state.h * grid.area))
    h = state.h
    for _ in range(n_steps):
        h = transport_step(h, ut, vt, dt, cdgrid, mass_target=mass_init)

    t_s = days * 86400.0
    h_exact = cosine_bell_exact(grid.lon, grid.lat, grid.radius, t_s, beta)
    err = np.asarray(h) - np.asarray(h_exact)
    abs_err = np.abs(err)

    l_inf = float(np.max(abs_err))
    area_np = np.asarray(grid.area)
    h_exact_np = np.asarray(h_exact)
    l2_num = float(np.sum(area_np * err ** 2))
    l2_den = float(np.sum(area_np * h_exact_np ** 2))
    l2 = float(np.sqrt(l2_num / l2_den)) if l2_den > 0 else np.nan

    return {
        'l_inf': l_inf,
        'l2': l2,
        'h_min': float(np.min(np.asarray(h))),
        'h_max': float(np.max(np.asarray(h))),
    }


n = 36

print(f"Iter-810 cosine bell LEGACY vs DUOGRID at C{n}, 1 day")
print(f"(post-iter-808 sign-flip sync fix)")
print()

r_L = _run_cb(n, use_duogrid=False)
r_D = _run_cb(n, use_duogrid=True)

print(f"{'metric':>14}  {'LEGACY':>12}  {'DUOGRID':>12}  {'ratio':>10}")
print("-" * 55)
for k in ('l_inf', 'l2', 'h_min', 'h_max'):
    L = r_L[k]; D = r_D[k]
    if abs(L) > 1e-30:
        ratio = f"{D / L:.3f}"
    else:
        ratio = 'N/A'
    print(f"  {k:>12}  {L:>12.3e}  {D:>12.3e}  {ratio:>10}")

print()
print("Interpretation cues (observational only):")
print("- If DUOGRID cosine bell matches LEGACY closely:")
print("  iter-808 fix unlocks DUOGRID for cosine bell too.")
print("- If DUOGRID cosine bell is substantially worse:")
print("  cosine-bell-specific duogrid issue remains.")
