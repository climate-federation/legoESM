"""Iter-799 diagnostic: DUOGRID t=0 tendency audit.

Iter-787 showed DUOGRID W2 is 800x worse than LEGACY.  Iter-788
ruled out config-level fixes.  iter-799 measures the t=0
tendencies under DUOGRID vs LEGACY to locate which operator
produces the blowup.

Per iter-792's methodology (t=0 tendency audit for W2 solid-body
rotation where analytical tendencies are all ZERO), iter-799
computes dh/dt, du/dt, dv/dt at t=0 under both paths and reports
peak magnitudes.  A large DUOGRID tendency indicates the broken
operator.

Scope: observational only.
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
from legoesm.core.operators_cdgrid import fv3_sw_tendencies
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def _run_t0(n, use_duogrid):
    div_damp = 8.0 * _div_damp_cube(n)
    grid = create_cubed_sphere(n=n, use_duogrid=use_duogrid)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=div_damp,
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2)
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))

    dh_dt, du_dt, dv_dt = fv3_sw_tendencies(
        sw.h.data, u_d, v_d, sw.h_s.data, cdgrid,
        g=cfg.g, div_damp=cfg.div_damp,
        hyperdiff_coeff=cfg.hyperdiff_coeff,
        boundary_fix=cfg.boundary_fix,
    )

    return {
        'dh_dt_peak': float(np.max(np.abs(np.asarray(dh_dt)))),
        'du_dt_peak': float(np.max(np.abs(np.asarray(du_dt)))),
        'dv_dt_peak': float(np.max(np.abs(np.asarray(dv_dt)))),
    }


n = 36

print(f"Iter-799 DUOGRID vs LEGACY t=0 tendency audit for W2 C36")
print(f"Analytical tendencies for W2 steady state are all ZERO.")
print()
print(f"{'path':>18}  {'dh/dt peak':>12}  {'du/dt peak':>12}  {'dv/dt peak':>12}")
print("-" * 60)

r_legacy = _run_t0(n, use_duogrid=False)
print(f"{'LEGACY':>18}  {r_legacy['dh_dt_peak']:>12.3e}  "
      f"{r_legacy['du_dt_peak']:>12.3e}  {r_legacy['dv_dt_peak']:>12.3e}")

r_duogrid = _run_t0(n, use_duogrid=True)
print(f"{'DUOGRID':>18}  {r_duogrid['dh_dt_peak']:>12.3e}  "
      f"{r_duogrid['du_dt_peak']:>12.3e}  {r_duogrid['dv_dt_peak']:>12.3e}")

print()
for label in ('dh_dt_peak', 'du_dt_peak', 'dv_dt_peak'):
    ratio = r_duogrid[label] / r_legacy[label] if r_legacy[label] > 0 else np.nan
    print(f"  DUOGRID/LEGACY {label}: {ratio:.2f}x")

print()
print("Interpretation cues (observational only):")
print("- If DUOGRID tendency peaks are MUCH larger than LEGACY at")
print("  t=0: the duogrid-dispatched operator chain produces")
print("  immediate spurious tendencies, not accumulation.")
print("- If DUOGRID tendency peaks are similar to LEGACY at t=0:")
print("  the blowup is from time-step accumulation, not the")
print("  instantaneous operator chain.")

print()
print("What iter-799 DOES measure:")
print("- DUOGRID vs LEGACY t=0 peak tendency magnitudes for W2.")
print("What it does NOT establish:")
print("- WHICH specific duogrid-dispatched operator (dgrid_to_cgrid,")
print("  halo exchanges, KE, pressure gradient) produces the peak.")
print("- Whether the observed magnitudes CAUSE iter-787's 1-day")
print("  v_ll_Linf=100 m/s (requires tracking over timesteps).")
