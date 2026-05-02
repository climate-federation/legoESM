"""Iter-803 diagnostic: force `_fill_corner_region_averaging`
(the LEGACY-like 2-pt averaging fallback) under DUOGRID and test
whether DUOGRID dh/dt blowup is fixed.

Iter-802's monotone_clip approach was ineffective for dh/dt due
to neighbour-set design flaw.  iter-803 tries a simpler approach:
monkey-patch `fill_corner_region` to always return the averaging
fallback output, bypassing Lagrange extrapolation entirely.  This
matches what LEGACY does via `_fill_corners_h2`.

If DUOGRID dh/dt drops close to LEGACY (~1e-4) with averaging
only: the Lagrange extrapolation IS the root cause; fix is to
switch duogrid corner fill to averaging.
If unchanged: the issue is elsewhere in the duogrid chain.
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

from legoesm.grids import duogrid as duogrid_mod
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel)
from legoesm.core.operators_cdgrid import fv3_sw_tendencies
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


orig_fill_corner_region = duogrid_mod.fill_corner_region


def _measure(label, use_duogrid, use_averaging_fallback, n=36):
    # Apply or remove the monkey-patch to bypass Lagrange.
    if use_averaging_fallback:
        duogrid_mod.fill_corner_region = (
            duogrid_mod._fill_corner_region_averaging)
    else:
        duogrid_mod.fill_corner_region = orig_fill_corner_region

    grid = create_cubed_sphere(n=n, use_duogrid=use_duogrid)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
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


print(f"Iter-803 force averaging fallback test for DUOGRID fill_corner_region")
print(f"C36 W2 IC, iter-761 config")
print()
print(f"{'variant':>40}  {'dh/dt peak':>11}  {'du/dt peak':>11}  {'dv/dt peak':>11}")
print("-" * 80)

r_legacy = _measure("LEGACY (use_duogrid=False)",
                     use_duogrid=False, use_averaging_fallback=False)
print(f"{'LEGACY (use_duogrid=False)':>40}  "
      f"{r_legacy['dh_dt_peak']:>11.3e}  "
      f"{r_legacy['du_dt_peak']:>11.3e}  "
      f"{r_legacy['dv_dt_peak']:>11.3e}")

r_duogrid = _measure("DUOGRID (Lagrange, baseline)",
                      use_duogrid=True, use_averaging_fallback=False)
print(f"{'DUOGRID (Lagrange, baseline)':>40}  "
      f"{r_duogrid['dh_dt_peak']:>11.3e}  "
      f"{r_duogrid['du_dt_peak']:>11.3e}  "
      f"{r_duogrid['dv_dt_peak']:>11.3e}")

r_duogrid_avg = _measure("DUOGRID (averaging fallback)",
                          use_duogrid=True, use_averaging_fallback=True)
print(f"{'DUOGRID (averaging fallback)':>40}  "
      f"{r_duogrid_avg['dh_dt_peak']:>11.3e}  "
      f"{r_duogrid_avg['du_dt_peak']:>11.3e}  "
      f"{r_duogrid_avg['dv_dt_peak']:>11.3e}")

print()
print(f"Ratios vs LEGACY:")
for k in ('dh_dt_peak', 'du_dt_peak', 'dv_dt_peak'):
    r_d = r_duogrid[k] / r_legacy[k]
    r_da = r_duogrid_avg[k] / r_legacy[k]
    print(f"  {k}: DUOGRID Lagrange {r_d:.2f}x, DUOGRID averaging {r_da:.2f}x")

# Restore.
duogrid_mod.fill_corner_region = orig_fill_corner_region

print()
print("Interpretation cues (observational only):")
print("- If averaging fallback dh/dt ≈ LEGACY: Lagrange fill is the culprit.")
print("- If averaging fallback dh/dt is still ≫ LEGACY: Lagrange fill is")
print("  not the (only) issue.  Check `cube_rmp_vectorized` or edge-halo.")
