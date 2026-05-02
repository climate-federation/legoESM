"""Iter-804 diagnostic: bypass `cube_rmp_vectorized` under DUOGRID
to test whether the kinked-to-extended remap is the dh/dt blowup
source.

Iter-803 found that forcing averaging fallback in
`fill_corner_region` fixed DUOGRID du/dt and dv/dt to near-LEGACY
levels, but NOT dh/dt (still 1172x).  This points to the other
duogrid halo operator: `cube_rmp_vectorized` (Lagrange remap on
edge-halo cells).

iter-804 tests:
  (a) disable cube_rmp_vectorized entirely under DUOGRID
  (b) disable it AND use averaging-fallback for fill_corner_region
Results tell us which operator actually broke dh/dt.
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
orig_cube_rmp = duogrid_mod.cube_rmp_vectorized


def _measure(label, use_duogrid, disable_rmp, disable_lagrange, n=36):
    if disable_rmp:
        duogrid_mod.cube_rmp_vectorized = lambda p, dg, h: p
    else:
        duogrid_mod.cube_rmp_vectorized = orig_cube_rmp

    if disable_lagrange:
        duogrid_mod.fill_corner_region = duogrid_mod._fill_corner_region_averaging
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


print(f"Iter-804 DUOGRID halo operator decomposition test")
print(f"C36 W2 IC, iter-761 config")
print()
print(f"{'variant':>45}  {'dh/dt peak':>11}  {'du/dt peak':>11}  {'dv/dt peak':>11}")
print("-" * 85)

cases = [
    ('LEGACY', False, False, False),
    ('DUOGRID (full baseline)', True, False, False),
    ('DUOGRID (no Lagrange; rmp ON)', True, False, True),
    ('DUOGRID (Lagrange ON; rmp OFF)', True, True, False),
    ('DUOGRID (no Lagrange; no rmp)', True, True, True),
]

results = {}
for label, use_dg, disable_rmp, disable_lagrange in cases:
    r = _measure(label, use_dg, disable_rmp, disable_lagrange)
    results[label] = r
    print(f"{label:>45}  {r['dh_dt_peak']:>11.3e}  "
          f"{r['du_dt_peak']:>11.3e}  {r['dv_dt_peak']:>11.3e}")

print()
r_L = results['LEGACY']
print(f"Ratios vs LEGACY:")
for label, r in results.items():
    if label == 'LEGACY':
        continue
    dh = r['dh_dt_peak'] / r_L['dh_dt_peak']
    du = r['du_dt_peak'] / r_L['du_dt_peak']
    dv = r['dv_dt_peak'] / r_L['dv_dt_peak']
    print(f"  {label:>45}: dh {dh:.2f}x, du {du:.2f}x, dv {dv:.2f}x")

# Restore.
duogrid_mod.fill_corner_region = orig_fill_corner_region
duogrid_mod.cube_rmp_vectorized = orig_cube_rmp

print()
print("Interpretation cues (observational only):")
print("- If 'no rmp' drops dh/dt to LEGACY level: cube_rmp_vectorized")
print("  is the dh/dt culprit (kinked→extended remap breaks h halo).")
print("- If 'no rmp' and 'no Lagrange' both fail to fix dh/dt:")
print("  the issue is outside fill_corner_region and cube_rmp, maybe")
print("  in the initial `_pad_halo_local_h2` copy or in PPM itself.")
