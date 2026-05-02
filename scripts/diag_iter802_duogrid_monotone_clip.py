"""Iter-802 diagnostic: test `fill_corner_region(monotone_clip=True)`.

Per iter-801's finding that DUOGRID fill_corner_region overshoots
by 144 m at polar cube corners (causing 1172x dh/dt blowup),
iter-802 tests whether a monotonicity clip recovers sane
behaviour.

The new `monotone_clip` parameter on `fill_corner_region` (added
in iter-802 in src/legoesm/grids/duogrid.py) clips each Lagrange-
extrapolated cube-corner cell to the min/max of its 4 neighbours.
When monotone_clip=True is globally enabled (via monkey-patching
`fill_corner_region` at the module level), we re-run iter-800's
t=0 dh/dt audit to see if the blowup is eliminated.

If dh/dt drops close to LEGACY (~1e-4) with monotone_clip=True,
the clip is a viable production fix.  If it only partially helps
or doesn't help, other duogrid-chain issues remain.

Scope: observational only, no production source change (the
`fill_corner_region(monotone_clip=True)` parameter is added as
OPTIONAL; this diagnostic tests its impact before wiring it on
by default).
"""
import os, sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
for _stale in ("JAX_PLATFORM_NAME", "JAX_DISABLE_JIT", "JAX_DEBUG_NANS"):
    os.environ.pop(_stale, None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functools
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from legoesm.grids import duogrid as duogrid_mod
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel)
from legoesm.core.operators_cdgrid import (
    fv3_d2cc, fv3_cc2c, cgrid_mass_flux_divergence,
    fv3_sw_tendencies)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


orig_fill_corner_region = duogrid_mod.fill_corner_region

def _patched_fill_corner_region(padded, dg, halo):
    return orig_fill_corner_region(padded, dg, halo, monotone_clip=True)


def _measure(label, use_duogrid, clip_on, n=36):
    # Apply or remove the monkey-patch.
    if clip_on:
        duogrid_mod.fill_corner_region = _patched_fill_corner_region
        from legoesm.grids import halo as halo_mod
        halo_mod.fill_corner_region = _patched_fill_corner_region  # type: ignore
    else:
        duogrid_mod.fill_corner_region = orig_fill_corner_region
        from legoesm.grids import halo as halo_mod
        halo_mod.fill_corner_region = orig_fill_corner_region  # type: ignore

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

    # Measure t=0 tendencies.
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


print(f"Iter-802 DUOGRID t=0 tendency under fill_corner_region monotone clip")
print(f"C36 W2 IC, iter-761 config")
print()
print(f"{'variant':>32}  {'dh/dt peak':>11}  {'du/dt peak':>11}  {'dv/dt peak':>11}")
print("-" * 70)

r_legacy = _measure("LEGACY (clip off)",
                     use_duogrid=False, clip_on=False)
print(f"{'LEGACY (clip off)':>32}  "
      f"{r_legacy['dh_dt_peak']:>11.3e}  "
      f"{r_legacy['du_dt_peak']:>11.3e}  "
      f"{r_legacy['dv_dt_peak']:>11.3e}")

r_duogrid_off = _measure("DUOGRID (clip off, baseline)",
                          use_duogrid=True, clip_on=False)
print(f"{'DUOGRID (clip off, baseline)':>32}  "
      f"{r_duogrid_off['dh_dt_peak']:>11.3e}  "
      f"{r_duogrid_off['du_dt_peak']:>11.3e}  "
      f"{r_duogrid_off['dv_dt_peak']:>11.3e}")

r_duogrid_on = _measure("DUOGRID (clip ON)",
                         use_duogrid=True, clip_on=True)
print(f"{'DUOGRID (clip ON)':>32}  "
      f"{r_duogrid_on['dh_dt_peak']:>11.3e}  "
      f"{r_duogrid_on['du_dt_peak']:>11.3e}  "
      f"{r_duogrid_on['dv_dt_peak']:>11.3e}")

print()
print(f"DUOGRID clip ON / clip off ratio:")
for k in ('dh_dt_peak', 'du_dt_peak', 'dv_dt_peak'):
    r = r_duogrid_on[k] / r_duogrid_off[k]
    print(f"  {k}: {r:.3f}x")
print()
print(f"DUOGRID clip ON / LEGACY ratio:")
for k in ('dh_dt_peak', 'du_dt_peak', 'dv_dt_peak'):
    r = r_duogrid_on[k] / r_legacy[k]
    print(f"  {k}: {r:.3f}x")

# Restore original.
duogrid_mod.fill_corner_region = orig_fill_corner_region

print()
print("Interpretation cues (observational only):")
print("- If DUOGRID clip ON dh/dt ≈ LEGACY dh/dt (~1e-4):")
print("  the monotonicity clip FIXES the DUOGRID t=0 blowup.")
print("  Next step: wire monotone_clip=True by default in")
print("  pad_halo → fill_corner_region when duogrid is active.")
print("- If DUOGRID clip ON dh/dt is still >> LEGACY:")
print("  clip doesn't fully fix; other duogrid issues remain.")
