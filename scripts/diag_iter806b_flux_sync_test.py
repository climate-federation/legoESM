"""Iter-806b diagnostic: test whether `synchronize_cgrid_fluxes`
is the DUOGRID dh/dt blowup source.

Iter-806 found only rsin_u/rsin_v metric fields differ between
LEGACY and DUOGRID cdgrids, and those aren't used in
fv3_sw_tendencies.  A further inspection of
cgrid_mass_flux_divergence (operators_cdgrid.py:537-540) shows
there IS a duogrid-specific branch: when dg.ng >= 2, it applies
`synchronize_cgrid_fluxes(flux_x, flux_y, n)` to average boundary
fluxes across adjacent faces.  This is the Ralph loop's critical
duogrid constraint #1.

iter-806b tests: monkey-patch `synchronize_cgrid_fluxes` to a no-
op and measure DUOGRID t=0 dh/dt.  If dh/dt drops to LEGACY level,
the flux sync is broken (or its interaction with our PPM path is).

Note: if the flux sync IS the cause, we can't just disable it —
Fortran-faithfulness requires it.  But if we can confirm the
cause, we can audit the implementation for bugs.
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

from legoesm.grids import halo as halo_mod
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel)
from legoesm.core.operators_cdgrid import fv3_sw_tendencies
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


orig_sync = halo_mod.synchronize_cgrid_fluxes


def _measure(label, use_duogrid, disable_sync, n=36):
    if disable_sync:
        halo_mod.synchronize_cgrid_fluxes = lambda fx, fy, n_: (fx, fy)
        from legoesm.core import operators_cdgrid as ocd_mod
        if hasattr(ocd_mod, 'synchronize_cgrid_fluxes'):
            ocd_mod.synchronize_cgrid_fluxes = lambda fx, fy, n_: (fx, fy)
    else:
        halo_mod.synchronize_cgrid_fluxes = orig_sync
        from legoesm.core import operators_cdgrid as ocd_mod
        if hasattr(ocd_mod, 'synchronize_cgrid_fluxes'):
            ocd_mod.synchronize_cgrid_fluxes = orig_sync

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


print(f"Iter-806b synchronize_cgrid_fluxes disable test")
print(f"C36 W2 IC, iter-761 config")
print()
print(f"{'variant':>42}  {'dh/dt peak':>11}  {'du/dt peak':>11}  {'dv/dt peak':>11}")
print("-" * 85)

cases = [
    ('LEGACY (sync not applied)', False, False),
    ('DUOGRID (baseline, sync ON)', True, False),
    ('DUOGRID (sync DISABLED)', True, True),
]

results = {}
for label, use_dg, dis_sync in cases:
    r = _measure(label, use_dg, dis_sync)
    results[label] = r
    print(f"{label:>42}  {r['dh_dt_peak']:>11.3e}  "
          f"{r['du_dt_peak']:>11.3e}  {r['dv_dt_peak']:>11.3e}")

# Restore.
halo_mod.synchronize_cgrid_fluxes = orig_sync
from legoesm.core import operators_cdgrid as ocd_mod
if hasattr(ocd_mod, 'synchronize_cgrid_fluxes'):
    ocd_mod.synchronize_cgrid_fluxes = orig_sync

print()
r_L = results['LEGACY (sync not applied)']
for label, r in results.items():
    if 'LEGACY' in label:
        continue
    dh = r['dh_dt_peak'] / r_L['dh_dt_peak']
    du = r['du_dt_peak'] / r_L['du_dt_peak']
    dv = r['dv_dt_peak'] / r_L['dv_dt_peak']
    print(f"  {label}: dh {dh:.2f}x, du {du:.2f}x, dv {dv:.2f}x")

print()
print("Interpretation cues (observational only):")
print("- If DUOGRID (sync DISABLED) dh/dt ≈ LEGACY: sync IS the")
print("  culprit; audit synchronize_cgrid_fluxes for bugs.")
print("- If DUOGRID (sync DISABLED) dh/dt still ≫ LEGACY: sync is")
print("  not the issue; the blowup is elsewhere.")
