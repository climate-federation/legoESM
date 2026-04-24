"""Iter-808 diagnostic: test whether sign flip at rev-true/polar
edges fixes DUOGRID flux sync.

Per iter-807, 4 shared edges have flux discrepancy rel=2.0
(opposite signs):
  face 1 N ↔ face 4 E (rev=False)
  face 2 S ↔ face 5 S (rev=True)
  face 2 N ↔ face 4 N (rev=True)
  face 3 S ↔ face 5 W (rev=False)

Each of these has one face connected to polar face 4 or 5.  Under
the gnomonic cubed-sphere projection, the local +y direction can
invert between these neighbouring faces across the shared edge, so
fluxes with the same physical meaning have opposite signs in local
coordinates.

iter-808 tests a sign-aware flux sync: for each shared edge,
determine whether the neighbor's flux has opposite sign and apply
a flip BEFORE averaging.  An empirical "needs_sign_flip" lookup is
built from iter-807's data.
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
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from legoesm.core.operators_cdgrid import fv3_sw_tendencies
from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


# Empirical sign-flip table from iter-807 observations.
# For each (face, edge), if True then the neighbor's flux must be
# SIGN-FLIPPED before averaging (because the two faces use opposite
# sign conventions for the shared-edge mass flux).
#
# Observed problematic edges from iter-807 (rel=2.0):
#   face 1 N ↔ face 4 E (rev=False)
#   face 2 S ↔ face 5 S (rev=True)
#   face 2 N ↔ face 4 N (rev=True)
#   face 3 S ↔ face 5 W (rev=False)
NEEDS_SIGN_FLIP = {
    (1, NORTH), (4, EAST),
    (2, SOUTH), (5, SOUTH),
    (2, NORTH), (4, NORTH),
    (3, SOUTH), (5, WEST),
}


def _extract_bdy(fx, fy, face, edge, n):
    if edge == WEST:
        return fx[face, 0, :]
    elif edge == EAST:
        return fx[face, n, :]
    elif edge == SOUTH:
        return fy[face, :, 0]
    else:
        return fy[face, :, n]


def _signed_sync(fx, fy, n):
    """Flux sync with per-edge sign-flip awareness."""
    avgs = {}
    for face in range(6):
        for edge in (WEST, EAST, SOUTH, NORTH):
            nbr_face, nbr_edge, rev = CONNECTIVITY[face][edge]
            local = _extract_bdy(fx, fy, face, edge, n)
            nbr = _extract_bdy(fx, fy, nbr_face, nbr_edge, n)
            if rev:
                nbr = nbr[::-1]
            if (face, edge) in NEEDS_SIGN_FLIP:
                nbr = -nbr
            avgs[(face, edge)] = 0.5 * (local + nbr)

    for face in range(6):
        for edge in (WEST, EAST, SOUTH, NORTH):
            avg = avgs[(face, edge)]
            if edge == WEST:
                fx = fx.at[face, 0, :].set(avg)
            elif edge == EAST:
                fx = fx.at[face, n, :].set(avg)
            elif edge == SOUTH:
                fy = fy.at[face, :, 0].set(avg)
            else:
                fy = fy.at[face, :, n].set(avg)
    return fx, fy


orig_sync = halo_mod.synchronize_cgrid_fluxes


def _measure(label, use_duogrid, sync_mode, n=36):
    if sync_mode == 'default':
        halo_mod.synchronize_cgrid_fluxes = orig_sync
        from legoesm.core import operators_cdgrid as ocd_mod
        if hasattr(ocd_mod, 'synchronize_cgrid_fluxes'):
            ocd_mod.synchronize_cgrid_fluxes = orig_sync
    elif sync_mode == 'off':
        halo_mod.synchronize_cgrid_fluxes = lambda fx, fy, n_: (fx, fy)
        from legoesm.core import operators_cdgrid as ocd_mod
        if hasattr(ocd_mod, 'synchronize_cgrid_fluxes'):
            ocd_mod.synchronize_cgrid_fluxes = lambda fx, fy, n_: (fx, fy)
    elif sync_mode == 'signed':
        halo_mod.synchronize_cgrid_fluxes = _signed_sync
        from legoesm.core import operators_cdgrid as ocd_mod
        if hasattr(ocd_mod, 'synchronize_cgrid_fluxes'):
            ocd_mod.synchronize_cgrid_fluxes = _signed_sync
    else:
        raise ValueError(sync_mode)

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


print(f"Iter-808 sign-aware flux sync test")
print(f"C36 W2 IC, iter-761 config")
print()
print(f"{'variant':>45}  {'dh/dt peak':>11}  {'du/dt peak':>11}  {'dv/dt peak':>11}")
print("-" * 85)

cases = [
    ('LEGACY (sync not applied)', False, 'default'),
    ('DUOGRID (default sync, broken baseline)', True, 'default'),
    ('DUOGRID (sync off)', True, 'off'),
    ('DUOGRID (sign-aware sync)', True, 'signed'),
]

results = {}
for label, use_dg, mode in cases:
    r = _measure(label, use_dg, mode)
    results[label] = r
    print(f"{label:>45}  {r['dh_dt_peak']:>11.3e}  "
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
print("- If sign-aware sync dh/dt ≈ LEGACY: sign-flip is the fix.")
print("- If still too large: the fix is more nuanced (more edges")
print("  need treatment, or sign-flip alone doesn't suffice).")
