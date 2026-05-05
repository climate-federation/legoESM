"""Iter-800 diagnostic: DUOGRID u_c, v_c divergence vs LEGACY at t=0.

Per iter-799's finding that DUOGRID dh/dt at t=0 is 1172x LEGACY,
iter-800 measures u_c, v_c output from `fv3_cc2c` under both
paths and computes the divergence = cgrid_divergence(u_c, v_c).

For W2 steady-state solid-body rotation, analytical divergence = 0.
Both paths should give near-zero divergence numerically if correctly
implemented.  A large DUOGRID divergence at cube vertices confirms
the halo path is broken.
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
from legoesm.core.operators_cdgrid import (
    fv3_d2cc, fv3_cc2c, cgrid_divergence, cgrid_mass_flux_divergence)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


CUBE_VERTEX_LAT = np.arcsin(1.0 / np.sqrt(3.0))
CUBE_VERTEX_LATS_R = np.array([CUBE_VERTEX_LAT, -CUBE_VERTEX_LAT])
CUBE_VERTEX_LONS_R = np.deg2rad(np.array([45.0, 135.0, -45.0, -135.0]))


def _gc_dist(lat_deg, lon_deg):
    lat_r = np.deg2rad(lat_deg)
    lon_r = np.deg2rad(lon_deg)
    min_d = np.inf
    for vlat in CUBE_VERTEX_LATS_R:
        for vlon in CUBE_VERTEX_LONS_R:
            dlat = lat_r - vlat
            dlon = lon_r - vlon
            a = (np.sin(dlat / 2) ** 2
                 + np.cos(lat_r) * np.cos(vlat) * np.sin(dlon / 2) ** 2)
            d = 2.0 * np.arcsin(np.sqrt(np.maximum(a, 0.0)))
            min_d = min(min_d, np.rad2deg(d))
    return min_d


def _run(use_duogrid, n):
    grid = create_cubed_sphere(n=n, use_duogrid=use_duogrid)
    model = FV3EdgeShallowWaterModel(grid)
    cdgrid = model.cdgrid
    # Analytical W2 IC at edge-midpoint D-grid.
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))

    u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)
    u_c, v_c = fv3_cc2c(u_cc, v_cc, cdgrid)
    div = cgrid_divergence(u_c, v_c, cdgrid)
    sw = williamson_test2(grid)
    h = sw.h.data
    mass_div = cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid)

    return {
        'u_c': np.asarray(u_c),
        'v_c': np.asarray(v_c),
        'div': np.asarray(div),
        'mass_div': np.asarray(mass_div),
        'cdgrid': cdgrid,
        'grid': grid,
    }


n = 36

print(f"Iter-800 DUOGRID vs LEGACY u_c, v_c, divergence at t=0 for W2")
print(f"W2 solid-body rotation: analytical divergence = 0 everywhere.")
print()

r_legacy = _run(use_duogrid=False, n=n)
r_duogrid = _run(use_duogrid=True, n=n)

lat_cc = np.rad2deg(np.asarray(r_legacy['grid'].lat))
lon_cc = np.rad2deg(np.asarray(r_legacy['grid'].lon))
lon_cc = np.where(lon_cc > 180.0, lon_cc - 360.0, lon_cc)


def _report(label, field, lat, lon):
    a = np.abs(field)
    idx = np.unravel_index(np.argmax(a), a.shape)
    face, ci, cj = int(idx[0]), int(idx[1]), int(idx[2])
    peak = float(a[face, ci, cj])
    latd = float(lat[face, ci, cj])
    lond = float(lon[face, ci, cj])
    gc = _gc_dist(latd, lond)
    m = float(np.mean(a))
    print(f"  {label:>25}  peak={peak:.3e}  face={face}  ({ci},{cj})  "
          f"lat={latd:+.2f}° lon={lond:+.2f}°  GC={gc:.2f}°  "
          f"|mean|={m:.3e}")


for label, r in (('LEGACY', r_legacy), ('DUOGRID', r_duogrid)):
    print(f"{label}:")
    _report(f"{label} u_c", r['u_c'], lat_cc, lon_cc[:, :, :-1])
    # Note: u_c is at (6, n+1, n); use lat at (6, n+1, n) which is
    # lat_edge_y.  Use the cdgrid fields:
    cdg = r['cdgrid']
    lat_uc = np.rad2deg(np.asarray(cdg.lat_edge_y))
    lon_uc = np.rad2deg(np.asarray(cdg.lon_edge_y))
    lon_uc = np.where(lon_uc > 180.0, lon_uc - 360.0, lon_uc)
    _report(f"{label} u_c (re-loc)", r['u_c'], lat_uc, lon_uc)
    lat_vc = np.rad2deg(np.asarray(cdg.lat_edge_x))
    lon_vc = np.rad2deg(np.asarray(cdg.lon_edge_x))
    lon_vc = np.where(lon_vc > 180.0, lon_vc - 360.0, lon_vc)
    _report(f"{label} v_c", r['v_c'], lat_vc, lon_vc)
    _report(f"{label} div (∇·u_c)", r['div'], lat_cc, lon_cc)
    _report(f"{label} mass_div (dh/dt)", r['mass_div'], lat_cc, lon_cc)
    print()

ratio_div = float(np.max(np.abs(r_duogrid['div']))
                   / np.max(np.abs(r_legacy['div'])))
print(f"Divergence ratio (DUOGRID/LEGACY) = {ratio_div:.2f}x")

print()
print("Interpretation cues (observational only):")
print("- If DUOGRID div peaks at cube vertices with large magnitude:")
print("  the `pad_halo_vector(duogrid=dg)` or `fv3_cc2c` duogrid")
print("  dispatch is broken at cube corners.")
print("- If DUOGRID u_c or v_c has obviously wrong values at cube")
print("  vertices (e.g. much larger than analytical ~40 m/s):")
print("  confirms the halo exchange is the culprit.")
