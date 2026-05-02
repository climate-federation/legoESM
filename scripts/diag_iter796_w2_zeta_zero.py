"""Iter-796 diagnostic: W2 with zeta=0 (analytical) test.

Per iter-795's refutation of B-halo inconsistency as the W2 mode-
A mechanism, iter-796 tests the next candidate: the corner-wind
interpolation → relative vorticity (zeta) → Coriolis term.

For steady-state solid-body rotation, analytical zeta = 0.  Our
numerical zeta has small peak (1.2e-5 from iter-793) due to
corner-wind interpolation noise.  iter-796 replaces numerical
zeta with ZERO and measures the dv_cc residual impact.

If zeta→0 substantially reduces dv_cc residual: the corner-wind
chain is the mechanism; fix by improving corner-wind halo treatment.
If unchanged or larger: zeta noise is not the W2 mode-A source.

Scope: observational only, t=0 diagnostic.
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
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from legoesm.core.operators_cdgrid import (
    fv3_d2cc, fv3_cc2c, _arakawa_lamb_gradient, _interp_corner_to_center,
    dgrid_vorticity)
from legoesm.grids.halo import pad_halo_vector
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


CUBE_VERTEX_LAT = np.arcsin(1.0 / np.sqrt(3.0))
CUBE_VERTEX_LATS_R = np.array([CUBE_VERTEX_LAT, -CUBE_VERTEX_LAT])
CUBE_VERTEX_LONS_R = np.deg2rad(np.array([45.0, 135.0, -45.0, -135.0]))


def _gc_dist_to_nearest_cube_vertex_deg(lat_deg, lon_deg):
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


n = 36
grid = create_cubed_sphere(n=n, use_duogrid=False)
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
h = sw.h.data
h_s = sw.h_s.data
g = cfg.g

u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)
B = 0.5 * (u_cc ** 2 + v_cc ** 2) + g * (h + h_s)
dB_dx, dB_dy_perp = _arakawa_lamb_gradient(B, cdgrid)
dB_dx_cc = _interp_corner_to_center(dB_dx)
dB_dy_cc = _interp_corner_to_center(dB_dy_perp)

# Corner winds / zeta (default).
u_cc_pad, v_cc_pad = pad_halo_vector(
    u_cc, v_cc,
    grid.cos_angle, grid.sin_angle,
    grid.cos_angle_padded, grid.sin_angle_padded,
    interp_offsets=grid.halo_interp_offsets, duogrid=None,
)
u_corner = 0.25 * (u_cc_pad[:, :-1, :-1] + u_cc_pad[:, 1:, :-1]
                    + u_cc_pad[:, :-1, 1:] + u_cc_pad[:, 1:, 1:])
v_corner = 0.25 * (v_cc_pad[:, :-1, :-1] + v_cc_pad[:, 1:, :-1]
                    + v_cc_pad[:, :-1, 1:] + v_cc_pad[:, 1:, 1:])
zeta = dgrid_vorticity(u_corner, v_corner, cdgrid)
zeta_abs_default = zeta + cdgrid.base.f
zeta_abs_zeroed = jnp.zeros_like(zeta) + cdgrid.base.f

# Two dv_cc variants:
dv_default = -zeta_abs_default * u_cc - dB_dy_cc
dv_zeta_zero = -zeta_abs_zeroed * u_cc - dB_dy_cc

lat_cc = np.rad2deg(np.asarray(grid.lat))
lon_cc = np.rad2deg(np.asarray(grid.lon))
lon_cc = np.where(lon_cc > 180.0, lon_cc - 360.0, lon_cc)


def _report(label, dv):
    a = np.asarray(np.abs(dv))
    idx = np.unravel_index(np.argmax(a), a.shape)
    face, ci, cj = int(idx[0]), int(idx[1]), int(idx[2])
    peak = float(a[face, ci, cj])
    latd = float(lat_cc[face, ci, cj])
    lond = float(lon_cc[face, ci, cj])
    gc = _gc_dist_to_nearest_cube_vertex_deg(latd, lond)
    m = float(np.mean(a))
    hot = a > 0.5 * peak
    n_hot = int(np.sum(hot))
    n_near = 0
    if n_hot > 0:
        hot_lat = lat_cc[hot]
        hot_lon = lon_cc[hot]
        for i in range(n_hot):
            gi = _gc_dist_to_nearest_cube_vertex_deg(float(hot_lat[i]),
                                                      float(hot_lon[i]))
            if gi < 10.0:
                n_near += 1
    pct = 100.0 * n_near / max(n_hot, 1)
    print(f"  {label:>22}  peak={peak:.3e}  ({face},{ci},{cj})  "
          f"lat={latd:+.2f}° lon={lond:+.2f}°  GC={gc:.2f}°  "
          f"|mean|={m:.3e}  hot={n_hot} near_vert={n_near} ({pct:.1f}%)")


print(f"Iter-796 W2 dv_cc residual with zeta replaced by ZERO")
print(f"C36, β=0, LEGACY path, no div damp applied (isolate Coriolis+pressure)")
print()
print(f"{'variant':>22}  {'peak':>10}  {'face,(i,j)':>15}  "
      f"{'GC':>5}  {'|mean|':>10}  {'hot':>5}  {'near_vert':>15}")
print("-" * 115)
_report("default (numerical zeta)", dv_default)
_report("zeta = 0 (analytical)", dv_zeta_zero)

diff = np.asarray(dv_default - dv_zeta_zero)
print()
print(f"max |dv_default - dv_zeta_zero| = {float(np.max(np.abs(diff))):.3e}")
print(f"This equals max |zeta * u_cc| = {float(np.max(np.abs(np.asarray(zeta) * np.asarray(u_cc)))):.3e}")

print()
print("Interpretation cues (observational only):")
print("- If zeta→0 makes dv_zeta_zero MUCH SMALLER than dv_default:")
print("  zeta noise is the cube-vertex mechanism; fix corner-wind halo.")
print("- If dv_zeta_zero is SIMILAR to dv_default:")
print("  zeta noise is not the main contributor.")

print()
print("What iter-796 DOES measure (observational only):")
print("- dv_cc residual at t=0 with numerical zeta vs analytical (zero) zeta.")
print("What it does NOT establish:")
print("- 1-day v_ll_Linf impact of an improved zeta operator.")
