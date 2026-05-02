"""Iter-793 diagnostic: decompose W2 t=0 dv_cc tendency into
its constituent terms.

Per iter-792's finding that dv/dt at t=0 peaks 4.34° GC from a
cube vertex with 41.5% hot cells near vertices, iter-793 computes
each contribution to dv_cc separately and reports its peak
location + cube-vertex concentration:

  dv_cc_total = -zeta_abs * u_cc - dB_dy_cc
                [+ adaptive_coeff * ddiv_dy_perp_cc if div_damp>0]
                [- hyperdiff_coeff * bilap_v_local if hyperdiff>0]
                [+ boundary_fix smoothing]

For W2 steady state, each analytical component should individually
be small, and the SUM should be exactly zero (geostrophic balance).
Numerically, each term is computed with truncation error, and the
cancellation leaves a residual.  The residual's cube-vertex
localisation must come from one or more of the constituent terms.

Scope: observational only.  No source-code change, no new sentinel.
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
    fv3_d2cc, fv3_cc2c, cgrid_divergence, cgrid_mass_flux_divergence,
    dgrid_vorticity, _arakawa_lamb_gradient, _interp_corner_to_center)
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
div_damp = 8.0 * _div_damp_cube(n)

grid = create_cubed_sphere(n=n, use_duogrid=False)
cfg = CDGridShallowWaterConfig(
    hyperdiff_coeff=0.0,
    div_damp=div_damp,
    boundary_fix=True,  # matches iter-761
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

# Replicate fv3_sw_tendencies internals (minus boundary_fix) to
# extract each term.

# (a) cell-centre + C-grid velocities
u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)
u_c, v_c = fv3_cc2c(u_cc, v_cc, cdgrid)

# (c) Bernoulli
KE = 0.5 * (u_cc ** 2 + v_cc ** 2)
B = KE + g * (h + h_s)
B_geopot_only = g * (h + h_s)  # analogue without KE for decomposition

# (d) A-L gradient of B (and of geopotential only for decomposition)
dB_dx, dB_dy_perp = _arakawa_lamb_gradient(B, cdgrid)
dBg_dx, dBg_dy_perp = _arakawa_lamb_gradient(B_geopot_only, cdgrid)
dKE_dx = dB_dx - dBg_dx
dKE_dy_perp = dB_dy_perp - dBg_dy_perp

# (e) corner winds
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
zeta_abs = zeta + cdgrid.base.f

# (g) decomposition terms
dB_dx_cc = _interp_corner_to_center(dB_dx)
dB_dy_cc = _interp_corner_to_center(dB_dy_perp)
dBg_dx_cc = _interp_corner_to_center(dBg_dx)
dBg_dy_cc = _interp_corner_to_center(dBg_dy_perp)
dKE_dx_cc = _interp_corner_to_center(dKE_dx)
dKE_dy_cc = _interp_corner_to_center(dKE_dy_perp)

# For dv_cc: v-eqn terms
dv_coriolis = -zeta_abs * u_cc
dv_pressure = -dBg_dy_cc
dv_ke_grad = -dKE_dy_cc
dv_balance = dv_coriolis + dv_pressure  # should be ~0 for geostrophic
dv_total_nodamp = -zeta_abs * u_cc - dB_dy_cc  # sum before div damping

# Divergence damping dv term
div_field = cgrid_divergence(u_c, v_c, cdgrid)
da_min_c = jnp.min(cdgrid.area_corner)
d2_bg = div_damp / da_min_c
dddmp = 0.2
div_abs = jnp.abs(div_field)
adaptive_coeff = da_min_c * jnp.maximum(
    d2_bg, jnp.minimum(0.20, dddmp * div_abs))
ddiv_dx, ddiv_dy_perp = _arakawa_lamb_gradient(div_field, cdgrid)
dv_div_damp = adaptive_coeff * _interp_corner_to_center(ddiv_dy_perp)

dv_total = dv_total_nodamp + dv_div_damp

lat_cc = np.rad2deg(np.asarray(grid.lat))
lon_cc = np.rad2deg(np.asarray(grid.lon))
lon_cc = np.where(lon_cc > 180.0, lon_cc - 360.0, lon_cc)


def _report(label, field):
    f = np.asarray(field)
    a = np.abs(f)
    idx = np.unravel_index(np.argmax(a), a.shape)
    face, ci, cj = int(idx[0]), int(idx[1]), int(idx[2])
    peak = float(a[face, ci, cj])
    latd = float(lat_cc[face, ci, cj])
    lond = float(lon_cc[face, ci, cj])
    gc = _gc_dist_to_nearest_cube_vertex_deg(latd, lond)
    m = float(np.mean(a))
    # hot cells
    hot = a > 0.5 * peak
    n_hot = int(np.sum(hot))
    n_near = 0
    if n_hot > 0:
        hot_lat = lat_cc[hot]
        hot_lon = lon_cc[hot]
        for i in range(n_hot):
            g = _gc_dist_to_nearest_cube_vertex_deg(float(hot_lat[i]),
                                                     float(hot_lon[i]))
            if g < 10.0:
                n_near += 1
    pct = 100.0 * n_near / max(n_hot, 1)
    print(f"  {label:>25}  peak={peak:.3e}  "
          f"({face},{ci},{cj})  lat={latd:+6.2f}°  lon={lond:+7.2f}°  "
          f"GC={gc:5.2f}°  |mean|={m:.3e}  hot={n_hot:4d}  "
          f"near_vert={n_near:4d} ({pct:5.1f}%)")


print(f"Iter-793 W2 t=0 dv_cc decomposition at C36 (LEGACY)")
print(f"For steady-state W2, analytical total dv_cc = 0 (geostrophic balance).")
print(f"Each constituent term is computed separately to locate the cube-vertex source.")
print()
print(f"{'term':>25}  {'peak':>10}  {'face,(i,j)':>14}  "
      f"{'lat':>8}  {'lon':>9}  {'GC':>5}  {'|mean|':>10}  "
      f"{'hot':>5}  {'near_vert':>13}")
print("-" * 130)
_report("dv total (with div damp)", dv_total)
_report("dv total (NO div damp)", dv_total_nodamp)
_report("dv Coriolis (-zeta*u)", dv_coriolis)
_report("dv pressure (-grad_y g h)", dv_pressure)
_report("dv KE-grad (-grad_y KE)", dv_ke_grad)
_report("dv balance (Cor+press)", dv_balance)
_report("dv div-damp", dv_div_damp)
_report("zeta (relative vort)", zeta)
_report("zeta_abs (=zeta+f)", zeta_abs)

print()
print("Interpretation cues (observational only):")
print("- `dv balance` should be the dominant error source for the ")
print("  total tendency, since Cor + press gradient is supposed to ")
print("  cancel for geostrophic W2.  If dv_balance has the same")
print("  cube-vertex concentration as dv_total, the imbalance is the ")
print("  driver.")
print("- `dv_ke_grad` should be near-zero analytically (KE is")
print("  smooth for solid-body) but not numerically.  Its cube-")
print("  vertex concentration indicates KE-gradient truncation bias.")
print("- `dv_div_damp` magnitude shows whether damping is")
print("  transiently triggered at cube vertices.")
print("- If `zeta` (relative vort) is large at cube vertices, the ")
print("  corner-wind halo interpolation + dgrid_vorticity is biased.")

print()
print("What iter-793 DOES measure (observational only):")
print("- Peak values and cube-vertex concentration of each dv_cc")
print("  contribution term at t=0 on W2 IC.")
print("What it does NOT establish:")
print("- A repair path for whichever term dominates.  A Fortran-")
print("  faithful replacement would be iter-794+ material.")
print("- The impact of each term on the FULL W2 1-day run's")
print("  v_ll_Linf (accumulation is nonlinear).")
