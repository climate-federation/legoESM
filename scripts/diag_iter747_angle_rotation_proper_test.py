"""Iter-747 diagnostic: PROPERLY test whether the angle-rotation
pipeline injects a spurious hyperdiff tendency at face 4 near-pole
cells.

Iter-746b's constant-field test was flagged as tautological: passing
a constant through the same forward+inverse rotation gives the same
constant by trigonometric identity, so the pipeline trivially
preserves it.  That says NOTHING about the pipeline's behaviour on a
non-trivial field.

Iter-747 runs TWO proper tests:

(A) **W2 exact IC through the hyperdiff pipeline.**  The W2 exact
    solution has `u_east = u_0 * cos(lat)`, `v_north = 0`.  The
    LAPLACIAN of these analytic fields on a sphere is known:
      ∇²(u_0*cos(lat)) = -u_0*cos(lat)/a² * 2
      ∇²(0) = 0
    The BILAPLACIAN is
      ∇⁴(u_0*cos(lat)) = u_0*cos(lat) * (2/a²)²
      ∇⁴(0) = 0
    The hyperdiff tendency should scale ~uniformly with cos(lat).  A
    Fortran-faithful pipeline applied to the W2 IC should give a
    SMOOTH, latitude-dependent, cube-symmetric hyperdiff tendency.
    Any cube-face or polar signature is an ARTIFACT of the pipeline.

(B) **Stepped simulation state at t=1 d.**  The actual simulation
    state has accumulated error.  Apply the hyperdiff pipeline at
    t=1d and measure where the largest contributions come from per
    face.

Compare each face's max |hyperdiff contribution to du_cc and dv_cc|
against the (smooth expected, constant-cos(lat)) baseline.  If face
4/5 polar interior has contributions that EXCEED the equatorial
cos(lat)-scaled baseline significantly, the pipeline IS injecting a
polar-specific artifact.

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/diag_iter747_angle_rotation_proper_test.py
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import (
    create_cubed_sphere_cdgrid, cell_centre_angles_from_4edge)
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    FV3EdgeShallowWaterModel, FV3EdgeShallowWaterState,
    CDGridShallowWaterConfig)
from tests.test_cases.williamson import williamson_test2
from scripts.run_atmosphere_test_matrix import (
    _hyperdiff_cube, _div_damp_cube)
from legoesm.core.operators import laplacian_compact


n = 36
grid = create_cubed_sphere(n)
cdgrid = create_cubed_sphere_cdgrid(grid)

lat = np.asarray(grid.lat, dtype=np.float64)  # radians
lat_deg = np.degrees(lat)
lon_deg = np.asarray(grid.lon, dtype=np.float64) * 180 / np.pi
angle = np.asarray(grid.angle, dtype=np.float64)
radius = float(grid.radius)
hyp_coeff = _hyperdiff_cube(n)

print("=== Iter-747 angle-rotation proper test (non-tautological) ===\n")

# ----------------------------------------------------------------------
# TEST A: W2 exact IC through the hyperdiff pipeline.
# ----------------------------------------------------------------------
print("--- Test A: W2 exact IC through the pipeline ---\n")

u0 = 2.0 * np.pi * radius / (12.0 * 86400.0)
u_east_exact = u0 * np.cos(lat)      # (6, n, n) — the W2 IC in geographic
v_north_exact = np.zeros_like(lat)

# Rotate into face-local (what u_cc, v_cc would be at t=0 in the
# production code).
cos_a = np.cos(angle)
sin_a = np.sin(angle)
u_cc = cos_a * u_east_exact + sin_a * v_north_exact
v_cc = -sin_a * u_east_exact + cos_a * v_north_exact

# Now run the EXACT hyperdiff pipeline.
u_cc_jx = jnp.asarray(u_cc)
v_cc_jx = jnp.asarray(v_cc)
cos_a_jx = jnp.asarray(cos_a)
sin_a_jx = jnp.asarray(sin_a)

ue_cc_jx = cos_a_jx * u_cc_jx - sin_a_jx * v_cc_jx   # back to u_east
vn_cc_jx = sin_a_jx * u_cc_jx + cos_a_jx * v_cc_jx   # back to v_north

# Round-trip check.
print(f"  Round-trip check: max|ue_cc - u_east_exact| = "
      f"{float(jnp.max(jnp.abs(ue_cc_jx - u_east_exact))):.3e}")
print(f"  Round-trip check: max|vn_cc - v_north_exact| = "
      f"{float(jnp.max(jnp.abs(vn_cc_jx - v_north_exact))):.3e}")

# Apply laplacian_compact and bilaplacian (matches the production
# fv3_sw_tendencies hyperdiff block).
lap_ue = np.asarray(laplacian_compact(ue_cc_jx, grid))
bilap_ue = np.asarray(laplacian_compact(jnp.asarray(lap_ue), grid))
lap_vn = np.asarray(laplacian_compact(vn_cc_jx, grid))
bilap_vn = np.asarray(laplacian_compact(jnp.asarray(lap_vn), grid))

# Expected analytic values for the smooth field u_east = u_0*cos(lat):
#   ∇²(u_0 cos(lat)) = -u_0 cos(lat) / a^2   [NOT 2* — spherical Laplacian on cos(lat) is
#                                             specifically (1/cos(lat))*d/dlat(cos(lat)*d cos(lat)/dlat)
#                                             = (1/cos)*d/dlat(-sin*cos) = (1/cos)*(-cos^2+sin^2)/.. hmm let me redo]
# Let f = cos(lat).  ∇²f = (1/cos(lat)) d/dlat(cos(lat) * df/dlat) / a^2
#                        = (1/cos(lat)) d/dlat(cos(lat)*(-sin(lat))) / a^2
#                        = (1/cos(lat)) * (-cos^2(lat) + sin^2(lat)) / a^2
#                        = (sin²(lat) - cos²(lat)) / (a² cos(lat))
# At lat=0: (0-1)/1 = -1/a².  At lat=86° (cos(lat)=0.07), it blows up.
# This is because cos(lat) is a zonal harmonic that doesn't vanish at the pole, and
# its Laplacian has a singularity there (the Y_1^0 mode does NOT have this
# problem — I may be conflating).
# Actually cos(lat) is NOT a spherical harmonic; sin(lat) is (Y_1^0).  For the W2
# IC u_east = u_0*cos(lat), the geographic u itself does NOT satisfy ∇² ∝ f because
# u_east is NOT a scalar — it's the east component of a vector.  On a sphere the
# vector Laplacian has different formulae than the scalar Laplacian.
# So the "expected Laplacian" is not straightforward.  We use the Laplacian as
# computed by `laplacian_compact`, which is a FD cell-centre Laplacian.  What we
# CAN compare is the per-face pattern.

print(f"\n  Per-face max |bilap_ue| (bilaplacian of u_east for W2 IC):")
for f in range(6):
    val = np.abs(bilap_ue[f]).max()
    argmax = np.argmax(np.abs(bilap_ue[f]))
    i, j = np.unravel_index(argmax, bilap_ue[f].shape)
    print(f"    face{f}: max={val:.3e}  at (i={i},j={j}), "
          f"lat={lat_deg[f,i,j]:.1f}°, lon={lon_deg[f,i,j]:.1f}°")

print(f"\n  Per-face max |bilap_vn| (bilaplacian of v_north=0 for W2 IC):")
for f in range(6):
    val = np.abs(bilap_vn[f]).max()
    argmax = np.argmax(np.abs(bilap_vn[f]))
    i, j = np.unravel_index(argmax, bilap_vn[f].shape)
    print(f"    face{f}: max={val:.3e}  at (i={i},j={j}), "
          f"lat={lat_deg[f,i,j]:.1f}°, lon={lon_deg[f,i,j]:.1f}°")

# Rotate bilap back to face-local (the actual hyperdiff contribution).
bilap_u_local = cos_a * bilap_ue + sin_a * bilap_vn
bilap_v_local = -sin_a * bilap_ue + cos_a * bilap_vn

# Compute spurious tendency.
print(f"\n  Hyperdiff tendency (−hyp_coeff * bilap_*_local) on W2 IC:")
hyp_du = -hyp_coeff * bilap_u_local
hyp_dv = -hyp_coeff * bilap_v_local
print(f"  Global max|hyp_du|: {np.abs(hyp_du).max():.3e} m/s/s")
print(f"  Global max|hyp_dv|: {np.abs(hyp_dv).max():.3e} m/s/s")
for f in range(6):
    du_max = np.abs(hyp_du[f]).max()
    dv_max = np.abs(hyp_dv[f]).max()
    i = np.argmax(np.abs(hyp_dv[f]))
    ii, jj = np.unravel_index(i, hyp_dv[f].shape)
    print(f"    face{f}: max|hyp_du|={du_max:.3e}  max|hyp_dv|={dv_max:.3e}"
          f"  (dv argmax at (i={ii},j={jj}), lat={lat_deg[f,ii,jj]:.1f}°)")

# ----------------------------------------------------------------------
# TEST B: Stepped simulation state at t=1 d.
# ----------------------------------------------------------------------
print("\n--- Test B: hyperdiff tendency on actual W2 state at t=1 d ---\n")

model = FV3EdgeShallowWaterModel(grid, CDGridShallowWaterConfig(
    hyperdiff_coeff=hyp_coeff,
    div_damp=_div_damp_cube(n),
    boundary_fix=True))
cdgrid = model.cdgrid

sw = williamson_test2(grid)
u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
state = FV3EdgeShallowWaterState(
    h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
model.set_initial_mass(state)

dt = 300.0
n_steps = int(86400 / dt)
for _ in range(n_steps):
    state = model.step(state, dt)

# Extract simulation u_cc, v_cc at t=1d.
u_cc_sim = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                  + np.asarray(state.u_d)[:, :, 1:])
v_cc_sim = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                  + np.asarray(state.v_d)[:, 1:, :])

# Apply hyperdiff pipeline.
u_cc_sim_jx = jnp.asarray(u_cc_sim)
v_cc_sim_jx = jnp.asarray(v_cc_sim)
ue_sim = cos_a_jx * u_cc_sim_jx - sin_a_jx * v_cc_sim_jx
vn_sim = sin_a_jx * u_cc_sim_jx + cos_a_jx * v_cc_sim_jx

lap_ue_sim = np.asarray(laplacian_compact(ue_sim, grid))
bilap_ue_sim = np.asarray(laplacian_compact(jnp.asarray(lap_ue_sim), grid))
lap_vn_sim = np.asarray(laplacian_compact(vn_sim, grid))
bilap_vn_sim = np.asarray(laplacian_compact(jnp.asarray(lap_vn_sim), grid))

bilap_u_local_sim = cos_a * bilap_ue_sim + sin_a * bilap_vn_sim
bilap_v_local_sim = -sin_a * bilap_ue_sim + cos_a * bilap_vn_sim

hyp_du_sim = -hyp_coeff * bilap_u_local_sim
hyp_dv_sim = -hyp_coeff * bilap_v_local_sim

# Per face max.
print(f"  Per-face max|hyp_dv| on W2 state at t=1 d:")
for f in range(6):
    dv_max = np.abs(hyp_dv_sim[f]).max()
    i = np.argmax(np.abs(hyp_dv_sim[f]))
    ii, jj = np.unravel_index(i, hyp_dv_sim[f].shape)
    print(f"    face{f}: max|hyp_dv|={dv_max:.3e}  at (i={ii},j={jj}) "
          f"lat={lat_deg[f,ii,jj]:.1f}° lon={lon_deg[f,ii,jj]:.1f}°")

# Compute per-cell integrated contribution over dt (one step):
#   Δv = dt * hyp_dv
# and compare to the observed polar peak of 0.303 m/s.
#
# Actually v evolves via RK3 with all tendencies; hyperdiff is ONE of
# them.  The relevant question is: at face 4 polar, is max|hyp_dv|
# MUCH LARGER than at equatorial faces?  If yes, it's amplifying
# structure at the polar face beyond what it does at equatorial.
print(f"\n  Ratio of face 4 max|hyp_dv| to face 0 max|hyp_dv|:")
r = np.abs(hyp_dv_sim[4]).max() / np.abs(hyp_dv_sim[0]).max()
print(f"    {r:.2f}  (>> 1 means polar face is DISPROPORTIONATELY hit)")

# Compare to the W2 IC hyperdiff pattern (Test A).
r_ic = np.abs(hyp_dv[4]).max() / np.abs(hyp_dv[0]).max() if np.abs(hyp_dv[0]).max() > 1e-30 else float('inf')
print(f"  Same ratio for W2 IC (Test A): {r_ic:.2f}")
print(f"  Ratio-of-ratios (t=1 / IC): "
      f"{r / r_ic if r_ic > 0 else float('inf'):.2f}")
print()
print("Interpretation:")
print("  - If TEST A face 4 is ~same as face 0, the pipeline does NOT")
print("    intrinsically favor the polar face — angle rotation is cleared.")
print("  - If TEST B face 4 is LARGE while TEST A face 4 is small,")
print("    the polar amplification comes from ACCUMULATED STATE error")
print("    interacting with the hyperdiff stencil, not from the")
print("    rotation itself.")
print("  - If TEST A face 4 is already larger than face 0, the angle")
print("    rotation IS biasing the hyperdiff toward the polar face")
print("    EVEN on the smooth exact IC — a real bug.")
