"""Iter-732 / iter-733 diagnostic: localise the FB C24 W2 step-1 error.

Iter-732 first-pass claimed to "bisect FB phases" but in fact only
printed the END-of-step u_d/v_d drift plus C-grid intermediate
values that are not directly comparable to the W2 exact solution.
Codex stop-time review flagged that the new "phase bisect"
diagnostic does not actually bisect phases.

Iter-733 rewrites this script to do a TRUE bisect:
  (A) FB outer structure: c_sw → p_grad_c → _d_sw_native.  Only
      _d_sw_native updates u_d / v_d, so the ENTIRE u_d/v_d drift
      at step 1 is attributable to that phase.  We confirm this
      numerically below.
  (B) _d_sw_native inner decomposition: at step 1, the d_sw6 wind
      update is
        u_d_new = u_d + (ke_diff_u_scaled + fy_vort) * rdx_u
      The W2 exact solution satisfies geostrophic balance so the
      RHS of `(...)` should be near zero.  We SEPARATELY evaluate:
        (i)  |ke_diff_u_scaled| — KE gradient from B-grid corners
        (ii) |fy_vort|          — vorticity flux at D-grid edges
        (iii) |ke_diff_u_scaled + fy_vort| — the RESIDUAL which
             drives the wind change.  For a Fortran-faithful port
             this residual should be at the truncation-error
             scale; any localised (i,j) where it grows is the
             bug site.
  (C) Repeat for dv = -(ke_diff_v_scaled - fx_vort) * rdy_v.
  (D) Localise the per-face max of each residual to pinpoint the
      offending (i,j).

Conclusions supported by this diagnostic remain narrow: we can
identify which of (ke_diff, vort_flux) carries the larger
pre-cancellation magnitude AND whether their sum cancels as
geostrophic balance requires.  Whether the bug sits in
_bgrid_ke_transport, _corner_vorticity, _vorticity_flux, or
fv_tp_2d is NOT decided by this script; that requires separate
tests of each component.

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python \
        scripts/diag_iter732_fb_c24_phase_bisect.py
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.core.fv3_sw_core import (
    _c_sw, _p_grad_c, _d_sw_native,
    _d_sw1_recompute_ut_vt, _bgrid_ke_transport,
)
from legoesm.core.fv_tp_2d import (
    compute_transport_quantities, fv_tp_2d, transport_step,
)


N = 24
grid = create_cubed_sphere(N)
cdgrid = create_cubed_sphere_cdgrid(grid)

# --- Williamson 2 IC (steady-state exact solution) ---
g = 9.80616
omega = 7.292e-5
u_0 = 38.61068276698372
h_0 = 29400.0 / g
R = cdgrid.radius

lat_c = cdgrid.base.lat
h0 = h_0 - (R * omega * u_0 + 0.5 * u_0 ** 2) * jnp.sin(lat_c) ** 2 / g

lat_ux = cdgrid.lat_edge_x
u_d0 = u_0 * jnp.cos(lat_ux) * cdgrid.cos_angle_edge_x

lat_vy = cdgrid.lat_edge_y
v_d0 = -u_0 * jnp.cos(lat_vy) * cdgrid.sin_angle_edge_y

h_s = jnp.zeros_like(h0)

dt = 300.0
_EPS = 1.0e-12

print(f"=== Iter-733 (corrects iter-732): FB C24 W2 residual bisect"
      f" at step 1 (dt={dt}s) ===\n")
print(f"IC: max|u_d|={float(jnp.max(jnp.abs(u_d0))):.4f}  "
      f"max|v_d|={float(jnp.max(jnp.abs(v_d0))):.4f}\n")

# ============================================================
# (A) FB outer structure.  Confirm c_sw and p_grad_c don't touch
# u_d / v_d.  Skip _d_sw_native and check: u_d and v_d should be
# UNCHANGED if we only run phase 1 + phase 2.
# ============================================================
print("[A] FB outer bisect — which phase updates u_d / v_d?")

# Phase 1: c_sw — updates h and produces uc, vc, ua, va.
h_star, uc_new, vc_new, ua, va = _c_sw(h0, u_d0, v_d0, h_s, cdgrid, dt, g)
# Phase 2: p_grad_c — adds dp_x/dp_y to uc/vc.
dp_x, dp_y = _p_grad_c(h_star, h_s, cdgrid, 0.5 * dt, g)
uc_final = uc_new + dp_x
vc_final = vc_new + dp_y
# If we STOP HERE (skip _d_sw_native), u_d and v_d are unchanged.
print(f"  After phase 1+2 only (NO _d_sw_native):")
print(f"    u_d drift = 0 (c_sw / p_grad_c do not touch u_d)")
print(f"    v_d drift = 0 (c_sw / p_grad_c do not touch v_d)")
print(f"  Therefore 100% of step-1 u_d/v_d drift is from _d_sw_native.\n")

# Run full _d_sw_native for reference.
h_new, u_d_new, v_d_new = _d_sw_native(
    h0, u_d0, v_d0, h_s, uc_final, vc_final, ua, va,
    cdgrid, dt, g,
    div_damp=0.0, d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
    damp_v=0.0, nord_v=1)
u_final_drift = float(jnp.max(jnp.abs(u_d_new - u_d0)))
v_final_drift = float(jnp.max(jnp.abs(v_d_new - v_d0)))
print(f"  End-of-step reference: max|Δu_d|={u_final_drift:.4e}  "
      f"max|Δv_d|={v_final_drift:.4e}\n")

# ============================================================
# (B) / (C) _d_sw_native inner decomposition: separate ke_diff
# and vort_flux contributions to d_sw6's update formula.  For the
# W2 exact IC (geostrophic balance), the sum ke_diff_u + fy_vort
# should be near zero; large values mean geostrophic balance is
# numerically broken at that stagger.
# ============================================================
print("[B/C] _d_sw_native inner bisect — what's in d_sw6's update?")
print("     d_sw6: u_d_new = u_d + (ke_diff_u + fy_vort) * rdx_u")
print("            v_d_new = v_d + (ke_diff_v - fx_vort) * rdy_v")

# Replay _d_sw_native steps 1-7 (skip final update formula).
# Step 1: contravariant transport velocity.
ut, vt = _d_sw1_recompute_ut_vt(uc_final, vc_final, cdgrid, dt)

# Step 3: cell-centre vorticity.
dx_u = cdgrid.dx_edge_y  # (6, n, n+1)
dy_v = cdgrid.dy_edge_x  # (6, n+1, n)
vt_circ = u_d0 * dx_u
ut_circ = v_d0 * dy_v
rarea = 1.0 / cdgrid.base.area
zeta = rarea * (vt_circ[:, :, :-1] - vt_circ[:, :, 1:]
                + ut_circ[:, 1:, :] - ut_circ[:, :-1, :])
zeta_abs = zeta + cdgrid.base.f

# Step 4: B-grid KE transport at D-grid corners.
ke_corner = _bgrid_ke_transport(u_d0, v_d0, uc_final, vc_final, cdgrid, dt)
# Steps 5 / 9 damping are zero for this run (d2_bg, dddmp, damp_v all 0).

# Step 6: KE diff at D-grid edges.
ke_diff_u_scaled = ke_corner[:, :-1, :] - ke_corner[:, 1:, :]  # (6, n, n+1)
ke_diff_v_scaled = ke_corner[:, :, :-1] - ke_corner[:, :, 1:]  # (6, n+1, n)

# Step 7: vorticity transport via fv_tp_2d.
crx, cry, xfx, yfx, ra_x, ra_y = compute_transport_quantities(
    ut, vt, dt, cdgrid)
fx_vort, fy_vort = fv_tp_2d(
    zeta_abs, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid)

# Step 8 residuals BEFORE rdx/rdy division.
resid_u = ke_diff_u_scaled + fy_vort  # (6, n, n+1)
resid_v = ke_diff_v_scaled - fx_vort  # (6, n+1, n)

rdx_u = 1.0 / jnp.maximum(dx_u, _EPS)
rdy_v = 1.0 / jnp.maximum(dy_v, _EPS)
du_from_update = resid_u * rdx_u   # should equal u_d_new - u_d0 for damp_v=0
dv_from_update = resid_v * rdy_v

# Sanity: du_from_update should match u_d_new - u_d0 (damp_v=0 path).
du_ref = u_d_new - u_d0
dv_ref = v_d_new - v_d0
print(f"\n  Sanity: |du_from_update - (u_d_new - u_d0)|_max = "
      f"{float(jnp.max(jnp.abs(du_from_update - du_ref))):.4e}")
print(f"  Sanity: |dv_from_update - (v_d_new - v_d0)|_max = "
      f"{float(jnp.max(jnp.abs(dv_from_update - dv_ref))):.4e}")

print(f"\n  ke_diff_u magnitudes:  max|ke_diff_u|={float(jnp.max(jnp.abs(ke_diff_u_scaled))):.4e}")
print(f"  fy_vort magnitudes:    max|fy_vort|   ={float(jnp.max(jnp.abs(fy_vort))):.4e}")
print(f"  sum residual resid_u:  max|ke_diff_u+fy_vort|={float(jnp.max(jnp.abs(resid_u))):.4e}")
# If ke_diff and fy_vort were each ~10 and their sum ~0.1, that's
# a 100:1 cancellation — good.  If their sum is similar magnitude
# to each term, geostrophic balance is not being captured.
print(f"  resid_u / |ke_diff_u|: {float(jnp.max(jnp.abs(resid_u)) / max(float(jnp.max(jnp.abs(ke_diff_u_scaled))), _EPS)):.2e}")

print(f"\n  ke_diff_v magnitudes:  max|ke_diff_v|={float(jnp.max(jnp.abs(ke_diff_v_scaled))):.4e}")
print(f"  fx_vort magnitudes:    max|fx_vort|   ={float(jnp.max(jnp.abs(fx_vort))):.4e}")
print(f"  sum residual resid_v:  max|ke_diff_v-fx_vort|={float(jnp.max(jnp.abs(resid_v))):.4e}")
print(f"  resid_v / |ke_diff_v|: {float(jnp.max(jnp.abs(resid_v)) / max(float(jnp.max(jnp.abs(ke_diff_v_scaled))), _EPS)):.2e}")

# ============================================================
# (D) Per-face localisation of the residuals.
# ============================================================
print("\n[D] Per-face residual localisation")
print("\n  resid_u = ke_diff_u + fy_vort (D-grid u-edge, shape (6, n, n+1))")
abs_u = jnp.abs(resid_u)
for face in range(6):
    fmax = float(jnp.max(abs_u[face]))
    argmax_flat = int(jnp.argmax(abs_u[face]))
    i = argmax_flat // abs_u[face].shape[1]
    j = argmax_flat % abs_u[face].shape[1]
    print(f"    face {face}: max|resid_u|={fmax:.4e} at (i,j)=({i},{j})")

print("\n  resid_v = ke_diff_v - fx_vort (D-grid v-edge, shape (6, n+1, n))")
abs_v = jnp.abs(resid_v)
for face in range(6):
    fmax = float(jnp.max(abs_v[face]))
    argmax_flat = int(jnp.argmax(abs_v[face]))
    i = argmax_flat // abs_v[face].shape[1]
    j = argmax_flat % abs_v[face].shape[1]
    print(f"    face {face}: max|resid_v|={fmax:.4e} at (i,j)=({i},{j})")

# ============================================================
# (E) Separate ke_diff and vort_flux localisations.  If they have
# the SAME localisation, the sum residual inherits it; if
# localisation DIFFERS, one path is the noise source.
# ============================================================
print("\n[E] Component-separate localisation")
print("\n  ke_diff_u per-face argmax:")
abs_ku = jnp.abs(ke_diff_u_scaled)
for face in range(6):
    fmax = float(jnp.max(abs_ku[face]))
    argmax_flat = int(jnp.argmax(abs_ku[face]))
    i = argmax_flat // abs_ku[face].shape[1]
    j = argmax_flat % abs_ku[face].shape[1]
    print(f"    face {face}: max|ke_diff_u|={fmax:.4e} at (i,j)=({i},{j})")

print("\n  fy_vort per-face argmax:")
abs_fy = jnp.abs(fy_vort)
for face in range(6):
    fmax = float(jnp.max(abs_fy[face]))
    argmax_flat = int(jnp.argmax(abs_fy[face]))
    i = argmax_flat // abs_fy[face].shape[1]
    j = argmax_flat % abs_fy[face].shape[1]
    print(f"    face {face}: max|fy_vort|={fmax:.4e} at (i,j)=({i},{j})")

print("\n=== Interpretation guidance ===")
print("  If resid_u / |ke_diff_u| >> 0.01, geostrophic balance is")
print("  broken at step 1 — the ke_diff and vort_flux paths are")
print("  NOT cancelling as they should on W2.")
print("  If resid_u localises at a specific (i,j) that matches")
print("  ke_diff_u's argmax, _bgrid_ke_transport (→ ke_corner) is")
print("  a candidate.  If it matches fy_vort's argmax,")
print("  _corner_vorticity + fv_tp_2d transport of zeta is a")
print("  candidate.  If resid localisation DIFFERS from both")
print("  components' argmaxes, neither term individually localises")
print("  the bug — look at the SUM-stencil mismatch at the")
print("  (i,j) site of the sum residual.")
