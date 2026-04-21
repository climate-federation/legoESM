"""Iter-732 diagnostic: bisect FB C24 W2 blowup to the offending phase.

Context: iter-730 showed FB W2 at C24 blows up at step 60 for any
damp_v ∈ {0, 0.06, 0.12}.  Iter-731 noted this only rules out a
narrow slice of the parameter space.  Iter-732 attacks the
structural side by running ONE FB step at C24 with the W2 exact IC
and measuring, after each of the three FB phases (c_sw, p_grad_c,
d_sw), how far the updated state has drifted from the analytic
steady-state solution.  Whichever phase produces the largest
per-step drift at step 1 is the likely bug source.

W2 is an exact steady-state solution of the SW equations; after
one FB step the state should differ from the IC by at most
round-off (for a Fortran-faithful implementation) or by the small
truncation error inherent to the discretisation (for a faithful
port).  A O(1) drift in one phase localises the bug.

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
from legoesm.core.fv3_sw_core import _c_sw, _p_grad_c, _d_sw_native


N = 24
grid = create_cubed_sphere(N)
cdgrid = create_cubed_sphere_cdgrid(grid)

# --- Williamson 2 IC (same as diag_iter730_fb_c24_stability.py) ---
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

def norms(label, current, exact):
    diff = current - exact
    linf = float(jnp.max(jnp.abs(diff)))
    l2 = float(jnp.sqrt(jnp.mean(diff ** 2)))
    ref = float(jnp.max(jnp.abs(exact)))
    print(f"  {label}: max|Δ|={linf:10.4e}  RMS={l2:10.4e}  "
          f"max|Δ|/max|ref|={linf / max(ref, 1e-30):.2e}")
    return {"label": label, "max_abs": linf, "rms": l2, "ref": ref}

print(f"=== Iter-732: FB C24 W2 phase bisect at step 1 (dt={dt}s) ===\n")
print(f"IC: max|u_d|={float(jnp.max(jnp.abs(u_d0))):.4f}  "
      f"max|v_d|={float(jnp.max(jnp.abs(v_d0))):.4f}")

# Phase 1: c_sw (dt/2 forward C-grid)
print("\n[Phase 1] c_sw (C-grid forward half-step)")
h_star, uc_new, vc_new, ua, va = _c_sw(h0, u_d0, v_d0, h_s, cdgrid, dt, g)
print(f"  h_star max|Δ|={float(jnp.max(jnp.abs(h_star - h0))):.4e}")
print(f"  uc_new max|Δ|={float(jnp.max(jnp.abs(uc_new))):.4e}  "
      f"(|uc| is a tendency increment; starts at 0 in SW sense)")
print(f"  vc_new max|Δ|={float(jnp.max(jnp.abs(vc_new))):.4e}")

# Phase 2: p_grad_c — backward pressure gradient (dt/2)
print("\n[Phase 2] p_grad_c (backward pressure gradient)")
dt2 = 0.5 * dt
dp_x, dp_y = _p_grad_c(h_star, h_s, cdgrid, dt2, g)
uc_after_pgrad = uc_new + dp_x
vc_after_pgrad = vc_new + dp_y
print(f"  dp_x max|Δ|={float(jnp.max(jnp.abs(dp_x))):.4e}")
print(f"  dp_y max|Δ|={float(jnp.max(jnp.abs(dp_y))):.4e}")
print(f"  uc+dp_x max|Δ|={float(jnp.max(jnp.abs(uc_after_pgrad))):.4e}")
print(f"  vc+dp_y max|Δ|={float(jnp.max(jnp.abs(vc_after_pgrad))):.4e}")

# Phase 3: _d_sw_native — full D-grid step using ORIGINAL u_d0/v_d0
print("\n[Phase 3] _d_sw_native (D-grid full step)")
h_new, u_d_new, v_d_new = _d_sw_native(
    h0, u_d0, v_d0, h_s, uc_after_pgrad, vc_after_pgrad, ua, va,
    cdgrid, dt, g,
    div_damp=0.0, d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
    damp_v=0.0, nord_v=1)
print(f"  h_new max|Δ|={float(jnp.max(jnp.abs(h_new - h0))):.4e}")
print(f"  u_d_new-u_d0 max|Δ|={float(jnp.max(jnp.abs(u_d_new - u_d0))):.4e}")
print(f"  v_d_new-v_d0 max|Δ|={float(jnp.max(jnp.abs(v_d_new - v_d0))):.4e}")

print("\n=== Per-variable summary (step 1 drift from W2 exact) ===")
print("  h: ", end=""); norms("after FB step 1", h_new, h0)
print("  u_d:", end=" "); norms("after FB step 1", u_d_new, u_d0)
print("  v_d:", end=" "); norms("after FB step 1", v_d_new, v_d0)

# Which D-grid coordinates carry the largest drift?
print("\n=== u_d max-error localisation ===")
u_diff = u_d_new - u_d0
u_err_abs = jnp.abs(u_diff)
for face in range(6):
    face_max = float(jnp.max(u_err_abs[face]))
    argmax_flat = int(jnp.argmax(u_err_abs[face]))
    i = argmax_flat // u_err_abs[face].shape[1]
    j = argmax_flat % u_err_abs[face].shape[1]
    print(f"  face {face}: max|Δu_d|={face_max:.4e} at (i,j)=({i},{j})")

print("\n=== v_d max-error localisation ===")
v_diff = v_d_new - v_d0
v_err_abs = jnp.abs(v_diff)
for face in range(6):
    face_max = float(jnp.max(v_err_abs[face]))
    argmax_flat = int(jnp.argmax(v_err_abs[face]))
    i = argmax_flat // v_err_abs[face].shape[1]
    j = argmax_flat % v_err_abs[face].shape[1]
    print(f"  face {face}: max|Δv_d|={face_max:.4e} at (i,j)=({i},{j})")

print("\n=== Analysis hint ===")
print("For W2 (exact steady state):")
print("  h drift ~ O(truncation error per dt) for good discretisation")
print("  u_d, v_d drift ~ O(truncation error per dt)")
print("  Localisation (i,j) near panel edges (0 or n-1) → edge/halo bug")
print("  Localisation far from edges → interior stencil bug")
print("  Face-dependent asymmetry → metric / rotation bug")
