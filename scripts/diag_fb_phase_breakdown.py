"""Iter-662 diagnostic: per-phase spatial residual breakdown.

FB chain has three phases:
  1. c_sw (forward, dt/2)
  2. p_grad_c (backward, dt/2)
  3. d_sw_native (full dt)

Feed a balanced Williamson 2 IC into each phase in isolation and
measure the tendency. For each phase, also identify WHERE the
residual concentrates (face boundaries vs interior).
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"; os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import numpy as np
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.core.fv3_sw_core import _c_sw, _p_grad_c, _d_sw_native

N = 36
grid = create_cubed_sphere(N)
cdgrid = create_cubed_sphere_cdgrid(grid)

g = 9.80616; omega = 7.292e-5; u_0 = 38.61068276698372
h_0 = 29400.0 / g; R = cdgrid.radius
lat_c = cdgrid.base.lat
h = h_0 - (R*omega*u_0 + 0.5*u_0**2) * jnp.sin(lat_c)**2 / g
u_d = u_0 * jnp.cos(cdgrid.lat_edge_x) * cdgrid.cos_angle_edge_x
v_d = -u_0 * jnp.cos(cdgrid.lat_edge_y) * cdgrid.sin_angle_edge_y
h_s = jnp.zeros_like(h)

dt = 600.0
dt2 = 0.5 * dt

def classify(field_err, n):
    """Classify per-cell residual into vertex/edge/interior."""
    # field_err shape (6, n+1, n) for u-face, (6, n, n+1) for v-face.
    # Generalize: cells with i in {0, last} OR j in {0, last} are boundary.
    a = np.abs(field_err)
    s = a.shape
    # Last axis-1 idx is s[1]-1, last axis-2 idx is s[2]-1.
    i_last = s[1] - 1; j_last = s[2] - 1
    v_mask = np.zeros_like(a, dtype=bool)
    for i in (0, i_last):
        for j in (0, j_last):
            v_mask[:, i, j] = True
    e_mask = np.zeros_like(a, dtype=bool)
    for i in (0, i_last):
        e_mask[:, i, :] = True
    for j in (0, j_last):
        e_mask[:, :, j] = True
    e_mask = e_mask & ~v_mask
    int_mask = ~(v_mask | e_mask)
    return {
        'vertex_max': float(a[v_mask].max()),
        'edge_max':   float(a[e_mask].max()),
        'interior_max': float(a[int_mask].max()) if int_mask.sum() else 0.0,
        'vertex_mean': float(a[v_mask].mean()),
        'edge_mean':   float(a[e_mask].mean()),
        'interior_mean': float(a[int_mask].mean()) if int_mask.sum() else 0.0,
    }

# --- Phase 1: c_sw alone ---
h_star, uc_new, vc_new, ua, va = _c_sw(h, u_d, v_d, h_s, cdgrid, dt, g)
# c_sw output: h_star, uc_new, vc_new are new state after half-step.
# Tendency: (new - old)/(dt/2).  But uc/vc are C-grid while u_d/v_d are D-grid.
# Just measure h_star vs h, and the uc_new magnitude (no uc baseline to subtract).
dh_csw = np.asarray(h_star - h) / dt2
print(f'--- Phase 1 (c_sw, dt/2) ---')
print(f'  max|dh|/dt   = {np.abs(dh_csw).max():.3e} m/s')
print(f'  max|uc_new|  = {float(jnp.abs(uc_new).max()):.3e}')
print(f'  max|vc_new|  = {float(jnp.abs(vc_new).max()):.3e}')

# --- Phase 2: p_grad_c alone ---
dp_x, dp_y = _p_grad_c(h_star, h_s, cdgrid, dt2, g)
print(f'\n--- Phase 2 (p_grad_c, dt/2) ---')
# dp_x is the PGF increment already multiplied by dt2.
# So dp_x/dt gives the pgf tendency.
print(f'  max|dp_x|/dt = {float(jnp.abs(dp_x).max())/dt:.3e} m/s²')
print(f'  max|dp_y|/dt = {float(jnp.abs(dp_y).max())/dt:.3e} m/s²')
dp_y_cls = classify(np.asarray(dp_y) / dt, N)
print(f'  dp_y/dt dist: vertex max={dp_y_cls["vertex_max"]:.3e}  '
      f'edge max={dp_y_cls["edge_max"]:.3e}  interior max={dp_y_cls["interior_max"]:.3e}')

# --- Phase 3: d_sw_native alone ---
# Apply C-grid updates first (c_sw's uc_new, vc_new + p_grad_c's dp_x, dp_y).
uc_after = uc_new + dp_x
vc_after = vc_new + dp_y
h_new, u_d_new, v_d_new = _d_sw_native(
    h, u_d, v_d, h_s, uc_after, vc_after, ua, va, cdgrid, dt, g,
    div_damp=0.0, d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
    damp_v=0.0, nord_v=0)
du = np.asarray(u_d_new - u_d) / dt
dv = np.asarray(v_d_new - v_d) / dt
dh_total = np.asarray(h_new - h) / dt
print(f'\n--- Phase 3 (d_sw_native, full dt, after c_sw+p_grad_c) ---')
print(f'  max|dh_total|/dt = {np.abs(dh_total).max():.3e} m/s')
print(f'  max|du|/dt       = {np.abs(du).max():.3e} m/s²')
print(f'  max|dv|/dt       = {np.abs(dv).max():.3e} m/s²')
du_cls = classify(du, N)
dv_cls = classify(dv, N)
print(f'  du/dt dist: vertex={du_cls["vertex_max"]:.3e}  '
      f'edge={du_cls["edge_max"]:.3e}  interior={du_cls["interior_max"]:.3e}')
print(f'  dv/dt dist: vertex={dv_cls["vertex_max"]:.3e}  '
      f'edge={dv_cls["edge_max"]:.3e}  interior={dv_cls["interior_max"]:.3e}')
