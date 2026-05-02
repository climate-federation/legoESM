"""Iter-984: term-by-term decomposition of c_sw + p_grad_c on W2."""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
sys.path.insert(0, '/home/gentine/Documents/Code/legoESM/legoESM')
import jax.numpy as jnp
import numpy as np

from legoesm.core.fv3_sw_core import (
    _d2a2c_vect, _ke_upwind, _corner_vorticity, _vorticity_flux, _p_grad_c,
)
from legoesm.core.operators_cdgrid import _pad_halo_auto
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from tests.atmosphere.shallow_water.test_cases.williamson import williamson_test2

N = 36
dt = 300.0
grid = create_cubed_sphere(N, use_duogrid=True)
cdgrid = create_cubed_sphere_cdgrid(grid)
sw = williamson_test2(grid)
u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))

# Reproduce c_sw step-by-step
dt2 = 0.5 * dt
g = 9.80616
ua, va, uc, vc, ut, vt = _d2a2c_vect(u_d, v_d, cdgrid)
use_duogrid = True

# KE term
ke_u, ke_v = _ke_upwind(uc, vc, ua, va, u_d, v_d, cdgrid, use_duogrid)
ke_total = dt2 * 0.5 * (ua * ke_u + va * ke_v)

# Vorticity at corners
vort_abs = _corner_vorticity(uc, vc, cdgrid, use_duogrid)

# Vorticity flux at C-grid faces
fy1, vort_x, fx1, vort_y = _vorticity_flux(v_d, u_d, uc, vc, vort_abs, cdgrid, use_duogrid)
fy1 = dt2 * fy1
fx1 = dt2 * fx1

# KE gradient
ke_pad = _pad_halo_auto(ke_total, cdgrid)
dke_x = cdgrid.rdxc * (ke_pad[:, :-1, 1:-1] - ke_pad[:, 1:, 1:-1])  # at u-faces (n+1, n)
dke_y = cdgrid.rdyc * (ke_pad[:, 1:-1, :-1] - ke_pad[:, 1:-1, 1:])  # at v-faces (n, n+1)

# Vorticity flux contribution to uc, vc
vort_contrib_uc = fy1 * vort_x   # at u-faces
vort_contrib_vc = -fx1 * vort_y

# p_grad_c
h_star = sw.h.data  # NOTE: c_sw transports h with first-order upwind first; we approximate with IC h here
# For W2 IC (analytical steady), h_star should ≈ h_ic. Let me use h_ic for the p_grad_c test.
dp_x, dp_y = _p_grad_c(h_star, sw.h_s.data, cdgrid, dt2, g)

print(f"Term-by-term decomposition of c_sw + p_grad_c on W2 IC:")
print(f"  vort_contrib_uc (= fy1*vort_x): |max| = {float(np.abs(vort_contrib_uc).max()):.4f}")
print(f"  dke_x:                          |max| = {float(np.abs(dke_x).max()):.4f}")
print(f"  dp_x (p_grad_c):                 |max| = {float(np.abs(dp_x).max()):.4f}")
print(f"  SUM (should ≈ 0 for steady):     |max| = {float(np.abs(vort_contrib_uc + dke_x + dp_x).max()):.4f}")
print()
print(f"  vort_contrib_vc (= -fx1*vort_y): |max| = {float(np.abs(vort_contrib_vc).max()):.4f}")
print(f"  dke_y:                          |max| = {float(np.abs(dke_y).max()):.4f}")
print(f"  dp_y:                           |max| = {float(np.abs(dp_y).max()):.4f}")
print(f"  SUM (should ≈ 0 for steady):     |max| = {float(np.abs(vort_contrib_vc + dke_y + dp_y).max()):.4f}")

# Top 5 SUM_uc locations
sum_uc = np.asarray(vort_contrib_uc + dke_x + dp_x)
flat_sorted = np.argsort(np.abs(sum_uc).ravel())[::-1][:5]
print(f"\nTop 5 |SUM_uc| (residual imbalance) locations:")
for idx in flat_sorted:
    f, i, j = np.unravel_index(idx, sum_uc.shape)
    print(f"  face={f} i={i:2d} j={j:2d} |sum|={abs(sum_uc[f, i, j]):.4f} = vort:{vort_contrib_uc[f, i, j]:.3f} + dke:{dke_x[f, i, j]:.3f} + dp:{dp_x[f, i, j]:.3f}")

sum_vc = np.asarray(vort_contrib_vc + dke_y + dp_y)
flat_sorted = np.argsort(np.abs(sum_vc).ravel())[::-1][:5]
print(f"\nTop 5 |SUM_vc| (residual imbalance) locations:")
for idx in flat_sorted:
    f, i, j = np.unravel_index(idx, sum_vc.shape)
    print(f"  face={f} i={i:2d} j={j:2d} |sum|={abs(sum_vc[f, i, j]):.4f} = vort:{vort_contrib_vc[f, i, j]:.3f} + dke:{dke_y[f, i, j]:.3f} + dp:{dp_y[f, i, j]:.3f}")
