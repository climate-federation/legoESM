"""Iter-983: trace c_sw + p_grad_c output on W2 IC at cube vertices."""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
sys.path.insert(0, '/home/gentine/Documents/Code/legoESM/legoESM')
import jax.numpy as jnp
import numpy as np

from legoesm.core.fv3_sw_core import _c_sw, _p_grad_c, _d2a2c_vect
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

# Get OLD uc, vc directly from d2a2c (no c_sw modifications)
_, _, uc_old, vc_old, _, _ = _d2a2c_vect(u_d, v_d, cdgrid)

# Run c_sw → uc_new
h_star, uc_new, vc_new, ua, va = _c_sw(sw.h.data, u_d, v_d, sw.h_s.data, cdgrid, dt, 9.80616)

# Run p_grad_c
dt2 = 0.5 * dt
dp_x, dp_y = _p_grad_c(h_star, sw.h_s.data, cdgrid, dt2, 9.80616)
uc_after_pgc = uc_new + dp_x
vc_after_pgc = vc_new + dp_y

# Compute the increments
duc_csw = np.asarray(uc_new - uc_old)
dvc_csw = np.asarray(vc_new - vc_old)
duc_total = np.asarray(uc_after_pgc - uc_old)
dvc_total = np.asarray(vc_after_pgc - vc_old)

print(f"c_sw alone increment to uc, vc:")
print(f"  |duc|_max = {np.abs(duc_csw).max():.4f} m/s")
print(f"  |dvc|_max = {np.abs(dvc_csw).max():.4f} m/s")
print(f"\nc_sw + p_grad_c TOTAL increment to uc, vc:")
print(f"  |duc|_max = {np.abs(duc_total).max():.4f} m/s")
print(f"  |dvc|_max = {np.abs(dvc_total).max():.4f} m/s")
print(f"\nFor W2 SOLID BODY in steady balance, total c_sw+p_grad_c should be ≈0")

# Top 5 |duc| locations (after c_sw + p_grad_c)
flat_sorted = np.argsort(np.abs(duc_total).ravel())[::-1][:5]
print(f"\nTop 5 |duc| (c_sw + p_grad_c) locations:")
for idx in flat_sorted:
    f, i, j = np.unravel_index(idx, duc_total.shape)
    print(f"  face={f} i={i:2d} j={j:2d} |duc|={abs(duc_total[f, i, j]):.4f}")

flat_sorted = np.argsort(np.abs(dvc_total).ravel())[::-1][:5]
print(f"\nTop 5 |dvc| (c_sw + p_grad_c) locations:")
for idx in flat_sorted:
    f, i, j = np.unravel_index(idx, dvc_total.shape)
    print(f"  face={f} i={i:2d} j={j:2d} |dvc|={abs(dvc_total[f, i, j]):.4f}")
