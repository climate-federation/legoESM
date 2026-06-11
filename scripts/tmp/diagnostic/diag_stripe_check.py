#!/usr/bin/env python
"""Investigate horizontal striping in csw+dg d2a2c_vect output."""
import os
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax.numpy as jnp
import numpy as np
from legoesm.grids.cubed_sphere import create_cubed_sphere, rotate_winds_geo_to_grid
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import pad_halo
from legoesm import constants

N = 16
G = constants.g

grid = create_cubed_sphere(N, use_duogrid=True)
cdgrid = create_cubed_sphere_cdgrid(grid)

# Solid body rotation IC
u0 = 2*jnp.pi*grid.radius/(12*86400)
u_east = u0*grid.cos_lat
v_north = jnp.zeros_like(u_east)
u_grid, v_grid = rotate_winds_geo_to_grid(u_east, v_north, grid.angle)
u_pad = pad_halo(u_grid)
v_pad = pad_halo(v_grid)
u_d = 0.5*(u_pad[:,1:-1,:-1]+u_pad[:,1:-1,1:])
v_d = 0.5*(v_pad[:,:-1,1:-1]+v_pad[:,1:,1:-1])

# Run d2a2c_vect
from legoesm.core.fv3_sw_core import d2a2c_vect
ua, va, uc, vc, ut, vt = d2a2c_vect(u_d, v_d, cdgrid)

# Check ua pattern
ua_np = np.asarray(ua)
va_np = np.asarray(va)

print(f"=== Stripe analysis at C{N} ===")
print(f"ua shape: {ua_np.shape}")

# Check row-by-row std (stripes would show as varying std across rows)
for face in [0, 4]:
    print(f"\nFace {face} ua:")
    for j in range(N):
        row = ua_np[face, :, j]
        print(f"  j={j:2d}: mean={row.mean():8.3f} std={row.std():8.5f}")

# Check if the 4th-order D→A introduces edge effects
print(f"\n=== 4th-order D→A check ===")
# Compare 4th-order vs 2nd-order D→A at cell j=0 and j=N-1
from legoesm.grids.duogrid import pad_halo_dgrid
u_d_ext, v_d_ext = pad_halo_dgrid(
    u_d, v_d,
    cdgrid.cos_angle_edge_x, cdgrid.sin_angle_edge_x,
    cdgrid.cos_angle_edge_y, cdgrid.sin_angle_edge_y,
    grid.duogrid)

_A1, _A2 = 0.5625, -0.0625
utmp_4th = _A2*(u_d_ext[:,:,:-3]+u_d_ext[:,:,3:]) + _A1*(u_d_ext[:,:,1:-2]+u_d_ext[:,:,2:-1])
utmp_2nd = 0.5*(u_d_ext[:,:,1:-2]+u_d_ext[:,:,2:-1])

diff = np.asarray(utmp_4th - utmp_2nd)
print(f"4th-2nd diff: max_abs={np.abs(diff).max():.6f}")
print(f"  Face 0 row j=0: {diff[0,:,0]}")
print(f"  Face 0 row j=N//2: {diff[0,:,N//2]}")
print(f"  Face 0 row j=N-1: {diff[0,:,N-1]}")

# Check geographic rotation asymmetry
print(f"\n=== Geographic rotation check ===")
# Compare utmp → u_east → utmp roundtrip error
cos_a = np.asarray(grid.cos_angle)
sin_a = np.asarray(grid.sin_angle)
cos_theta = np.asarray(cdgrid.cos_sg[:,:,:,4])
sin_theta = np.asarray(cdgrid.sin_sg[:,:,:,4])

utmp_np = np.asarray(utmp_4th)
vtmp_np = np.asarray(_A2*(v_d_ext[:,:-3,:]+v_d_ext[:,3:,:]) + _A1*(v_d_ext[:,1:-2,:]+v_d_ext[:,2:-1,:]))

# Non-orthogonal forward
u_east_np = cos_a*utmp_np + sin_a*(utmp_np*cos_theta - vtmp_np)/sin_theta
v_north_np = sin_a*utmp_np + cos_a*(vtmp_np - utmp_np*cos_theta)/sin_theta

# Orthogonal back
utmp_back_orth = cos_a*u_east_np + sin_a*v_north_np
vtmp_back_orth = -sin_a*u_east_np + cos_a*v_north_np

# Non-orthogonal back
cos_beta = cos_a*cos_theta - sin_a*sin_theta
sin_beta = sin_a*cos_theta + cos_a*sin_theta
utmp_back_nonorth = cos_a*u_east_np + sin_a*v_north_np  # same! (x_hat projection)
vtmp_back_nonorth = cos_beta*u_east_np + sin_beta*v_north_np

print(f"Roundtrip error (non-orth forward + orth back):")
print(f"  utmp: max_abs={np.abs(utmp_back_orth - utmp_np).max():.3e}")
print(f"  vtmp: max_abs={np.abs(vtmp_back_orth - vtmp_np).max():.3e}")
print(f"Roundtrip error (non-orth forward + non-orth back):")
print(f"  utmp: max_abs={np.abs(utmp_back_nonorth - utmp_np).max():.3e}")
print(f"  vtmp: max_abs={np.abs(vtmp_back_nonorth - vtmp_np).max():.3e}")
