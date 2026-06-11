#!/usr/bin/env python
"""Check pad_halo_dgrid quality for all faces and edges."""
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
from legoesm.grids.duogrid import pad_halo_dgrid

N = 12

grid = create_cubed_sphere(N, use_duogrid=True)
cdgrid = create_cubed_sphere_cdgrid(grid)

# Solid body rotation
u0 = 2*jnp.pi*grid.radius/(12*86400)
u_east = u0*grid.cos_lat
v_north = jnp.zeros_like(u_east)
u_grid, v_grid = rotate_winds_geo_to_grid(u_east, v_north, grid.angle)
u_pad = pad_halo(u_grid)
v_pad = pad_halo(v_grid)
u_d = 0.5*(u_pad[:,1:-1,:-1]+u_pad[:,1:-1,1:])
v_d = 0.5*(v_pad[:,:-1,1:-1]+v_pad[:,1:,1:-1])

# Pad
u_d_ext, v_d_ext = pad_halo_dgrid(
    u_d, v_d,
    cdgrid.cos_angle_edge_x, cdgrid.sin_angle_edge_x,
    cdgrid.cos_angle_edge_y, cdgrid.sin_angle_edge_y,
    grid.duogrid)

# u_d_ext: (6, N, N+3), v_d_ext: (6, N+3, N)
# The original u_d occupies u_d_ext[:,:,1:-1] (N+1 edges)
# The halo is at j=0 (south) and j=N+2 (north)

u_ext_np = np.asarray(u_d_ext)
v_ext_np = np.asarray(v_d_ext)
u_d_np = np.asarray(u_d)
v_d_np = np.asarray(v_d)

print(f"=== pad_halo_dgrid quality check at C{N} ===")
print(f"u_d_ext shape: {u_ext_np.shape}, v_d_ext shape: {v_ext_np.shape}")

# Check that interior is preserved
u_interior = u_ext_np[:,:,1:-1]
u_interior_err = np.abs(u_interior - u_d_np).max()
print(f"\nInterior preservation: u_d max_err={u_interior_err:.2e}")

v_interior = v_ext_np[:,1:-1,:]
v_interior_err = np.abs(v_interior - v_d_np).max()
print(f"                       v_d max_err={v_interior_err:.2e}")

# Check halo values for each face
print(f"\nHalo values per face (u_d south/north j-halo):")
for face in range(6):
    south_halo = u_ext_np[face, :, 0]
    north_halo = u_ext_np[face, :, -1]
    interior_mean = u_ext_np[face, :, N//2+1].mean()
    print(f"  Face {face}: south mean={south_halo.mean():8.3f} "
          f"north mean={north_halo.mean():8.3f} "
          f"interior mean={interior_mean:8.3f}")

print(f"\nHalo values per face (v_d west/east i-halo):")
for face in range(6):
    west_halo = v_ext_np[face, 0, :]
    east_halo = v_ext_np[face, -1, :]
    interior_mean = v_ext_np[face, N//2+1, :].mean()
    print(f"  Face {face}: west mean={west_halo.mean():8.3f} "
          f"east mean={east_halo.mean():8.3f} "
          f"interior mean={interior_mean:8.3f}")

# Check 4th-order D→A at boundary cells
_A1, _A2 = 0.5625, -0.0625
utmp_4th = _A2*(u_ext_np[:,:,:-3]+u_ext_np[:,:,3:]) + _A1*(u_ext_np[:,:,1:-2]+u_ext_np[:,:,2:-1])
print(f"\n4th-order D→A utmp at boundary vs interior (face 4 = north pole):")
for j in [0, 1, N//2, N-2, N-1]:
    row = utmp_4th[4, :, j]
    print(f"  j={j:2d}: mean={row.mean():8.3f} std={row.std():.5f}")

# Check expected symmetry on face 4 (polar)
# For solid body rotation, face 4 should have antisymmetry about j=N//2
print(f"\nFace 4 symmetry check (j vs N-1-j):")
for j in range(N//2):
    val_j = utmp_4th[4, :, j].mean()
    val_sym = utmp_4th[4, :, N-1-j].mean()
    print(f"  j={j:2d}: {val_j:8.3f}, j={N-1-j:2d}: {val_sym:8.3f}, "
          f"sum={val_j+val_sym:8.5f} (should be ~0)")
