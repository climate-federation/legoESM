"""Iter-661 diagnostic: step-1 tendency magnitudes for FB chain on
the balanced Williamson 2 IC.  For a geostrophically balanced state,
the instantaneous tendency should be zero (up to truncation error).
If it's O(1 m/s²), the FB chain scheme/IC combination produces
spurious forcing.
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"; os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import numpy as np
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    FV3FBShallowWaterModel, FV3EdgeShallowWaterState, CDGridShallowWaterConfig,
)

N = 36
grid = create_cubed_sphere(N)
cdgrid = create_cubed_sphere_cdgrid(grid)
config = CDGridShallowWaterConfig(A_h=0.0, hyperdiff_coeff=0.0, div_damp=0.0)
model = FV3FBShallowWaterModel(grid, config)

g=9.80616; omega=7.292e-5; u_0=38.61068276698372
h_0=29400.0/g; R=cdgrid.radius
lat_c=cdgrid.base.lat
h=h_0 - (R*omega*u_0 + 0.5*u_0**2) * jnp.sin(lat_c)**2/g
u_d = u_0*jnp.cos(cdgrid.lat_edge_x)*cdgrid.cos_angle_edge_x
v_d = -u_0*jnp.cos(cdgrid.lat_edge_y)*cdgrid.sin_angle_edge_y
state0 = FV3EdgeShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=jnp.zeros_like(h))
model.set_initial_mass(state0)

# Run short dt to compute tendency.
for dt in [1.0, 10.0, 100.0, 600.0]:
    state1 = model.step(state0, dt)
    dh = np.asarray(state1.h - state0.h)
    du = np.asarray(state1.u_d - state0.u_d)
    dv = np.asarray(state1.v_d - state0.v_d)
    print(f'dt={dt:>6.1f}s  max|dh|/dt={np.abs(dh).max()/dt:.3e} m/s  '
          f'max|du|/dt={np.abs(du).max()/dt:.3e} m/s²  '
          f'max|dv|/dt={np.abs(dv).max()/dt:.3e} m/s²')

# For reference: balanced geostrophic target is 0.  The analytic
# residual from rotation-Coriolis on the cubed-sphere is machine eps
# at interior cells; any O(1e-2) or larger tendency is scheme.
print()
print('For reference: on a geostrophically balanced state, all three '
      'tendencies should be O(epsilon) ~ 1e-15.  Values >> 1e-3 indicate '
      'spurious forcing from the FB scheme or an IC-scheme mismatch.')
