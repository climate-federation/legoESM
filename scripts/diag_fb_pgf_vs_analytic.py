"""Iter-665 follow-up: compare PGF magnitude on ORIGINAL h vs
AFTER-c_sw h_star.  If PGF on h is near analytic (~2.94e-3) but PGF
on h_star is 2x larger, c_sw's mass tendency is steepening h.
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"; os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import numpy as np
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.core.fv3_sw_core import _c_sw, _p_grad_c

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

dt = 600.0; dt2 = 0.5 * dt

# PGF directly on ORIGINAL h (skipping c_sw).
dp_x_h, dp_y_h = _p_grad_c(h, h_s, cdgrid, dt2, g)
pgf_u_h = float(jnp.abs(dp_x_h).max()) / dt2
pgf_v_h = float(jnp.abs(dp_y_h).max()) / dt2

# PGF on h_star (after c_sw).
h_star, _, _, _, _ = _c_sw(h, u_d, v_d, h_s, cdgrid, dt, g)
dp_x_star, dp_y_star = _p_grad_c(h_star, h_s, cdgrid, dt2, g)
pgf_u_star = float(jnp.abs(dp_x_star).max()) / dt2
pgf_v_star = float(jnp.abs(dp_y_star).max()) / dt2

dh_csw = float(jnp.abs(h_star - h).max())

# Analytic balanced PGF at lat=45:
omega_u_0 = omega * u_0
u_0_sq_term = u_0**2 / (2 * R)
pgf_analytic = omega_u_0 + u_0_sq_term  # * sin(2*lat) which is max 1 at lat=45

print(f'Analytic balanced |PGF| max  = {pgf_analytic:.3e} m/s²')
print()
print(f'PGF on ORIGINAL h:')
print(f'  max|PGF_u|/dt2 = {pgf_u_h:.3e} m/s²')
print(f'  max|PGF_v|/dt2 = {pgf_v_h:.3e} m/s²')
print()
print(f'PGF on h_star (AFTER c_sw):')
print(f'  max|PGF_u|/dt2 = {pgf_u_star:.3e} m/s²')
print(f'  max|PGF_v|/dt2 = {pgf_v_star:.3e} m/s²')
print()
print(f'c_sw mass change max|h_star - h| = {dh_csw:.3e} m (over dt/2 = {dt2}s)')
