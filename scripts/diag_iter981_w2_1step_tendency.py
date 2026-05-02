"""Iter-981: 1-step FB chain tendency on W2 IC. For solid body,
analytic dt = 0 so any nonzero tendency identifies the bug source."""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
sys.path.insert(0, '/home/gentine/Documents/Code/legoESM/legoESM')
import jax.numpy as jnp
import numpy as np
import warnings

from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterState, FV3FBShallowWaterModel,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from tests.atmosphere.shallow_water.test_cases.williamson import williamson_test2

N = 36
dt = 300.0
grid = create_cubed_sphere(N, use_duogrid=True)
cdgrid = create_cubed_sphere_cdgrid(grid)
sw = williamson_test2(grid)
u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
u_d_init = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
v_d_init = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
state0 = FV3EdgeShallowWaterState(h=sw.h.data, u_d=u_d_init, v_d=v_d_init, h_s=sw.h_s.data)

cfg = CDGridShallowWaterConfig(
    hyperdiff_coeff=0.0, d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
    damp_v=0.06, nord_v=2,
)
model = FV3FBShallowWaterModel(grid, cfg)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    state1 = model.step(state0, dt)

du_d = np.asarray(state1.u_d - u_d_init)
dv_d = np.asarray(state1.v_d - v_d_init)
dh = np.asarray(state1.h - sw.h.data)
print(f"After 1 step (dt=300s, W2 solid body, analytic d/dt=0):")
print(f"  |du_d|_max = {np.abs(du_d).max():.6f} m/s")
print(f"  |dv_d|_max = {np.abs(dv_d).max():.6f} m/s")
print(f"  |dh|_max = {np.abs(dh).max():.6f} m")

# Find locations of largest deviations
idx_u = np.argmax(np.abs(du_d))
idx_v = np.argmax(np.abs(dv_d))
idx_h = np.argmax(np.abs(dh))
fu, iu, ju = np.unravel_index(idx_u, du_d.shape)
fv_, iv, jv = np.unravel_index(idx_v, dv_d.shape)
fh, ih, jh = np.unravel_index(idx_h, dh.shape)
print(f"  du_d max at face={fu}, i={iu}, j={ju}, val={du_d[fu, iu, ju]:.4f}")
print(f"  dv_d max at face={fv_}, i={iv}, j={jv}, val={dv_d[fv_, iv, jv]:.4f}")
print(f"  dh   max at face={fh}, i={ih}, j={jh}, val={dh[fh, ih, jh]:.4f}")

# Top 5 du_d locations for pattern
flat_sorted = np.argsort(np.abs(du_d).ravel())[::-1][:5]
print(f"\nTop 5 |du_d| locations (cells & |du_d|):")
for idx in flat_sorted:
    f, i, j = np.unravel_index(idx, du_d.shape)
    print(f"  face={f} i={i:2d} j={j:2d} |du|={abs(du_d[f, i, j]):.4f}")
