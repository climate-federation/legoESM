"""Iter-1025: W2 robustness check with alpha=pi/4 rotation."""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
sys.path.insert(0, '/home/gentine/Documents/Code/legoESM/legoESM')
import jax, jax.numpy as jnp
import numpy as np
import warnings

from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    iter1009_dual_target_config, FV3EdgeShallowWaterState, FV3EdgeShallowWaterModel,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.regridding import (
    apply_cubedsphere_to_latlon, get_cubedsphere_to_latlon_weights,
)
from tests.atmosphere.shallow_water.test_cases.williamson import williamson_test2
from tests.test_iter921_w2_v_vs_h_pareto_sentinel import cell_centre_angles_from_4edge
from legoesm import constants

N = 36; DT = 300.0
cfg = iter1009_dual_target_config(N)

def run_w2_alpha(alpha):
    """W2 IC with rotation angle alpha."""
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    
    R = grid.radius
    Omega = constants.Omega
    g = constants.g
    u_0 = 2.0 * jnp.pi * R / (12.0 * 86400.0)
    gh_0 = 2.94e4
    h_0 = gh_0 / g
    
    lat = grid.lat
    lon = grid.lon
    
    # General W2 IC with rotation alpha (Williamson 1992 eqs 75-79)
    u_east = u_0 * (jnp.cos(lat) * jnp.cos(alpha) +
                     jnp.cos(lon) * jnp.sin(lat) * jnp.sin(alpha))
    v_north = -u_0 * jnp.sin(lon) * jnp.sin(alpha)
    
    # Geostrophic h (Williamson eq. 79)
    h_data = h_0 - (R * Omega * u_0 + u_0**2/2) * (
        -jnp.cos(lon)*jnp.cos(lat)*jnp.sin(alpha) + jnp.sin(lat)*jnp.cos(alpha)
    )**2 / g
    
    # Convert to D-grid (edge-midpoint) using cell-centre angles
    # Use simple approximation via cell-centre rotation angles
    # We need u at u-edges and v at v-edges; for simplicity use IC at cell centre then average
    cos_a = jnp.cos(grid.angle)
    sin_a = jnp.sin(grid.angle)
    u_grid = cos_a * u_east + sin_a * v_north
    v_grid = -sin_a * u_east + cos_a * v_north
    
    # Pad to D-grid edge positions (interior only)
    n = N
    u_d = jnp.zeros((6, n, n+1))
    u_d = u_d.at[:, :, 1:n].set(0.5 * (u_grid[:, :, :n-1] + u_grid[:, :, 1:n]))
    u_d = u_d.at[:, :, 0].set(u_grid[:, :, 0])
    u_d = u_d.at[:, :, n].set(u_grid[:, :, n-1])
    
    v_d = jnp.zeros((6, n+1, n))
    v_d = v_d.at[:, 1:n, :].set(0.5 * (v_grid[:, :n-1, :] + v_grid[:, 1:n, :]))
    v_d = v_d.at[:, 0, :].set(v_grid[:, 0, :])
    v_d = v_d.at[:, n, :].set(v_grid[:, n-1, :])
    
    h_s_data = jnp.zeros_like(h_data)
    state = FV3EdgeShallowWaterState(h=h_data, u_d=u_d, v_d=v_d, h_s=h_s_data)
    
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(int(86400/DT)):
            state = model.step(state, DT)
    
    h_arr = np.asarray(state.h)
    h_err = float(np.max(np.abs(h_arr - np.asarray(h_data))))
    return h_err, np.isfinite(h_arr).all()

print("=== Iter-1025 W2 alpha sweep (iter-1021 calibration) ===", flush=True)
print(f"alpha   | h_err_max | finite", flush=True)
for alpha_deg in [0, 15, 30, 45, 60, 90]:
    alpha = jnp.radians(alpha_deg)
    h_err, finite = run_w2_alpha(alpha)
    print(f"{alpha_deg:>5}°  | {h_err:>9.2f} | {finite}", flush=True)
