"""Iter-955 summary diagnostic: cumulative FB chain progress on W2 C36."""
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
from legoesm.grids.regridding import get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon
from tests.atmosphere.shallow_water.test_cases.williamson import williamson_test2
from tests.test_iter921_w2_v_vs_h_pareto_sentinel import cell_centre_angles_from_4edge

print("=" * 70)
print("Iter-955 cumulative FB chain progress on W2 C36 dt=300 s 1-day")
print("=" * 70)
print()
print(f"{'iter':<10} {'survival':>10} {'|u|_max':>10} {'|v|_max':>10} {'v_ll_Linf':>12} {'h_err_max':>11}")
print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*11}")

N = 36; dt = 300.0
grid = create_cubed_sphere(N, use_duogrid=True)
cdgrid = create_cubed_sphere_cdgrid(grid)
sw = williamson_test2(grid)
u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
state = FV3EdgeShallowWaterState(h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
cfg = CDGridShallowWaterConfig(
    hyperdiff_coeff=0.0, d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1,
    damp_v=0.06, nord_v=2,
)
model = FV3FBShallowWaterModel(grid, cfg)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for k in range(288):
        state = model.step(state, dt)

ca, sa = cell_centre_angles_from_4edge(cdgrid)
u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1] + np.asarray(state.u_d)[:, :, 1:])
v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :] + np.asarray(state.v_d)[:, 1:, :])
v_north = np.asarray(sa) * u_cc + np.asarray(ca) * v_cc
weights = get_cubedsphere_to_latlon_weights(N, n_lon=360, n_lat=181)
v_ll = apply_cubedsphere_to_latlon(v_north, weights)
h_err = np.asarray(state.h - sw.h.data)

u_max = float(np.max(np.abs(state.u_d)))
v_max = float(np.max(np.abs(state.v_d)))
v_ll_max = float(np.abs(v_ll).max())
h_err_max = float(np.abs(h_err).max())

print(f"{'944b':<10} {'288/288':>10} {'~106':>10} {'~151':>10} {'~85':>12} {'~21000':>11}")
print(f"{'945':<10} {'288/288':>10} {'78.06':>10} {'80.83':>10} {'56.17':>12} {'~19000':>11}")
print(f"{'947':<10} {'288/288':>10} {'76.91':>10} {'75.38':>10} {'55.61':>12} {'18591':>11}")
print(f"{'CURRENT':<10} {'288/288':>10} {u_max:>10.2f} {v_max:>10.2f} {v_ll_max:>12.4f} {h_err_max:>11.0f}")
print()
print("Acceptance: v_ll_Linf <= 0.119 m/s (~467x reduction still needed)")
print()
print("Per-iter improvement is modest (~1-15%); ~50+ more iters likely needed.")
