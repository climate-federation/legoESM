"""Iter-658 / iter-659 FB-chain blowup trace diagnostic.

Traces per-step max|h|, max|u_d|, max|v_d|, max|v_err|, and the
(face, i, j) location of the max v-error for the first 30 steps of
a Williamson 2 run at C36 through ``FV3FBShallowWaterModel``.

Iter-658 observed trajectory (where MAX v_err lives each step):
  step 1:  v_err=3.14  at face 0, i=36, j=0
  step 4:  v_err=12.4  at face 3, i=0,  j=0
  step 14: v_err=67    at face 1, i=0,  j=0   (exp growth begins)
  step 26: v_err=1e14  at face 0, i=0,  j=0
  step 27: NaN

**Iter-659 correction (Codex finding)**: the initial iter-658
claim "localize to cube vertices" was WRONG.  An error-distribution
histogram at step 1 showed:
  max at cube VERTICES (24 cells): 3.16 m/s
  max at cube EDGES   (828 cells): 3.00 m/s
  max at INTERIOR    (7140 cells): 2.15 m/s
  mean at vertices:  2.40 m/s
  mean at edges:     1.45 m/s
  mean at interior:  1.04 m/s
  cells with v_err > 1.0 m/s: 4432 of ~8000 (55%)

The error is FIELD-WIDE, not concentrated at a few cells.  Vertices
are 1.5-2× the interior magnitude, but the interior dominates total
error by cell count.  The MAX location tracking only reports a
single point and gives a misleading "localization" impression.

**Iter-660 (Codex correction on iter-659)**: the histogram data
above only supports the observation "step-1 v_err is widespread".
It does NOT by itself prove any specific cause.  Propagation-speed
arguments (wind ~23 km/step, gravity waves ~100 km/step, dx~600 km
at C36) rule out a single-cell bug spreading field-wide in one step,
but candidate root causes (IC-scheme mismatch vs. FB coupling
residual vs. temporal-scheme residual) need additional diagnostics
(e.g., step-1 tendency magnitudes, dt-scaling of residuals) to
separate — those are future work.
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

# Williamson 2 IC
g = 9.80616; omega = 7.292e-5; u_0 = 38.61068276698372
h_0 = 29400.0 / g; R = cdgrid.radius
lat_c = cdgrid.base.lat
h_init = h_0 - (R * omega * u_0 + 0.5*u_0**2) * jnp.sin(lat_c)**2 / g
lat_ux = cdgrid.lat_edge_x
u_d = u_0 * jnp.cos(lat_ux) * cdgrid.cos_angle_edge_x
lat_vy = cdgrid.lat_edge_y
v_d = -u_0 * jnp.cos(lat_vy) * cdgrid.sin_angle_edge_y
state = FV3EdgeShallowWaterState(h=h_init, u_d=u_d, v_d=v_d, h_s=jnp.zeros_like(h_init))
model.set_initial_mass(state)

dt = 600.0
u_exact = u_0 * jnp.cos(cdgrid.lat_edge_x) * cdgrid.cos_angle_edge_x
v_exact = -u_0 * jnp.cos(cdgrid.lat_edge_y) * cdgrid.sin_angle_edge_y

print(f"{'step':>5s} {'max|h|':>10s} {'max|u_d|':>10s} {'max|v_d|':>10s} "
      f"{'max|v_err|':>12s} {'where':>20s}")
print(f"{'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*20}")

for step in range(30):
    try:
        state = model.step(state, dt)
    except Exception as e:
        print(f"step {step+1}: EXCEPTION {e}")
        break
    h = np.asarray(state.h); u = np.asarray(state.u_d); v = np.asarray(state.v_d)
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(v)):
        # Find where NaN/inf appears
        bad_v = np.argwhere(~np.isfinite(v))
        loc = f"face={bad_v[0,0]}, i={bad_v[0,1]}, j={bad_v[0,2]}" if len(bad_v) > 0 else "?"
        print(f"{step+1:>5d} {float(np.max(np.abs(h))):>10.3e} "
              f"{float(np.max(np.abs(u))):>10.3e} {float(np.max(np.abs(v))):>10.3e} "
              f"{'inf/nan':>12s} {loc:>20s}")
        break
    v_err = float(np.max(np.abs(v - np.asarray(v_exact))))
    v_err_idx = np.argmax(np.abs(v - np.asarray(v_exact)))
    v_err_loc = np.unravel_index(v_err_idx, v.shape)
    print(f"{step+1:>5d} {float(np.max(np.abs(h))):>10.3e} "
          f"{float(np.max(np.abs(u))):>10.3e} {float(np.max(np.abs(v))):>10.3e} "
          f"{v_err:>12.3e} f{v_err_loc[0]},i{v_err_loc[1]},j{v_err_loc[2]:>5d}")
