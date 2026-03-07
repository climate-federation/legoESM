#!/usr/bin/env python
"""Diagnose FV cubed-sphere face-boundary artifacts on Williamson TC2.

TC2 is a steady geostrophic flow — any deviation is numerical error.
This script measures:
1. Error growth over time
2. Error concentration at face boundaries vs interior
3. Effect of different fixes
"""

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv import (
    FVShallowWaterModel,
    FVShallowWaterConfig,
    fv_shallow_water_tendencies,
)
from legoesm.atmosphere.dynamics.shallow_water import (
    ShallowWaterModel,
    shallow_water_tendencies,
)
from legoesm.core.operators_fv_cubed import default_div_damp_coeffs
from tests.test_cases.williamson import williamson_test2, compute_error_norms

N = 24  # C24
grid = create_cubed_sphere(N)
ic = williamson_test2(grid)
ref = ic  # TC2 is steady

# Build face-boundary mask: cells within `depth` of any edge
def face_boundary_mask(n, depth=2):
    m = np.zeros((6, n, n), dtype=bool)
    m[:, :depth, :] = True   # WEST
    m[:, -depth:, :] = True  # EAST
    m[:, :, :depth] = True   # SOUTH
    m[:, :, -depth:] = True  # NORTH
    return jnp.array(m)

bnd_mask = face_boundary_mask(N, depth=2)
int_mask = ~bnd_mask

# --- 1. Measure single-step tendency errors ---
print("=" * 60)
print("Single-step tendency analysis (TC2 should have zero dh/dt)")
print("=" * 60)

# FV tendencies
nu2, nu4 = default_div_damp_coeffs(grid, dt=450.0)
fv_config = FVShallowWaterConfig(
    div_damp_2=nu2, div_damp_4=nu4,
    hyperdiff_coeff=0.0,
    use_conservation_fixer=True,
)
tend_fv = fv_shallow_water_tendencies(ic, grid, fv_config)

# Centered tendencies for comparison
tend_cen = shallow_water_tendencies(ic, grid)

print(f"\nFV dh/dt:  max={float(jnp.max(jnp.abs(tend_fv.dh_dt.data))):.6e}")
print(f"  boundary: max={float(jnp.max(jnp.abs(tend_fv.dh_dt.data[bnd_mask]))):.6e}")
print(f"  interior: max={float(jnp.max(jnp.abs(tend_fv.dh_dt.data[int_mask]))):.6e}")
print(f"  ratio bnd/int: {float(jnp.max(jnp.abs(tend_fv.dh_dt.data[bnd_mask]))) / max(float(jnp.max(jnp.abs(tend_fv.dh_dt.data[int_mask]))), 1e-30):.1f}x")

print(f"\nFV du/dt:  max={float(jnp.max(jnp.abs(tend_fv.du_dt.data))):.6e}")
print(f"  boundary: max={float(jnp.max(jnp.abs(tend_fv.du_dt.data[bnd_mask]))):.6e}")
print(f"  interior: max={float(jnp.max(jnp.abs(tend_fv.du_dt.data[int_mask]))):.6e}")

print(f"\nFV dv/dt:  max={float(jnp.max(jnp.abs(tend_fv.dv_dt.data))):.6e}")
print(f"  boundary: max={float(jnp.max(jnp.abs(tend_fv.dv_dt.data[bnd_mask]))):.6e}")
print(f"  interior: max={float(jnp.max(jnp.abs(tend_fv.dv_dt.data[int_mask]))):.6e}")

print(f"\nCentered dh/dt: max={float(jnp.max(jnp.abs(tend_cen.dh_dt.data))):.6e}")
print(f"  boundary: max={float(jnp.max(jnp.abs(tend_cen.dh_dt.data[bnd_mask]))):.6e}")
print(f"  interior: max={float(jnp.max(jnp.abs(tend_cen.dh_dt.data[int_mask]))):.6e}")

print(f"\nCentered du/dt: max={float(jnp.max(jnp.abs(tend_cen.du_dt.data))):.6e}")
print(f"  boundary: max={float(jnp.max(jnp.abs(tend_cen.du_dt.data[bnd_mask]))):.6e}")
print(f"  interior: max={float(jnp.max(jnp.abs(tend_cen.du_dt.data[int_mask]))):.6e}")


# --- 2. Time integration error growth ---
print("\n" + "=" * 60)
print("Error growth over time (5 days, dt=450s)")
print("=" * 60)

dt = 450.0
n_steps = int(5 * 86400 / dt)
model_fv = FVShallowWaterModel(grid, fv_config)
model_cen = ShallowWaterModel(grid)

state_fv = ic
state_cen = ic

for i in range(n_steps):
    state_fv = model_fv.step(state_fv, dt)
    state_cen = model_cen.step(state_cen, dt)

    if (i + 1) % (n_steps // 5) == 0:
        day = (i + 1) * dt / 86400.0

        err_fv = state_fv.h.data - ref.h.data
        err_cen = state_cen.h.data - ref.h.data

        norms_fv = compute_error_norms(state_fv, ref, grid)
        norms_cen = compute_error_norms(state_cen, ref, grid)

        fv_bnd = float(jnp.max(jnp.abs(err_fv[bnd_mask])))
        fv_int = float(jnp.max(jnp.abs(err_fv[int_mask])))
        cen_bnd = float(jnp.max(jnp.abs(err_cen[bnd_mask])))
        cen_int = float(jnp.max(jnp.abs(err_cen[int_mask])))

        print(f"\nDay {day:.1f}:")
        print(f"  FV  L2={norms_fv['l2']:.4e}  Linf={norms_fv['linf']:.4e}  "
              f"bnd={fv_bnd:.3e}  int={fv_int:.3e}  ratio={fv_bnd/max(fv_int,1e-30):.1f}x")
        print(f"  Cen L2={norms_cen['l2']:.4e}  Linf={norms_cen['linf']:.4e}  "
              f"bnd={cen_bnd:.3e}  int={cen_int:.3e}  ratio={cen_bnd/max(cen_int,1e-30):.1f}x")

        # Check velocity errors
        u_err_fv = float(jnp.max(jnp.abs(state_fv.u.data - ic.u.data)))
        v_err_fv = float(jnp.max(jnp.abs(state_fv.v.data - ic.v.data)))
        u_err_cen = float(jnp.max(jnp.abs(state_cen.u.data - ic.u.data)))
        v_err_cen = float(jnp.max(jnp.abs(state_cen.v.data - ic.v.data)))
        print(f"  FV  |du|={u_err_fv:.3e}  |dv|={v_err_fv:.3e}")
        print(f"  Cen |du|={u_err_cen:.3e}  |dv|={v_err_cen:.3e}")

print("\nDone.")
