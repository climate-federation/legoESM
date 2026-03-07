#!/usr/bin/env python
"""Test different fixes for FV cubed-sphere face-boundary artifacts.

Runs Williamson TC2 at C24 for 5 days with different configurations
to identify the most effective fix.
"""

import sys
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "tests")

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import pad_halo
from legoesm.atmosphere.dynamics.shallow_water_fv import (
    FVShallowWaterModel,
    FVShallowWaterConfig,
    fv_shallow_water_tendencies,
)
from legoesm.core.operators_fv_cubed import default_div_damp_coeffs
from test_cases.williamson import williamson_test2, compute_error_norms

N = 24
grid = create_cubed_sphere(N)
ic = williamson_test2(grid)
ref = ic

dt = 450.0
n_steps = int(5 * 86400 / dt)

def face_boundary_mask(n, depth=2):
    m = np.zeros((6, n, n), dtype=bool)
    m[:, :depth, :] = True
    m[:, -depth:, :] = True
    m[:, :, :depth] = True
    m[:, :, -depth:] = True
    return jnp.array(m)

bnd_mask = face_boundary_mask(N, depth=2)

dx_min = float(jnp.min(grid.dx / 2.0))

# --- Configurations to test ---
configs = {}

# 1. Baseline: div damping only (no hyperdiff)
nu2, nu4 = default_div_damp_coeffs(grid, dt=dt)
configs["baseline"] = FVShallowWaterConfig(
    div_damp_2=nu2, div_damp_4=nu4,
    hyperdiff_coeff=0.0,
)

# 2. Add moderate hyperdiffusion
hyp = 0.02 * dx_min**4 / dt
configs["hyperdiff_0.02"] = FVShallowWaterConfig(
    div_damp_2=nu2, div_damp_4=nu4,
    hyperdiff_coeff=hyp,
)

# 3. Stronger hyperdiffusion
hyp2 = 0.05 * dx_min**4 / dt
configs["hyperdiff_0.05"] = FVShallowWaterConfig(
    div_damp_2=nu2, div_damp_4=nu4,
    hyperdiff_coeff=hyp2,
)

# 4. Double div damping + moderate hyperdiff
configs["2x_divdamp+hyp"] = FVShallowWaterConfig(
    div_damp_2=2*nu2, div_damp_4=2*nu4,
    hyperdiff_coeff=hyp,
)

print(f"Grid: C{N}, dx_min={dx_min:.0f} m, dt={dt} s")
print(f"nu2={nu2:.3e}, nu4={nu4:.3e}")
print(f"hyp_0.02={hyp:.3e}, hyp_0.05={hyp2:.3e}")
print(f"Steps: {n_steps}")
print()

for name, config in configs.items():
    print(f"--- {name} ---")
    model = FVShallowWaterModel(grid, config)
    state = ic

    for i in range(n_steps):
        state = model.step(state, dt)
        if jnp.any(~jnp.isfinite(state.h.data)):
            print(f"  BLEW UP at step {i+1}")
            break

    if jnp.all(jnp.isfinite(state.h.data)):
        norms = compute_error_norms(state, ref, grid)
        err_h = state.h.data - ref.h.data
        bnd_err = float(jnp.max(jnp.abs(err_h[bnd_mask])))
        int_err = float(jnp.max(jnp.abs(err_h[~bnd_mask])))
        u_err = float(jnp.max(jnp.abs(state.u.data - ic.u.data)))
        v_err = float(jnp.max(jnp.abs(state.v.data - ic.v.data)))

        print(f"  L2={norms['l2']:.4e}  Linf={norms['linf']:.4e}")
        print(f"  h_bnd={bnd_err:.1f}  h_int={int_err:.1f}  ratio={bnd_err/max(int_err,1e-30):.2f}x")
        print(f"  |du|={u_err:.2f}  |dv|={v_err:.2f}")
    print()

print("Done.")
