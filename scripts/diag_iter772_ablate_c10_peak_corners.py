"""Iter-772 diagnostic: ablate grad_c10 at the 16 D-grid corners
that surround the top-4 iter-770 peak cells.  Measures W2 C36 1-d
v_ll_Linf with the ablation vs without.

Motivation.  Iter-771b measured |c10| ~= 9x the interior median at
the D-grid corners that update the top-4 |v_north| peak cells.
If c10 (the cross-coupling from dB_raw_x into dB_dy_perp) is a
causal contributor to mode A, zeroing it at these 16 D-grid
corners should reduce v_ll_Linf.

Scope.  This is an ABLATION diagnostic: it mutates the cdgrid
metric in-place after construction, then runs W2.  The mutation
is LOCAL to 16 D-grid corner positions (the 4 surrounding each of
the top-4 iter-770 peak cells) and does not affect the rest of
the grid.  No source-code change to operators.

Expected outcome.
- v_ll_Linf drops: c10 is a causal mode-A contributor.  Iter-773+
  can explore alternative stencil formulations that reduce c10.
- v_ll_Linf unchanged: c10 is benign at these corners.  Mechanism
  is elsewhere.
- v_ll_Linf rises: zeroing c10 misrepresents the grid metric and
  makes the stencil inconsistent; ablation contraindicates this
  direction.
"""
import os, sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
for _stale in ("JAX_PLATFORM_NAME", "JAX_DISABLE_JIT", "JAX_DEBUG_NANS"):
    os.environ.pop(_stale, None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import (
    create_cubed_sphere_cdgrid, cell_centre_angles_from_4edge,
    CubedSphereCDGrid)
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from legoesm.grids.regridding import (
    get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2, williamson_test2_exact)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


n = 36
dt = 300.0
days = 1
n_steps = int(days * 86400 / dt)

grid = create_cubed_sphere(n)
u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
sw = williamson_test2(grid)
exact = williamson_test2_exact(grid, days * 86400.0)
weights = get_cubedsphere_to_latlon_weights(n, 360, 181)


def _build_ablated_cdgrid(ablate_c10_peak_corners: bool):
    """Construct cdgrid, optionally zero grad_c10 at the 16 peak-
    updating D-grid corners (4 surrounding each of 4 peak cells
    from iter-770)."""
    cdg = create_cubed_sphere_cdgrid(grid)
    if not ablate_c10_peak_corners:
        return cdg
    c10 = np.array(cdg.grad_c10, copy=True)  # shape (6, n+1, n+1)
    # Iter-770 top-4 peak cells (face, cell i, cell j):
    peak_cells = [(0, 34, 0), (0, 34, 35), (2, 34, 0), (2, 34, 35)]
    # For each peak cell (ci, cj), the 4 surrounding D-grid corners
    # are at (ci, cj), (ci, cj+1), (ci+1, cj), (ci+1, cj+1).
    for face, ci, cj in peak_cells:
        for di in (0, 1):
            for dj in (0, 1):
                c10[face, ci + di, cj + dj] = 0.0
    return cdg._replace(grad_c10=jnp.asarray(c10))


def run_and_measure(ablate: bool):
    cdg = _build_ablated_cdgrid(ablate)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
    )
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    # Replace the model's internal cdgrid with our ablated one.
    model.cdgrid = cdg
    cdgrid = model.cdgrid

    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    h_exact = np.asarray(exact.h.data)
    area = np.asarray(grid.area)
    for _ in range(n_steps):
        state = model.step(state, dt)

    h_err = np.asarray(state.h) - h_exact
    h_l2 = float(np.sqrt(np.sum(h_err ** 2 * area)
                          / np.sum(h_exact ** 2 * area)))
    h_linf = float(np.max(np.abs(h_err)) / np.max(np.abs(h_exact)))

    ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdgrid)
    u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                   + np.asarray(state.u_d)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                   + np.asarray(state.v_d)[:, 1:, :])
    v_north = np.asarray(sa_4edge) * u_cc + np.asarray(ca_4edge) * v_cc
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    v_ll_linf = float(np.max(np.abs(v_ll)))
    return h_l2, h_linf, v_ll_linf


for flag, label in [(False, "OFF (unablated, default)   "),
                     (True,  "ON  (c10=0 at peak corners)")]:
    l2, linf, vllinf = run_and_measure(flag)
    print(f"[{label}]  h_L2={l2:.3e}  h_Linf={linf:.3e}  "
          f"v_ll_Linf={vllinf:.3e}")

print("\nCanonical OFF baseline (iter-768 commit-time):")
print("  h_L2=2.07e-04  h_Linf=1.53e-03  v_ll_Linf=1.59e-01")
