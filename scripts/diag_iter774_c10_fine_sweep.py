"""Iter-774 diagnostic: fine-grained sweep of `grad_c10` scale
factor alpha near 1.0 at the 16 peak-updating D-grid corners.

Iter-773 sampled alpha in {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}
with 0.25 steps.  Of those 7, alpha=1.0 gave the smallest
v_ll_Linf.  Iter-773b's explicit scope-limit: "finer-grained
alpha near 1.0 (e.g. alpha=0.95, 1.05) were not measured" was
listed as an iter-774+ candidate.

Iter-774 runs the same W2 C36 1-d sweep at
  alpha in {0.90, 0.95, 0.98, 0.99, 1.00, 1.01, 1.02, 1.05, 1.10}

If the fine-grained minimum is at alpha != 1.0 (e.g. alpha=0.99
or 1.01), a small fixed correction factor could reduce mode A.
If it remains at alpha = 1.0, the default value is the minimum
among the 9 fine-grained points as well.
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
    create_cubed_sphere_cdgrid, cell_centre_angles_from_4edge)
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

PEAK_CELLS = [(0, 34, 0), (0, 34, 35), (2, 34, 0), (2, 34, 35)]


def run_and_measure(alpha: float):
    cdg = create_cubed_sphere_cdgrid(grid)
    c10 = np.array(cdg.grad_c10, copy=True)
    for face, ci, cj in PEAK_CELLS:
        for di in (0, 1):
            for dj in (0, 1):
                c10[face, ci + di, cj + dj] *= alpha
    cdg = cdg._replace(grad_c10=jnp.asarray(c10))

    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
    )
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    model.cdgrid = cdg

    u_d = cdg.cos_angle_edge_x * (u0 * jnp.cos(cdg.lat_edge_x))
    v_d = -cdg.sin_angle_edge_y * (u0 * jnp.cos(cdg.lat_edge_y))
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

    ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdg)
    u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                   + np.asarray(state.u_d)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                   + np.asarray(state.v_d)[:, 1:, :])
    v_north = np.asarray(sa_4edge) * u_cc + np.asarray(ca_4edge) * v_cc
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    return h_l2, float(np.max(np.abs(v_ll)))


alphas = [0.90, 0.95, 0.98, 0.99, 1.00, 1.01, 1.02, 1.05, 1.10]
print(f"Iter-774 fine-grained c10 sweep near alpha=1.0 at 16")
print(f"peak-updating D-grid corners.")
print()
print(f"{'alpha':>8}  {'h_L2':>10}  {'v_ll_Linf':>11}  {'vs alpha=1':>11}")
print("-" * 50)
results = []
baseline_v = None
for alpha in alphas:
    l2, vll = run_and_measure(alpha)
    if abs(alpha - 1.0) < 1e-9:
        baseline_v = vll
    results.append((alpha, l2, vll))
for alpha, l2, vll in results:
    rel = vll / baseline_v if baseline_v else float("nan")
    marker = " [default]" if abs(alpha - 1.0) < 1e-9 else ""
    print(f"  {alpha:>6.3f}  {l2:>10.3e}  {vll:>11.3e}  {rel:>9.3f}x{marker}")

best = min(results, key=lambda r: r[2])
print()
print(f"Smallest v_ll_Linf among 9 samples: alpha = {best[0]:.3f}, "
      f"v_ll_Linf = {best[2]:.3e}")
