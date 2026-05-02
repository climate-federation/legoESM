"""Iter-773 diagnostic: ranged ablation sweep of `grad_c10` at the
16 peak-updating D-grid corners from iter-770.

Iter-772 tested ONE substitution (full zero) at the 16 D-grid
corners surrounding the top-4 iter-770 peak cells.  Zero made W2
v_ll_Linf 31x worse.  Iter-772b's scope-limits note explicitly
flagged this test: "a smaller non-zero value might be better"
was not examined.

Iter-773 runs a scalar sweep:
  c10[16 corners] = alpha * c10_default
for alpha in {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}.  Measures
W2 C36 1-d v_ll_Linf at each alpha.

If the minimum of v_ll_Linf is at alpha = 1.0 (current value),
the current c10 magnitude is the best on this 1D slice.  If the
minimum is elsewhere, that alpha's value is the better choice
and iter-774+ could ship the corresponding scale factor.

Scope.  Same as iter-772: mutation of cdgrid.grad_c10 at 16
specific D-grid corner positions; no source-code change; no
effect on the rest of the grid.
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


alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
print(f"Iter-773 c10 ranged ablation at 16 peak-updating D-grid corners")
print(f"(top-4 iter-770 peak cells surrounded by 4 corners each).")
print()
print(f"{'alpha':>8}  {'h_L2':>10}  {'v_ll_Linf':>10}  {'vs alpha=1':>12}")
print("-" * 50)
baseline_v = None
results = []
for alpha in alphas:
    l2, vll = run_and_measure(alpha)
    if alpha == 1.0:
        baseline_v = vll
    results.append((alpha, l2, vll))
for alpha, l2, vll in results:
    rel = vll / baseline_v if baseline_v else float("nan")
    marker = " [default]" if alpha == 1.0 else ""
    print(f"  {alpha:>6.2f}  {l2:>10.3e}  {vll:>10.3e}  {rel:>10.3f}x{marker}")
