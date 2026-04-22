"""Iter-775 diagnostic: cross-test validation of iter-774's
alpha=0.98 scaling of grad_c10 at 16 peak-updating D-grid
corners.

Iter-774 showed alpha=0.98 reduces W2 v_ll_Linf by 30 percent on
C36 1-day.  But a correction that only helps W2 and hurts W5 is
not a universal fix.  This diagnostic runs W5 at both alpha=1.0
(default) and alpha=0.98 and compares mass drift, max |h-h_ic|,
and h range.  Cosine bell is NOT affected because it uses
transport_step directly, not model.step (NOTE in
scripts/run_atmosphere_test_matrix.py:1568-1576).

If alpha=0.98 significantly worsens W5 or produces blow-up:
iter-774's finding is W2-specific, not a universal fix.
If alpha=0.98 leaves W5 essentially unchanged or improves it:
the W2 reduction may generalize (still not Fortran-faithful
without derivation comparison — see iter-774 caveats).
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
    create_cubed_sphere_cdgrid)
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test5)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


n = 36
dt = 300.0
days = 1
n_steps = int(days * 86400 / dt)

grid = create_cubed_sphere(n)
u0 = 20.0    # W5 reference zonal wind speed
sw = williamson_test5(grid)

PEAK_CELLS = [(0, 34, 0), (0, 34, 35), (2, 34, 0), (2, 34, 35)]


def run_w5(alpha: float):
    cdg = create_cubed_sphere_cdgrid(grid)
    if alpha != 1.0:
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

    h_ic = np.asarray(sw.h.data)
    area = np.asarray(grid.area)
    mass_ic = float(np.sum(h_ic * area))

    for _ in range(n_steps):
        state = model.step(state, dt)

    h_final = np.asarray(state.h)
    mass_final = float(np.sum(h_final * area))
    mass_drift = abs(mass_final - mass_ic) / mass_ic

    # W5 has no analytic solution; measure h range + diff vs IC
    h_diff = h_final - h_ic
    h_diff_linf = float(np.max(np.abs(h_diff)))
    h_min = float(np.min(h_final))
    h_max = float(np.max(h_final))

    return mass_drift, h_diff_linf, h_min, h_max


print(f"Iter-775 W5 cross-test of iter-774's alpha=0.98 c10")
print(f"correction at 16 peak-updating D-grid corners.")
print(f"C36, dt=300s, 1-day integration, same matrix config.")
print()
print(f"{'alpha':>8}  {'mass_drift':>12}  {'|h-h_ic|_Linf':>14}  "
      f"{'h_range (min, max)':>24}")
print("-" * 75)
for alpha in [1.00, 0.98]:
    md, hlinf, hmin, hmax = run_w5(alpha)
    marker = " [default]" if alpha == 1.00 else " [iter-774 cand]"
    print(f"  {alpha:>6.3f}  {md:>12.3e}  {hlinf:>14.3e}  "
          f"({hmin:>.4e}, {hmax:>.4e}){marker}")
