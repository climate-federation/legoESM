"""Iter-779 diagnostic: cosine bell 1-day sweep at FIXED dt across
4 resolutions.

Iter-778 measured L1/L2/Linf at C16/C24/C36/C48 using
dt=1800*(36/n) (CFL-scaled).  Result: non-monotone error plateau
above C24.  One candidate mechanism listed was:
  (b) dt scaling vs spatial-error crossover.

Iter-779 re-runs the same sweep at FIXED dt across all 4
resolutions.  dt=1350s is CFL-safe at C48 (the highest
resolution); using it at lower resolutions just means more
sub-steps, not a stability issue.

If the plateau PERSISTS at fixed dt, candidate (b) is ruled out
at that dt setting.  If the plateau DISAPPEARS at fixed dt, the
plateau in iter-778 was due to dt scaling.

Scope limits (same as iter-778): 4 discrete n values, 1 IC,
1 horizon, 1 dt policy.  A Fortran-oracle comparison is NOT
performed here.
"""
import os, sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
for _stale in ("JAX_PLATFORM_NAME", "JAX_DISABLE_JIT", "JAX_DEBUG_NANS"):
    os.environ.pop(_stale, None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel)
from legoesm.core.fv3_sw_core import _d2a2c_vect
from legoesm.core.fv_tp_2d import transport_step
from tests.test_cases.cosine_bell import (
    cosine_bell_cubesphere, cosine_bell_exact, cosine_bell_error_norms)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def run_cb(n, dt, days=1):
    n_steps = int(round(days * 86400 / dt))
    dt_exact = days * 86400 / n_steps

    beta = jnp.pi / 4.0
    grid = create_cubed_sphere(n)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=_div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2)
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid
    state = cosine_bell_cubesphere(grid, cdgrid, beta)
    _ua, _va, _uc, _vc, ut, vt = _d2a2c_vect(state.u_d, state.v_d, cdgrid)
    mass_init = float(jnp.sum(state.h * grid.area))

    h = state.h
    for _ in range(n_steps):
        h = transport_step(h, ut, vt, dt_exact, cdgrid,
                            mass_target=mass_init)

    t_s = days * 86400.0
    h_exact = cosine_bell_exact(grid.lon, grid.lat, grid.radius, t_s, beta)
    norms = cosine_bell_error_norms(h, h_exact, grid.area)
    peak_h = float(np.max(np.asarray(h)))
    peak_ex = float(np.max(np.asarray(h_exact)))
    und = (peak_ex - peak_h) / peak_ex
    return (float(norms['l1']), float(norms['l2']),
            float(norms['linf']), und, dt_exact, n_steps)


DT_FIXED = 1350.0   # CFL-safe at C48

print(f"Iter-779 cosine bell FIXED-dt convergence sweep")
print(f"β=π/4, 1 day, dt={DT_FIXED:.0f}s at all n (CFL-safe at C48)")
print()
print(f"{'n':>4}  {'dt':>7}  {'nsteps':>6}  "
      f"{'L1':>10}  {'L2':>10}  {'Linf':>10}  "
      f"{'undershoot':>11}")
print("-" * 74)
results = []
for n in (16, 24, 36, 48):
    l1, l2, linf, und, dt_out, nsteps = run_cb(n, DT_FIXED)
    results.append((n, l1, l2, linf, und))
    print(f"  {n:>2}  {dt_out:>7.1f}  {nsteps:>6}  "
          f"{l1:>10.3e}  {l2:>10.3e}  {linf:>10.3e}  "
          f"{und:>10.3%}")

print()
print(f"Observed convergence order p = log(err(n1)/err(n2)) / log(n2/n1):")
print(f"{'pair':>10}  {'p(L1)':>8}  {'p(L2)':>8}  {'p(Linf)':>10}")
for i in range(len(results) - 1):
    n1, l1_1, l2_1, linf_1, _ = results[i]
    n2, l1_2, l2_2, linf_2, _ = results[i + 1]
    ratio_n = n2 / n1
    p_l1 = math.log(l1_1 / l1_2) / math.log(ratio_n)
    p_l2 = math.log(l2_1 / l2_2) / math.log(ratio_n)
    p_linf = math.log(linf_1 / linf_2) / math.log(ratio_n)
    print(f"  {n1:>3}->{n2:<3}  "
          f"{p_l1:>8.3f}  {p_l2:>8.3f}  {p_linf:>10.3f}")

print()
print(f"Iter-778 CFL-scaled reference (dt = 1800*(36/n)):")
print(f"   n   dt    L1         L2         Linf       und")
print(f"  16  4114  1.72e-01   1.52e-01   1.73e-01   15.0%")
print(f"  24  2700  1.24e-01   1.16e-01   1.17e-01   11.2%")
print(f"  36  1800  1.20e-01   1.17e-01   1.23e-01    9.5%")
print(f"  48  1350  1.21e-01   1.20e-01   1.27e-01    8.8%")
