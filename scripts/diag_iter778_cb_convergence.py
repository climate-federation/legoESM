"""Iter-778 diagnostic: cosine bell 1-day convergence sweep at
C16, C24, C36, C48.

PPM (Lin-Rood) is expected to be ~3rd-order accurate on smooth
solutions (linear problems with constant advection).  That means
the error norms should scale as h^3, where h ~ 1/n is the grid
spacing.  Doubling n should reduce error by factor ~8.

Iter-776 measured the C36 baseline:
  L1 = 1.20e-1, L2 = 1.17e-1, Linf = 1.23e-1, undershoot = 9.5%

Iter-778 measures the same quantities at C16, C24, C36, C48 and
reports:
- Error norms at each resolution
- Observed convergence order between consecutive resolutions
  p = log(err(n1)/err(n2)) / log(n2/n1)

Expected outcomes:
- p ≈ 3: PPM truncation is the dominant error; higher C reduces it.
- p ≈ 1 or 2: super-ideal (only PPM shape-preserving limiters active)
  or sub-ideal (structural error dominates at low resolution, PPM
  truncation at high).
- p < 1: likely a structural error that does not converge — would
  indicate a bug in halo handling, _d2a2c_vect, or cube-corner
  treatment that doesn't decay with resolution.

Uses the canonical matrix cosine bell setup (β=π/4, dt=1800s for
C36, CFL-scaled for other resolutions).
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


def run_cb(n, days=1, dt=None):
    # CFL-scale dt by resolution if not given.  Matrix uses dt=1800
    # for C36; scale by ref/n.
    if dt is None:
        dt = 1800.0 * (36 / n)
    n_steps = int(days * 86400 / dt)
    # Ensure n_steps * dt covers the full day.
    dt = days * 86400 / n_steps

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
        h = transport_step(h, ut, vt, dt, cdgrid,
                            mass_target=mass_init)

    t_s = days * 86400.0
    h_exact = cosine_bell_exact(grid.lon, grid.lat, grid.radius, t_s, beta)
    norms = cosine_bell_error_norms(h, h_exact, grid.area)
    peak_h = float(np.max(np.asarray(h)))
    peak_ex = float(np.max(np.asarray(h_exact)))
    und = (peak_ex - peak_h) / peak_ex
    return (float(norms['l1']), float(norms['l2']),
            float(norms['linf']), und, dt, n_steps)


print(f"Iter-778 cosine bell convergence sweep at β=π/4, 1 day")
print()
print(f"{'n':>4}  {'dt':>7}  {'nsteps':>6}  "
      f"{'L1':>10}  {'L2':>10}  {'Linf':>10}  "
      f"{'undershoot':>11}")
print("-" * 74)
results = []
for n in (16, 24, 36, 48):
    l1, l2, linf, und, dt, nsteps = run_cb(n)
    results.append((n, l1, l2, linf, und))
    print(f"  {n:>2}  {dt:>7.1f}  {nsteps:>6}  "
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
