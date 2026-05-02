"""Iter-784 diagnostic: PPM limiter bypass for the cosine bell run.

Iter-783 ruled out `_d2a2c_vect`'s cube-vertex error as the
dominant driver of the C36 1-day cosine bell distortion.
Remaining iter-778 candidates are:
  (a) resolution-invariant structural error
  (d) PPM limiter.

Iter-784 tests candidate (d).  The current `_ppm_1d` applies
`_pert_ppm_iv0` (positive-definite constraint, FV3 hord=9) to
the PPM parabola coefficients.  Near the bell's zero transition
(r = R_0 boundary), the parabola has a gradient that the
limiter may clip, dissipating sub-grid structure and causing
long-term error growth.

Method:
  1. BASELINE: run with the normal `_pert_ppm_iv0` limiter.
  2. NO_LIMITER: monkey-patch `_pert_ppm_iv0` to return bl, br
     unchanged (no positivity clip, no extremum zeroing).  Also
     bypass the face-boundary `_pert_ppm(iv=1)` calls.

If NO_LIMITER Linf drops by ≥ 30%, candidate (d) is supported:
the PPM limiter is the dominant driver.  If Linf is unchanged
or worse, (d) is not primary and candidate (a) is more likely.

Scope: observational only.  The monkey-patch is a DIAGNOSTIC-
ONLY modification; the production code is not changed.  Without
the limiter, h CAN go negative; the `mass_target` fix in
`transport_step` then re-clips those to zero and rescales — so
we measure Linf on the limiter-bypassed-then-rescaled h-field.
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
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel)
from legoesm.core.fv3_sw_core import _d2a2c_vect
from legoesm.core import fv_tp_2d as fv_tp_2d_mod
from legoesm.core.fv_tp_2d import transport_step
from tests.test_cases.cosine_bell import (
    cosine_bell_cubesphere, cosine_bell_exact)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def _run(n, dt, days, disable_limiter=False):
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

    # Monkey-patch limiter functions if requested.
    orig_pert_iv0 = fv_tp_2d_mod._pert_ppm_iv0
    orig_pert_iv1 = fv_tp_2d_mod._pert_ppm
    if disable_limiter:
        fv_tp_2d_mod._pert_ppm_iv0 = lambda q, bl, br: (bl, br)
        fv_tp_2d_mod._pert_ppm = lambda bl, br: (bl, br)

    try:
        h = state.h
        for _ in range(n_steps):
            h = transport_step(h, ut, vt, dt_exact, cdgrid,
                                mass_target=mass_init)
    finally:
        # Restore originals.
        fv_tp_2d_mod._pert_ppm_iv0 = orig_pert_iv0
        fv_tp_2d_mod._pert_ppm = orig_pert_iv1

    t_s = days * 86400.0
    h_exact = cosine_bell_exact(grid.lon, grid.lat, grid.radius, t_s, beta)
    err = np.asarray(h) - np.asarray(h_exact)
    abs_err = np.abs(err)

    l_inf = float(np.max(abs_err))
    area_np = np.asarray(grid.area)
    h_exact_np = np.asarray(h_exact)
    l2_num = float(np.sum(area_np * err ** 2))
    l2_den = float(np.sum(area_np * h_exact_np ** 2))
    l2 = float(np.sqrt(l2_num / l2_den)) if l2_den > 0 else np.nan

    idx = np.unravel_index(np.argmax(abs_err), abs_err.shape)
    face, ci, cj = int(idx[0]), int(idx[1]), int(idx[2])
    lat = float(np.rad2deg(np.asarray(grid.lat)[face, ci, cj]))
    lon = float(np.rad2deg(np.asarray(grid.lon)[face, ci, cj]))
    if lon > 180: lon -= 360

    # Additional diagnostics: min / max of final h, and peak-error
    # SIGN (undershoot vs overshoot).
    h_np = np.asarray(h)
    h_min = float(np.min(h_np))
    h_max = float(np.max(h_np))
    peak_err_signed = float(err[face, ci, cj])

    return {
        'l_inf': l_inf,
        'l2': l2,
        'peak_face': face,
        'peak_cell': (ci, cj),
        'peak_lat': lat,
        'peak_lon': lon,
        'peak_err_signed': peak_err_signed,
        'h_min': h_min,
        'h_max': h_max,
    }


n = 36
dt = 1440.0
days = 1.0

print(f"Iter-784 cosine bell C36 with PPM limiter bypass")
print(f"dt={dt}s, β=π/4, {days} day")
print()
print(f"{'case':>24}  {'L_inf':>9}  {'L2':>9}  {'peak cell':>15}  "
      f"{'peak err':>10}  {'h_min':>8}  {'h_max':>8}")
print("-" * 95)

baseline = _run(n, dt, days, disable_limiter=False)
print(f"{'BASELINE (limiter ON)':>24}  {baseline['l_inf']:>9.3e}  "
      f"{baseline['l2']:>9.3e}  "
      f"{'face '+str(baseline['peak_face'])+' ('+str(baseline['peak_cell'][0])+','+str(baseline['peak_cell'][1])+')':>15}  "
      f"{baseline['peak_err_signed']:>+10.3e}  "
      f"{baseline['h_min']:>8.2f}  {baseline['h_max']:>8.2f}")

no_limiter = _run(n, dt, days, disable_limiter=True)
delta_linf = 100.0 * (no_limiter['l_inf'] - baseline['l_inf']) / baseline['l_inf']
delta_l2 = 100.0 * (no_limiter['l2'] - baseline['l2']) / baseline['l2']
print(f"{'NO LIMITER':>24}  {no_limiter['l_inf']:>9.3e}  "
      f"{no_limiter['l2']:>9.3e}  "
      f"{'face '+str(no_limiter['peak_face'])+' ('+str(no_limiter['peak_cell'][0])+','+str(no_limiter['peak_cell'][1])+')':>15}  "
      f"{no_limiter['peak_err_signed']:>+10.3e}  "
      f"{no_limiter['h_min']:>8.2f}  {no_limiter['h_max']:>8.2f}"
      f"  [Linf {delta_linf:+.1f}%, L2 {delta_l2:+.1f}%]")

print()
print("Interpretation cues (observational only):")
print("- If NO_LIMITER Linf is MUCH SMALLER (>=30% reduction):")
print("  candidate (d) 'PPM limiter' is supported as a primary")
print("  driver of the cosine bell distortion.")
print("- If NO_LIMITER Linf is similar or larger: the limiter is")
print("  NOT the primary driver.  Candidate (a) 'resolution-")
print("  invariant structural error' becomes more likely.")
print("- NO_LIMITER h_min / h_max may go below 0 / above H0=1000.")
print("  The mass_target fix then re-clips negatives to zero and")
print("  rescales, so the reported Linf is on the re-clipped field.")
print("- The peak_err sign (+/- signs before Linf) distinguishes")
print("  over- from undershoot; BASELINE typically undershoots (-)")
print("  because the limiter clips the PPM parabola.")

print()
print("What iter-784 DOES measure (observational only):")
print("- Linf and L2 of the C36 1-day cosine bell with and without")
print("  the `_pert_ppm_iv0` + face-boundary `_pert_ppm` limiters.")
print("- Shift in peak-error location and sign.")
print("What it does NOT establish:")
print("- Whether a different limiter (e.g. hord=8 2*dm monotone vs")
print("  the current hord=9 positive-definite) would outperform the")
print("  current choice; iter-784 only tests OFF vs ON.")
print("- Whether other components (PPM reconstruction, flux")
print("  integration) contribute independently of the limiter.")
print("- Whether the limiter bypass preserves long-term stability")
print("  (mass_target fix restores non-negativity, but bell shape")
print("  may be unphysical).")
