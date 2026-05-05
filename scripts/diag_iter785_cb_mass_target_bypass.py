"""Iter-785 diagnostic: mass_target fix bypass for the cosine bell.

Iter-784 ruled out the PPM limiter as the dominant driver.  The
next suspect in the transport pipeline is the `mass_target` fix at
the end of `transport_step` (src/legoesm/core/fv_tp_2d.py:571-577),
which clips h < 0 to 0 then rescales positive values to match the
initial total mass.  This "clip-and-rescale" step is NOT Fortran-
faithful (FV3 does not apply this post-step correction in
`fv_tp_2d`); it was added to the Python port for numerical
positivity.

Iter-785 tests three cases:
  1. BASELINE: normal mass-target fix (clip + rescale to initial
     mass).
  2. NO_MASS_FIX: `mass_target=None` — no clip, no rescale.  h can
     go negative; raw PPM output reported.
  3. CLIP_ONLY: clip h < 0 to 0 but DO NOT rescale.  This is a
     middle ground: enforces positivity but doesn't inflate to
     compensate for conservation loss.

If NO_MASS_FIX Linf is much larger than BASELINE, the mass fix is
MASKING a large intrinsic transport error by clip-and-rescale.
If NO_MASS_FIX Linf is similar to BASELINE, the mass fix is
mostly cosmetic and the residual error is inherent to the
transport scheme.

Scope: observational only.  mass_target bypass is a diagnostic-
only call variant; the production code is not changed.
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
from legoesm.core.fv_tp_2d import (
    transport_step, compute_transport_quantities, fv_tp_2d)
from tests.test_cases.cosine_bell import (
    cosine_bell_cubesphere, cosine_bell_exact)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def _transport_step_clip_only(h, ut, vt, dt, cdgrid):
    """Replica of transport_step but with CLIP_ONLY (no rescale)."""
    area = cdgrid.base.area
    crx, cry, xfx, yfx, ra_x, ra_y = compute_transport_quantities(
        ut, vt, dt, cdgrid)
    fx, fy = fv_tp_2d(h, crx, cry, xfx, yfx, ra_x, ra_y, cdgrid)
    h_new = h + (fx[:, :-1, :] - fx[:, 1:, :]
                 + fy[:, :, :-1] - fy[:, :, 1:]) / area
    # Clip only (no rescale)
    h_new = jnp.maximum(h_new, 0.0)
    return h_new


def _run(n, dt, days, mode):
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
        if mode == 'baseline':
            h = transport_step(h, ut, vt, dt_exact, cdgrid,
                                mass_target=mass_init)
        elif mode == 'no_mass_fix':
            h = transport_step(h, ut, vt, dt_exact, cdgrid,
                                mass_target=None)
        elif mode == 'clip_only':
            h = _transport_step_clip_only(h, ut, vt, dt_exact, cdgrid)
        else:
            raise ValueError(f"unknown mode: {mode}")

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
    h_np = np.asarray(h)
    h_min = float(np.min(h_np))
    h_max = float(np.max(h_np))
    mass_final = float(np.sum(h_np * area_np))
    mass_drift = (mass_final - mass_init) / mass_init
    peak_err_signed = float(err[face, ci, cj])

    return {
        'l_inf': l_inf,
        'l2': l2,
        'peak_face': face,
        'peak_cell': (ci, cj),
        'peak_err_signed': peak_err_signed,
        'h_min': h_min,
        'h_max': h_max,
        'mass_drift': mass_drift,
    }


n = 36
dt = 1440.0
days = 1.0

print(f"Iter-785 cosine bell C36 with mass_target fix variants")
print(f"dt={dt}s, β=π/4, {days} day")
print()
print(f"{'case':>14}  {'L_inf':>9}  {'L2':>9}  {'peak cell':>15}  "
      f"{'peak err':>10}  {'h_min':>8}  {'h_max':>8}  {'mass drift':>11}")
print("-" * 100)

results = {}
for mode, label in (('baseline', 'BASELINE'),
                     ('no_mass_fix', 'NO_MASS_FIX'),
                     ('clip_only', 'CLIP_ONLY')):
    r = _run(n, dt, days, mode)
    results[mode] = r
    print(f"{label:>14}  {r['l_inf']:>9.3e}  {r['l2']:>9.3e}  "
          f"{'face '+str(r['peak_face'])+' ('+str(r['peak_cell'][0])+','+str(r['peak_cell'][1])+')':>15}  "
          f"{r['peak_err_signed']:>+10.3e}  "
          f"{r['h_min']:>8.2f}  {r['h_max']:>8.2f}  "
          f"{r['mass_drift']:>+11.3e}")

print()
base_linf = results['baseline']['l_inf']
for mode in ('no_mass_fix', 'clip_only'):
    d = 100.0 * (results[mode]['l_inf'] - base_linf) / base_linf
    print(f"  {mode} vs baseline: Linf {d:+.1f}%")

print()
print("Interpretation cues (observational only):")
print("- NO_MASS_FIX reveals the RAW PPM output.  h_min may be")
print("  negative, mass_drift should be near zero (PPM is inherently")
print("  conservative) but not exactly zero due to truncation.")
print("- CLIP_ONLY keeps positivity but skips the rescale.  If it's")
print("  close to BASELINE, the rescale is cosmetic; if it differs")
print("  much from BASELINE but matches NO_MASS_FIX, the rescale is")
print("  the main post-fix component.")

print()
print("What iter-785 DOES measure:")
print("- Linf, L2, mass drift, h_min / h_max under three variants")
print("  of the `transport_step` post-step correction.")
print("What it does NOT establish:")
print("- Whether the non-Fortran-faithful mass_target fix should be")
print("  removed from production.  FV3 does not apply this post-step")
print("  rescale in `fv_tp_2d`; removal is a fidelity question that")
print("  iter-785 does not itself resolve.")
print("- Whether OTHER parts of the transport step (cross-term, flux-")
print("  form coupling, face-boundary PPM handling) contribute to")
print("  the remaining error.")
