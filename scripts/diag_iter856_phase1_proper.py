"""Iter-856 diagnostic: PROPERLY-PLUMBED Phase 1 trial.

***NOTE (post-iter-856 revert):*** the source-modified
`fv3_sw_tendencies(phase1_div_swap=...)` kwarg this script depended
on has been REVERTED.  The script will now FAIL with TypeError on
the `phase1_div_swap` / `phase1_dt` kwargs.  Run output from the
trial is captured in the iter-856 commit message + doc entry.
To re-run, re-apply the temporary source patch in
operators_cdgrid.py (see iter-856 doc entry for the diff).

iter-855 retracted iter-854's quantitative findings due to an RK3-
staleness bug in the monkey-patched cgrid_divergence (substeps 2/3
saw start-of-step winds for divergence while the rest of the tendency
saw intermediate state).

iter-856 fixes the plumbing properly via an OPT-IN keyword
`phase1_div_swap=True` added to `fv3_sw_tendencies` (default-off,
production unchanged).  When True, the divergence-damping branch
computes the Fortran-faithful Fortran-cc divergence using the actual
`u_d, v_d` at the call site — so each RK3 substep sees the correct
intermediate state.

This script runs its OWN SSP-RK3 loop calling `fv3_sw_tendencies`
directly with `phase1_div_swap=True`, so we don't depend on the
model's internal integrator + tendency wrapping.

After iter-856 measurement, the source change to `fv3_sw_tendencies`
should be REVERTED (per Ralph loop's "no production source-code
change" discipline for trial wire-ins).

Method:
  1. W2 LEGACY at C36, dt=300s.
  2. Sweep damp_scale ∈ {0, 0.001, 0.01, 0.1, 1.0, 2.0} × iter-761
     canonical (= 2.13e+08).
  3. For each scale, run own SSP-RK3 for 60 steps.  Record stability
     + final |h - h0|.
  4. Compare to un-patched baseline at canonical (~0.641 m).

Observational only.  Source change is reversible.
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

from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from legoesm.core.operators_cdgrid import fv3_sw_tendencies
from legoesm.grids.cubed_sphere import create_cubed_sphere
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def _ssp_rk3_step(state, dt, tendency_fn):
    """SSP-RK3 step (mirrors src/legoesm/timestepping/ssp_rk3.py).

    Parameters: tendency_fn(state) → state-shaped tendency.
    """
    k1 = tendency_fn(state)
    s1 = FV3EdgeShallowWaterState(
        h=state.h + dt * k1.h,
        u_d=state.u_d + dt * k1.u_d,
        v_d=state.v_d + dt * k1.v_d,
        h_s=state.h_s,
    )
    k2 = tendency_fn(s1)
    s2 = FV3EdgeShallowWaterState(
        h=0.75 * state.h + 0.25 * (s1.h + dt * k2.h),
        u_d=0.75 * state.u_d + 0.25 * (s1.u_d + dt * k2.u_d),
        v_d=0.75 * state.v_d + 0.25 * (s1.v_d + dt * k2.v_d),
        h_s=state.h_s,
    )
    k3 = tendency_fn(s2)
    s_new = FV3EdgeShallowWaterState(
        h=(1.0 / 3.0) * state.h
          + (2.0 / 3.0) * (s2.h + dt * k3.h),
        u_d=(1.0 / 3.0) * state.u_d
            + (2.0 / 3.0) * (s2.u_d + dt * k3.u_d),
        v_d=(1.0 / 3.0) * state.v_d
            + (2.0 / 3.0) * (s2.v_d + dt * k3.v_d),
        h_s=state.h_s,
    )
    return s_new


def run_phase1(damp_scale, n=36, dt=300.0, n_steps=60, phase1=True):
    """Run W2 LEGACY for n_steps under given damping scale.

    phase1=True uses fv3_sw_tendencies(phase1_div_swap=True);
    phase1=False uses production (un-patched divergence).
    """
    div_damp_canonical = 8.0 * 1.5e7 * (48.0 / n) ** 2
    div_damp = damp_scale * div_damp_canonical

    grid = create_cubed_sphere(n=n, use_duogrid=False)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0, div_damp=div_damp,
        boundary_fix=True, damp_v=0.06, nord_v=2)
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)

    h0_np = np.asarray(state.h)

    def tendency_fn(s):
        dh, du, dv = fv3_sw_tendencies(
            s.h, s.u_d, s.v_d, s.h_s, cdgrid,
            g=9.80616,
            div_damp=div_damp,
            hyperdiff_coeff=0.0,
            boundary_fix=True,
            boundary_fix_skip_corners=False,
            fortran_a2b_corner_avg=False,
            fortran_vector_corner_fill=False,
            phase1_div_swap=phase1,
            phase1_dt=dt,
        )
        return FV3EdgeShallowWaterState(
            h=dh, u_d=du, v_d=dv, h_s=jnp.zeros_like(s.h_s))

    blew_up_at = None
    last_err = float("nan")
    for step_i in range(n_steps):
        state = _ssp_rk3_step(state, dt, tendency_fn)
        h_now = np.asarray(state.h)
        if not np.all(np.isfinite(h_now)):
            blew_up_at = step_i
            break
        last_err = float(np.max(np.abs(h_now - h0_np)))
        if last_err > 1.0e6:
            blew_up_at = step_i
            break

    return {
        "completed": blew_up_at is None,
        "blew_up_at": blew_up_at,
        "final_peak_h_err": last_err,
    }


def main():
    n = 36
    dt = 300.0
    n_steps = 60

    print(f"Iter-856 PROPERLY-PLUMBED Phase 1 trial (W2 LEGACY C{n}, "
          f"dt={dt}s, n_steps={n_steps} = {n_steps*dt/3600:.1f} h)")
    print(f"Source-modified: fv3_sw_tendencies has opt-in "
          f"phase1_div_swap kwarg (default-off, production unchanged).")
    print(f"Custom SSP-RK3 loop calls fv3_sw_tendencies directly so each")
    print(f"substep sees correct intermediate u_d/v_d (NO staleness).")
    print()

    # Reference: un-patched at canonical.
    ref = run_phase1(damp_scale=1.0, n=n, dt=dt, n_steps=n_steps,
                     phase1=False)
    print(f"(reference) un-patched (phase1=False) at canonical 1.0×: "
          f"completed={ref['completed']}, "
          f"final |h-h0| = {ref['final_peak_h_err']:.3e} m")
    print()
    print(f"Phase 1 patched runs (phase1_div_swap=True):")
    print(f"{'damp_scale':>10}  {'completed?':>10}  "
          f"{'final |h-h0|':>14}  {'note':>30}")
    print("-" * 75)
    for damp_scale in (0.0, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 2.0):
        r = run_phase1(damp_scale, n=n, dt=dt, n_steps=n_steps,
                       phase1=True)
        if r["completed"]:
            note = "stable"
        elif r["blew_up_at"] is not None:
            note = f"blew up at step {r['blew_up_at']+1}"
        else:
            note = "?"
        print(f"{damp_scale:>10.4g}  "
              f"{'yes' if r['completed'] else 'NO':>10}  "
              f"{r['final_peak_h_err']:>14.3e}  "
              f"{note:>30}")

    print()
    print("Compare to iter-854 (BUGGY hybrid, RETRACTED):")
    print("- iter-854 had 1.0× → 301m (470× worse than baseline 0.64m).")
    print("- iter-854 had 0.001× → 0.886m (≈ baseline).")
    print("- iter-854 reported all damp_scales 0..2× as STABLE.")
    print()
    print("If iter-856 results match iter-854 qualitatively (stable+ratio),")
    print("the staleness bug was numerically inconsequential.  If they")
    print("differ markedly, iter-854 was indeed misleading.")


if __name__ == "__main__":
    main()
