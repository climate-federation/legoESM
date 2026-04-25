"""Iter-854 diagnostic: short-run stability test of iter-853 patched config.

Per iter-853b's iter-854+ priority: take iter-853's monkey-patched
`cgrid_divergence` (Fortran-cc replacing production cell-centre flux-
form) and run W2 LEGACY for 60 timesteps (5 h at dt=300s) under a
damping-coefficient sweep.  Goal: disentangle "stencil-instability
signature" from "over-damping artefact under iter-761 canonical
damping coefficient."

Live-state plumbing (Codex iter-854 review fix to iter-853 captured-
state issue): the patched `cgrid_divergence` reads u_d, v_d, ua, va
from a MUTABLE module-level dict updated BEFORE each model.step()
call.  This way the patched function uses the CURRENT integration
state, not the initial state captured in closure.

Damping sweep range (Codex recommendation): {0, 1e-3, 1e-2, 1e-1, 1,
2}× iter-761 canonical.  Exact 0 matters because the damping branch
is skipped only when div_damp==0 in operators_cdgrid.py:1690.  2× is
the high-side bracket; iter-794 24h sweep already crashed at 16×base
(= 2× canonical).

Caveats inherited from iter-853:
- Fortran helper still has `mode='edge'` midpoint-halo gap.
- Patched config has iter-849 Checks 1+3+4 still UNFIXED.
- "Per-step peak |dv/dt|" requires instrumenting inside fv3_sw_
  tendencies, which is not feasible without source modification.
  iter-854 reports peak |state.h - h0| (mass error) as the proxy
  observable per step.

***STALENESS BUG (iter-854b self-review):*** The mutable `LIVE_STATE`
dict is updated BEFORE each `model.step()` call, but `model.step()`
internally executes a 3-stage SSP-RK3 that calls `fv3_sw_tendencies`
3 times at DIFFERENT intermediate states (y₀, y₁, y₂).  The patched
`cgrid_divergence` reads `u_d, v_d` from `LIVE_STATE` which holds the
y₀ (start-of-step) state ONLY.  So substeps 2 and 3 use STALE
u_d/v_d for the Fortran-cc divergence while everything else in
`fv3_sw_tendencies` sees the actual intermediate state.  This makes
iter-854 a HYBRID computation, not a faithful Phase 1 trial.  The
quantitative findings (470× larger 5h error at canonical, "stable at
2.0×") are therefore UNRELIABLE.  An honest fix requires patching
INSIDE `fv3_sw_tendencies` so the divergence sees the actual y₁/y₂
state in substeps 2/3.

Observational only.  No production source-code change.
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
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel)
from legoesm.core import operators_cdgrid as ocd
from legoesm.core.fv3_sw_core import (
    _d2a2c_vect, _d_sw5_corner_divergence)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


# Mutable state read by patched cgrid_divergence at each tendency call
LIVE_STATE = {"u_d": None, "v_d": None, "cdgrid": None, "dt": None,
              "da_min_c": None}


def patched_cgrid_divergence(u_c, v_c, cdg):
    """Patched cgrid_divergence using Fortran-cc construction with
    live u_d/v_d from LIVE_STATE updated each step."""
    u_d = LIVE_STATE["u_d"]
    v_d = LIVE_STATE["v_d"]
    dt = LIVE_STATE["dt"]
    da_min_c = LIVE_STATE["da_min_c"]
    # Recompute ua, va from CURRENT u_d, v_d each call.
    ua, va, _, _, _, _ = _d2a2c_vect(u_d, v_d, cdg)
    ke_damping = _d_sw5_corner_divergence(
        u_d, v_d, ua, va, cdg, dt,
        d2_bg=1.0, dddmp=0.0, nord=0,
    )
    delpc_corner = ke_damping / da_min_c
    delpc_cc = 0.25 * (
        delpc_corner[:, :-1, :-1] + delpc_corner[:, 1:, :-1]
        + delpc_corner[:, :-1, 1:] + delpc_corner[:, 1:, 1:])
    return delpc_cc


def run_stability_test(damp_scale, n=36, dt=300.0, n_steps=60,
                       verbose=False):
    """Run W2 LEGACY for n_steps with patched div_damp at given scale."""
    div_damp_canonical = 8.0 * 1.5e7 * (48.0 / n) ** 2
    div_damp = damp_scale * div_damp_canonical

    grid = create_cubed_sphere(n=n, use_duogrid=False)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=div_damp,
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
    )
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdgrid = model.cdgrid

    LIVE_STATE["cdgrid"] = cdgrid
    LIVE_STATE["dt"] = dt
    LIVE_STATE["da_min_c"] = float(jnp.min(cdgrid.area_corner))

    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))

    h0 = sw.h.data
    state = type(sw)(h=h0, u_d=u_d, v_d=v_d, h_s=sw.h_s.data) if False else None
    # Build the state via the model's expected type
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterState)
    state = FV3EdgeShallowWaterState(
        h=h0, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    h0_np = np.asarray(state.h)

    # Apply patch ONLY when div_damp > 0 (otherwise the branch is
    # skipped and the patch wouldn't fire anyway).
    apply_patch = (div_damp > 0)
    if apply_patch:
        ocd.cgrid_divergence = patched_cgrid_divergence
    else:
        # Restore to original.  No-op patch when damping is off.
        pass

    peak_h_err_history = []
    blew_up_at = None

    try:
        for step_i in range(n_steps):
            # Update live state BEFORE the step (so the patched
            # cgrid_divergence sees the current u_d, v_d).
            LIVE_STATE["u_d"] = state.u_d
            LIVE_STATE["v_d"] = state.v_d

            state = model.step(state, dt)

            h_now = np.asarray(state.h)
            if not np.all(np.isfinite(h_now)):
                blew_up_at = step_i
                if verbose:
                    print(f"    Step {step_i+1}: NaN — blew up.")
                break

            peak_h_err = float(np.max(np.abs(h_now - h0_np)))
            peak_h_err_history.append(peak_h_err)

            if peak_h_err > 1000.0:
                blew_up_at = step_i
                if verbose:
                    print(f"    Step {step_i+1}: peak |h - h0| = "
                          f"{peak_h_err:.2e} m > 1000 — diverging.")
                break

            if verbose and (step_i + 1) % 10 == 0:
                print(f"    Step {step_i+1}: peak |h - h0| = "
                      f"{peak_h_err:.3e} m")
    finally:
        # Restore original on every run regardless of outcome.
        ocd.cgrid_divergence = orig_cgrid_divergence

    return {
        "completed": blew_up_at is None,
        "blew_up_at": blew_up_at,
        "n_steps_completed": (
            n_steps if blew_up_at is None else blew_up_at + 1),
        "final_peak_h_err": (
            peak_h_err_history[-1] if peak_h_err_history else float("nan")),
        "peak_h_err_history": peak_h_err_history,
    }


# Capture original cgrid_divergence ONCE at module import.
orig_cgrid_divergence = ocd.cgrid_divergence


def _run_unpatched(n=36, dt=300.0, n_steps=60, damp_scale=1.0):
    """Run UN-patched production W2 LEGACY at the given damp_scale."""
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
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterState)
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)
    h0_np = np.asarray(state.h)
    blew = None
    last_err = float("nan")
    for step_i in range(n_steps):
        state = model.step(state, dt)
        h_now = np.asarray(state.h)
        if not np.all(np.isfinite(h_now)):
            blew = step_i
            break
        last_err = float(np.max(np.abs(h_now - h0_np)))
    return {"completed": blew is None, "final_peak_h_err": last_err}


def main():
    n = 36
    dt = 300.0
    n_steps = 60

    print(f"Iter-854 short-run stability test (W2 LEGACY C{n}, "
          f"dt={dt}s, n_steps={n_steps} = {n_steps*dt/3600:.1f} h)")
    print(f"Patched config: cgrid_divergence → Fortran-cc "
          f"(_d_sw5_corner_divergence + 4-pt avg)")
    print()
    print(f"{'damp_scale':>10}  {'completed?':>10}  {'n_steps':>8}  "
          f"{'final |h-h0|':>14}  {'note':>20}")
    print("-" * 80)

    # Run un-patched baseline at canonical for context.
    print("(reference) un-patched production at canonical damping:")
    LIVE_STATE["u_d"] = None  # unused in unpatched
    base_result = _run_unpatched(n=n, dt=dt, n_steps=n_steps,
                                  damp_scale=1.0)
    print(f"  un-patched 1.0×: completed={base_result['completed']}, "
          f"final |h-h0| = {base_result['final_peak_h_err']:.3e} m")
    print()
    print(f"Patched runs (Fortran-cc divergence with various damp scales):")
    print(f"{'damp_scale':>10}  {'completed?':>10}  {'n_steps':>8}  "
          f"{'final |h-h0|':>14}  {'note':>20}")
    print("-" * 80)
    for damp_scale in (0.0, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 2.0):
        result = run_stability_test(damp_scale, n=n, dt=dt, n_steps=n_steps,
                                     verbose=False)
        if result["completed"]:
            note = "stable"
        elif result["blew_up_at"] is not None:
            note = f"blew up at step {result['blew_up_at']+1}"
        else:
            note = "?"
        print(f"{damp_scale:>10.4g}  "
              f"{'yes' if result['completed'] else 'NO':>10}  "
              f"{result['n_steps_completed']:>8}  "
              f"{result['final_peak_h_err']:>14.3e}  "
              f"{note:>20}")

    print()
    print("Caveats (iter-852/853 inherited):")
    print("- Patched config has iter-849 Checks 1, 3, 4 still UNFIXED.")
    print("  Concretely: no *dt factor in cap, no corner correction stencil,")
    print("  damping applied directly to du/dv (not via ke→d_sw6).")
    print("- _d_sw5_corner_divergence nord=0 still uses mode='edge' halo")
    print("  for vort/ptc (fv3_sw_core.py:~1044).")
    print("- Comparison baseline (un-patched) at same damp_scales would")
    print("  also be useful for context.  iter-855+ candidate.")
    print()
    print("Interpretation:")
    print("- If damp_scale=0 completes 60 steps stably AND damp_scale=1.0")
    print("  blows up: the 1.0× canonical damping coefficient is the")
    print("  destabilising factor under the patched stencil; a smaller")
    print("  coefficient could maintain stability.")
    print("- If damp_scale=0 also blows up: the patched stencil is")
    print("  intrinsically unstable in this configuration.")
    print("- If all scales survive 60 steps: the iter-853 t=0 |dv/dt|")
    print("  measurement was a misleading surface-level reading.")


if __name__ == "__main__":
    main()
