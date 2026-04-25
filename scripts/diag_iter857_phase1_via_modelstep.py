"""Iter-857 diagnostic: Phase 1 trial via model.step() (preserves
post-RK3 damp_v + mass fixer).

***NOTE (post-iter-857 revert):*** the source patches in
`fv3_sw_tendencies` (phase1_div_swap kwarg) AND in
`shallow_water_fv3_cdgrid.py`'s tendency_fn (toggle forwarding) have
been REVERTED.  Re-running this script will fail because the kwarg
is gone.  Run output is captured in the iter-857 commit message +
doc entry.

iter-856 ran the Phase 1 trial via a hand-rolled SSP-RK3 loop that
SKIPPED the production post-RK3 corrections (`damp_v` del-n vorticity
damping + mass fixer).  It blew up at canonical damping (step 10).

iter-854 ran the Phase 1 trial via `model.step()` (post-RK3 corrections
ACTIVE) but had an RK3-staleness bug (substeps 2/3 saw start-of-step
winds).  It was stable at all damp_scales.

iter-857 fixes both: uses `model.step()` (post-RK3 corrections active)
AND a properly-plumbed Phase 1 swap (each substep sees correct
intermediate state via the `phase1_div_swap` kwarg threaded through
`tendency_fn`).

If iter-857 at canonical damping is STABLE: post-RK3 corrections were
the dominant stabiliser; iter-856's blow-up was due to skipping them.

If iter-857 at canonical damping BLOWS UP: the staleness bug was
masking real instability that the post-RK3 corrections alone cannot
suppress.

Method:
- Source patch (REVERT after measurement): re-add `phase1_div_swap`
  and `phase1_dt` kwargs to `fv3_sw_tendencies`; forward via module-
  level toggle in `model.step`'s `tendency_fn` closure.
- Set `_PHASE1_DIV_SWAP_ENABLED = True` on `legoesm.core.operators_cdgrid`.
- Run W2 LEGACY C36 60 steps (5h) with damp sweep.
- Compare to (i) un-patched 1.0× ref baseline, (ii) iter-854 results,
  (iii) iter-856 results.
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
from legoesm.core import operators_cdgrid as ocd
from legoesm.grids.cubed_sphere import create_cubed_sphere
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def run_via_modelstep(damp_scale, n=36, dt=300.0, n_steps=60,
                      phase1_enabled=True):
    """Run W2 LEGACY via model.step() with optional phase1 swap."""
    div_damp_canonical = 8.0 * 1.5e7 * (48.0 / n) ** 2
    div_damp = damp_scale * div_damp_canonical

    # Set the module-level toggle BEFORE constructing the model
    # (so the JIT trace sees the right value).
    ocd._PHASE1_DIV_SWAP_ENABLED = bool(phase1_enabled)
    ocd._PHASE1_DT_VALUE = float(dt)

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
    model.set_initial_mass(state)
    h0_np = np.asarray(state.h)

    blew = None
    last_err = float("nan")
    try:
        for step_i in range(n_steps):
            state = model.step(state, dt)
            h_now = np.asarray(state.h)
            if not np.all(np.isfinite(h_now)):
                blew = step_i
                break
            last_err = float(np.max(np.abs(h_now - h0_np)))
            if last_err > 1.0e6:
                blew = step_i
                break
    finally:
        # Always restore toggle to OFF after the run.
        ocd._PHASE1_DIV_SWAP_ENABLED = False

    return {
        "completed": blew is None,
        "blew_up_at": blew,
        "final_peak_h_err": last_err,
    }


def main():
    n = 36
    dt = 300.0
    n_steps = 60

    print(f"Iter-857 Phase 1 trial via model.step() "
          f"(W2 LEGACY C{n}, dt={dt}s, n_steps={n_steps} = "
          f"{n_steps*dt/3600:.1f} h)")
    print(f"Source-modified: fv3_sw_tendencies has phase1_div_swap kwarg;")
    print(f"model wrapper threads it via _PHASE1_DIV_SWAP_ENABLED toggle.")
    print(f"Post-RK3 damp_v + mass fixer ACTIVE (vs iter-856 own-RK3 skip).")
    print()

    # Reference: un-patched (phase1=False) at canonical.
    ref = run_via_modelstep(damp_scale=1.0, n=n, dt=dt, n_steps=n_steps,
                            phase1_enabled=False)
    print(f"(reference) un-patched (phase1=False) at canonical 1.0× via model.step():")
    print(f"  completed={ref['completed']}, "
          f"final |h-h0| = {ref['final_peak_h_err']:.3e} m")
    print()
    print(f"Phase 1 patched runs (phase1=True) via model.step():")
    print(f"{'damp_scale':>10}  {'completed?':>10}  "
          f"{'final |h-h0|':>14}  {'note':>30}")
    print("-" * 75)
    for damp_scale in (0.0, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 2.0):
        r = run_via_modelstep(damp_scale, n=n, dt=dt, n_steps=n_steps,
                              phase1_enabled=True)
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
    print("Three-way comparison:")
    print("                     iter-854           iter-856           iter-857")
    print("                     (model.step,       (own-RK3,          (model.step,")
    print("                      stale)            no post-RK3)        properly plumbed)")
    print("  1.0×:              stable, 301 m     BLEW UP at step 10  see above")
    print("  2.0×:              stable, 639 m     BLEW UP at step 6   see above")
    print()
    print("Interpretation:")
    print("- If iter-857 1.0× IS STABLE: post-RK3 corrections were the")
    print("  dominant stabiliser in iter-854; iter-856's blow-up was due")
    print("  to skipping those corrections, not the staleness fix.")
    print("- If iter-857 1.0× BLOWS UP: the staleness fix exposed real")
    print("  instability that post-RK3 corrections alone cannot suppress.")


if __name__ == "__main__":
    main()
