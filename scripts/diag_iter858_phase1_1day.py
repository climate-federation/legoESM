"""Iter-858 diagnostic: 1-day Phase 1 W2 LEGACY at sub-canonical damping.

***NOTE (post-iter-858 revert):*** the source patches in
`fv3_sw_tendencies` (phase1_div_swap kwarg) AND in
`shallow_water_fv3_cdgrid.py` (toggle-forwarding tendency_fn) have
been REVERTED.  Re-running this script will fail.  Run output is
captured in the iter-858 commit message + doc entry.

Per iter-857b's iter-858+ Priority 1: with Phase 1 stencil swap +
sub-canonical damping (where iter-857 showed 60-step stability), run
the full W2 LEGACY 1-day integration via model.step() and measure
v_ll_Linf vs the iter-761 canonical sentinel value of 0.159 m/s.

If v_ll_Linf < 0.159 m/s at any sub-canonical damp_scale, the Phase 1
stencil + reduced-coefficient path is a viable mode-A reduction at
the cost of partial Fortran fidelity (Checks 1+3+4 still unfixed).

If v_ll_Linf ≥ 0.159 m/s at every tested damp_scale, the Phase 1
stencil swap alone (with the iter-849 Checks 1+3+4 unfixed) does NOT
reduce W2 LEGACY mode-A.

Method:
- Re-applied iter-857's source patches (phase1_div_swap kwarg in
  fv3_sw_tendencies + module-toggle forwarding in shallow_water_fv3_
  cdgrid.py).  Default off → production unchanged.
- Run W2 LEGACY C36 1 day (288 steps at dt=300s) via model.step().
- Sweep damp_scale ∈ {0.001, 0.01, 0.1} × iter-761 canonical
  (sub-canonical only; iter-857 showed 1.0× and 2.0× blow up).
- Report L2, v_ll_Linf, v_cc_Linf, h_max.
- Compare to iter-761 canonical sentinel: L2=2.18e-4, v_ll_Linf=0.159 m/s.
- After: REVERT both source patches.

***NOTE (post-iter-858 revert):*** the source patches will be
reverted after measurement; re-running this script will fail.
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
from legoesm.grids.cubed_sphere_cdgrid import (
    cell_centre_angles_from_4edge)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2)


def _project_v_ll(u_d, v_d, cdgrid):
    """Project D-grid (u_d, v_d) to v_north at cell centres using the
    sentinel's cell_centre_angles_from_4edge convention."""
    ca_4edge, sa_4edge = cell_centre_angles_from_4edge(cdgrid)
    u_cc = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])
    v_cc = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])
    v_north = sa_4edge * u_cc + ca_4edge * v_cc
    return np.asarray(v_north), np.asarray(v_cc)


def run_1day(damp_scale, n=36, dt=300.0, n_steps=288, phase1_enabled=True):
    """Run W2 LEGACY for 1 day via model.step() with phase1 toggle."""
    div_damp_canonical = 8.0 * 1.5e7 * (48.0 / n) ** 2
    div_damp = damp_scale * div_damp_canonical

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
    try:
        for step_i in range(n_steps):
            state = model.step(state, dt)
            if not np.all(np.isfinite(np.asarray(state.h))):
                blew = step_i
                break
    finally:
        ocd._PHASE1_DIV_SWAP_ENABLED = False

    if blew is not None:
        return {"completed": False, "blew_up_at": blew,
                "L2": float("nan"),
                "v_ll_Linf": float("nan"),
                "v_cc_Linf": float("nan"),
                "h_max": float("nan")}

    h_now = np.asarray(state.h)
    h_mean = float(np.mean(np.abs(h0_np)))
    err = h_now - h0_np
    L2 = float(np.sqrt(np.mean(err ** 2)) / h_mean)
    h_max = float(np.max(np.abs(h_now)))
    v_north, v_cc = _project_v_ll(state.u_d, state.v_d, cdgrid)
    return {
        "completed": True,
        "L2": L2,
        "v_ll_Linf": float(np.max(np.abs(v_north))),
        "v_cc_Linf": float(np.max(np.abs(v_cc))),
        "h_max": h_max,
    }


def main():
    n = 36
    dt = 300.0
    n_steps = 288  # 1 day at dt=300s

    print(f"Iter-858 1-day Phase 1 W2 LEGACY (C{n}, dt={dt}s, "
          f"n_steps={n_steps} = 24 h)")
    print(f"Sub-canonical damping sweep via model.step()")
    print(f"Reference: iter-761 canonical L2=2.18e-4, v_ll_Linf=0.159 m/s")
    print()
    print(f"{'damp_scale':>10}  {'phase1':>6}  {'completed?':>10}  "
          f"{'L2':>10}  {'v_ll_Linf':>10}  {'h_max':>8}")
    print("-" * 80)

    # Reference: iter-761 canonical (phase1=False, damp_scale=1.0×).
    ref = run_1day(damp_scale=1.0, n=n, dt=dt, n_steps=n_steps,
                   phase1_enabled=False)
    print(f"{'1.0':>10}  {'OFF':>6}  "
          f"{'yes' if ref['completed'] else 'NO':>10}  "
          f"{ref['L2']:>10.3e}  {ref['v_ll_Linf']:>10.3e}  "
          f"{ref['h_max']:>8.0f}  (iter-761 ref)")
    print()

    for damp_scale in (0.001, 0.01, 0.1):
        r = run_1day(damp_scale, n=n, dt=dt, n_steps=n_steps,
                     phase1_enabled=True)
        if r["completed"]:
            mark = ""
            if r["v_ll_Linf"] < 0.159:
                mark = "  ← BELOW reference 0.159"
            elif r["v_ll_Linf"] > 0.234:
                mark = "  ← above iter-794's 0.234 (=damp=0)"
            print(f"{damp_scale:>10.4g}  {'ON':>6}  {'yes':>10}  "
                  f"{r['L2']:>10.3e}  {r['v_ll_Linf']:>10.3e}  "
                  f"{r['h_max']:>8.0f}{mark}")
        else:
            print(f"{damp_scale:>10.4g}  {'ON':>6}  {'NO':>10}  "
                  f"blew up at step {r['blew_up_at']+1}")

    print()
    print("Interpretation:")
    print("- If any phase1=ON v_ll_Linf < 0.159: Phase 1 + sub-canonical")
    print("  damping IS a viable mode-A reduction (cost: partial Fortran")
    print("  fidelity, Checks 1+3+4 still unfixed).")
    print("- If all v_ll_Linf ≥ 0.159: Phase 1 stencil swap alone does")
    print("  NOT reduce W2 LEGACY mode-A; iter-849 Check 1 (*dt) and/or")
    print("  Check 4 (ke→d_sw6 routing) are needed concurrently.")


if __name__ == "__main__":
    main()
