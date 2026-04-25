"""Iter-859 diagnostic: Phase 1 (stencil swap) + Check 1 (*dt factor).

***NOTE (post-iter-859 revert):*** the source patches in
`fv3_sw_tendencies` (phase1_div_swap, phase1_check1_dt kwargs) AND
in `shallow_water_fv3_cdgrid.py` (toggle forwarding) have been
REVERTED.  Re-running this script will fail.  Run output is in the
iter-859 commit message + doc entry.

iter-858 ruled out Phase 1 stencil-swap-alone at sub-canonical damp_
scales as a viable mode-A reduction (v_ll_Linf got worse, not better).

iter-859 tests Phase 1 + iter-849 Check 1 (the `*dt` factor in the
adaptive cap).  Per Fortran sw_core.F90:1720:
    damp = da_min_c * max(d2_bg, min(0.20, dddmp * abs(delpc(i,j) * dt)))
Production has `dddmp * abs(div_field)` (no *dt).  iter-849's Check 1
identifies this as one of 4 d_sw5 fidelity gaps.  iter-758c noted
*dt alone (with production stencil) breaks RK3.  iter-859 tests Phase
1 + Check 1 together.

Configurations:
A) Phase 1 OFF, Check 1 OFF: production reference (= iter-858 ref).
B) Phase 1 OFF, Check 1 ON:  Check-1-alone (production stencil with *dt).
   iter-758c says this breaks; iter-859 quantifies the magnitude.
C) Phase 1 ON,  Check 1 OFF: iter-858 (already ruled out).
D) Phase 1 ON,  Check 1 ON:  Phase 1 + Check 1.  THIS IS THE TARGET.

Sweep damp_scale ∈ {0.001, 0.01, 0.1, 1.0} for D (Phase 1 + Check 1).
W2 LEGACY C36 1 day (288 steps).

Source patches (REVERT after measurement):
- fv3_sw_tendencies: phase1_div_swap, phase1_dt, phase1_check1_dt kwargs.
- shallow_water_fv3_cdgrid: tendency_fn forwarding via module toggles.

***NOTE (post-iter-859 revert):*** patches are reverted after the run;
re-running fails until the patches are re-applied.
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
    ca, sa = cell_centre_angles_from_4edge(cdgrid)
    u_cc = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])
    v_cc = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])
    return np.asarray(sa * u_cc + ca * v_cc), np.asarray(v_cc)


def run_1day(damp_scale, phase1=False, check1=False,
             n=36, dt=300.0, n_steps=288):
    div_damp_canonical = 8.0 * 1.5e7 * (48.0 / n) ** 2
    div_damp = damp_scale * div_damp_canonical

    ocd._PHASE1_DIV_SWAP_ENABLED = bool(phase1)
    ocd._PHASE1_DT_VALUE = float(dt)
    ocd._PHASE1_CHECK1_DT = bool(check1)

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
        ocd._PHASE1_CHECK1_DT = False

    if blew is not None:
        return {"completed": False, "blew_up_at": blew,
                "L2": float("nan"),
                "v_ll_Linf": float("nan")}
    h_now = np.asarray(state.h)
    h_mean = float(np.mean(np.abs(h0_np)))
    err = h_now - h0_np
    L2 = float(np.sqrt(np.mean(err ** 2)) / h_mean)
    v_north, _ = _project_v_ll(state.u_d, state.v_d, cdgrid)
    return {"completed": True, "L2": L2,
            "v_ll_Linf": float(np.max(np.abs(v_north)))}


def main():
    n = 36
    dt = 300.0
    n_steps = 288

    print(f"Iter-859 Phase 1 (stencil swap) + Check 1 (*dt factor) trial")
    print(f"W2 LEGACY C{n}, dt={dt}s, n_steps={n_steps} = 24 h")
    print(f"Reference: iter-858 un-patched 1.0× → v_ll_Linf=0.188 m/s")
    print()
    print(f"{'config':>30}  {'damp_scale':>10}  {'completed?':>10}  "
          f"{'L2':>10}  {'v_ll_Linf':>10}")
    print("-" * 90)

    # A: production reference (already known, but recompute for sanity)
    r = run_1day(damp_scale=1.0, phase1=False, check1=False,
                 n=n, dt=dt, n_steps=n_steps)
    print(f"{'A: prod (no Phase1, no C1)':>30}  {'1.0':>10}  "
          f"{'yes' if r['completed'] else 'NO':>10}  "
          f"{r['L2']:>10.3e}  {r['v_ll_Linf']:>10.3e}")

    # B: Check 1 alone at canonical (production stencil + *dt)
    r = run_1day(damp_scale=1.0, phase1=False, check1=True,
                 n=n, dt=dt, n_steps=n_steps)
    if r['completed']:
        print(f"{'B: prod + C1 only':>30}  {'1.0':>10}  yes  "
              f"{r['L2']:>10.3e}  {r['v_ll_Linf']:>10.3e}")
    else:
        print(f"{'B: prod + C1 only':>30}  {'1.0':>10}  NO  "
              f"blew up at step {r['blew_up_at']+1}")

    # D: Phase 1 + Check 1 at sub-canonical and canonical
    print()
    print(f"D: Phase 1 + Check 1 (the new combo):")
    for damp_scale in (0.001, 0.01, 0.1, 1.0):
        r = run_1day(damp_scale=damp_scale, phase1=True, check1=True,
                     n=n, dt=dt, n_steps=n_steps)
        if r['completed']:
            mark = ""
            if r['v_ll_Linf'] < 0.188:
                mark = "  ← BELOW reference 0.188"
            print(f"{'D: Phase1 + C1':>30}  {damp_scale:>10.4g}  yes  "
                  f"{r['L2']:>10.3e}  {r['v_ll_Linf']:>10.3e}{mark}")
        else:
            print(f"{'D: Phase1 + C1':>30}  {damp_scale:>10.4g}  NO  "
                  f"blew up at step {r['blew_up_at']+1}")

    print()
    print("Interpretation:")
    print("- If D at canonical (1.0×) gives v_ll_Linf < 0.188 m/s:")
    print("  Phase 1 + Check 1 is a viable mode-A reduction; Check 1")
    print("  was the missing piece that Phase 1 alone (iter-858) lacked.")
    print("- If D at canonical blows up: Check 1 alone (without Check 4")
    print("  ke→d_sw6 routing) is insufficient; multi-iter port required.")
    print("- If D at sub-canonical gives same regression as iter-858:")
    print("  Check 1 doesn't change the qualitative picture at sub-canonical.")


if __name__ == "__main__":
    main()
