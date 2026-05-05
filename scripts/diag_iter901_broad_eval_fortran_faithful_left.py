"""Iter-901 broad measurement: extend iter-900's W2 measurement to
W5 and an ocean-rest-state sanity check.

iter-900 measured ONLY W2 and found that `fortran_faithful_ppm_left=
True` worsens W2 v_ll_Linf by 53.7 % (0.132 -> 0.203 m/s).  iter-901
fills in the rest of the Ralph-protocol evaluations:

  - W5 (Williamson case 5 — flow over an isolated mountain).
  - Ocean rest state (h=h_const, u=v=0, h_s=0; should stay at rest).
  - Cosine bell is structurally INERT to the iter-900 flag because
    `run_cosine_bell` invokes `transport_step` directly without going
    through `fv3_sw_tendencies` (per iter-775 note in
    `scripts/run_atmosphere_test_matrix.py:1568-1576`).  iter-901
    skips it explicitly with that justification.

For each case we run TWO trajectories — iter-892 default
(`fortran_faithful_ppm_left=False`) and iter-900 candidate
(`fortran_faithful_ppm_left=True`) — using the iter-893-aligned
production matrix config (`apply_fortran_xppm_boundary=True`,
`hyperdiff_coeff=0.0`, `div_damp=8*_div_damp_cube(n)`,
`boundary_fix=True`, `damp_v=0.06`, `nord_v=2`).  Output a comparison
table.

Conclusion gate: if any case shows iter-900 IMPROVES the metric, the
iter-892-vs-iter-900 trade-off is multi-dimensional and a future
hybrid configuration may be warranted.  If iter-900 worsens or is
neutral on all cases, the iter-900 doc's negative-result conclusion
extends to the broader matrix.
"""
import os, sys
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2, williamson_test5)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


def _build_model(grid, n, faithful):
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        apply_fortran_xppm_boundary=True,
        fortran_faithful_ppm_left=faithful,
    )
    return FV3EdgeShallowWaterModel(grid, config=cfg)


def run_w5(faithful: bool):
    n = 36
    dt = 300.0
    n_steps = int(86400 / dt)
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    sw = williamson_test5(grid)
    model = _build_model(grid, n, faithful)
    cdg = model.cdgrid
    u0 = 20.0
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
    h_diff_linf = float(np.max(np.abs(h_final - h_ic)))
    h_min = float(np.min(h_final))
    h_max = float(np.max(h_final))
    return {
        'mass_drift': mass_drift,
        'h_diff_linf': h_diff_linf,
        'h_min': h_min,
        'h_max': h_max,
    }


def run_ocean_rest(faithful: bool):
    """Ocean rest state: flat h, zero winds, zero topography.  Should
    stay at rest within numerical noise.  Diagnostics: max|u| at end
    of 1 day, h drift from initial."""
    n = 36
    dt = 300.0
    n_steps = int(86400 / dt)
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    model = _build_model(grid, n, faithful)
    cdg = model.cdgrid

    h0 = 1.0e4
    h = jnp.ones((6, n, n)) * h0
    h_s = jnp.zeros_like(h)
    u_d = jnp.zeros_like(cdg.cos_angle_edge_x)
    v_d = jnp.zeros_like(cdg.cos_angle_edge_y)

    state = FV3EdgeShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)
    model.set_initial_mass(state)

    h_ic = np.asarray(h)
    area = np.asarray(grid.area)
    mass_ic = float(np.sum(h_ic * area))

    for _ in range(n_steps):
        state = model.step(state, dt)

    h_final = np.asarray(state.h)
    mass_final = float(np.sum(h_final * area))
    mass_drift = abs(mass_final - mass_ic) / mass_ic
    h_diff_linf = float(np.max(np.abs(h_final - h_ic)))
    u_max = float(np.max(np.abs(np.asarray(state.u_d))))
    v_max = float(np.max(np.abs(np.asarray(state.v_d))))
    return {
        'mass_drift': mass_drift,
        'h_diff_linf': h_diff_linf,
        'u_max': u_max,
        'v_max': v_max,
    }


def main():
    """Run the broad evaluation and print the comparison table.

    Iter-901c (Codex iter-901b stop-time fix): wrapped in a `main()`
    under `if __name__ == "__main__"` guard so importing this module
    (e.g., from the smoke test in
    `tests/test_iter901_diag_smoke.py`) does NOT trigger the heavy
    W5 + ocean-rest trajectories.  Pre-iter-901c, the smoke test's
    `_load_script_module()` would `exec_module` the top-level prints
    + `run_w5(False); run_w5(True); run_ocean_rest(False);
    run_ocean_rest(True)` — ~60 s — defeating the smoke-test purpose.
    """
    print("Iter-901 broad evaluation: fortran_faithful_ppm_left flag effect")
    print("on W5 and ocean-rest-state at C36 dt=300s 1-day.")
    print()
    print("Cosine bell SKIPPED — `run_cosine_bell` uses `transport_step`")
    print("directly without going through `fv3_sw_tendencies`, so the")
    print("iter-900 flag is structurally inert (per iter-775 note in")
    print("scripts/run_atmosphere_test_matrix.py:1568-1576).")
    print()

    # === W5 ===
    print(f"### W5 (Williamson case 5 — flow over isolated mountain) ###")
    print(f"{'config':<35}  {'mass_drift':>12}  {'|h-h_ic|_Linf':>14}  "
          f"{'h_range (min, max)':>30}")
    print("-" * 100)
    for faithful in (False, True):
        label = ("(A) iter-892 default (production)" if not faithful
                 else "(B) iter-900 fortran-faithful left")
        r = run_w5(faithful)
        print(f"{label:<35}  "
              f"{r['mass_drift']:>12.3e}  "
              f"{r['h_diff_linf']:>14.3e}  "
              f"({r['h_min']:>10.4e}, {r['h_max']:>10.4e})")
    print()

    # === Ocean rest state ===
    print(f"### Ocean rest (h=10000, u=v=0, h_s=0; should stay at rest) ###")
    print(f"{'config':<35}  {'mass_drift':>12}  {'|h-h_ic|_Linf':>14}  "
          f"{'max|u|':>10}  {'max|v|':>10}")
    print("-" * 100)
    for faithful in (False, True):
        label = ("(A) iter-892 default (production)" if not faithful
                 else "(B) iter-900 fortran-faithful left")
        r = run_ocean_rest(faithful)
        print(f"{label:<35}  "
              f"{r['mass_drift']:>12.3e}  "
              f"{r['h_diff_linf']:>14.3e}  "
              f"{r['u_max']:>10.3e}  {r['v_max']:>10.3e}")
    print()

    print("Conclusion: if any 'iter-900 fortran-faithful' row is BETTER")
    print("than its iter-892 counterpart, the iter-892-vs-iter-900 trade-")
    print("off is multi-dimensional and a hybrid config is warranted.")
    print("If iter-900 is worse or neutral on all 3 cases (W2 from")
    print("iter-900 + W5 + ocean rest), iter-892's empirical advantage")
    print("extends to the broader test matrix and the negative-result")
    print("conclusion is robust.")


if __name__ == "__main__":
    main()
