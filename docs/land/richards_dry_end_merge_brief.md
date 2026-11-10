# Richards dry-end fix: merge of `jisu/richards-dry-end-fix` with main

Written 2026-11-06.  Status: **current `origin/main` (`ee54a853`) is merged
into this branch; conflicts resolved; both test sets green apart from two
failures that fail on unmodified main.**  The PR into main is a plain merge.

## 1. What the branch changes (vs main)

| file | change |
|---|---|
| `packages/land/legoesm/land/richards.py` | Picard update applied in theta (`theta += C dpsi`, psi recovered by the inverse retention curve) instead of psi; per-layer ceiling at the water held under the hydrostatic head of the pond, excess cascaded up the column and onto the surface store (every iterate conserves water); no exfiltration through the surface link; damping factor measured on the applied (bounded) theta change; dry floor bounded at -1e6 m; `dry_floor()` returns the theta companion.  Main's damping, convergence flag, `refill`/`water_created` bookkeeping, PDI/Lu psi shift, frozen-soil impedance, `k_sat_decay`, field-capacity limiter and CFL caps are kept. |
| `packages/land/legoesm/land/soil_hydraulics.py` | elastic (specific-storage) branch starts at air entry (`air_entry_psi`), so `theta_from_psi` is strictly increasing and `psi_from_theta` its exact inverse through saturation; `moisture_capacity` consistent with it. |
| `packages/land/legoesm/land/multilayer_land.py` | extractable water measured above the solver's theta floor (`dry_floor`), not `theta_r`. |
| `scripts/data/cap_land_restart.py` | new: one-off capping of restarts written by the pre-fix solver (over-pressured / sub-floor layers), with `--verify`. |
| tests | `test_richards_dry_wetting.py` (new, 13), edits to `test_multilayer_water_balance.py`, `test_richards_dry_convergence.py`, `tests/unit/test_corrections.py`, `tests/unit/test_diff_land_ice_schemes.py`. |

## 2. Why (the mechanism, measured)

The global LMIP runs produced dry-season runoff spikes of up to 1e10 mm/day
in ~1.5 % of land cells.  Cause: the psi-stepping Picard update
`dpsi = residual / (C/dt)` on a bone-dry Clapp-Hornberger top layer, where the
capacity collapses like Se^(b+1) (C/dt ~ 1e-16 1/s): 1 mm of rain became
dpsi ~ +1e10 m, the elastic branch turned the pressure into water and the
surface link into runoff.  Main's October rework (damped psi iterations,
convergence flag, post-solve take-back) removes the spike but leaves the
solve unconverged on that cell and places the rain through the whole 3 m
column within one hour.

Spike cell (15S 32E Cosby Clapp-Hornberger clay: theta_sat 0.426, psi_sat
-0.169 m, b 7.1, K_sat 5.36e-6 m/s, theta_r 0; 10 layers over 3 m, growth 2,
2.93 mm top layer; dry profile floored at psi = -1e6 m; one hour of rain;
default `RichardsConfig()`; reference = the same solver at 60 s x 60):

| rain | pre-fix base (31 Aug) | main, one 3600 s step | this merge, one 3600 s step | 60 s reference |
|---|---|---|---|---|
| 1 mm/h | runoff 1.5e8 mm/day, top layer 14 x porosity | top theta/theta_sat 0.11, **not converged** (30 it.), water in layers 3-5 | top 0.53, converged (20 it.) | top 0.60 (both solvers) |
| 10 mm/h | runoff 1.3e9 mm/day | all 10 mm through 3 m (4.75 mm in the bottom 1.5 m), top 0.12, not converged | top three layers 1.00/1.00/0.93, 7.9 mm stored in the top 4 cm, 2.1 mm ponded | top 0.81-0.87, 10 mm in the top 9 cm, no pond |
| 50 mm/h | runoff 1e10 mm/day | 50 mm through 3 m (23.7 mm in the bottom layer), no pond, not converged | 7.9 mm stored in the top 4 cm, 42 mm ponded | 26-33 mm stored, 10-16 mm ponded |

Runoff 0 and the column budget closed to 1e-14 mm at every rate for main and
for this merge.  This merge keeps the wetting front at the surface and errs on
the conservative side (ponds water that infiltrates in later steps); main's
hourly answer is an unconverged iterate whose budget closes through the
take-back term.

## 3. Conflict resolution (done)

Main's `richards.py` / `multilayer_land.py` taken as the base; the branch's
hunks re-applied: `_surface_terms` exfiltration cap, `dry_floor`, the theta
update block (`_theta_update`: linearised storage change, hydrostatic
ceiling, upward cascade, floor/cap clip, exact pin at the floor), the
extractable-water floor.  `soil_hydraulics.py` merged clean.
`submit_lmip_ginsburg.sh` (deleted on main) dropped.

Independent adversarial review of the resolution (Claude subagent; the
codex CLI and GLM reviewer are not installed on this machine) found two
defects and one regression, all fixed in the merge:

| finding | fix |
|---|---|
| damping trial measured on raw `C*dpsi` ignored the floor clip: a layer held at the dry floor under a draw never counted as converged (lam < 1 forever) | trial = the applied change, `clip(theta + C dpsi, floor, cap) - theta` |
| the merge had silently taken main's `_PSI_FLOOR_MIN = -1e30`; for the spike cell that is psi = -4e27 m / theta = 4e-5, and the K clip times a 1e27 m gradient created 4.9e7 mm of water with damping off | the branch's `-1e6 m` restored, with its rationale (cancellation needs ~4 digits at -1e6 m vs ~25 at -1e30 m; pore RH < 1e-30 there) |
| PDI / Lu retention curves: their theta -> psi inverse is approximate, so inverting every iterate decoupled theta from psi by ~0.05 | those two curves keep main's psi update (static dispatch on `_shift_psi`); main's tests for them pass |
| round-off could leave a floor layer a few ulp above the psi floor, so the post-solve "pinned" test missed it | `psi = psi_floor` exactly where `theta <= theta_floor` |
| stale comment ("psi_from_theta has no inverse above saturation") | rewritten |

Tests updated for the new behaviour: main's
`test_unconverged_column_is_flagged_without_a_spike` (column B now
converges) and `test_former_undamped_ten_iterations_spike_on_these_columns`
(the former settings no longer spike — now the regression pin);
`test_richards_drainage_reported_at_solver_debited_K` runs 4/3 iterations
instead of 10/9 so its non-vacuity guard still sees a draining column; the
branch's `test_budget_residual_reports_infeasible_sink...` reads
`refill + water_created` (main's names for the branch's `budget_residual`).

## 4. Test status

Run with `JAX_ENABLE_X64=1` on this worktree:
`tests/land/unit/test_richards_dry_wetting.py`,
`test_richards_dry_convergence.py`, `test_richards_per_column.py`,
`test_ic_soil_hydraulics.py`, `test_multilayer_water_balance.py`,
`tests/unit/test_corrections.py`, `test_diff_land_ice_schemes.py`,
`test_diff_land.py`, `test_multilayer_land_driver.py`,
`test_soil_ice_impedance.py`: see the PR for the final count.  Two failures
are pre-existing on unmodified `origin/main` (verified by running them
there): `tests/unit/test_diff_land.py::test_slab_land_grad_wrt_sw_down_cross_component`
and `tests/land/unit/test_soil_ice_impedance.py::test_float32_frozen_dry_column_under_jit_scan`.

## 5. Known limits, not fixed here

* PDI / Lu curves stay on the psi update (no spike protection on that path;
  production uses Clapp-Hornberger).
* van Genuchten / Brooks-Corey `psi_from_theta` clips Se at 1 - 1e-6, so for
  theta within 3.5e-7 of theta_sat psi sticks at ~-7.6e-5 m for one iterate
  (self-corrects next iterate).
* `test_moist_subsoil_keeps_capillary_supply` ET pin moved by 1.26e-6
  relative (tolerance 1e-6) — see the PR for its final state.
* Hourly steps still carry time-step error against a 60 s reference (section
  2); the direction is conservative.
* `cap_land_restart.py` has no test of its own; needed only for runs
  restarting from files written before this fix.

## 6. For the merger

Nothing to resolve: the branch already contains `origin/main`.  If main
changes `richards.py`, `soil_hydraulics.py` or `multilayer_land.py` before
the merge, re-merge main into this branch and rerun the test list above.
The canopy / P-model work on `jisu/land-p-model` is not part of this branch;
`canopy/photosynthesis.py` is untouched.
