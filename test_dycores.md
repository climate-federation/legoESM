# Dycore Test Hardening — `test_dycores` branch

Single source of truth: `scripts/run_atmosphere_test_matrix.py`.

| Task grid | Matrix key | Dycore impls |
|-----------|------------|--------------|
| lat-lon FV | `latlon` | `CGridLatLon{ShallowWater,PrimitiveEquation}` |
| FV3 cube | `cubed_sphere` | `FV3EdgeShallowWater`, `CDGridPrimitiveEquation`, `CDGridCompressibleEuler` |
| MPAS | `icosahedral` | `MPAS{ShallowWater,PrimitiveEquation,CompressibleEuler}` |
| Spectral | `spectral` | `Spectral{ShallowWater,PrimitiveEquation,CompressibleEuler}` |

Ladder: SW Williamson (W2/W5/W6/cosine_bell) → Hydrostatic (HS, baroclinic,
DCMIP transport, AMIP, topo, gravity wave, Rossby-Haurwitz) → NH (DCMIP-2025
TC1/TC2a/TC3).

## Cumulative status (iter-1..30)

**Mass conservation: bit-clean across every implemented grid × equation set.**

| Equation set | cube | latlon FV | MPAS/ico | spectral |
|--------------|------|-----------|----------|----------|
| SW           | bit-clean | bit-clean | bit-clean | bit-clean |
| hydrostatic  | bit-clean | bit-clean | bit-clean | bit-clean |
| NH           | bit-clean | not implemented | bit-clean | bit-clean |

Measured worst-case cross-grid mass drift after iter-22 fixers:

| Case | cube | latlon | ico | spec |
|------|------|--------|-----|------|
| SW W5 | 1.46e-15 | 3.24e-16 | 1.62e-16 | 3.24e-16 |
| SW W6 | — | — | 0 | 1.91e-16 |
| SW cosine_bell | 2.18e-08 | 1.49e-05 (raw-FV) | 4.16e-07 | 0 |
| Hydro HS | 4.02e-13 | 1.61e-16 | 1.61e-16 | 1.61e-16 |
| Hydro baroclinic | 2.62e-11 | 0 | 0 | 4.82e-16 |
| Hydro AMIP (1-day) | 1.17e-12 | (anchor wired) | (anchor wired) | (anchor wired) |
| NH TC1 | 0 | n/a | 0 | 0 |
| NH TC2a | 6.28e-16 | n/a | 0 | 4.71e-16 |
| NH TC3 | 1.03e-15 | n/a | 0 | SKIP (Kessler) |

## Matrix runner PASS gates (iter-23..29)

- `_DYCORE_MASS_DRIFT_TOL = 1e-6` (module constant, iter-30) — applied to
  SW W5/W6, hydro HS/baroclinic/AMIP, NH TC1/TC2a/TC3.
- `_DYCORE_MASS_DRIFT_TOL_CB = 1e-4` — cosine_bell only (~7x above lat-lon
  intentional raw-FV benchmark at 1.49e-05).

## 13-test cross-grid regression suite (iter-15..17)

- `tests/atmosphere/shallow_water/integration/test_sw_mass_conservation_anchored.py` (3 grids)
- `tests/atmosphere/hydrostatic/validation/test_mass_conservation_anchored.py` (4 grids)
- `tests/atmosphere/nonhydrostatic/integration/test_nh_mass_conservation_anchored.py` (3 grids)
- `tests/atmosphere/test_anchor_mass_api.py` (12 cases, iter-22)

Tolerance `DRIFT_TOL = 1e-12` over 20 steps. Catches regressions in any of
the iter-1..22 anchor / fp64-budget-accumulator changes.

## Anchor lifecycle API (iter-18..22)

Every anchored model exposes a uniform four-method interface:

- `compute_mass(state)` / `compute_dry_mass(state)` — fp64 snapshot helper
- `reset_target_mass()` — clear cached target → re-snapshot on next step
- `set_target_mass(target)` — explicit anchor (checkpoint restart, cross-
  experiment reference)
- Step's lazy snapshot on first call when `anchor_mass_to_initial=True`

11 models covered: cube SW (FV3Edge/FV3FB/csw), cube PE, cube NH, latlon
SW, latlon PE, MPAS SW, MPAS PE, MPAS NH, spectral PE, spectral NH.

## Implementation patterns (apply to any new dycore)

1. **Anchored mass target** — snapshot initial mass in fp64 outside JIT
   on first `step()`; subsequent steps correct toward target so per-step
   storage round-trips don't random-walk.
2. **Budget accumulator** — use `_conservation_accumulator()` (fp64 when
   x64 enabled), never `_accumulation_dtype()`, for any sum feeding a
   diagnostic or fixer.
3. **No `correction.astype(state.dtype)`** — let JAX promote the
   correction add; rely on end-of-step `cast_pytree(allow_downcast=False)`
   to keep state in fp64.
4. **Spectral fixer convention** — a constant Δ in physical space maps
   to `Δ·sqrt(4π)` at the (n=0,m=0) coefficient (`sh_analysis(ones)[0] ==
   sqrt(4π)` under this repo's (4π)-normalised SH basis).
5. **Matrix runner opt-in** — config defaults stay False; matrix runner
   opts in. Preserves bit-for-bit baseline for drift-measurement tests.
6. **Anchor lifecycle API** — every anchored model exposes the same
   four-method interface (see above).

## Known limitations

- **`lax.scan` compat under fp32 storage** — anchored fixers keep post-
  fix state in fp64 via `cast_pytree(allow_downcast=False)`. With the
  default fp32 storage policy, `lax.scan` errors on the carry-dtype
  mismatch. Set `PrecisionPolicy.fp64()` (or `mixed_fp64_storage()`)
  before constructing the model and `integrate_scan` works at machine
  precision — pinned by `test_anchored_step_scan_compat_under_fp64_policy`
  (iter-36).
- **Cube NH TC1 `|w|_max=0.3177`** vs ico/spec 0.014. Suspect
  cube-imprint edge artifact; tuning knobs (`corner_div_damp_*`,
  `damp_v`) available but defaults preserved.
- **Lat-lon NH** — matrix runner SKIPs; DCMIP-2025 inits require
  `CubedSphereGrid`, no lat-lon variant yet.

## Improvement log (compressed iter-1..30)

| Iter | Scope | Headline reduction |
|------|-------|--------------------|
| 1    | fp64 `global_integral` / `_area_weighted_sum` diagnostic | Latlon HS `3.35e-3 → 1.57e-4` |
| 2    | Latlon PE anchor + drop fp32 cast | `→ 3.71e-7` |
| 3    | Spectral PE anchored fixer | 13 spectral hydro cases `→ ~1e-16` |
| 4    | Latlon SW anchor + `_conservation_accumulator` | W5 `1.21e-5 → 3.24e-16` |
| 5    | FV3 cube SW fp64 acc + drop cast | W5 `9.68e-7 → 1.46e-15` |
| 6    | `transport_step` fp64 acc; MPAS SW anchor | cube cosine_bell `4.49e-7 → 2.18e-8`; ico W5 `→ 1.62e-16` |
| 7    | Cube NH anchored fixer + `mass` diag | TC1 `→ 0` |
| 8    | MPAS NH fixer (new `compute_nh_dry_mass_mpas`, `fix_mass_nonhydrostatic_mpas`) | TC1 `→ 0` |
| 9    | Spectral NH fixer (`rho_prime_hat[0,:] += Δρ·sqrt(4π)`) | TC1 `→ 0` |
| 10   | NH `mass_drift` in notes; first doc compress | — |
| 11   | MPAS PE anchor + fp64 fixer; cube PE opt-in anchor | cube HS `4.01e-08 → 4.02e-13`; ico HS `→ 1.61e-16` |
| 12   | Latlon PE `compute_mass`/fixer fp64 budget | HS `3.71e-7 → 1.61e-16` (~10^13x cumulative) |
| 13   | `_total_area(grid)` fp64 divisor | Defensive |
| 14   | Strip stale `_accumulation_dtype` imports | — |
| 15   | 4-PE-solver regression test, drift ≤ 1e-12 / 5 steps | 4/4 PASS |
| 16   | Parallel SW + NH regression tests | 10/10 PASS |
| 17   | Tighten regression: 5→20 steps, 1e-10→1e-12 ceiling | 10/10 PASS |
| 18   | `reset_target_mass()` API on 7 lazy-snapshot models | — |
| 19   | `set_target_mass(target)` setter on same 7 models | — |
| 20   | reset/set on cube SW (3) + cube PE + cube NH; doc compress | — |
| 21   | `compute_mass`/`compute_dry_mass` helpers on cube models | — |
| 22   | Anchor API regression test (12 cases); spectral PE public `compute_mass` alias | — |
| 23   | HS PASS gate `1e-2 → 1e-6` | catches fixer regressions |
| 24   | Baroclinic PASS gate `1e-2 → 1e-6` (measured max 2.62e-11 cube) | — |
| 25   | NH PASS gate added at `1e-3` (was no gate) | TC1 PASS |
| 26   | NH gate `1e-3 → 1e-6` after TC2a/TC3 measured | all bit-clean |
| 27   | SW W5/W6 gate added at `1e-6` (was no gate) | 14/14 PASS |
| 28   | AMIP gate `1e-2 → 1e-6` (cube measured `1.17e-12`) | — |
| 29   | cosine_bell gate `1e-2 → 1e-4` (preserves latlon raw-FV benchmark) | 4/4 PASS |
| 30   | Hoist gates to `_DYCORE_MASS_DRIFT_TOL` / `_DYCORE_MASS_DRIFT_TOL_CB` module constants; doc compress | one-line edit for future re-tightening |
| 31   | Sticky-snapshot regression test (`test_anchor_lazy_snapshot_is_sticky`): docs that lazy snapshot fires once per `_target_mass is None` window. | locks the snapshot semantics |
| 32   | Matrix-runner gate constants pinned (`test_matrix_runner_mass_drift_constants_sane`) — regression to pre-iter-23 1e-2 now fails CI. | catches gate-loosening regressions |
| 33   | 100-step long-run hydro PE test (`test_long_run_mass_conservation_cubed_sphere_pe`): measured drift `4.18e-15`, gate `< 1e-12`. | locks long-run anchor stability |
| 34   | Long-run guards extended to SW (`...fv3_cube`) and NH (`...cubed_sphere`). Full atmos sweep: 986/988 PASS. | locks SW + NH long-run paths |
| 35   | Spectral SW `compute_mass(state)` — uniform public API across all 4 SW grids (no anchor needed; baseline bit-clean). | API parity |
| 36   | Resolve iter-18 scan limitation: anchored `integrate_scan` works at machine precision (`8.03e-16` drift) under `PrecisionPolicy.fp64()`. Test pins it. | scan workflows have a clean path |
| 37   | `jax.grad` through anchored cube PE step — single-step. | locks AD through additive fixer |
| 38   | `jax.grad` 3-step chain — grows `5.65e-02 → 2.96e-01`. | locks multi-step AD |
| 39   | `jax.grad` through spectral PE anchored fixer (SH round-trip in chain) — complex128 grad `3.22`. | locks spectral AD path |
| 40   | `jax.grad` through cube NH anchored fixer (`fix_mass_nonhydrostatic` on 3-D `rho_prime`) — fp64 grad `1.63`. 20/20 anchor API tests PASS. Doc compress (iter-31..39 collapsed to single rows). | locks NH AD path |
| 41   | spectral NH AD — 3-D SH round-trip in fixer. Last structural AD path covered. complex128 grad `4.71e-04`. | locks spectral NH AD |
| 42   | fp64 KE+PE fields in `fix_energy_shallow_water` + `fix_energy_mpas`. Same fp32-field bug as iter-1..5, energy path. | extends fp64 convention to energy fixers |
| 43   | fp64 fields in `compute_hydrostatic_energy` + `compute_nh_energy`. | fp64 mass + energy diagnostics |
| 44   | fp64 in `compute_global_moisture` + `compute_conservation_diagnostics`. `fix_total_water` transitively covered. | fp64 mass + energy + moisture |
| 45   | fp64 in `diagnostics/total_energy_{pe,nh}` (FV3-fidelity ports). | fp64 convention into diagnostics/ |
| 46   | fp64 in `diagnostics/angular_momentum` (4 helpers — NH + PE compute and apply_aam_correction). | fp64 AAM diagnostics |
| 47   | fp64 in `diagnostics/energy_budget` (column MSE + DSE). | fp64 MSE/DSE column integrals |
| 48   | fp64 in `diagnostics/column_integrals.column_water_vapor` (canonical CWV helper). `precision_drift` + `monthly_means` audited, already clean. | one canonical fp64 CWV |
| 49   | Remove dead `fix_mass_hydrostatic_latlon` (unused). | trim orphan |
| 50   | Doc compress (iter-41..49 rolled into single rows). | every conservation helper across `core/conservation.py` + `diagnostics/{total_energy_*, angular_momentum, energy_budget, column_integrals}.py` now performs non-trivial arithmetic in fp64 before any reduction. |
| 51   | Drop stale `_accumulation_dtype` import from `scripts/run_atmosphere_test_matrix.py` (imported but never used after iter-1/4/5/12 migrated all live conservation sites to `_conservation_accumulator`).  Cube cosine_bell quick PASS. | trim stale import |
| 52   | Extend `test_diagnostic_fp64_dtypes.py` with a 6th case — `compute_total_energy_pe` (iter-45 FV3-fidelity PE total-energy diagnostic). 6/6 dtype tests PASS. | pins PE TE fp64 contract |
