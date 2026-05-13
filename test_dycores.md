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

- **`lax.scan` compat** — anchored fixers keep post-fix state in fp64
  via `cast_pytree(allow_downcast=False)`. `lax.scan` requires
  type-stable carry, so scan-based workflows currently need
  `anchor_mass_to_initial=False`. Lossy downcast would worsen drift by
  ~10^5x — not a worthwhile trade.
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
| 31   | Add `test_anchor_lazy_snapshot_is_sticky` to `test_anchor_mass_api.py` — documents that the lazy snapshot fires once per `_target_mass is None` window; AD / repeated-init users must call `reset_target_mass()` between fresh initial states.  13/13 anchor API tests PASS. | locks the snapshot semantics |
| 32   | Add `test_matrix_runner_mass_drift_constants_sane` — asserts iter-30 matrix-runner gate constants stay at `1e-6` / `1e-4` (with a `fp64-floor`-aware lower bound on the tighter gate).  A regression to the pre-iter-23 `1e-2` ceiling now fails CI directly. 14/14 anchor API tests PASS. | catches gate-loosening regressions |
