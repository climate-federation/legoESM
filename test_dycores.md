# Dycore Test Hardening — `test_dycores` branch

Single source of truth: `scripts/run_atmosphere_test_matrix.py`.

| Task grid | Matrix key | Dycore impls |
|-----------|------------|--------------|
| lat-lon FV | `latlon` | `CGridLatLon{ShallowWater,PrimitiveEquation}` |
| FV3 cube | `cubed_sphere` | `FV3EdgeShallowWater`, `CDGridPrimitiveEquation`, `CDGridCompressibleEuler` |
| MPAS | `icosahedral` | `MPAS{ShallowWater,PrimitiveEquation,CompressibleEuler}` |
| Spectral | `spectral` | `Spectral{ShallowWater,PrimitiveEquation,CompressibleEuler}` |

Ladder (`tests/validation/README_DYCORE_PROGRESSION.md`):
1. SW Williamson — W2 / W5 / W6 / cosine_bell
2. Hydrostatic — Held-Suarez, baroclinic, DCMIP transport, AMIP, topo, gravity wave, Rossby-Haurwitz
3. NH — DCMIP 2025 TC1 / TC2a / TC3

## Cumulative status (iter-1..20)

**Mass conservation: bit-clean across every implemented grid × equation set**

| Equation set | cube | latlon FV | MPAS/ico | spectral |
|--------------|------|-----------|----------|----------|
| SW           | bit-clean (iter-5/6/20) | bit-clean (iter-1/4) | bit-clean (iter-6) | bit-clean (iter-1) |
| hydrostatic  | bit-clean (iter-11/12/20) | bit-clean (iter-2/12) | bit-clean (iter-11) | bit-clean (iter-3) |
| NH           | bit-clean (iter-7/20) | not implemented | bit-clean (iter-8) | bit-clean (iter-9) |

NH cross-grid TC1 mass-drift after iter-10 (matrix-runner `notes` line):
- cube  |w|_max=0.3177 m/s, mass_drift=0.00e+00
- ico   |w|_max=0.0145 m/s, mass_drift=0.00e+00
- spec  |w|_max=0.0144 m/s, mass_drift=0.00e+00

10-test cross-grid regression suite (iter-15/16/17) at:
- `tests/atmosphere/shallow_water/integration/test_sw_mass_conservation_anchored.py` (3 grids)
- `tests/atmosphere/hydrostatic/validation/test_mass_conservation_anchored.py` (4 grids)
- `tests/atmosphere/nonhydrostatic/integration/test_nh_mass_conservation_anchored.py` (3 grids)

Tolerance `DRIFT_TOL = 1e-12` over 20 steps (iter-17).  Each PE/SW/NH dycore
exposes `reset_target_mass()` (iter-18) and `set_target_mass(target)`
(iter-19) for anchor lifecycle management; cube SW classes additionally keep
the pre-existing `set_initial_mass(state)` convenience (iter-20 API parity).

## Implementation patterns established

1. **Anchored mass target.** Snapshot the initial dry mass in fp64
   outside JIT on the first `step()` call (`_target_mass` slot on the
   model).  Subsequent steps correct toward that target so per-step
   storage-cast rounding cannot random-walk the global integral.
2. **Budget accumulator.** Use `_conservation_accumulator()` (fp64 when
   x64 is enabled) — never `_accumulation_dtype()` — for any sum that
   feeds a conservation diagnostic or fixer.  A fp32 sum over 10^5+
   cells leaks ~`N·eps` noise that masquerades as physics drift.
3. **No `correction.astype(state.dtype)`.** Add the fp64 correction to
   the state directly; let JAX promote, and rely on the end-of-step
   `cast_pytree(..., allow_downcast=False, role="storage")` to keep
   post-fixer state in fp64.  Casting the correction BEFORE the add
   drops it below ULP and silently defeats the fixer.
4. **Spectral fixer convention.** A constant `Δ` in physical space
   maps to `Δ·sqrt(4π)` at the (n=0,m=0) coefficient under this repo's
   (4π)-normalised real-SH basis (verified empirically:
   `sh_analysis(ones)[0] == sqrt(4π)`).
5. **Matrix runner opt-in.** New flags default-off in the config; the
   matrix runner explicitly opts in.  Preserves bit-for-bit baseline
   for tests that intentionally measure raw drift, while production
   runs get clean conservation.
6. **Anchor lifecycle API.** Every anchored model exposes
   `reset_target_mass()` (clear → re-snapshot on next step) and
   `set_target_mass(target)` (explicit anchor — restart from checkpoint
   or cross-experiment reference).

## Known limitations

- **`lax.scan` compat.** Anchored fixers keep post-fix state in fp64
  via `cast_pytree(allow_downcast=False)` for the iter-1..12 fp64 floor.
  `jax.lax.scan` requires type-stable carry, so scan-based workflows
  (`integrate_scan`, `jax.grad` through long integrations) currently
  need `anchor_mass_to_initial=False`.  Iter-18 confirmed lossy-downcast
  would re-enable scan but worsen drift by ~10^5x — not yet a worthwhile
  trade.
- **Cube NH TC1 `|w|_max=0.3177`** vs ico/spectral 0.014.  Suspect cube-
  imprint edge artifact; tuning knobs available
  (`corner_div_damp_*`, `damp_v`) but defaults preserved.
- **Lat-lon NH:** matrix runner SKIPs — DCMIP-2025 inits require
  `CubedSphereGrid`, no lat-lon variant yet.

## Improvement log (compressed iter-1..20)

| Iter | Scope | Headline reduction |
|------|-------|--------------------|
| 1    | fp64 `global_integral` / `_area_weighted_sum` diagnostic | Latlon HS `3.35e-3 → 1.57e-4` |
| 2    | Latlon PE anchor + drop fp32 cast | `→ 3.71e-7` |
| 3    | Spectral PE anchored fixer | 13 spectral hydro cases all `→ ~1e-16` |
| 4    | Latlon SW anchor + `_conservation_accumulator` | W5 `1.21e-5 → 3.24e-16` |
| 5    | FV3 cube SW fp64 acc + drop cast | W5 `9.68e-7 → 1.46e-15` |
| 6    | `transport_step` fp64 acc; MPAS SW anchor | cube cosine_bell `4.49e-7 → 2.18e-8`; ico W5 `→ 1.62e-16` |
| 7    | Enable cube NH anchored fixer + `mass` diag | TC1 `(not tracked) → 0.00e+00` |
| 8    | MPAS NH fixer (`compute_nh_dry_mass_mpas`, `fix_mass_nonhydrostatic_mpas`) | TC1 `→ 0.00e+00` |
| 9    | Spectral NH fixer (`rho_prime_hat[0,:] += Δρ·sqrt(4π)`) | TC1 `→ 0.00e+00` |
| 10   | Surface `mass_drift` in NH `notes`; doc compress 482→102 | — |
| 11   | MPAS PE anchor + fp64 fixer; cube PE opt-in anchor | cube HS `4.01e-08 → 4.02e-13`; ico HS `1.31e-12 → 1.61e-16` |
| 12   | Latlon PE `compute_mass` + fixer fp64 budget | HS sigma `3.71e-7 → 1.61e-16` (~10^13x cumulative) |
| 13   | `_total_area(grid)` fp64 divisor in mass fixers | Defensive (cube was already fp64 via `allow_downcast=False`) |
| 14   | Strip stale `_accumulation_dtype` imports | — |
| 15   | 4-PE-solver regression test, drift ≤ 1e-12 / 5 steps | 4/4 PASS |
| 16   | Parallel SW + NH regression tests (3 + 3 grids) | 10/10 PASS |
| 17   | Tighten regression: 5→20 steps, 1e-10→1e-12 ceiling | 10/10 PASS |
| 18   | `reset_target_mass()` API on 7 lazy-snapshot models | — |
| 19   | `set_target_mass(target)` setter on same 7 models | — |
| 20   | `reset/set_target_mass` API on cube SW (3 classes) + cube PE + cube NH; doc compress | — |
| 21   | `compute_mass` / `compute_dry_mass` helpers on the four cube model classes (FV3EdgeSW, FV3FBSW, CDGridPE, CDGridCE) — API parity with MPAS/latlon/spectral twins. | — |
