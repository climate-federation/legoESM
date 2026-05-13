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

## Cumulative cross-grid status (iter-1..10)

| Equation set | cube | latlon FV | MPAS/ico | spectral |
|--------------|------|-----------|----------|----------|
| SW           | bit-clean (iter-5/6)  | bit-clean (iter-1/4) | bit-clean (iter-6) | bit-clean (iter-1)    |
| hydrostatic  | bit-clean (4e-8 baseline) | bit-clean (iter-2)   | bit-clean (1e-12 baseline) | bit-clean (iter-3) |
| NH           | bit-clean (iter-7)    | not implemented       | bit-clean (iter-8) | bit-clean (iter-9)    |

NH cross-grid TC1 mass-drift after iter-10 (matrix runner `notes` line):
- cube  |w|_max=0.3177 m/s, mass_drift=0.00e+00
- ico   |w|_max=0.0145 m/s, mass_drift=0.00e+00
- spec  |w|_max=0.0144 m/s, mass_drift=0.00e+00

## Improvement log

- **iter-1**: fp64 accumulator in `core/operators_latlon.global_integral` +
  matrix-runner `_area_weighted_sum` helper.  Stopped the fp32 reduction
  noise that surfaced as spurious `mass_drift` on lat-lon and spectral.
  Lat-lon HS 30-day drift `3.35e-3 → 1.57e-4` (1-day equivalent).
- **iter-2**: lat-lon PE `anchor_mass_to_initial` flag + lazy fp64 snapshot
  + drop `correction.astype(p_s.dtype)` cast in `_apply_safety_rails`.
  Lat-lon HS drift `1.57e-4 → 3.71e-7`.
- **iter-3**: spectral PE anchored mass fixer
  (`lnps_hat[0] += log(target/now)·sqrt(4π)`).  All 13 spectral hydro
  cases collapse to ~10^-16 (rossby_haurwitz `2.65e-3 → 1.12e-15`).
- **iter-4**: lat-lon SW `anchor_mass_to_initial` + use
  `_conservation_accumulator` (fp64) in fixer + drop `correction.astype`.
  Williamson-5 `1.21e-5 → 3.24e-16`.
- **iter-5**: FV3 cube SW (three model classes) `set_initial_mass` + fixer
  switch to fp64 budget acc + drop `correction.astype`.
  Cube Williamson-5 `9.68e-7 → 1.46e-15`.
- **iter-6**: cube `transport_step` `mass_pos` to fp64; MPAS SW
  `anchor_mass_to_initial` + fp64 fixer.
  Cube cosine_bell `4.49e-7 → 2.18e-8`; ico W5 `3.52e-10 → 1.62e-16`;
  ico W6 `2.14e-9 → 0`.
- **iter-7**: enable cube NH anchored mass fixer
  (`CompressibleEulerConfig.fix_mass=True, anchor_mass_to_initial=True`)
  at TC1/TC2a/TC3; expose `mass` in scalar_fn.  Cube NH TC1 mass drift
  `(not tracked) → 0.00e+00`.
- **iter-8**: MPAS NH anchored mass fixer.  New `compute_nh_dry_mass_mpas`
  / `fix_mass_nonhydrostatic_mpas` / `_batch_global_area_sums_voronoi` in
  `core/conservation.py`; config flags + lazy snapshot + step wrapper in
  `compressible_euler_mpas.py`.  Ico NH TC1 `(not tracked) → 0.00e+00`.
- **iter-9**: spectral NH anchored mass fixer
  (`rho_prime_hat[0,:] += Δρ·sqrt(4π)` mirrors iter-3 PE).
  Spectral NH TC1 `(not tracked) → 0.00e+00`.  All three NH grids
  bit-conserve.
- **iter-10**: surface `mass_drift` in the NH `notes` line of
  `run_atmosphere_test_matrix.py` (parallel to SW/hydro runners) so
  conservation is visible without parsing `mean_timeseries.csv`.
- **iter-11**: MPAS PE `anchor_mass_to_initial` flag + fp64 budget acc
  in `_fix_mass_mpas_hydro` (target_mass kwarg, fp32-cast on
  correction dropped); cube PE matrix-runner sites opt in to anchor.
  Ico HS 1-day mass drift `1.31e-12 → 1.61e-16`; cube HS 1-day
  `4.01e-08 → 4.02e-13` (~10^5x).
- **iter-12**: lat-lon PE `compute_mass` + in-step fixer + HS-path
  snapshot all switched from `_accumulation_dtype` (fp32 default) to
  `_conservation_accumulator` (fp64). The anchored target itself was
  contaminated with fp32 reduction noise — iter-2's 3.71e-7 residual
  was that, not the dynamics. Latlon HS sigma now `3.71e-07 → 1.61e-16`;
  hybrid `4.79e-07 → 1.61e-16`; topo `1.73e-07 → 0.00e+00`. Cumulative
  iter-0→iter-12 lat-lon HS reduction: ~10^13x.
- **iter-13**: `_total_area(grid)` in `core/conservation.py` casts
  `grid.grid_total_area` to the fp64 budget accumulator before being
  used as the divisor of every mass-fixer correction. All five
  `grid.total_area` callsites in this module migrated to the helper;
  MPAS SW `fix_mass_mpas` does the same on `mesh.grid_total_area`.
  Defensive change — on cube the storage cast is currently a no-op
  due to `cast_pytree(..., allow_downcast=False)` keeping post-fixer
  state in fp64 (the residual 4e-13 cube HS drift is fp64 accumulation
  noise across ~430 steps, not fp32 quantization).  Protects against
  future allow_downcast=True paths or fp32-only storage policies.
- **iter-14**: audit + cleanup.  Confirmed `zero_mean_tendency`,
  `fix_mass_hydrostatic_target`, `fix_mass_shallow_water` all already
  use fp64 accumulators / iter-13 `_total_area` helper.  Confirmed no
  remaining `_accumulation_dtype()` callers outside
  `core/conservation.py` itself; removed the three now-stale dycore
  imports (`shallow_water_latlon_cgrid.py`,
  `shallow_water_fv3_cdgrid.py`, `primitive_eq_latlon_cgrid.py`).
  829 unit tests PASS.
- **iter-15**: regression test
  `tests/atmosphere/hydrostatic/validation/test_mass_conservation_anchored.py`
  asserts mass drift ≤ 1e-10 over a short HS-init integration on all
  four PE solvers (cube C12 / lat-lon 36x72 / MPAS ico4 / spectral T21,
  5 steps each).  Catches regressions in any of the iter-1..14 fixer
  changes; 4/4 PASS in ~27s.

## Implementation patterns established (apply to any new dycore)

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
   `cast_pytree(..., "storage")` for the single round-trip.  Casting
   the correction BEFORE the add drops it below ULP and silently
   defeats the fixer.
4. **Spectral fixer convention.** A constant `Δ` in physical space
   maps to `Δ·sqrt(4π)` at the (n=0,m=0) coefficient under this repo's
   (4π)-normalised real-SH basis (verified empirically:
   `sh_analysis(ones)[0] == sqrt(4π)`).
5. **Matrix runner opt-in.** New flags default-off in the config; the
   matrix runner explicitly opts in.  Preserves bit-for-bit baseline
   for tests that intentionally measure drift, while production runs
   get clean conservation.

## Iteration 11 plan

- TC2a / TC3 mass-drift validation across cube/ico/spectral now that
  iter-10 surfaces drift in the notes line.
- Williamson-2 v-wind visual snapshot regression (CLAUDE.md guidance).
- Lat-lon NH not yet implemented (matrix runner returns SKIP).  Audit
  whether the lat-lon PE infrastructure can host a non-hydrostatic
  variant or whether the existing skip is structurally correct.
- Cross-grid summary plot of `mean_timeseries.csv` mass columns for one
  long (30-day) HS run to confirm post-iter-1..10 there's no remaining
  per-grid drift.
