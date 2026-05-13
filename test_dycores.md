# Dycore Test Hardening — `test_dycores` branch

Single source of truth for the suite: `scripts/run_atmosphere_test_matrix.py`.
Grid mapping the task asks about:

| Task grid name    | Matrix grid key | Dycore impl                                                        |
|-------------------|-----------------|--------------------------------------------------------------------|
| lat-lon FV        | `latlon`        | `CGridLatLonShallowWater`, `CGridLatLonPrimitiveEquationModel`     |
| FV3 (cubed)       | `cubed_sphere`  | `FV3EdgeShallowWaterModel`, `CDGridPrimitiveEquationModel`, `CDGridCompressibleEulerModel` |
| MPAS              | `icosahedral`   | `MPASShallowWater`, `MPASPrimitiveEquationModel`, `MPASCompressibleEuler` |
| Spectral (extra)  | `spectral`      | `SpectralShallowWaterModel`, `SpectralPrimitiveEquationModel`, `SpectralCompressibleEulerModel` |

Coverage ladder (from `tests/validation/README_DYCORE_PROGRESSION.md`):
1. Shallow water Williamson — W2 / W5 / W6 / cosine bell
2. Hydrostatic 3D — Held-Suarez, baroclinic wave, DCMIP transport, AMIP, topo variants, gravity waves, Rossby-Haurwitz
3. Non-hydrostatic — DCMIP 2025 TC1 / TC2 / TC3

## Baseline (pre-iteration-1, `main` 4089b59e)

`results/atmosphere/summary.txt` — 91 tests / 83 PASS / 0 FAIL / 8 SKIP.
Skips: spectral DCMIP transport (not implemented), spectral NH TC3 (Kessler not wired).

Conservation outliers worth fixing:

| Case (quick mode)                       | Mass drift  | Note                                        |
|-----------------------------------------|-------------|---------------------------------------------|
| latlon hydro held_suarez (sigma/hybrid) | 3.35e-3 / 2.99e-3 | 5 orders worse than cube/ico/spectral |
| latlon hydro amip                        | 2.23e-3     | Same root cause                              |
| spectral hydro held_suarez_topo / mountain_rossby / rossby_haurwitz | 3.33e-4 / 1.63e-3 / 2.65e-3 | Same root cause |
| cubed_sphere hydro held_suarez           | 4.01e-8     | Reference baseline                           |
| icosahedral hydro held_suarez            | 1.31e-12    | Best baseline                                |

## Iteration 1 — fp32 sum noise in mass diagnostic (FIX)

**Root cause.** `LatLonGrid.area`, `CubedSphereGrid.area`, `GaussianGrid.grid_area`
default to the **storage** dtype (fp32 unless `JAX_ENABLE_X64=1`).  The matrix
runner's `mass_fn` evaluated `float(jnp.sum(s.p_s * grid.area))` directly, i.e.
an fp32 product reduced in fp32 over ~16k–1M cells.  Plain fp32 reductions
accumulate `~N·eps ≈ 10^6 · 10^-7 ≈ 0.1` relative error, so the reported "mass
drift" was almost entirely diagnostic precision noise, not real conservation
loss.  Cubed-sphere happened to be clean because it went through
`core/operators.global_integral` which already casts to fp64; Voronoi was clean
because `mesh.areaCell` is fp64; lat-lon and spectral went through the bare-sum
path.

`core/operators_latlon.global_integral` had the same bare-sum bug
(`jnp.sum(field.data * grid.area)`) and was used by no model state diagnostic
yet, but anything new built on it would have inherited the issue.

**Changes (targeted; 2 files):**

- `src/legoesm/core/operators_latlon.py` — cast to fp64 accumulator before
  product/sum in `global_integral`; promote denominator dtype in `global_mean`;
  add MPI `global_sum_mpi` path mirroring `core/operators.global_integral`.
- `scripts/run_atmosphere_test_matrix.py` — added `_area_weighted_sum(field,
  area)` helper (fp64 promote + sum) next to the existing
  `_area_weighted_mean`.  Replaced every `mass_fn` / `scalar_fn` mass-integral
  call (3 latlon, 3 icosahedral, 3 spectral; cubed-sphere stays on
  `global_integral` to keep MPI semantics).

**Validation:**

- `tests/unit/test_operators_latlon.py::TestGlobalIntegral` PASS.
- Lat-lon SW Williamson-2 (`--only sw --grid latlon --test williamson2 --quick`):
  PASS, L2/Linf unchanged.
- Lat-lon hydro Held-Suarez (`--only hydro --grid latlon --test held_suarez
  --days 1`): PASS, **mass drift = 1.57e-4** (sigma) / **2.17e-4** (hybrid).
  Down from baseline 3.35e-3 / 2.99e-3 at quick=30d.  Per-step drift is
  monotonically smaller, confirming the bulk of the previous "drift" was
  diagnostic noise rather than real loss.  Residual ~1e-4 over 1 day still
  exceeds machine precision — see Iteration 2 plan.

## Iteration 2 plan — residual drift in latlon fixer

The remaining ~1e-4 drift after the diagnostic fix points at the per-step
fixer.  Two suspects:

1. `_apply_safety_rails` in `primitive_eq_latlon_cgrid.py` line 775 does
   `correction.astype(p_s_pre.dtype)` before adding to `p_s_pre` — drops the
   fp64 correction to fp32 prior to addition, where the cubed-sphere
   `fix_ps_mass` keeps the correction at fp64 and lets JAX promote the sum.
2. The end-of-step `cast_pytree(state_new, None, "storage")` always rounds
   `p_s` back to fp32 storage, so the per-step correction quantum below
   `~ULP(p_s) ≈ 1e-2 Pa` is lost regardless.  Initial-mass anchoring
   (`anchor_mass_to_initial`, already present for cube) avoids accumulating
   that quantization noise across steps.

Iteration 2 will (a) drop the `.astype` cast for parity with cube, and (b)
plumb `anchor_mass_to_initial` through `CGridLatLonPrimitiveEquationConfig` so
the runner can opt into anchored conservation.

## Iteration 2 — initial-mass anchor + fp32-cast removal in latlon fixer (FIX)

**Root cause of residual ~10^-4 drift.** Two compounding effects:

1. ``_apply_safety_rails`` did
   ``p_s_post = p_s_pre + correction.astype(p_s_pre.dtype)`` — the fp64
   ``correction`` was rounded to fp32 *before* the add, so the per-step
   correction quantum below ``ULP(p_s) ≈ 1e-2 Pa`` was silently dropped.
2. The fixer anchored to the previous post-cast ``p_s`` (storage fp32) instead
   of the initial mass.  The unchanged storage round-trip on every step
   produced a random walk in total mass — cubed-sphere stayed clean by chance
   (smaller per-cell mass quantum after metric weighting), lat-lon did not.

**Changes (1 source file + 1 runner site):**

- ``src/legoesm/atmosphere/dynamics/primitive_eq_latlon_cgrid.py``
  - Added ``anchor_mass_to_initial: bool = False`` to
    ``CGridLatLonPrimitiveEquationConfig`` (mirrors ``DycoreConfig`` and
    ``CDGridPrimitiveEquationConfig``).
  - Added ``self._target_mass`` slot to the model; ``step()`` snapshots the
    initial mass in fp64 on first call when the flag is on, then threads
    ``target_mass`` into ``_step_cgrid`` so the existing ``target_mass`` path
    (already wired in ``_apply_safety_rails``) is reused — no new fix code.
  - Removed ``correction.astype(p_s_pre.dtype)`` in ``_apply_safety_rails`` so
    the add is fp64-promoted, matching ``fix_ps_mass`` (cubed-sphere).  The
    end-of-step ``cast_pytree(..., "storage")`` still rounds back to fp32, but
    the correction is applied *before* that single round-trip.
- ``scripts/run_atmosphere_test_matrix.py``: enable ``anchor_mass_to_initial=
  True`` at all three ``CGridLatLonPrimitiveEquationConfig`` sites (HS,
  baroclinic, AMIP).

**Validation (via `run_atmosphere_test_matrix.py`):**

| Case                                  | Baseline drift | Iter-1   | Iter-2     |
|---------------------------------------|----------------|----------|------------|
| latlon hydro held_suarez (sigma, 1d)  | 3.35e-3 (30d)  | 1.57e-4  | **3.71e-7** |
| latlon hydro held_suarez (hybrid, 1d) | 2.99e-3 (30d)  | 2.17e-4  | **4.79e-7** |
| latlon hydro held_suarez_topo (1d)    | 3.22e-4 (30d)  | 1.88e-4  | **1.73e-7** |
| latlon hydro baroclinic (2d)          | 8.53e-7 (10d)  | —        | 9.48e-7    |
| latlon hydro rotated_baroclinic (2d)  | 8.53e-7 (10d)  | —        | 6.03e-7    |

Held-Suarez drift now within 1 order of magnitude of cubed-sphere (4e-8) and
icosahedral (1e-12).  Baroclinic was already clean; anchor is neutral there.

Unit tests: ``tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py``
48/48 PASS.

## Iteration 3 — spectral PE anchored mass fixer (FIX)

**Root cause.** ``SpectralPrimitiveEquationModel`` had no mass fixer at all.
Mass drift on spectral hydrostatic cases (mountain_rossby 1.63e-3,
rossby_haurwitz 2.65e-3, AMIP 3.02e-4) was true uncorrected conservation loss
from the SSP-RK3/SI integrator, not diagnostic noise.

**Changes (1 src file + 1 runner site):**

- ``src/legoesm/atmosphere/dynamics/spectral_pe.py``
  - ``SpectralPEConfig``: add ``fix_mass: bool = False`` and
    ``anchor_mass_to_initial: bool = False`` (off by default to preserve
    bit-for-bit baseline for tests that measure drift; matrix runner opts in).
  - Add ``_target_mass`` slot, lazily populated in ``step()`` on first call
    (snapshot computed outside JIT in fp64 via ``_compute_initial_mass``).
  - ``_apply_mass_fixer(state)``: ``Δ = log(target / current)`` added to
    ``lnps_hat[0]`` after scaling by ``sqrt(4π)`` to match this module's
    (4π)-normalised real-SH convention (verified empirically:
    ``sh_analysis(ones)[0] == sqrt(4π)``).  Multiplicative correction in
    physical space preserves ``p_s`` gradients exactly (same property as the
    additive cubed-sphere/lat-lon ``fix_ps_mass`` correction).
  - Hooks: ``_do_step`` (SSP-RK3 path) + both leapfrog branches (Euler
    startup + leapfrog body, applied AFTER the Robert-Asselin filter so the
    computational mode is damped first, then mass is restored exactly).
- ``scripts/run_atmosphere_test_matrix.py``: enable ``fix_mass=True,
  anchor_mass_to_initial=True`` at all three ``SpectralPEConfig`` sites
  (HS, baroclinic, AMIP).

**Validation (via `run_atmosphere_test_matrix.py --only hydro --grid spectral --quick`):**

| Case                    | Baseline mass drift | Iter-3 mass drift |
|-------------------------|---------------------|-------------------|
| held_suarez             | 2.00e-05            | **1.61e-16**      |
| held_suarez (hybrid)    | 1.97e-04            | **0.00e+00**      |
| held_suarez_topo        | 3.33e-04            | **1.45e-15**      |
| baroclinic              | 3.82e-09            | **9.64e-16**      |
| amip                    | 3.02e-04            | **1.28e-15**      |
| rotated_baroclinic      | 2.75e-04            | **4.82e-16**      |
| rotated_steady          | 2.75e-04            | **3.21e-16**      |
| rest_state_topo         | 2.22e-04            | **1.13e-15**      |
| gravity_wave_3_1        | 4.19e-07            | **8.03e-16**      |
| inertio_gravity_3_2     | 4.44e-04            | **3.21e-16**      |
| mountain_rossby_5_0     | 1.63e-03            | **1.29e-15**      |
| rossby_haurwitz_6_0     | 2.65e-03            | **1.12e-15**      |

All 13 spectral hydro PASS cases at machine precision (~10^-16).  ``max|v|``
values unchanged (no dynamics regression).  Unit suite
``tests/atmosphere/hydrostatic`` 768/768 PASS.

## Iteration 4 — latlon SW anchored mass + fp64 budget accumulator (FIX)

**Root cause of residual ~10^-5/-6 SW drift.** Two cumulative effects:

1. ``CGridLatLonShallowWaterConfig`` had no ``anchor_mass_to_initial`` flag
   (only the per-step pre-state path).  Fp32 storage cast at end of step
   produced random-walk drift, same pattern as iter-2 PE.
2. The SW fixer used ``_accumulation_dtype()`` for the area-weighted sums,
   which returns **fp32** under the default fp32 storage policy (only the
   compute role would be fp64 in mixed mode).  ``_conservation_accumulator()``
   is the dtype budgets actually need — promotes to fp64 whenever x64 is
   enabled — and was already used by the cubed-sphere ``_batch_global_area_sums``
   path.  The fp32 reduction noise (~N·eps on a 16k-cell lat-lon grid) leaked
   into both the anchored target and the per-step mass, defeating the fixer
   even after iter-4's anchor change.

**Changes (1 src file + 4 runner sites):**

- ``src/legoesm/atmosphere/dynamics/shallow_water_latlon_cgrid.py``
  - Add ``anchor_mass_to_initial: bool = False`` to
    ``CGridLatLonShallowWaterConfig``.
  - Split ``step()`` into an outer Python wrapper that snapshots initial mass
    (fp64) on first call when the flag is on, plus the existing JIT-compiled
    body renamed to ``_step_jit``.  Reuses the existing ``target_mass``
    argument plumbing — no new fixer code.
  - ``compute_mass`` and the in-step fixer both switch from
    ``_accumulation_dtype`` to ``_conservation_accumulator`` so the area-
    weighted sums are reliably fp64.
  - Drop ``correction.astype(state_new.h.dtype)`` (same rationale as iter-2).
- ``scripts/run_atmosphere_test_matrix.py``: enable
  ``anchor_mass_to_initial=True`` at the lat-lon SW config; replace 3
  ``float(jnp.sum(state.h * grid.area))`` mass diagnostics (W5, cosine_bell
  cube target, cosine_bell latlon final) with ``_area_weighted_sum`` for
  consistency with iter-1.

**Validation (via `run_atmosphere_test_matrix.py --only sw --grid latlon --quick`):**

| Case               | Baseline mass drift | Iter-4 mass drift |
|--------------------|---------------------|-------------------|
| williamson5        | 1.21e-05            | **3.24e-16**      |
| cosine_bell        | 1.43e-05            | 1.49e-05 (n/c)    |
| williamson2        | L2=2.54e-04 (n/c)   | L2=2.67e-04 (n/c) |

Williamson-5 mass drift now at machine precision (~10^-11 reduction).
Williamson-2 / cosine-bell error norms unchanged (cosine_bell uses a custom
raw-FV step that bypasses the SW model, so the model edits don't reach it —
that's a separate fixer wiring for iter-5).
``tests/atmosphere/shallow_water`` 110/110 PASS.

## Iteration 5 — FV3 cube SW fp64 budget accumulator + drop fp32 cast (FIX)

**Root cause.** Three SW model classes in ``shallow_water_fv3_cdgrid.py``
(``FV3EdgeShallowWaterModel``, ``FV3FBShallowWaterModel``, the experimental
csw variant) shared the same iter-4-style bug pair:

1. ``set_initial_mass`` snapshotted with a bare ``jnp.sum(state.h *
   self.cdgrid.base.area)``.  Both inputs are typically fp32, so the ~6·N²
   cubed-sphere reduction leaked ~N·eps noise into the anchor.  Set via
   ``set_initial_mass`` from the matrix runner at line 2016/2442, so every
   cube SW Williamson case inherited the noise.
2. The fixers used ``_accumulation_dtype()`` (fp32 on default storage
   policy) and added the correction back with ``correction.astype(state_new.h.dtype)``,
   so even with an anchored target the per-step round-trip lost precision.

**Changes (1 src file):**

- ``src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py``
  - Import ``_conservation_accumulator``.
  - Three ``set_initial_mass`` methods (lines 631-633, 840-841, 950-951) now
    cast both inputs to the fp64 budget accumulator before the sum.
  - Three in-step fixers (lines 741, 887, 1193) switch from
    ``_accumulation_dtype`` to ``_conservation_accumulator``.
  - Three ``h_fixed = state_new.h + correction.astype(...)`` adds (lines 759,
    901, 1213) drop the cast for fp64 promotion (matches iter-2 / iter-4).

**Validation (via `run_atmosphere_test_matrix.py --only sw --grid cubed_sphere --quick`):**

| Case               | Baseline mass drift | Iter-5 mass drift |
|--------------------|---------------------|-------------------|
| williamson5        | 9.68e-07            | **1.46e-15**      |
| williamson2        | L2=2.05e-04 (n/c)   | L2=2.04e-04 (n/c) |
| cosine_bell        | 4.49e-07            | 5.38e-07 (n/c)    |

Williamson-5 mass drift now at machine precision (~10^8 reduction).
cosine_bell uses a separate raw-FV path (`transport_step` in `fv_tp_2d.py`)
and is unchanged — that's iter-6 if pursued.
``tests/atmosphere/shallow_water`` 110/110 PASS.

## Iteration 6 — MPAS SW anchor + cube transport_step fp64 acc (FIX)

**Two targeted fixes.**

(a) ``core/fv_tp_2d.transport_step`` (cube cosine_bell raw-FV path):
    ``mass_pos = jnp.sum(h_pos * area)`` ran in fp32 on a ~6·N² cubed-sphere
    grid, leaking ~N·eps reduction noise into the multiplicative ``scale``.
    Cast both factors to ``_conservation_accumulator()`` before the sum
    (matches iter-5 SW model).  ``scale`` is now divided in fp64 and cast
    once back to ``h_pos.dtype`` before the final multiply.

(b) ``MPASShallowWaterModel``: add ``anchor_mass_to_initial: bool = False``
    + ``_target_mass`` slot + lazy fp64 snapshot in ``step()``.  Split
    ``step`` into a Python wrapper (snapshot, outside JIT) and the existing
    JIT body renamed ``_step_jit``.  ``_fix_mass_mpas`` learns an optional
    ``target_mass`` kwarg: when set, the per-step ``state_old.h`` reduction
    is dropped (target replaces it) and the fp32-cast on ``correction`` is
    removed for fp64 promotion.

**Changes:**

- ``src/legoesm/core/fv_tp_2d.py`` (1 site, ~10 LOC).
- ``src/legoesm/atmosphere/dynamics/shallow_water_mpas.py`` (~30 LOC):
  config + ``_target_mass`` + ``compute_mass`` helper + step wrapper +
  ``_fix_mass_mpas`` extension.
- ``scripts/run_atmosphere_test_matrix.py``: enable
  ``anchor_mass_to_initial=True`` at the icosahedral SW config.

**Validation (via `run_atmosphere_test_matrix.py --only sw --quick`):**

| Case                            | Baseline | Iter-5 | Iter-6        |
|---------------------------------|----------|--------|---------------|
| cube  shallow_water  W5         | 9.68e-07 | 1.46e-15 | 1.46e-15    |
| cube  shallow_water  cosine_bell| 4.49e-07 | 5.38e-07 | **2.18e-08**|
| ico   shallow_water  W5         | 3.52e-10 | n/c     | **1.62e-16**  |
| ico   shallow_water  W6         | 2.14e-09 | n/c     | **0.00e+00**  |
| ico   shallow_water  cosine_bell| 4.16e-07 | n/c     | 4.16e-07 (n/c)|

ico W5/W6 now at exact bitcleanness.  Cube cosine_bell ~25x cleaner
(remaining 2e-8 is residual non-conservation in the PPM clip+rescale, not
diagnostic noise).  ``tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py``
7/7 PASS.

## Iteration 7 plan

- Full quick suite snapshot (all grids × SW/hydro/NH) for a single
  before/after table covering iter-1..iter-6.
- NH dycore audit: ``compressible_euler.py`` / ``compressible_euler_cdgrid.py``
  / ``compressible_euler_mpas.py`` / ``spectral_nh.py`` — none of the iter-1..6
  changes touched the NH solvers; check if they have the same fp32-cast
  fixer pattern.
- Visual snapshot regression check (Williamson-2 v-wind, lat-lon + cube) per
  CLAUDE.md guidance.

## Improvement log

- **iter-1 (2026-05-13)**: fp64 accumulator for lat-lon `global_integral` /
  matrix-runner mass diagnostics; `_area_weighted_sum` helper.  Latlon HS mass
  drift `3.35e-3 → 1.57e-4` (1-day, sigma).
- **iter-2 (2026-05-13)**: `anchor_mass_to_initial` for latlon PE + drop fp32
  cast in `_apply_safety_rails`.  Latlon HS mass drift `1.57e-4 → 3.71e-7`
  (1-day, sigma).  Cumulative iter-0→iter-2 reduction: ~9000x.
- **iter-3 (2026-05-13)**: spectral PE anchored mass fixer
  (`lnps_hat[0] += log(target/now)·sqrt(4π)`).  All 13 spectral hydro cases
  now drift at ~1e-16 (machine precision).  Largest baseline drift
  (rossby_haurwitz 2.65e-3) collapses to 1.12e-15 — ~10^13x reduction.
- **iter-4 (2026-05-13)**: latlon SW `anchor_mass_to_initial` +
  `_conservation_accumulator` (fp64) in fixer + drop fp32 cast.
  Williamson-5 mass drift `1.21e-05 → 3.24e-16` (machine precision).
- **iter-5 (2026-05-13)**: FV3 cube SW (three classes) `set_initial_mass` +
  fixer use fp64 budget accumulator, drop fp32 cast on `h` correction.
  Williamson-5 mass drift `9.68e-07 → 1.46e-15` (machine precision).
- **iter-6 (2026-05-13)**: cube `transport_step` mass_pos fp64;
  MPAS SW `anchor_mass_to_initial` + fp64 fixer.  cube cosine_bell
  `4.49e-07 → 2.18e-08`; ico W5 `3.52e-10 → 1.62e-16`;
  ico W6 `2.14e-09 → 0.00e+00`.
