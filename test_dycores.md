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

## Improvement log

- **iter-1 (2026-05-13)**: fp64 accumulator for lat-lon `global_integral` /
  matrix-runner mass diagnostics; `_area_weighted_sum` helper.  Latlon HS mass
  drift `3.35e-3 → 1.57e-4` (1-day, sigma).
