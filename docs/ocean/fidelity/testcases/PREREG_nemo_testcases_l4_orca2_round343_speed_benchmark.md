# ORCA2 Round 343 Preregistration — Speed Benchmark of legoESM ORCA2 (Exact vs Normal Mode)

- **Round**: `orca2_r343`
- **Author**: Argon
- **Branch**: `fidelity/nemo-testcases-orca2-rounds`
- **Base commit**: `665e6a64a084` (`orca2_r341: day-1..30 prefix rescore walk on in-flight 1-year ORCA2 K0 run`)
- **Date**: 2026-10-28

## 1. Objective & Scope

Measure the execution speed and 8-day accuracy cost of legoESM's normal (non-bit-exact) mode relative to exact-NEMO mode on the standalone ORCA2 K0 composed card (`build_nemo_orca2_k0_ocean_card`, `nemo_orca2_k0_ocean_step`).

**Hard constraints**:
- Measurement only: no model code or default in `packages/` is modified.
- No NEMO execution; never disturb the in-flight 1-year CPU run (PID `4171654`).
- GPU benchmarks run exclusively on an idle GPU (`CUDA_VISIBLE_DEVICES=0`, 0 MiB used, 0% utilization prior to launch).

## 2. Pre-Registered Switch Inventory Classification

By reading `packages/core/legoesm/core/` and `packages/ocean/legoesm/ocean/`, every switch that places `build_nemo_orca2_k0_ocean_card` into exact-NEMO mode is classified into four categories:

1. **Category A — Hard `libm`-guarded switches (6 switches)**:
   Switches that raise `RuntimeError` unless `PrecisionPolicy.transcendentals == "libm"` (or set the `libm` policy itself):
   - `PrecisionPolicy.transcendentals`: `"libm"` -> `"native"`
   - `NEMOTestcaseCard.transcendentals`: `"libm"` -> `"native"`
   - `TKEConfig.tke_htau_evaluation`: `"nemo_scalar_libm"` -> `"jax_expression"`
   - `TKEConfig.tke_etau_exponential_evaluation`: `"nemo_scalar_libm"` -> `"jax_expression"`
   - `DoubleDiffusionConfig.(enabled, form)`: `(True, "nemo_literal")` -> `(False, "hybrid")`
   - `IWMConfig.(form, tsdiff)`: `("nemo_literal", True)` -> `("generic", False)`
2. **Category B — Source-rounding, operand-association, and literal-order switches with normal defaults (36 switches)**:
   Switches on `TKEConfig` (8), `GMRediConfig` (5), `MLEConfig` (1), `ZStarCoordinate` (1), `LatLonCGridOceanConfig` (19), and `BarotropicConfig` (2) that insert `nemo_source_round` / `optimization_barrier` or enforce NEMO's Fortran statement association order and have a working normal default on the ORCA2 K0 card.
3. **Category C — Unconditional `nemo_source_round` / `nemo_source_barrier` sites**:
   Direct calls to `nemo_source_round` / `nemo_source_barrier` (`packages/core/legoesm/core/source_rounding.py:14-35`) in `bulk_flux_omip.py`, `eos.py`, `shortwave_penetration.py`, and `freshwater.py` that have no config toggle.
4. **Category D — Structurally coupled WS-RK3 / partial-cell / tripolar scheme selectors**:
   Scheme-level selectors (`zdf_implicit_solver_evaluation="nemo_literal"` and its face-thickness/barotropic-removal companions, stage-3 L8/L9 Krhs bundle, tripolar EEN/Redi/TKE-carry/barotropic-continuity selectors) required by `LatLonCGridOceanModel` constructor validation or 3-D partial-cell staircase masking on the ORCA2 `rk3_ws` + `OceanPartialCellCoordinate` + `NorthFold` card.

Normal mode in arms (b), (c), and (d) sets all 42 switches in Categories A and B to their normal class defaults while leaving `packages/` defaults untouched.

## 3. Pre-Registered Benchmark Arms & Protocol

All arms use the identical standalone ORCA2 K0 composed card and SHA-verified inputs:
- Deck root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2`
- CORE manifest: `input_files.sha256`
- Ice cover climatology: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round339/ice_cover_producer/orca2_full_si3_year1_np2/ice_cover_clim.nc`
- Reference exact-mode daily states: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/legoesm_year1_2920step/candidate_step{kt:08d}.npz` (`kt = 8, 16, 24, 32, 40, 48, 56, 64`).

Warm-up / compile window: steps `kt = 1..2` (step 1 compiles the cold-start `eta_rk3_after=None, u_before=None` trace; step 2 compiles the carried `eta_rk3_after=Field, u_before=Field` trace; steps `kt >= 3` execute without recompilation).

| Arm | Mode | Backend | Precision | Steps Run | Warm-up Steps | Steady-State Window | Accuracy Comparison Steps |
|---|---|---|---|---|---|---|---|
| **(a)** `exact_cpu_fp64` | Exact (`libm`, base card) | CPU | `float64` (`JAX_ENABLE_X64=1`) | 22 (`kt=1..22`) | `kt=1..2` | `kt=3..22` (20 steps) | `kt=8, 16` (Days 1–2) |
| **(b)** `normal_cpu_fp64` | Normal (`native`, Cat A+B defaults) | CPU | `float64` (`JAX_ENABLE_X64=1`) | 64 (`kt=1..64`) | `kt=1..2` | `kt=3..64` (62 steps) | `kt=8, 16, ..., 64` (Days 1–8) |
| **(c)** `normal_gpu_fp64` | Normal (`native`, Cat A+B defaults) | 1 GPU (`CUDA_VISIBLE_DEVICES=0`) | `float64` (`JAX_ENABLE_X64=1`) | 64 (`kt=1..64`) | `kt=1..2` | `kt=3..64` (62 steps) | `kt=8, 16, ..., 64` (Days 1–8) |
| **(d)** `normal_gpu_fp32` | Normal (`native`, Cat A+B defaults) | 1 GPU (`CUDA_VISIBLE_DEVICES=0`) | `float32` (`JAX_ENABLE_X64=0`) | 64 (`kt=1..64`) | `kt=1..2` | `kt=3..64` (62 steps) | `kt=8, 16, ..., 64` (Days 1–8) |

For each arm, the benchmark records:
- **Compile / warm-up time [s]**: `t(kt=1) + t(kt=2)` (and individual `t(kt=1)`, `t(kt=2)`), plus net compile overhead `t(kt=1) + t(kt=2) - 2 * median_step_s`.
- **Steady-state seconds per step [s/step]**: median over the steady-state window (`>= 20` steps after warm-up), along with mean, IQR, min, max, odd-step median (`nn_fsbc=2` surface-forcing update steps), and even-step median (`nn_fsbc=2` surface-forcing hold steps).
- **Memory**: peak host RSS [GiB] from `/proc/self/status` (`VmHWM`) / `resource.getrusage`, plus GPU peak memory in-use [MiB / GiB] from `jax.devices()[0].memory_stats()["peak_bytes_in_use"]` on GPU arms.
- **Projected wall-clock for 1 year (2,920 steps)**: both steady-state-only (`2920 * median_step_s`) and cold-start-inclusive (`t(kt=1) + t(kt=2) + 2918 * median_step_s`).
- **Daily accuracy cost vs exact mode**: for each day `d in 1..8` (`kt = 8*d`), SST (`T[..., 0]`) and 3-D `T` RMS difference (over all grid cells and over wet cells `is_active`) and max absolute difference (`max |T_cand - T_ref|`).

## 4. Verification & Controls

- **Confirm**:
  - `git diff packages/` is empty; `build_nemo_orca2_k0_ocean_card` default card fields remain unchanged.
  - Arm (a) matches `orca2_r342` `candidate_step00000008.npz` and `candidate_step00000016.npz` with `0.0` RMS and `0.0` max difference.
  - Arms (b), (c), and (d) complete 64 steps with all prognostic arrays finite and produce complete timing, memory, 1-year projection, and 8-day SST/T accuracy tables.
  - Direct unit tests in `tests/ocean/fidelity/test_orca2_round343_speed_benchmark.py` (including planted-violation controls) and citation gate tests in `tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py` pass.
- **Refute**:
  - Any modification to `packages/`, non-finite state in any arm, fewer than 20 steady-state steps after warm-up, or non-zero difference in arm (a) vs `orca2_r342`.
