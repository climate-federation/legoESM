# SI3 Ladder Round 2 (`si3_r2`) — Rung D5 Preregistration

**Round**: `si3_r2` (SI3 ladder rung D5, executing Decision 156)  
**Branch**: `fidelity/nemo-testcases-si3-ladder`  
**Target record**: `/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_scalarmath_v2_a` (8,760 hourly steps of 3,600 s, `cfgs/C1D/EXP_SASICE` with the lane-3 3+3/salinity-2/pond-off scalar-math v2 overlay; existing immutable record reused without running NEMO).

---

## 1. Starting facts to verify

1. **Subnormal snow thickness in NEMO**: Starting at `kt = 5841` (`POST_DH.h_s`) and entering vertical heat diffusion (`ice_thd_zdf_bl99`) at `kt = 5842..5883`, NEMO 5.0.2 carries `h_s_1d = 5e-324 m` (`0x0000000000000001`, `2^-1074`, the smallest positive IEEE-754 binary64 subnormal) because `zh_s(jk) = h_s_1d(ji) * r1_nlay_s` (`icethd_dh.F90:161`) underflows `5e-324 * (1/3)` to `0.0`, preventing `icethd_dh.F90:212` from decrementing `h_s_1d`.
2. **NEMO snow-present predicate**: In `icethd_zdf_bl99.F90:178,346,443,581,586,612`, NEMO evaluates `h_s_1d(ji) > 0._wp` as `.TRUE.`, sets `isnow(ji) = 1._wp`, clamps the conductive snow layer thickness to `zh_s(ji) = MAX(zh_min, h_s_1d(ji)) * r1_nlay_s = 1.0e-3 * (1.0 / 3.0) m` (`icethd_zdf_bl99.F90:179`), and solves the snow-covered BL99 tridiagonal system.
3. **Phase 6 CPU discrepancy**: In lane 3's Phase 6 CPU year gate (`docs/ocean/fidelity/testcases/nemo_testcases_l3thd_phase6_receipt.md:194-209`), 25 of 1,302,135 stage rows exceed `1e-15`:
   - 18 normal-valued `POST_ZDF.e_s` rows in `kt = 5842..5883`, with worst normal row at `kt = 5860 POST_ZDF.e_s`: `norm_l2 = 0.02510099530281747` (`max_abs = 2,832,427.961354971 J m-3`, `max_ref = 112,841,260.96135496 J m-3`, where `112,841,260.96135496 - 2,832,427.961354971 = 110,008,833.0 = rhos * rLfus`, the `T_s = 273.15 K` enthalpy returned by the `no_snow_*` branch).
   - 7 subnormal-valued `h_s` rows (`kt = 5841 POST_DH.h_s` and `kt = 5842 PRE_ZDF.h_s`, `POST_ZDF.h_s`, `POST_DH.h_s`, `POST_TEMP.h_s`, `POST_SAL.h_s`, `POST_COR.h_s`) where `max_ref = 5e-324 m` and `max_abs = 5e-324 m` (`norm_l2 = 1.0`).

---

## 2. Preregistered platform probe

Before changing any physics or scoring the year, we run a committed probe (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py`) under `JAX_ENABLE_X64=1` and `PrecisionPolicy.fp64()` on:
1. **CPU** (`JAX_PLATFORMS=cpu`, the production XLA CPU JIT path).
2. **One idle GPU** (`JAX_PLATFORMS=cuda`, pinned to a single idle GPU via `CUDA_VISIBLE_DEVICES` after checking `nvidia-smi`).

On each target platform, the probe executes three compiled (`jax.jit`) stages and stamps backend, device kind, JAX/jaxlib versions, and FTZ/DAZ disposition:

1. **Stage A — Compiled smallest-subnormal identity and arithmetic bit-pattern probe**:
   - Input `x_sub = np.uint64(1).view(np.float64)` (`5e-324`, bit pattern `0x0000000000000001`), plus normal inputs `a = 2.0**-1022` (`0x0010000000000000`) and `b = 2.0**-52` (`0x3cb0000000000000`) whose exact product is `5e-324` (`0x0000000000000001`).
   - Compiled functions:
     - `jit_identity(x) = x` (tests buffer copy / parameter-to-return boundary).
     - `jit_mul_one(x) = x * jnp.float64(1.0)` (tests DAZ on subnormal input / FTZ on subnormal output).
     - `jit_add_zero(x) = x + jnp.float64(0.0)` (tests DAZ/FTZ under addition).
     - `jit_produce_subnormal(a, b) = a * b` (tests FTZ when a subnormal result is produced inside a compiled kernel from normal operands).
   - Records raw uint64 hex bit patterns (`0x...`) of inputs and outputs.

2. **Stage B — Compiled `h_s > 0` predicate on the recorded C1D operand**:
   - Reads the exact recorded `h_s` operand (`0x0000000000000001` at `kt = 5842`, `kt = 5860`, and all 42 subnormal-snow steps in `kt = 5842..5883`) from `oracle_si3_thd_frames.bin` (`PRE_ZDF.h_s`) and `oracle_si3_zdf_inputs.bin`.
   - Evaluates compiled `jit_pred(h_s) = (h_s > jnp.float64(0.0))` and `jit_isnow(h_s) = jnp.where(h_s > jnp.float64(0.0), jnp.float64(1.0), jnp.float64(0.0))` on the recorded operand, recording boolean outputs and `isnow` uint64 bit patterns.

3. **Stage C — Exact ZDF (`_si3_zdf_bl99_step`) outputs with the recorded operand**:
   - Evaluates compiled `_si3_zdf_bl99_step` on the exact recorded `PRE_ZDF` state and `oracle_si3_zdf_inputs.bin` forcing at `kt = 5842`, `kt = 5860`, and across all 42 subnormal-snow steps (`kt = 5842..5883`), recording output uint64 bit patterns, max ULP distance, max absolute error, and normalized L2 error against NEMO's recorded `POST_ZDF` frame (`t_su`, `e_s`, `e_i`, `h_s`, `h_i`, `cnd_ice`, `qcn_ice`, `qtr_ice_bot`).

---

## 3. Predictions and falsifiers

### Prediction P1 — CPU (`JAX_PLATFORMS=cpu`, `x86_64` XLA CPU JIT)

- **Prediction**:
  - XLA CPU compiled kernels execute with hardware MXCSR `FTZ=1`/`DAZ=1` (Decision 127).
  - While `jit_identity(x_sub)` may copy the 64-bit buffer unchanged (`0x0000000000000001`) if XLA emits a memory move without an FP instruction, every compiled floating-point arithmetic or comparison instruction (`jit_mul_one`, `jit_add_zero`, `jit_produce_subnormal`, `jit_pred`, `jit_isnow`) flushes `0x0000000000000001` to `0x0000000000000000` (`0.0`) and evaluates `h_s > 0.0` to `False` (`isnow = 0.0`, `0x0000000000000000`).
  - In Stage C on CPU, `_si3_zdf_bl99_step` at `kt = 5860` takes the `no_snow_*` branch (`bitz_lipscomb.py:444,580-591`), yielding `POST_ZDF.e_s = [110008833.0, 110008833.0, 110008833.0] J m-3` and `norm_l2 = 0.02510099530281747` (`max_abs = 2,832,427.961354971 J m-3`).
- **CONFIRM vs REFUTE**:
  - **CONFIRM P1**: `jit_mul_one(x_sub)` and `jit_produce_subnormal(a, b)` return `0x0000000000000000`, `jit_pred(h_s)` returns `False` on `0x0000000000000001`, and `kt = 5860 POST_ZDF.e_s` has `norm_l2 = 0.02510099530281747`. Because per-statement subnormal workarounds are forbidden (`docs/ice/fidelity/si3_ladder_design.md:349-350`), literal closure on CPU is impossible; we declare **`D5 PLATFORM_FLOOR`** on CPU and keep the 25 downstream/subnormal rows reported as unresolved on CPU.
  - **REFUTE P1**: `jit_pred(h_s)` returns `True` and `jit_mul_one(x_sub)` returns `0x0000000000000001` on CPU. In that case CPU preserves subnormals and must close the snow branch literally.

### Prediction P2 — GPU (`JAX_PLATFORMS=cuda`, NVIDIA CUDA `f64` JIT)

- **Prediction**:
  - NVIDIA CUDA binary64 (`f64`) instructions implement IEEE-754 gradual underflow without FTZ/DAZ:
    - Stage A (`jit_identity`, `jit_mul_one`, `jit_add_zero`, `jit_produce_subnormal`) returns `0x0000000000000001` (`5e-324`).
    - Stage B (`jit_pred`, `jit_isnow`) on the recorded `h_s = 0x0000000000000001` returns `True` and `isnow = 1.0` (`0x3ff0000000000000`).
  - **First departing statement in the snow-present branch (`P2b`)**:
    - Once `h_snow > 0.0` is `True` on GPU, `_si3_zdf_bl99_step` enters the snow-present branch (`packages/ice/legoesm/ice/bitz_lipscomb.py:442-591`), dropping `kt = 5860 POST_ZDF.e_s` normalized L2 error from `0.02510099530281747` to `< 1e-15` even before statement-level edits.
    - However, comparing `icethd_zdf_bl99.F90` against `packages/ice/legoesm/ice/bitz_lipscomb.py` predicts a 1-ULP / few-ULP statement departure inside the clamped snow-thickness branch when `h_s_1d = 5e-324 < zh_min = 1.0e-3`:
      1. In `icethd_zdf_bl99.F90:172-173,179-180`, NEMO computes:
         - `zh_i(ji) = MAX( zh_min , h_i_1d(ji) ) * r1_nlay_i` ; `z1_h_i(ji) = 1._wp / zh_i(ji)`
         - `zh_s(ji) = MAX( zh_min , h_s_1d(ji) ) * r1_nlay_s` ; `z1_h_s(ji) = 1._wp / zh_s(ji)`
         where `r1_nlay_i = r1_nlay_s = 1.0_wp / 3.0_wp`, and then multiplies by `z1_h_s(ji)` and `z1_h_i(ji)` in `zkappa_s`, `zkappa_i`, `zeta_i`, and `zeta_s` (`icethd_zdf_bl99.F90:332,334-335,341,366,374`).
      2. In `packages/ice/legoesm/ice/bitz_lipscomb.py:442-443,456-468`, legoESM instead writes `dz_i = jnp.maximum(h_ice, _H_MIN) / 3.0` and `dz_s = jnp.maximum(h_snow, _H_MIN) / 3.0` (`1.0e-3 / 3.0 = 0x1.5d867c3ec37fcp-12` vs `1.0e-3 * (1.0 / 3.0) = 0x1.5d867c3ec37fbp-12`, a 1-ULP difference in `zh_s`) and divides by `dz_s` / `dz_i` instead of multiplying by `z1_h_s` / `z1_h_i`.
    - Fixing the departing snow-branch statement(s) as a selectable SI3 statement in `packages/ice/legoesm/ice/bitz_lipscomb.py` (enabled by the SI3 card/config, leaving default callers bit-identical) will make `kt = 5842..5883 POST_ZDF` snow-branch outputs match NEMO bit-for-bit (or down to any isolated compiler/intrinsic floor if present) and reduce the full-year C1D over-bar count (`> 1e-15`) on GPU from `25` to `0` across all `1,302,135` stage rows.
- **CONFIRM vs REFUTE**:
  - **CONFIRM P2**: Stage A/B on GPU preserves `0x0000000000000001` and `h_s > 0.0 == True`; identifying and fixing the first departing snow-branch statement in `packages/ice/legoesm/ice/bitz_lipscomb.py` as a selectable SI3 option yields `0` rows over the `1e-15` bar across all `1,302,135` C1D stage rows on GPU (`kt = 5860 POST_ZDF.e_s` error reduced from `0.02510099530281747` to `0.0`), while default non-SI3 callers remain bit-for-bit unchanged.
  - **REFUTE P2**: Either (a) GPU flushes `0x0000000000000001` (`h_s > 0.0 == False`), in which case GPU is also `D5 PLATFORM_FLOOR`, or (b) after fixing the `zh_s` / `z1_h_s` statement in `bitz_lipscomb.py`, `kt = 5842..5883 POST_ZDF.e_s` still differs from NEMO, which will falsify `zh_s` / `z1_h_s` as the sole statement departure and require isolating the next statement in `icethd_zdf_bl99.F90`.
