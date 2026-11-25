# SI3 Ladder Round 3 (`si3_r3`) — Rung D5 Continuation Preregistration

**Round**: `si3_r3` (SI3 ladder rung D5 continuation, executing Decision 156 and Decision 158)  
**Branch**: `fidelity/nemo-testcases-si3-ladder` (starting from `a8c2f7f21cf1`, Round 2 landed)  
**Target record**: `/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_scalarmath_v2_a` (8,760 hourly steps of 3,600 s, `621,960` stage-field rows across 8 stages; existing immutable record reused without running NEMO).

---

## 1. Scope and Ordered Measurement Protocol

Per **DECISION 158**, the SI3 sea-ice ladder certifies on the CPU (`fp64`, production JIT, `XLA_FLAGS=""`) with XLA's fused multiply-add (`FMA`) prevented **in code** at the statements where NEMO's `-O3` build does not fuse (using `nemo_source_round` from `packages/core/legoesm/core/source_rounding.py:14-27`, never a global `--xla_cpu_max_isa` flag); binary64 subnormal cases (`kt = 5842..5867`, 26 steps where `0 < h_s < 2.2250738585072014e-308`) remain a stated platform floor (`D5_PLATFORM_FLOOR`) on CPU; and each rung is spot-checked on an idle GPU (recording `nvidia-smi` before and after).

This round executes four ordered steps, each measured alone on the full 8,760-step C1D year (`621,960` stage-field rows) on CPU production JIT (`XLA_FLAGS=""`), CPU diagnostic no-FMA (`XLA_FLAGS="--xla_cpu_max_isa=AVX"`), and an idle GPU:

1. **Step 1 — Committed reproduction of the Round 2 reviewer's numbers**:
   - Extend the committed D5 probe (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py`) to reproduce from committed code the reviewer-measured numbers noted in `docs/ice/fidelity/si3_r2_d5_receipt.md:180-190`:
     - Across the `7,385` snow-present steps (`h_s > 0` with recorded `t_s` in `oracle_si3_zdf_inputs.bin`), the shipped Round 2 code (`(e_s * 3.0) / v_s` and `icethd_dh.F90:498-503` snow-temperature inversion) reproduces NEMO's recorded ZDF-entry `t_s` to `0` ULP on **`1,811` of `7,385`** steps (`5,574` mismatching).
     - With NEMO-exact ZDF entry inputs (`(e / v) * 3.0`, recorded/profile `sz_i`, and `src/ICE/icevar.F90:412-413` entry snow temperature), all `7,385` of `7,385` snow steps reproduce `t_s` to `0` ULP, and across the `8,734` normal-snow steps (`26,202` active `POST_ZDF` rows `t_su`, `e_i`, `e_s`), `POST_ZDF` over-bar rows drop from **`1,625` (`_nemo_zdf_layer_reciprocals=False`) to `0` (`_nemo_zdf_layer_reciprocals=True`)** and all **`26,202` rows are at `0` ULP** on GPU and on CPU without FMA (`--xla_cpu_max_isa=AVX`). On GPU (where the 26 subnormal steps `kt = 5842..5867` also preserve subnormals), all `26,280` active `POST_ZDF` rows (`61,320` total `POST_ZDF` rows) across all `8,760` steps are at `0` ULP (`0` over-bar).

2. **Step 2 — Harness entry conversion (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:201-219`)**:
   - In NEMO's `ice_var_glo2eqv` (`src/ICE/icevar.F90:368,374,390,412-413,488-506`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:368,374,404,426-427,502-520`) and `ice_thd_1d2d` (`scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd.F90:453,521,528`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd.f90:452,520,527`), layer enthalpies are converted from global fields via `(e_i / v_i) * zlay_i` and `(e_s / v_s) * zlay_s` (`zlay_i = zlay_s = 3.0`), and for `nn_icesal == 2` layer salinity `sz_i` comes from the option-2 salinity profile (`ice_var_salprof` on `s_i = sv_i / v_i`, not `szv_i * 3.0 / v_i`).
   - Update `_entry_arrays` in `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:201-219` (mimicry glue in the harness, not model code) to compute `(g["e_i"] / vi) * 3.0`, `(g["e_s"] / vs) * 3.0`, and layer salinity from the option-2 salinity profile (`icevar.F90:488-506`), and measure the full-year C1D re-score (`621,960` rows) after Step 2 alone (with Step 3 and Step 4 still off).

3. **Step 3 — Model statement: solver-entry snow temperature (`packages/ice/legoesm/ice/bitz_lipscomb.py:321-340,436-438`)**:
   - At solver entry (`ice_var_glo2eqv`, `src/ICE/icevar.F90:412-413`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:426-427`), NEMO computes:
     `t_s(ji,jj,jk,jl) = rt0 + MAX( -100._wp , MIN( r1_rcpi*( -r1_rhos*( e_s(ji,jj,jk,jl) / v_s(ji,jj,jl) * zlay_s ) + rLfus ) , 0._wp ) )`
     whereas `snow_temperature_from_enthalpy` in `packages/ice/legoesm/ice/bitz_lipscomb.py:331-340` currently evaluates `constants.T0 + (-e * inverse_rho_snow * inverse_c_ice + constants.latent_fusion * inverse_c_ice)` from `src/ICE/icethd_dh.F90:511` (`C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:511`).
   - Transcribe `icevar.F90:412-413` in `snow_temperature_from_enthalpy` under the selectable SI3 flag `_nemo_glo2eqv_snow_temp: bool = True` (forwarded by `_si3_zdf_bl99_step` and `si3_column_step_arrays`, default `True` on the SI3 path, `False` retaining the `icethd_dh.F90:511` association for revert control; non-SI3 callers bit-for-bit unchanged).
   - Measure the full-year C1D re-score (`621,960` rows) after Step 2 + Step 3 (with Step 4 still off).

4. **Step 4 — CPU FMA contraction prevention in code (`DECISION 158`)**:
   - Identify every statement in the BL99 / SI3 column path (`packages/core/legoesm/timestepping/tridiagonal.py:135-170` `nemo_unnormalised` arm and `packages/ice/legoesm/ice/bitz_lipscomb.py`) where XLA's default CPU codegen (`XLA_FLAGS=""`) fuses multiply-add/subtract relative to NEMO's `-O3` build and the `--xla_cpu_max_isa=AVX` diagnostic.
   - Prevent FMA contraction in code at those statements using `nemo_source_round` (`packages/core/legoesm/core/source_rounding.py:14-27`), gated by a selectable SI3 flag `_nemo_source_round: bool = True` (default `True` on the SI3 path, `False` for revert control; non-SI3 callers bit-for-bit unchanged), citing each NEMO statement in `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90`, `icevar.f90`, `icethd.f90`, `icethd_dh.f90`, and `icethd_do.f90`.
   - Measure the full-year C1D re-score (`621,960` rows) after Step 2 + Step 3 + Step 4 on CPU production JIT (`XLA_FLAGS=""`), CPU diagnostic (`XLA_FLAGS="--xla_cpu_max_isa=AVX"`), and GPU.

---

## 2. Frozen Predictions and Falsifiers

1. **P1 (Step 1 — Reviewer reproduction)**:
   - **CONFIRM**: Shipped Round 2 reproduces recorded ZDF-entry `t_s` at `0` ULP on `1,811 / 7,385` snow steps; with NEMO-exact ZDF inputs (`(e/v)*3`, profile `sz_i`, `icevar.F90:412-413` `t_s`), all `7,385 / 7,385` snow steps reproduce `t_s` at `0` ULP and all `26,202` active `POST_ZDF` rows (`8,734` normal steps × 3 fields) are at `0` ULP (`1,625 -> 0` over-bar) on GPU and CPU (`--xla_cpu_max_isa=AVX`).
   - **REFUTE**: Any of the counts (`1,811`, `7,385`, `26,202`, `1,625 -> 0`) differs when measured by the committed probe.

2. **P2 (Step 2 — Harness entry conversion)**:
   - **CONFIRM**: Updating `_entry_arrays` to `(e_i / v_i) * 3.0`, `(e_s / v_s) * 3.0` (`icevar.F90:390,412-413`), and option-2 salinity profile `sz_i` (`icevar.F90:488-506`) increases ZDF-entry `t_s` `0`-ULP steps from `1,811 / 7,385` to `3,547 / 7,385` (`3,838` mismatching) and reduces `POST_ZDF` over-bar rows on GPU from `471` to `60` (and on CPU `--xla_cpu_max_isa=AVX` outside `kt = 5842..5867` from `470` to `60`, total `545 -> 135`; on CPU production JIT `646 -> 209`).
   - **REFUTE**: `ENTRY` stage opens any over-bar debt row (`ENTRY` over-bar > `0`), or `POST_ZDF` over-bar rows do not decrease.

3. **P3 (Step 3 — Solver-entry snow temperature `icevar.F90:412-413`)**:
   - **CONFIRM**: Adding `_nemo_glo2eqv_snow_temp=True` (with Step 2 active) increases ZDF-entry `t_s` `0`-ULP steps from `3,547 / 7,385` to **`7,385 / 7,385`** (`0` mismatching), and reduces `POST_ZDF` over-bar rows to **`0` of `61,320`** (all **`61,320` rows at `0` ULP**) on GPU and **`0` of `61,138`** outside `kt = 5842..5867` (all `61,138` normal-step `POST_ZDF` rows at `0` ULP; `75` over-bar in `kt = 5842..5867` `D5_PLATFORM_FLOOR`) on CPU (`--xla_cpu_max_isa=AVX`).
   - **REFUTE**: Any normal-step `POST_ZDF` row remains `> 0` ULP on GPU or CPU (`--xla_cpu_max_isa=AVX`), or any non-SI3 caller changes by `>= 1` ULP.

4. **P4 (Step 4 — In-code CPU FMA prevention via `nemo_source_round`, Decision 158)**:
   - **CONFIRM**: Wrapping the FMA-contracted products in `thomas_solve(..., "nemo_unnormalised")` (`icethd_zdf_bl99.f90:548,555,560,566-567`) and the BL99 / SI3 column statements in `bitz_lipscomb.py` with `nemo_source_round` makes the CPU production-JIT (`XLA_FLAGS=""`) full-year C1D re-score (`621,960` rows) match the CPU no-fusion diagnostic (`XLA_FLAGS="--xla_cpu_max_isa=AVX"`) in both over-bar row count and `0`-ULP row count across every stage (`POST_ZDF` over-bar on CPU production JIT drops from `78` to `75` = the exact `75` `D5_PLATFORM_FLOOR` rows in `kt = 5842..5867`, with all `61,138` normal-step `POST_ZDF` rows at `0` ULP), while GPU and `--xla_cpu_max_isa=AVX` counts remain invariant and all reverse-mode/forward-mode gradients through every touched function remain finite and agree.
   - **REFUTE**: CPU production JIT (`XLA_FLAGS=""`) still disagrees with `--xla_cpu_max_isa=AVX` on over-bar or `0`-ULP counts after `nemo_source_round` is applied, or breaks `jax.grad` / `jax.jvp`.
