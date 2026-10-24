# SI3 Ladder Round 5 (`si3_r5`) — Rung D5 Debts Preregistration

**Round**: `si3_r5` (SI3 ladder rung D5 debt closure, executing Decision 156, Decision 158, and Decision 159)  
**Branch**: `fidelity/nemo-testcases-si3-ladder` (starting from `a572f4b25403`, Round 3 landed)  
**Target record**: `/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_scalarmath_v2_a` (8,760 hourly steps of 3,600 s, `621,960` stage-field rows across 8 stages; existing immutable record reused without running NEMO).

---

## 1. Scope and Ordered Measurement Protocol

Per **DECISION 156**, **DECISION 158**, and **DECISION 159**, Round 5 closes the four D5 debt items recorded in `docs/ice/fidelity/si3_r3_d5_receipt.md:247-266`, measuring each one statement at a time across the full 8,760-step C1D year (`621,960` stage-field rows) under compiled `jax.jit` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:1819`) on CPU production JIT (`XLA_FLAGS=""`), compared against the `--xla_cpu_max_isa=AVX` diagnostic (requiring full-year output SHA-256 bitwise equality), and spot-checked by counts on an idle GPU (recording `nvidia-smi` before and after):

1. **Item 1 — Ice-layer vertical enthalpy remap (`src/ICE/icevar.F90:957-1034`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:971-1048`)**:
   - Transcribe NEMO's `ice_var_vremap` cumulative-content interpolation literally in `packages/ice/legoesm/ice/bitz_lipscomb.py` (`_nemo_ice_enthalpy_remap`), called from `_dh_step` (`src/ICE/icethd_dh.F90:511`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:521`) under the selectable SI3 flag `_nemo_ice_remap: bool = True` (forwarded by `si3_column_step_arrays`, default `True` on the SI3 path, `False` retaining `_piecewise_remap` for revert control; non-SI3 callers bit-for-bit unchanged).
   - Remove the non-NEMO `_nemo_source_round` insertion from `_piecewise_remap` (`packages/ice/legoesm/ice/bitz_lipscomb.py:880-903`), restoring `_piecewise_remap` to its unrounded overlap-sum form.

2. **Item 2 — Ice density reciprocal in `_dh_step` (`src/ICE/icethd_dh.F90:254,391`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:264,401`)**:
   - Multiply by NEMO's reciprocal `r1_rhoi = 1.0 / constants.rho_ice` (`sr(-zdf.e_ice[..., k] * inverse_ice_density)`) at the surface-melt (`src/ICE/icethd_dh.F90:254`) and basal-melt (`src/ICE/icethd_dh.F90:391`) specific-enthalpy evaluations in `_dh_step` instead of dividing by `constants.rho_ice`, gated by `_nemo_dh_ice_density_reciprocal: bool = True` (forwarded by `si3_column_step_arrays`, default `True` on the SI3 path, `False` evaluating IEEE-754 division `/ constants.rho_ice` for revert control).
   - Correct the misquoted line in `docs/ice/fidelity/si3_r3_d5_receipt.md:110` so it accurately states what Round 3 wrapped (`-zdf.e_ice[..., k] / constants.rho_ice`) and that Round 5 transcribes NEMO's reciprocal multiplication (`-e_i_1d(ji,jk) * r1_rhoi`).

3. **Item 3 — Salinity profile bulk-salinity factoring (`src/ICE/icevar.F90:504-505,575-576`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:518-519,589-590`)**:
   - In `option2_salinity_profile` (`packages/ice/legoesm/ice/bitz_lipscomb.py`), transcribe NEMO's left-to-right association:
     `zalpha * s_i * 2._wp * (REAL(jk,wp) - 0.5_wp) * r1_nlay_i + (1._wp - zalpha) * s_i`
     instead of `S_bulk * (alpha * z + (1.0 - alpha))`, gated by `_nemo_salprof_factoring: bool = True` (forwarded by `si3_column_step_arrays`, default `True` on the SI3 path, `False` retaining the factored `S_bulk * (...)` form for revert control).

4. **Item 4 — Harness transition-constant naming and missing-SSS refusal (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py` & `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py`)**:
   - Rename the two salinity-profile transition parameters in `_nemo_salprof` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py`) from `rn_sal_gd: float = 4.5, rn_sal_fl: float = 3.5` (namelist parameter names whose record values are `5.0` and `2.0`) to NEMO's fixed local parameters `zsi1: float = 4.5, zsi0: float = 3.5` (`src/ICE/icevar.F90:471-472,546-547`; compiled `C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:485-486,560-561`), and correct the comment line references from `:495`/`:496` to `:499`/`:500`.
   - Remove the silent `sss_m = 34.0` default in `_exact_1d_entry` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py`) and `_nemo_salprof` / `_entry_arrays` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py`), refusing with `GateError` (`phase2.require(...)`) when sea-surface salinity (`sss_m` / `"sss"`) is absent or non-finite.

---

## 2. Frozen Predictions and Falsifiers

1. **P1 (Item 1 — `ice_var_vremap` transcription `src/ICE/icevar.F90:957-1034`)**:
   - **CONFIRM**: Replacing `_piecewise_remap` with `_nemo_ice_enthalpy_remap` (`_nemo_ice_remap=True`) and removing `_nemo_source_round` from `_piecewise_remap` reduces full-year C1D over-bar rows under `jax.jit` on CPU (`XLA_FLAGS=""` and `--xla_cpu_max_isa=AVX`) from **`965` to `663`** (`-302` over-bar rows; normal-step over-bar rows outside `kt = 5842..5867` drop from **`336` to `34`**, closing all `29 × 4 = 116` over-bar rows in `POST_DH..POST_TEMP2.e_i` to **`0`** and reducing `POST_DO.e_i` / `EXIT.e_i` over-bar rows from `99` each to `6` each) and on GPU from **`336` to `34`** (`-302` over-bar rows), while increasing full-year `0`-ULP rows on CPU from **`521,368` to `561,810`** (`+40,442` rows) and on GPU from **`521,679` to `562,223`** (`+40,544` rows).
   - **REFUTE**: `POST_DH..POST_TEMP2.e_i` retain any normal-step over-bar row (`> 0`), or reverting `_nemo_ice_remap=False` fails to restore the `965` (`336` normal) over-bar count.

2. **P2 (Item 2 — Ice density reciprocal `src/ICE/icethd_dh.F90:254,391`)**:
   - **CONFIRM**: Multiplying by `inverse_ice_density = 1.0 / constants.rho_ice` (`_nemo_dh_ice_density_reciprocal=True`) matches NEMO's `zEi = - e_i_1d(ji,jk) * r1_rhoi` (`src/ICE/icethd_dh.F90:254,391`) bit-for-bit on inexact operands (e.g., `kt = 4893` layer 1 `zEi = 0xc10ee367890ddc03` and `zdE = 0xc10edecdb63bb14a` vs IEEE-754 division `/ 917.0` `zEi = 0xc10ee367890ddc02` and `zdE = 0xc10edecdb63bb149`, differing on `4,969` of `8,760` steps for `zEi`, `4,958` steps for `zdE`, and `6` stage-field rows `POST_DH..EXIT.e_i` at `kt = 4893` when compared against true IEEE-754 division `/ rho_ice`), while preserving the compiled `jax.jit` full-year counts (`663` over-bar, `561,810` `0`-ULP on CPU) where XLA's `AlgebraicSimplifier` already rewrote constant `/ 917.0` to `* (1.0 / 917.0)`.
   - **REFUTE**: Reverting to IEEE-754 division (`/ rho_ice`, `_nemo_dh_ice_density_reciprocal=False`) fails to change the bits on inexact C1D operands (`kt = 4893`), or `docs/ice/fidelity/si3_r3_d5_receipt.md` retains the misquoted description.

3. **P3 (Item 3 — Salinity profile bulk-salinity factoring `src/ICE/icevar.F90:504-505,575-576`)**:
   - **CONFIRM**: Factoring `option2_salinity_profile` as `zalpha * s_i * 2.0 * (jk - 0.5) * r1_nlay_i + (1.0 - zalpha) * s_i` (`_nemo_salprof_factoring=True`) increases full-year C1D `0`-ULP rows under `jax.jit` by **`+5,776` rows** (`561,810 -> 567,586` on CPU; `562,223 -> 567,999` on GPU: `+1,921` rows each in `POST_SAL.sz_i` and `POST_TEMP2.sz_i` from `6,832 -> 8,753`, and `+967` rows each in `POST_DO.szv_i` and `EXIT.szv_i` from `5,702 -> 6,669`), while keeping over-bar rows at `663` (`34` normal + `629` floor on CPU; `34` on GPU).
   - **REFUTE**: `0`-ULP rows do not increase by `+5,776`, or reverting `_nemo_salprof_factoring=False` does not fail bit-exactness on inexact operands (e.g., `kt = 2` or `kt = 3961`).

4. **P4 (Item 4 — Harness `zsi1`/`zsi0` rename, missing-SSS refusal, CPU/AVX SHA-256 equality, AD & non-SI3 invariance)**:
   - **CONFIRM**: Renaming `zsi1`/`zsi0` in `_nemo_salprof` and making `_exact_1d_entry` / `_entry_arrays` / `_nemo_salprof` refuse (`GateError`) when `sss_m` is `None` or `"sss"` is absent leaves all valid full-year counts bit-identical (`663` over-bar, `567,586` `0`-ULP on CPU; `34` over-bar, `567,999` `0`-ULP on GPU) while failing closed when `sss_m` is omitted; CPU production JIT (`XLA_FLAGS=""`) and `--xla_cpu_max_isa=AVX` have identical full-year and per-stage output SHA-256 digests across every Round 5 arm; reverse-mode (`jax.grad`) and forward-mode (`jax.jvp`) gradients through every touched function are finite, nonzero, and agree while preserving forward primal bits (`0` ULP); and non-SI3 callers match parent commit bit-for-bit.
   - **REFUTE**: Omitting `sss_m` does not raise `GateError`, CPU production JIT and `--xla_cpu_max_isa=AVX` differ in full-year SHA-256, any non-SI3 caller changes by `>= 1` ULP, or `jax.grad` / `jax.jvp` fails.
