# SI3 Ladder Round 21 (`si3_r21`) — O8, O9, O10, O11, O12 Master-Record Scoring Preregistration

- Date: 2026-10-27. Lane `si3`, round 21 (`si3_r21`), authored by Argon.
- Branch: `fidelity/nemo-testcases-si3-ladder` at parent `3d5446cca2b4` (O12 model prep landed).
- Scope: score rungs **O8, O9, O10, O11, O12** in order against the admitted 100-step shared `ORCA2_SAS_ICE` master record (`docs/ice/fidelity/si3_r18_o8_o16_receipt.md:154-180`, `DECISION 173`).
- Status: committed **BEFORE** reading any binary record from `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/oracle_o8_o16_master/record_100step/o8_o16_master_records/`. No model code or configuration changes are permitted this round.

## 1. Pinned master-record provenance and admission digests

Master record root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/oracle_o8_o16_master` (`docs/ice/fidelity/si3_r18_o8_o16_receipt.md:154-180`):

- Producer commit: `cd34f1be648a05abbb23e4cb6bb2a9ed66e60fc5`. Amendment (2026-10-29 provenance audit): this was a pre-rebase commit and is no longer reachable.
- `manifest/run_manifest.json` SHA-256: `bfb0b71d1500b7bc91c68a9e39c8af7a654b139258c0635ad5b1f25284b31fd0`
- `manifest/resource_usage.json` SHA-256: `b30a5febe9248e5008999cb419953dad40c78658542a3c7221d301825270967c`
- `manifest/records.sha256` SHA-256: `fbd3cb4d3450f428d8444dcb145bec3c1a832554e3e7e05d7f5b6a159daadd85`
- Per-rung projection manifest SHA-256s (`manifest/per_rung/<RUNG>.json`):
  - `O8`: `303a29d76034c3856e4cf6fa60ed551b7146e5decd3618038b058501c818ed82` (`kinds = (1, 2)`, `200` records)
  - `O9`: `3d97b7bb71c91530a89f60498c8a84316829200e89d8c45470f5a8d1814ee0a3` (`kinds = (21, 3, 4, 5, 6, 17)`, `2,115` records)
  - `O10`: `ded24ddbc825a9a331d763665e51559022c97288cb7c7f516b3a587693deb554` (`kinds = (21, 6, 7, 8)`, `1,600` records)
  - `O11`: `0c5197c52331975ff77d813f0154e0df6ea0aa1e0593e9ca062990d664465666` (`kinds = (21, 8, 9)`, `1,100` records)
  - `O12`: `fe395cab681d59a8c28a722cd7fc8222ec3ac0ee8c4e569d4043643fd4a47e50` (`kinds = (10, 11)`, `200` records)

Every scalar and selector is parsed from the run's resolved `record_100step/output.namelist.ice` and `record_100step/output.namelist.dyn` and cross-checked against the 59-element binary record scalar header (`SCALAR_NAMES` in `scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o16_master_gate.py:40-55`).

## 2. Per-rung stage definitions, production functions, and scored exit fields (GIVEN-NEMO-ENTRY only)

All evaluations run on CPU (`jax.default_backend() == "cpu"`), `fp64` (`JAX_ENABLE_X64=1`), `PrecisionPolicy.fp64(transcendentals="libm")` (`DECISION 158`, `DECISION 162`), under `jax.jit` with array and scalar inputs passed as dynamic JIT arguments. Comparisons require `0` ULP (`uint64` bitwise identity including signed zeros; `DECISION 159`).

| Rung | Entry record kind(s) | Production implementation (`jax.jit`) | Scored exit record kind(s) and fields |
|---|---|---|---|
| **O8** | Kind `1` (`frazil_entry`, `kt = 1..100, jl = 0`) | `si3_ice_thd_frazil` (`packages/ice/legoesm/ice/bitz_lipscomb.py:2024-2042`) | Kind `2` (`frazil_exit`, `kt = 1..100, jl = 0`): all 2 fields (`ht_i_new`, `fraz_frac`, shape `(180, 148)`) |
| **O9** | Kind `3` (`category_entry`, `kt = 1..100, jl = 1..5`) + Kind `21` (`pre_category_state`, `kt = 1..100, jl = 0`) | `si3_ice_thd_zdf_bl99` (`packages/ice/legoesm/ice/bitz_lipscomb.py:834-919`), `si3_ice_thd_dh` (`:1288-1311`), `si3_ice_thd_temp` (`:1272-1277`), and composed `si3_ice_thd_category_stages` (`:1344-1382`) | Kind `4` (`zdf_exit`, `kt = 1..100, jl = 1..5`, all 16 fields); Kind `5` (`dh_exit`, `kt = 1..100, jl = 1..5`, all 67 fields); Kind `6` (`temp1_exit_sal_entry`, `kt = 1..100, jl = 1..5`, all 67 fields); Kind `17` (`picard_final`, `kt in (1, 2, 100), jl = 1..5`, all 12 fields + `max(n_picard) == iconv` per `DECISION 176`/`DECISION 179`) |
| **O10** | Kind `6` (`temp1_exit_sal_entry`, `kt = 1..100, jl = 1..5`) + Kind `21` (`pre_category_state`) | `si3_ice_thd_sal` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1705-1782`), `si3_ice_thd_temp` (`:1272-1277`), and composed `si3_ice_thd_o10_category_stages` (`:1785-1808`) | Kind `7` (`sal_exit`, `kt = 1..100, jl = 1..5`, all 67 fields); Kind `8` (`temp2_exit_da_entry`, `kt = 1..100, jl = 1..5`, all 67 fields) |
| **O11** | Kind `8` (`temp2_exit_da_entry`, `kt = 1..100, jl = 1..5`) + Kind `21` (`pre_category_state`) | `si3_ice_thd_da` (`packages/ice/legoesm/ice/si3_thd_da.py:65-185`) | Kind `9` (`da_exit`, `kt = 1..100, jl = 1..5`, all 67 fields) |
| **O12** | Kind `10` (`pond_entry_category_done`, `kt = 1..100, jl = 0`) | `si3_ice_thd_pnd` (`packages/ice/legoesm/ice/si3_thd_pnd.py:575-781`) and `step_ponds(..., scheme="si3_lev")` (`packages/ice/legoesm/ice/ponds.py:34-311`) | Kind `11` (`pond_exit_itd_entry`, `kt = 1..100, jl = 0`, all 96 fields) |

## 3. Frozen predictions and falsifiers

- **P1 (Admission & digest integrity)**: `admit` in `scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o16_master_gate.py:701-741` passes on `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/oracle_o8_o16_master`, and all projection manifest SHA-256 digests for `O8`, `O9`, `O10`, `O11`, and `O12` match Section 1.
  - **CONFIRM**: `status == "ADMITTED"` and all 5 projection digests match.
  - **REFUTE**: Any digest or admission failure.
- **P2 (Per-rung 0-ULP exit-field scores across all 100 steps and all categories)**:
  - **O8**: `unequal == 0` and `max_abs_diff == 0.0` across all `100` steps for both `ht_i_new` and `fraz_frac` (`26,640 * 100 = 2,664,000` cell comparisons per field).
  - **O9**: `unequal == 0` and `max_abs_diff == 0.0` across all `100 * 5 = 500` `(kt, jl)` category frames for every exit field in Kind `4` (16 fields), Kind `5` (67 fields), and Kind `6` (67 fields), plus all `3 * 5 = 15` `(kt, jl)` frames of Kind `17` (12 fields + `max(n_picard) == iconv`).
  - **O10**: `unequal == 0` and `max_abs_diff == 0.0` across all `100 * 5 = 500` `(kt, jl)` category frames for every exit field in Kind `7` (67 fields) and Kind `8` (67 fields).
  - **O11**: `unequal == 0` and `max_abs_diff == 0.0` across all `100 * 5 = 500` `(kt, jl)` category frames for every exit field in Kind `9` (67 fields).
  - **O12**: `unequal == 0` and `max_abs_diff == 0.0` across all `100` steps for every exit field in Kind `11` (96 fields).
  - **CONFIRM (per rung)**: `0` ULP (`unequal == 0` including signed-zero bit 63) across every exit field, step, category, and cell.
  - **REFUTE (per rung)**: Any `>= 1` ULP difference (outside a proved `DECISION 158` CPU subnormal-flush cell at the producing statement). If a rung is not bit-exact, localize the first differing statement (`CONFIRMED` vs `PLAUSIBLE`, record `ppsrc` line + `legoESM` line), record the first non-bit `(kt, category, cell, field)` and a fix proposal, and **STOP** that rung without modifying model code; continue scoring the next rung in `O8, O9, O10, O11, O12`.
- **P3 (Direct negative controls / plants)**:
  - A 1-ULP change planted in one exit field of each rung (`O8` `ht_i_new`, `O9` `t_su`/`h_i`/`t_i`, `O10` `sz_i`/`t_i`, `O11` `a_i`, `O12` `v_ip`) is caught by the scorer (`unequal >= 1`, `max_ulp >= 1`, nonzero exit status), and parameter/checksum/swapped-field plants fail closed.
