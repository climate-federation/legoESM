# SI3 Ladder Round 21 (`si3_r21`) — O8, O9, O10, O11, O12 Master-Record Scoring Receipt

- **Date**: 2026-10-27
- **Lane / Round**: `si3` / Round 21 (`si3_r21`), authored by Argon
- **Branch**: `fidelity/nemo-testcases-si3-ladder` (parent `3d5446cca2b4`)
- **Preregistration**: `docs/ice/fidelity/si3_r21_o8_o12_score_preregister.md:1-51` (committed at `9b814d537d` before reading any binary record)
- **Master record scored**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/oracle_o8_o16_master` (`docs/ice/fidelity/si3_r18_o8_o16_receipt.md:154-180`, `DECISION 173`)
- **Scoring driver**: `scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1-2147`
- **Direct test suite**: `tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:1-858` (`10 passed in 446.56s`)
- **Full 100-step JSON score report**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round21/o8_o12_master_score.json` (`137,492` bytes, SHA-256 `2ffd0cd63ac8503cd0abe6a2751834fc6bb19f2f76ec2b6e92248f220c8b7bd1`)
- **Final-code Option A re-measurement report**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round21/r21_option_a_final_remeasurement.json` (`4,979` bytes, SHA-256 `3b886af81161f2f0cf56c37ce9df39e67a87b634c3cfb731a9fe2e4697884ebb`)
- **Verdict**: **`PASS_ALL_O8_O12_0_ULP` (literal stage kernels and M4 production entry points)** — every EXIT field of the literal rungs **O8, O9, O10, O11, and O12** matches the admitted 100-step NEMO 5.0.2 `ORCA2_SAS_ICE` master record at **`0` ULP** (`unequal == 0`, `max |diff| == 0.0`, `first_non_bit == None`, including IEEE-754 signed-zero bit 63) across all `100` steps (`kt = 1..100`) and all categories (`jl = 1..5` and `jl = 0`), both through the stage kernels (Sections 2–3) and through the rewired production entry points `si3_column_step_arrays` (`0 / 306,731,360`, `0` ULP) and `step_ponds(scheme='si3_lev')` (`0 / 1,145,653,900`, `0` ULP) implemented per **`DECISION 186` (Option A)** (Section 8). Duplicate helpers (`_si3_zdf_bl99_step`, `_piecewise_remap`, `_nemo_snow_enthalpy_remap`, `_nemo_ice_enthalpy_remap`, `_c1d_dh_via_core`, `_dh_step`) are deleted from `packages/ice/legoesm/ice/bitz_lipscomb.py:1811-1998`, leaving one implementation (Model-Code Checklist **M4** satisfied), and all final-code re-measurements (D5 year hash `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2` `MATCHES_PINNED`, lane-3 cards, `O1`–`O7` admissions, `step_sea_ice`, and default non-SI3 runs) are unchanged at `0` ULP with finite gradients.
- **independent review: operator**

---

## 1. Master-record admission and resolved-namelist verification

Before scoring any rung, `verify_master_admission_manifests` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:338-393`) and `parse_resolved_ice_namelist` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:297-335`) verify the admitted master record at `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/oracle_o8_o16_master` (`docs/ice/fidelity/si3_r18_o8_o16_receipt.md:154-180`, `scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o16_master_gate.py:40-55`):

| Artifact / Projection | Pinned SHA-256 | Verified SHA-256 | Records verified | Status |
|---|---|---|---:|---|
| `manifest/run_manifest.json` | `bfb0b71d1500b7bc91c68a9e39c8af7a654b139258c0635ad5b1f25284b31fd0` | `bfb0b71d1500b7bc91c68a9e39c8af7a654b139258c0635ad5b1f25284b31fd0` | — | `PASS` |
| `manifest/resource_usage.json` | `b30a5febe9248e5008999cb419953dad40c78658542a3c7221d301825270967c` | `b30a5febe9248e5008999cb419953dad40c78658542a3c7221d301825270967c` | — | `PASS` |
| `manifest/records.sha256` | `fbd3cb4d3450f428d8444dcb145bec3c1a832554e3e7e05d7f5b6a159daadd85` | `fbd3cb4d3450f428d8444dcb145bec3c1a832554e3e7e05d7f5b6a159daadd85` | `5,115` | `PASS` |
| `manifest/per_rung/O8.json` | `303a29d76034c3856e4cf6fa60ed551b7146e5decd3618038b058501c818ed82` | `303a29d76034c3856e4cf6fa60ed551b7146e5decd3618038b058501c818ed82` | `200` (`kinds 1, 2`) | `PASS` |
| `manifest/per_rung/O9.json` | `3d97b7bb71c91530a89f60498c8a84316829200e89d8c45470f5a8d1814ee0a3` | `3d97b7bb71c91530a89f60498c8a84316829200e89d8c45470f5a8d1814ee0a3` | `2,115` (`kinds 21, 3, 4, 5, 6, 17`) | `PASS` |
| `manifest/per_rung/O10.json` | `ded24ddbc825a9a331d763665e51559022c97288cb7c7f516b3a587693deb554` | `ded24ddbc825a9a331d763665e51559022c97288cb7c7f516b3a587693deb554` | `1,600` (`kinds 21, 6, 7, 8`) | `PASS` |
| `manifest/per_rung/O11.json` | `0c5197c52331975ff77d813f0154e0df6ea0aa1e0593e9ca062990d664465666` | `0c5197c52331975ff77d813f0154e0df6ea0aa1e0593e9ca062990d664465666` | `1,100` (`kinds 21, 8, 9`) | `PASS` |
| `manifest/per_rung/O12.json` | `fe395cab681d59a8c28a722cd7fc8222ec3ac0ee8c4e569d4043643fd4a47e50` | `fe395cab681d59a8c28a722cd7fc8222ec3ac0ee8c4e569d4043643fd4a47e50` | `200` (`kinds 10, 11`) | `PASS` |

All 60 resolved namelist parameters parsed from `record_100step/output.namelist.ice` and `record_100step/output.namelist.dyn` (`ln_icedH=True`, `ln_icedA=True`, `ln_icedO=True`, `ln_leadhfx=True`, `rn_hinew=0.1`, `ln_frazil=False`, `rn_maxfraz=1.0`, `rn_vfraz=0.417`, `rn_Cfraz=5.0`, `rn_himin=0.1`, `rn_kappa_i=1.0`, `rn_kappa_s=10.0`, `ln_zdf_BL99=True`, `ln_cndi_U64=False`, `ln_cndi_P07=True`, `rn_cnd_s=0.31`, `nn_qtrice=0`, `ln_zdf_chkcvg=False`, `ln_cndflx=False`, `ln_cndemulate=False`, `nn_snwfra=2`, `rn_beta=1.0`, `rn_dmin=8.0`, `rn_snwblow=0.66`, `nn_icesal=4`, `rn_icesal=4.0`, `rn_sinew=0.75`, `rn_simin=0.1`, `ln_drainage=True`, `ln_flushing=True`, `nn_sal_scheme=1`, `nn_drainage=50`, `nn_flushing=5`, `rn_flushrate=0.3`, `rn_vbrc=0.05`, `nn_liquidus=2`, `rn_sal_himin=0.1`, `rn_alpha_RJW=0.03`, `rn_Rc_RJW=8.0`, `rn_alpha_GN=0.00046`, `rn_Rc_GN=8.1`, `rn_alpha_CW=5e-07`, `ln_sal_chk=False`, `ln_pnd=True`, `ln_pnd_LEV=True`, `rn_apnd_min=0.15`, `rn_apnd_max=0.8`, `rn_pnd_flush=0.1`, `ln_pnd_lids=True`, `ln_pnd_rain=True`, `ln_pnd_CST=False`, `rn_apnd=0.2`, `rn_hpnd=0.05`, `ln_pnd_TOPO=False`, `ln_pnd_alb=True`, `rn_pnd_hl_min=0.005`, `rn_pnd_hl_max=0.015`, `ln_virtual_itd=False`, `ln_icediachk=False`, `ln_drgice_imp=False`) agree bit-for-bit with the 59-element binary scalar header of every scored record (`_verify_record_scalars_against_namelist` in `scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:506-563`), alongside `rDt_ice = 21600.0`, `r1_Dt_ice = 1.0 / 21600.0`, and `r1_nlay_i = 0.1`.

---

## 2. Per-rung 100-step 0-ULP summary table (O8, O9, O10, O11, O12)

All evaluations ran on CPU (`jax.default_backend() == "cpu"`), `fp64` (`JAX_ENABLE_X64=1`), `PrecisionPolicy.fp64(transcendentals="libm")`, under `jax.jit` (`_build_compiled_kernels` in `scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:680-974`) with dynamic array inputs passed as runtime JIT arguments:

| Rung | Stage & NEMO `ppsrc` reference | Production function(s) (`jax.jit`) | Entry -> Exit kinds (`kt`, `jl`) | Exit fields scored | Total `unequal / total` elements | `max \|diff\|` | `max ULP` | First non-bit `(kt, category, cell, field)` | Verdict |
|---|---|---|---|---:|---:|---:|---:|---|---|
| **O8** | Frazil collection thickness (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:393-477`; `docs/ice/fidelity/si3_r15_o8_receipt.md:1-195`) | `si3_ice_thd_frazil` (`packages/ice/legoesm/ice/bitz_lipscomb.py:2024`) | Kind `1` -> Kind `2` (`kt=1..100, jl=0`) | `2` (`ht_i_new`, `fraz_frac`) | `0 / 5,328,000` | `0.0` | `0` | `None` | `PASS_0_ULP` |
| **O9** | Vertical conduction + thickness growth/melt + enthalpy remap + 1st diagnostic `T_i` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:201-205`; `docs/ice/fidelity/si3_r16_o9_receipt.md:1-336`) | `si3_ice_thd_zdf_bl99` (`packages/ice/legoesm/ice/bitz_lipscomb.py:834`), `si3_ice_thd_dh` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1288`), `si3_ice_thd_temp` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1272`), `si3_ice_thd_category_stages` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1344`) | Kind `3` (+ Kind `21`) -> Kinds `4`, `5`, `6` (`kt=1..100, jl=1..5`) + Kind `17` (`kt in (1,2,100), jl=1..5`) | Kind `4` (`16`), Kind `5` (`67`), Kind `6` (`67`), Composed (`59`), Kind `17` (`14` + `iconv`), Gather (`9`) | Kind `4`: `0 / 80,516,982`<br>Kind `5`: `0 / 195,541,242`<br>Kind `6`: `0 / 195,541,242`<br>Composed: `0 / 230,048,520`<br>Kind `17`: `0 / 3,769,716` (`15/15 iconv`)<br>Gather: `0 / 59,429,201` | `0.0` | `0` | `None` | `PASS_0_ULP` |
| **O10** | Gravity drainage + flushing (`nn_icesal=4`) + 2nd diagnostic `T_i` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:311-683`; `docs/ice/fidelity/si3_r19_o10_receipt.md:1-245`) | `si3_ice_thd_sal` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1705-1774`), `si3_ice_thd_temp` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1272`), `si3_ice_thd_o10_category_stages` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1785-1808`) | Kind `6` -> Kind `7` -> Kind `8` (`kt=1..100, jl=1..5`) | Kind `7` (`67`), Kind `8` (`67`), Unpadded composed spot-check (`4`) | Kind `7`: `0 / 195,541,242`<br>Kind `8`: `0 / 195,541,242`<br>Unpadded `kt=1`: `0 / 413,270` | `0.0` | `0` | `None` | `PASS_0_ULP` |
| **O11** | Lateral melting `ice_thd_da` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:167-170`; `docs/ice/fidelity/si3_r17_o11_receipt.md:1-251`) | `si3_ice_thd_da` (`packages/ice/legoesm/ice/si3_thd_da.py:65`) | Kind `8` -> Kind `9` (`kt=1..100, jl=1..5`) | Kind `9` (`67`: `6` updated + `61` pass-through) | `0 / 195,541,242` | `0.0` | `0` | `None` | `PASS_0_ULP` |
| **O12** | Level-ice melt ponds `ice_thd_pnd` / `pnd_LEV` / `ice_perm_eff` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:162-264`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:329-582`; `docs/ice/fidelity/si3_r20_o12_receipt.md:1-188`) | `si3_ice_thd_pnd` (`packages/ice/legoesm/ice/si3_thd_pnd.py:575-781`) / `step_ponds` (`packages/ice/legoesm/ice/ponds.py:34-311`) | Kind `10` -> Kind `11` (`kt=1..100, jl=0`) | Kind `11` (`96`: `8` updated + `88` pass-through) | `0 / 1,145,653,900` | `0.0` | `0` | `None` | `PASS_0_ULP` |

---

## 3. Per-rung field-by-field 0-ULP tables

### 3.1 Rung O8 (`Kind 1 -> Kind 2`, `si3_ice_thd_frazil`, `kt = 1..100, jl = 0`)

Scored across all `100` steps (`180 x 148 = 26,640` interior cells per step; `scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1027-1044`):

| Exit field (Kind `2`) | Extent / shape per record | Role | `unequal / total` | `max \|diff\|` | `max ULP` | First non-bit `(kt, category, cell, field)` |
|---|---|---|---:|---:|---:|---|
| `ht_i_new` | `I2` `(180, 148)` | Updated (`si3_ice_thd_frazil.ht_i_new`) | `0 / 2,664,000` | `0.0` | `0` | `None` |
| `fraz_frac` | `I2` `(180, 148)` | Updated (`si3_ice_thd_frazil.fraz_frac`) | `0 / 2,664,000` | `0.0` | `0` | `None` |

### 3.2 Rung O9 (`Kind 3 -> Kind 4 -> Kind 5 -> Kind 6` + `Kind 17` + `Kind 21 -> Kind 3` gather)

Across `kt = 1..100` and `jl = 1..5` (`500` category records), the sum of gathered active cells is $\sum_{kt=1}^{100}\sum_{jl=1}^{5} \mathrm{npti}(kt, jl) = 1,917,071$ (`G1`), giving $19,170,710$ ice-layer cells (`GI`, `10` layers) and $9,585,355$ snow-layer cells (`GS`, `5` layers) (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1201-1434`).

#### Kind `4` (`zdf_exit`, `si3_ice_thd_zdf_bl99`, all 16 fields)

| Exit field(s) (Kind `4`) | Extent | Role | `unequal / total` (each field) | `max \|diff\|` | `max ULP` | First non-bit `(kt, category, cell, field)` |
|---|---|---|---:|---:|---:|---|
| `t_su`, `qns_ice`, `qcn_ice_top`, `qcn_ice_bot`, `qtr_ice_bot`, `hfx_err_dif`, `hfx_dif`, `cnd_ice`, `t1_ice`, `t_si` (10 fields) | `G1` `(npti,)` | Updated by `si3_ice_thd_zdf_bl99` | `0 / 1,917,071` | `0.0` | `0` | `None` |
| `t_i`, `e_i` (2 fields) | `GI` `(npti, 10)` | Updated by `si3_ice_thd_zdf_bl99` | `0 / 19,170,710` | `0.0` | `0` | `None` |
| `t_s`, `e_s` (2 fields) | `GS` `(npti, 5)` | Updated by `si3_ice_thd_zdf_bl99` | `0 / 9,585,355` | `0.0` | `0` | `None` |
| `nptidx` (`i4`), `o_i` (`f8`) (2 fields) | `G1` `(npti,)` | Pass-through from Kind `3` | `0 / 1,917,071` | `0.0` | `0` | `None` |
| **Kind `4` total (16 fields)** | — | — | **`0 / 80,516,982`** | **`0.0`** | **`0`** | **`None`** |

#### Kind `5` (`dh_exit`, `si3_ice_thd_dh`, all 67 fields)

| Exit field(s) (Kind `5`) | Extent | Role | `unequal / total` (each field) | `max \|diff\|` | `max ULP` | First non-bit `(kt, category, cell, field)` |
|---|---|---|---:|---:|---:|---|
| `h_i`, `h_s`, `t_su`, `a_i`, `s_i`, `qml_ice` (6 fields) | `G1` `(npti,)` | Updated column state (`si3_ice_thd_dh`) | `0 / 1,917,071` | `0.0` | `0` | `None` |
| `e_i`, `sz_i` (2 fields) | `GI` `(npti, 10)` | Updated ice enthalpy & layer salinity (`si3_ice_thd_dh`) | `0 / 19,170,710` | `0.0` | `0` | `None` |
| `e_s`, `t_s` (2 fields) | `GS` `(npti, 5)` | Updated snow enthalpy & temperature (`si3_ice_thd_dh`) | `0 / 9,585,355` | `0.0` | `0` | `None` |
| `dh_s_tot`, `dh_s_sum`, `dh_s_itm`, `dh_s_snic`, `dh_i_sum`, `dh_i_itm`, `dh_i_bom`, `dh_i_bog`, `dh_i_sub` (9 `DH_DIAGS` fields) | `G1` `(npti,)` | Updated thickness increments (`si3_ice_thd_dh.dh`) | `0 / 1,917,071` | `0.0` | `0` | `None` |
| `wfx_snw_sni`, `wfx_snw_sum`, `wfx_sub`, `wfx_snw_sub`, `wfx_ice_sub`, `wfx_sum`, `wfx_bom`, `wfx_bog`, `wfx_dni`, `wfx_res`, `wfx_sni`, `wfx_opw`, `hfx_sum`, `hfx_bom`, `hfx_bog`, `hfx_dif`, `hfx_opw`, `hfx_snw`, `hfx_thd`, `hfx_sub`, `hfx_res`, `hfx_err_dif`, `sfx_bog`, `sfx_bom`, `sfx_sum`, `sfx_sni`, `sfx_opw`, `sfx_res`, `sfx_sub` (29 `DH_FLUXES` fields) | `G1` `(npti,)` | Updated 2D-gathered mass/heat/salt fluxes (`si3_ice_thd_dh.fluxes`) | `0 / 1,917,071` | `0.0` | `0` | `None` |
| `t_i` (1 field) | `GI` `(npti, 10)` | Pass-through from Kind `4` | `0 / 19,170,710` | `0.0` | `0` | `None` |
| `nptidx`, `o_i`, `at_i`, `t_bo`, `qns_ice`, `qsr_ice`, `dqns_ice`, `qtr_ice_top`, `qcn_ice_top`, `qcn_ice_bot`, `qtr_ice_bot`, `cnd_ice`, `t1_ice`, `t_si`, `qsb_ice_bot`, `fhld`, `frq_m`, `sprecip`, `qprec_ice`, `evap_ice`, `sss`, `sst` (22 fields; `hfx_err_dif`, `hfx_dif`, `qml_ice` already counted above) | `G1` `(npti,)` | Pass-through from Kind `3` / Kind `4` | `0 / 1,917,071` | `0.0` | `0` | `None` |
| **Kind `5` total (67 fields)** | — | — | **`0 / 195,541,242`** | **`0.0`** | **`0`** | **`None`** |

#### Kind `6` (`temp1_exit_sal_entry`, `si3_ice_thd_temp`, all 67 fields) and Composed `si3_ice_thd_category_stages`

| Substage / Exit field(s) | Extent | Role | `unequal / total` | `max \|diff\|` | `max ULP` | First non-bit `(kt, category, cell, field)` |
|---|---|---|---:|---:|---:|---|
| Kind `6` `t_i` | `GI` `(npti, 10)` | Updated by 1st `si3_ice_thd_temp` (`h_i`, `e_i`, `sz_i`) | `0 / 19,170,710` | `0.0` | `0` | `None` |
| Kind `6` remaining 66 fields (`62` `G1`, `2` `GI`, `2` `GS`) | `G1`, `GI`, `GS` | Pass-through from Kind `5` | `0 / 176,370,532` | `0.0` | `0` | `None` |
| **Kind `6` total (67 fields)** | — | — | **`0 / 195,541,242`** | **`0.0`** | **`0`** | **`None`** |
| **Composed `si3_ice_thd_category_stages` (all 59 updated outputs: `14` ZDF + `44` DH + `1` TEMP1: `50` `G1` + `5` `GI` + `4` `GS`)** | `G1`, `GI`, `GS` | End-to-end Kind `3` -> Kinds `4, 5, 6` under single `jax.jit` | **`0 / 230,048,520`** | **`0.0`** | **`0`** | **`None`** |
| **Kind `21` -> Kind `3` gather (`a_i`, `h_i`, `h_s`, `t_su`, `s_i`, `o_i`, `t_i`, `sz_i`, `t_s`)** | `G1`, `GI`, `GS` | Pre-category 2D -> 1D gather via `nptidx` | **`0 / 59,429,201`** | **`0.0`** | **`0`** | **`None`** |

#### Kind `17` (`picard_final`, `kt in (1, 2, 100), jl = 1..5`, `15` frames, `55,437` cells)

| Field / Check (Kind `17`) | Extent | `unequal / total` | `max \|diff\|` | `max ULP` | First non-bit `(kt, category, cell, field)` |
|---|---|---:|---:|---:|---|
| `nptidx` (`i4`), `l_t_converged` (`i4`), `t_su`, `qns_ice`, `zradtr_s_top`, `zradtr_i_bot` (6 fields) | `G1` `(npti,)` | `0 / 55,437` each | `0.0` | `0` | `None` |
| `t_i`, `zradab_i`, `zradtr_i_diff` (3 fields) | `GI` `(npti, 10)` | `0 / 554,370` each | `0.0` | `0` | `None` |
| `zkappa_i` | `GI0` `(npti, 11)` | `0 / 609,807` | `0.0` | `0` | `None` |
| `t_s`, `zradab_s`, `zradtr_s_diff` (3 fields) | `GS` `(npti, 5)` | `0 / 277,185` each | `0.0` | `0` | `None` |
| `zkappa_s` | `GS0` `(npti, 6)` | `0 / 332,622` | `0.0` | `0` | `None` |
| `max(n_picard) == picard_iterations` (`8,5,4,3,3` at `kt=1`; `26,6,4,3,3` at `kt=2`; `31,11,8,4,4` at `kt=100`) | Scalar per `(kt, jl)` | `0 / 15` (`15 / 15` match) | `0.0` | `0` | `None` |
| **Kind `17` array total (14 fields)** | — | **`0 / 3,769,716`** | **`0.0`** | **`0`** | **`None`** |

### 3.3 Rung O10 (`Kind 6 -> Kind 7 -> Kind 8`, `si3_ice_thd_sal`, `si3_ice_thd_temp`, `si3_ice_thd_o10_category_stages`, `kt = 1..100, jl = 1..5`)

Scored across all `500` `(kt, jl)` category frames (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1323-1348`, `scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1436-1456`):

| Record kind / Exit field(s) | Extent | Role | `unequal / total` | `max \|diff\|` | `max ULP` | First non-bit `(kt, category, cell, field)` |
|---|---|---|---:|---:|---:|---|
| Kind `7` `sz_i` | `GI` `(npti, 10)` | Updated by `si3_ice_thd_sal` (`sal_drain_flush`) | `0 / 19,170,710` | `0.0` | `0` | `None` |
| Kind `7` `sfx_bri` | `G1` `(npti,)` | Updated brine salt flux (`si3_ice_thd_sal.sfx_bri`) | `0 / 1,917,071` | `0.0` | `0` | `None` |
| Kind `7` `sfx_res` | `G1` `(npti,)` | Updated residual salt flux (`si3_ice_thd_sal.sfx_res`) | `0 / 1,917,071` | `0.0` | `0` | `None` |
| Kind `7` remaining 64 fields (`60` `G1`, `2` `GI`, `2` `GS`) | `G1`, `GI`, `GS` | Pass-through from Kind `6` | `0 / 172,536,390` | `0.0` | `0` | `None` |
| **Kind `7` total (67 fields)** | — | — | **`0 / 195,541,242`** | **`0.0`** | **`0`** | **`None`** |
| Kind `8` `t_i` | `GI` `(npti, 10)` | Updated by 2nd `si3_ice_thd_temp` (`h_i`, `e_i`, `sz_i`) | `0 / 19,170,710` | `0.0` | `0` | `None` |
| Kind `8` remaining 66 fields (`62` `G1`, `2` `GI`, `2` `GS`) | `G1`, `GI`, `GS` | Pass-through from Kind `7` | `0 / 176,370,532` | `0.0` | `0` | `None` |
| **Kind `8` total (67 fields)** | — | — | **`0 / 195,541,242`** | **`0.0`** | **`0`** | **`None`** |
| Unpadded `(npti,)` JIT spot-check of `si3_ice_thd_o10_category_stages` (`kt=1, jl=1..5`: `sz_i`, `sfx_bri`, `sfx_res`, `t_i`) | `G1`, `GI` | Unpadded `(npti,)` JIT verification (`18,785` cells at `kt=1`) | **`0 / 413,270`** | **`0.0`** | **`0`** | **`None`** |

### 3.4 Rung O11 (`Kind 8 -> Kind 9`, `si3_ice_thd_da`, `kt = 1..100, jl = 1..5`)

Scored across all `500` `(kt, jl)` category frames (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1458-1469`):

| Exit field(s) (Kind `9`) | Extent | Role | `unequal / total` (each field) | `max \|diff\|` | `max ULP` | First non-bit `(kt, category, cell, field)` |
|---|---|---|---:|---:|---:|---|
| `a_i` | `G1` `(npti,)` | Updated ice concentration (`si3_ice_thd_da.a_i`) | `0 / 1,917,071` | `0.0` | `0` | `None` |
| `h_i` | `G1` `(npti,)` | Updated ice thickness (`si3_ice_thd_da.h_i`) | `0 / 1,917,071` | `0.0` | `0` | `None` |
| `h_s` | `G1` `(npti,)` | Updated snow thickness (`si3_ice_thd_da.h_s`) | `0 / 1,917,071` | `0.0` | `0` | `None` |
| `sfx_lam` | `G1` `(npti,)` | Updated lateral-melt salt flux (`si3_ice_thd_da.sfx_lam`) | `0 / 1,917,071` | `0.0` | `0` | `None` |
| `hfx_thd` | `G1` `(npti,)` | Updated thermodynamic heat flux (`si3_ice_thd_da.hfx_thd`) | `0 / 1,917,071` | `0.0` | `0` | `None` |
| `wfx_lam` | `G1` `(npti,)` | Updated lateral-melt water flux (`si3_ice_thd_da.wfx_lam`) | `0 / 1,917,071` | `0.0` | `0` | `None` |
| Remaining 61 fields (`56` `G1`, `3` `GI`, `2` `GS`) | `G1`, `GI`, `GS` | Pass-through from Kind `8` | `0 / 184,038,816` (total) | `0.0` | `0` | `None` |
| **Kind `9` total (67 fields)** | — | — | **`0 / 195,541,242`** | **`0.0`** | **`0`** | **`None`** |

### 3.5 Rung O12 (`Kind 10 -> Kind 11`, `si3_ice_thd_pnd`, `kt = 1..100, jl = 0`)

Scored across all `100` steps (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1549-1581`):

| Exit field(s) (Kind `11`) | Extent / shape per record | Role | `unequal / total` (each field) | `max \|diff\|` | `max ULP` | First non-bit `(kt, category, cell, field)` |
|---|---|---|---:|---:|---:|---|
| `at_i` | `I2` `(180, 148)` | Updated total ice concentration (`si3_ice_thd_pnd.at_i`) | `0 / 2,664,000` | `0.0` | `0` | `None` |
| `a_ip` | `I3` `(180, 148, 5)` | Updated pond concentration (`si3_ice_thd_pnd.a_ip`) | `0 / 13,320,000` | `0.0` | `0` | `None` |
| `v_ip` | `I3` `(180, 148, 5)` | Updated pond volume (`si3_ice_thd_pnd.v_ip`) | `0 / 13,320,000` | `0.0` | `0` | `None` |
| `h_ip` | `I3` `(180, 148, 5)` | Updated pond depth (`si3_ice_thd_pnd.h_ip`) | `0 / 13,320,000` | `0.0` | `0` | `None` |
| `v_il` | `I3` `(180, 148, 5)` | Updated pond lid volume (`si3_ice_thd_pnd.v_il`) | `0 / 13,320,000` | `0.0` | `0` | `None` |
| `h_il` | `I3` `(180, 148, 5)` | Updated pond lid thickness (`si3_ice_thd_pnd.h_il`) | `0 / 13,320,000` | `0.0` | `0` | `None` |
| `a_ip_frac` | `I3` `(180, 148, 5)` | Updated relative pond area `a_ip / a_i` (`si3_ice_thd_pnd.a_ip_frac`) | `0 / 13,320,000` | `0.0` | `0` | `None` |
| `wfx_pnd` | `I2` `(180, 148)` | Updated pond freshwater flux (`si3_ice_thd_pnd.wfx_pnd`) | `0 / 2,664,000` | `0.0` | `0` | `None` |
| `STATE_I3` pass-through (`a_i`, `v_i`, `v_s`, `h_i`, `h_s`, `s_i`, `sv_i`, `o_i`, `oa_i`, `t_su`, `a_ip_eff`: 11 fields) | `I3` `(180, 148, 5)` | Pass-through from Kind `10` | `0 / 13,320,000` | `0.0` | `0` | `None` |
| `STATE_II4` pass-through (`t_i`, `e_i`, `sz_i`, `szv_i`: 4 fields) | `II4` `(180, 148, 10, 5)` | Pass-through from Kind `10` | `0 / 133,200,000` | `0.0` | `0` | `None` |
| `STATE_IS4` pass-through (`t_s`, `e_s`: 2 fields) | `IS4` `(180, 148, 5, 5)` | Pass-through from Kind `10` | `0 / 66,600,000` | `0.0` | `0` | `None` |
| `PRE_AGG_TOTALS` pass-through (`ato_i`, `vt_i`, `vt_s`, `st_i`, `et_i`, `et_s`, `tm_su`, `at_ip`, `at_ip_eff`, `vt_ip`, `vt_il`: 11 fields; `at_i` updated above) | `I2` `(180, 148)` | Pass-through from Kind `10` | `0 / 2,664,000` | `0.0` | `0` | `None` |
| `POND` pass-through (`dh_i_sum_2d`, `dh_s_sum_2d`: 2 fields; `a_ip`, `v_ip`, `h_ip`, `v_il`, `h_il`, `a_ip_frac` updated above) | `I3` `(180, 148, 5)` | Pass-through from Kind `10` | `0 / 13,320,000` | `0.0` | `0` | `None` |
| `POND_FORCING` pass-through (`sprecip`, `tprecip`: 2 `I2` fields; `qsr_ice`, `qns_ice`: 2 `I3` fields) | `I2` `(180, 148)`, `I3` `(180, 148, 5)` | Pass-through from Kind `10` | `I2`: `0 / 2,664,000`<br>`I3`: `0 / 13,320,000` | `0.0` | `0` | `None` |
| `GLOBAL_AUX` pass-through (`h_i_b`, `a_i_b`: 2 `I3` fields; `t_bo`, `ht_i_new`, `fraz_frac`, `sss_m`, `sst_m`, `rn_amax_2d`, `smask0`: 7 `I2` fields; `tmask_k1`: 1 `F2` field; `hi_max`: 1 `C0` field; `hi_mean`: 1 `C1` field) | `I3`, `I2`, `F2`, `C0`, `C1` | Pass-through from Kind `10` | `I3`: `0 / 13,320,000`<br>`I2`: `0 / 2,664,000`<br>`F2`: `0 / 2,796,800`<br>`C0`: `0 / 600`<br>`C1`: `0 / 500` | `0.0` | `0` | `None` |
| `FLUX` pass-through (42 `I2` fields; `wfx_pnd` updated above) | `I2` `(180, 148)` | Pass-through from Kind `10` | `0 / 2,664,000` | `0.0` | `0` | `None` |
| **Kind `11` total (96 fields)** | — | — | **`0 / 1,145,653,900`** | **`0.0`** | **`0`** | **`None`** |

---

## 4. Direct test and planted-violation verification

`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:1-858` exercises `scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1-2147` across 10 test cases (log saved at `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round21/pytest_o8_o12_master_scorer.log`, `499` bytes, SHA-256 `db453287cdeae8fc858730caf7ca02703d6b1a90a7ffa63c7c0961b0e933836a`):

| Test / Plant | Target | Expected fail-closed behavior | Observed result |
|---|---|---|---|
| `test_field_score_accumulator_catches_1ulp_and_signed_zero` (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:23-55`) | `FieldScoreAccumulator` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:573-663`) | Catches 1-ULP normal perturbation (`max_ulp == 1`, `first_non_bit` recorded) and `-0.0` vs `+0.0` signed-zero bit difference | `PASS` |
| `test_master_admission_manifests_and_plants` (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:58-85`) | `verify_master_admission_manifests` + `manifest-digest` & `namelist-mismatch` plants | All 8 admission & projection digests match; `manifest-digest` and `namelist-mismatch` (`rn_sinew + 0.05`) raise `STATUS PLANT-FIRED` | `PASS` |
| `test_score_o8_to_o12_representative_steps_0_ulp` (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:88-136`) | `score_o8_to_o12` on `steps=(1, 2, 100)` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:977-1698`) | `PASS_ALL_O8_O12_0_ULP` across `O8`, `O9` (including all 15 Kind 17 Picard frames), `O10`, `O11` (and `production_column_step_arrays_o9_o11`), and `O12` (and `production_step_ponds_si3_lev`) | `PASS` |
| `--plant o8-1ulp` (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:194-216`) | 1-ULP flip in O8 exit `ht_i_new[0, 0]` (`kt=1`) | Raises `STATUS PLANT-FIRED: O8 non-bit-exact: unequal=1/53280 ... 'field': 'ht_i_new'` | `PASS` |
| `--plant o9-1ulp` (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:194-216`) | 1-ULP flip in O9 Kind `4` exit `t_su[0]` (`kt=1, jl=1`) | Raises `STATUS PLANT-FIRED: O9 non-bit-exact ... 'field': 't_su'` | `PASS` |
| `--plant o10-1ulp` (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:194-216`) | 1-ULP flip in O10 Kind `7` exit `sz_i[0, 0]` (`kt=1, jl=1`) | Raises `STATUS PLANT-FIRED: O10 non-bit-exact ... 'field': 'sz_i'` | `PASS` |
| `--plant o11-1ulp` (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:194-216`) | 1-ULP flip in O11 Kind `9` exit `a_i[0]` (`kt=1, jl=1`) | Raises `STATUS PLANT-FIRED: O11 non-bit-exact ... 'field': 'a_i'` | `PASS` |
| `--plant o12-1ulp` (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:194-216`) | 1-ULP flip in O12 Kind `11` exit `a_ip[0, 0, 0]` (`kt=1, jl=0`) | Raises `STATUS PLANT-FIRED: O12 non-bit-exact ... 'field': 'a_ip'` | `PASS` |
| `--plant swapped-exit` (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:194-216`) | Swap O8 exit fields `ht_i_new` <-> `fraz_frac` (`kt=1`) | Raises `STATUS PLANT-FIRED: O8 non-bit-exact` | `PASS` |
| `test_production_call_sites_route_through_o8_o12_and_fail_when_reverted` (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:220-858`) | Production `si3_column_step_arrays` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1811-1998`), `step_sea_ice(thermo_scheme='si3_bl99')` (`packages/ice/legoesm/ice/sea_ice.py:286-298,500-517`), and `step_ponds(scheme='si3_lev')` (`packages/ice/legoesm/ice/ponds.py:34-311`) | Verifies deleted duplicates do not exist on `bitz_lipscomb`, spy-verifies `si3_column_step_arrays`, `step_sea_ice`, and `step_ponds` call the certified `O8`–`O12` functions, and verifies reverting `si3_column_step_arrays` O10 to `nn_icesal=2` (`19/19` `sz_i` elements differ at `kt=1, jl=1`, `AssertionError: O11 production_column_step_arrays_o9_o11 non-bit-exact`) or reverting `step_ponds(scheme='si3_lev')` to bypass `si3_ice_thd_pnd` (`AssertionError: O12 production step_ponds(scheme='si3_lev') non-bit-exact`) fails closed | `PASS` |

---

## 5. Production Entry-Point and Configuration Map (O8–O12, Review Finding (a) Item 1)

Model-Code Checklist **M4** requires that the production path use the certified implementation with no parallel or duplicate path. At `HEAD` `e79f43e5f994`, the certified literal stage functions scored in Sections 2–3 and the production model dispatcher (`step_sea_ice` in `packages/ice/legoesm/ice/sea_ice.py:286-298,500-517`) are wired as follows across all stages (`O8`–`O12`) and all configurations/cards:

### 5.1 Per-stage (`O8`–`O12`) function call map in `packages/ice/legoesm/ice/`

| Rung | Certified NEMO-literal stage function(s) | Callers of certified literal function(s) at `HEAD` | Production `si3_column_step_arrays` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1811-1998`) / `step_sea_ice` path at `HEAD` | Exact arithmetic / wiring divergence between `si3_column_step_arrays` / `step_sea_ice` and the certified literal functions |
|---|---|---|---|---|
| **O8** (`ice_thd_frazil`) | `si3_ice_thd_frazil` (`packages/ice/legoesm/ice/bitz_lipscomb.py:2024-2042`) | `si3_column_step_arrays` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1968`), `nemo_si3_o8_o12_master_scorer.py` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1027-1044`) | Calls `si3_ice_thd_frazil(forcing.qlead, rn_hinew=_NEW_ICE_THICKNESS, ln_frazil=False)` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1968`), where `_NEW_ICE_THICKNESS = 0.05` (`packages/ice/legoesm/ice/bitz_lipscomb.py:115`) | Shared kernel, **except** `si3_column_step_arrays` hardcodes `rn_hinew = 0.05` (C1D `c1d_omip_l3_namelist_ice_cfg` value) and `ln_frazil = False`, whereas `ORCA2_SAS_ICE` has `rn_hinew = 0.1` (`tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref`). |
| **O9 — ZDF** (`ice_thd_zdf_BL99`) | `si3_ice_thd_zdf_bl99` (`packages/ice/legoesm/ice/bitz_lipscomb.py:834-919`) | `si3_ice_thd_category_stages` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1344-1382`), `nemo_si3_o8_o12_master_scorer.py` (**0 callers in `packages/` outside `si3_ice_thd_category_stages`**) | Calls `_si3_zdf_bl99_step` (`packages/ice/legoesm/ice/bitz_lipscomb.py:545-804`, invoked at `packages/ice/legoesm/ice/bitz_lipscomb.py:1811-1927`) | Both call `_si3_zdf_bl99_core` (`packages/ice/legoesm/ice/bitz_lipscomb.py:545`), but `_si3_zdf_bl99_step` omits the 6 post-core diagnostic outputs (`qcn_ice_top`, `qcn_ice_bot`, `hfx_err_dif`, `hfx_dif`, `cnd_ice`, `t1_ice`, `t_si`; `packages/ice/legoesm/ice/bitz_lipscomb.py:906`) and accepts 11 legacy `_nemo_*` toggle flags. When all `_nemo_*` flags are `True` (default), `t_su`, `t_i`, `e_i`, `t_s`, `e_s`, `qns_ice`, `qtr_ice_bot` are bit-identical (`0` ULP). |
| **O9 — DH** (`ice_thd_dh`) | `si3_ice_thd_dh` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1288-1311`) | `si3_ice_thd_category_stages` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1344-1382`), `nemo_si3_o8_o12_master_scorer.py` (**0 callers in `packages/` outside `si3_ice_thd_category_stages`**) | Calls `_dh_step` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1015-1269`, invoked at `packages/ice/legoesm/ice/bitz_lipscomb.py:1811-1939`) | When all 11 `_nemo_*` flags are `True` (default), `_dh_step` delegates to `_c1d_dh_via_core` -> `_si3_ice_thd_dh_core` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1015`) with `nn_icesal=2`, `rn_sinew=0.5`, `rn_snwblow=0.5`, whereas `si3_ice_thd_dh` calls `_si3_ice_thd_dh_core` with namelist `nn_icesal` (`4` in `ORCA2_SAS_ICE`, `2` in `C1D_OMIP_L3`), `rn_sinew`, `rn_snwblow`, and returns the 9 `SI3DHDiagnostics` and 29 `SI3DHFluxes` fields. In addition, `_dh_step` retains `595` lines (`packages/ice/legoesm/ice/bitz_lipscomb.py:1015-1269`) of duplicate legacy DH code behind non-default `_nemo_*=False` flags. |
| **O9 — TEMP1** (`ice_thd_temp` 1st call) | `si3_ice_thd_temp` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1272-1277`) & `si3_ice_thd_category_stages` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1344-1382`) | `si3_ice_thd_o10_category_stages` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1785-1808`), `nemo_si3_o8_o12_master_scorer.py` (**0 callers in `packages/` outside `bitz_lipscomb.py` / `si3_thd_pnd.py`**) | Sets `post_temp1 = post_dh` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1811-1939`) without calling `si3_ice_thd_temp` (`SI3ColumnArrays` does not carry `t_i`) | Because `post_temp1 = post_dh` never computes diagnostic `t_i = ice_temperature_from_enthalpy(e_i, sz_i)` (`packages/ice/legoesm/ice/bitz_lipscomb.py:249-311`), no `t_i` array exists to feed O10 `nn_icesal = 4` brine-volume/liquidus calculations (`_sal4_vc19_brine`, `packages/ice/legoesm/ice/bitz_lipscomb.py:1385-1394`). |
| **O10 — SAL + TEMP2** (`ice_thd_sal` + `ice_thd_temp` 2nd call) | `si3_ice_thd_sal` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1705-1782`), `_si3_ice_thd_sal4_core` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1491-1594`, `:1597-1668`), `si3_ice_thd_o10_category_stages` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1785-1808`) | `si3_ice_thd_o12_stages` (`packages/ice/legoesm/ice/si3_thd_pnd.py:784-972`), `nemo_si3_o8_o12_master_scorer.py` (**0 production callers in `packages/`**) | Runs inline `nn_icesal = 2` bulk-salinity relaxation + `option2_salinity_profile` (`packages/ice/legoesm/ice/bitz_lipscomb.py:163-211`, invoked at `packages/ice/legoesm/ice/bitz_lipscomb.py:1750-1755`) and sets `post_temp2 = post_sal` | **Complete algorithmic divergence when `nn_icesal == 4`**: `si3_column_step_arrays` always runs the `nn_icesal = 2` exponential relaxation toward `[rn_simin, zsi1]` followed by the 2-parameter linear profile `option2_salinity_profile`, never calling `si3_ice_thd_sal` (`nn_icesal = 4` RJW14 gravity drainage + flushing subcycling over `nn_drainage=50`, `nn_flushing=5` with `sfx_bri` and `sfx_res`) or 2nd `si3_ice_thd_temp`. Conversely, `si3_ice_thd_sal` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1705-1774`) currently rejects `nn_icesal != 4` and `nlay_i != 10`. |
| **O11 — DA** (`ice_thd_da`) | `si3_ice_thd_da` (`packages/ice/legoesm/ice/si3_thd_da.py:65-185`) | `_si3_column_thd_da` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1952`, `:2077`), `si3_ice_thd_o12_stages` (`packages/ice/legoesm/ice/si3_thd_pnd.py:784-972`), `nemo_si3_o8_o12_master_scorer.py` | `si3_column_step_arrays` calls `_si3_column_thd_da` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1952`), which calls `si3_ice_thd_da` when `si3.lateral_melt = True` | Shared kernel when `si3.lateral_melt = True`. In `C1D_OMIP_L3`, `ln_icedA = False` (`scripts/validate/ocean_fidelity/testcases/configs/c1d_omip_l3_namelist_ice_cfg`), so `si3.lateral_melt = False` and O11 is a pass-through. |
| **O12 — PND** (`ice_thd_pnd` / `pnd_LEV` / `ice_perm_eff`) | `si3_ice_thd_pnd` (`packages/ice/legoesm/ice/si3_thd_pnd.py:575-781`), `si3_pnd_lev` (`packages/ice/legoesm/ice/si3_thd_pnd.py:235-572`), `si3_ice_perm_eff` (`packages/ice/legoesm/ice/si3_thd_pnd.py:168-232`), `si3_ice_thd_o12_stages` (`packages/ice/legoesm/ice/si3_thd_pnd.py:784-972`) | `si3_ice_thd_pnd` and `si3_ice_thd_o12_stages` have **0 callers in `packages/`**; `step_ponds` (`packages/ice/legoesm/ice/ponds.py:34-311`) calls `si3_pnd_lev` -> `si3_ice_perm_eff` | `si3_column_step_arrays` (`thermo_scheme='si3_bl99'`) does not call ponds (`packages/ice/legoesm/ice/config.py:291-338` forbids `ponds.enabled=True` with `thermo_scheme='si3_bl99'`). Under `thermo_scheme='zero_layer'` (`_thermo_v2`), `step_ponds(scheme='si3_lev')` (`packages/ice/legoesm/ice/ponds.py:34-311`) calls `si3_pnd_lev` per category | `step_ponds(scheme='si3_lev')` bypasses `si3_ice_thd_pnd` (`packages/ice/legoesm/ice/si3_thd_pnd.py:717-735`, `:738-781`): before calling `si3_pnd_lev`, `step_ponds` multiplies `a_ip_in = pond_area * a_i`, so when a category's ice vanishes (`a_i == 0`), `a_ip_in == 0` and `step_ponds` returns `drain_to_ocean_m = 0.0` instead of draining `(v_ip + v_il) * rhow * r1_Dt_ice` into `wfx_pnd` as `si3_ice_thd_pnd` does at `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:162-264`. |

### 5.2 Per-configuration / per-card call-path map

| Configuration / Certified Card | Production / Harness Entry Point | Calls `si3_column_step_arrays`? | Calls certified literal `O8`–`O12` functions? | Notes |
|---|---|---|---|---|
| **D5 `C1D_OMIP_L3` year** (`8,760` steps, `jpl=1, nlay_i=3, nlay_s=1, nn_icesal=2`) | `run_c1d_scam_trace` (`packages/ice/legoesm/ice/c1d_omip_l3.py`) & `run_year_rescore` (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:806-1883`) | **Yes** (`si3_column_step_arrays` at every step `kt=1..8760`) | **O8 only** (`si3_ice_thd_frazil`); **No** for O9 (`si3_ice_thd_zdf_bl99`, `si3_ice_thd_dh`, `si3_ice_thd_temp`), **No** for O10 (`si3_ice_thd_sal`), **No** for O11 (`lateral_melt=False`), **No** for O12 (`ln_pnd=False`) | Pinned full-year SHA-256: `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`. |
| **Production `step_sea_ice(thermo_scheme='si3_bl99')`** (standalone `IceColumnModel.step` & coupled `coupler.py` / `ocean_forcing.py`) | `step_sea_ice` -> `_si3_step_with_trace` (`packages/ice/legoesm/ice/sea_ice.py:286-298,500-517`) | **Yes** (`si3_column_step_arrays`) | **O8** (`si3_ice_thd_frazil`) and **O11** (`si3_ice_thd_da` when `si3.lateral_melt=True`); **No** for O9/O10/O12 | Only active for `SI3ColumnState` (`1` category, `3` ice layers, `1` snow layer). |
| **Lane-3 `ICE_ADV1D` & `ICE_ADV2D`** (Prather advection) | `step_prather_1d_Once` / `step_si3_adv2d_Once` (`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2091-2107`, `packages/ice/legoesm/ice/fidelity/nemo_adv2d_testcase_recipe.py:388-613`) | **No** | **No** (`ln_icethd = False`) | Pure Prather advection (`icedyn_adv_pra`); thermodynamics disabled. |
| **Lane-3 `ICE_ADV2D_RHG`** (EVP rheology + Prather advection) | `step_si3_adv2d_rhg_Once` (`packages/ice/legoesm/ice/fidelity/nemo_adv2d_rhg_testcase_recipe.py:152`) | **No** | **No** (`ln_icethd = False`) | EVP + Prather advection (`icedyn_rhg_evp` + `icedyn_adv_pra`); thermodynamics disabled. |
| **ORCA2 SAS `O1`, `O2`, `O3`, `O4`, `O5`, `O7` admissions** | `si3_o1_step`, `si3_o2_scalar_initial_state`, `si3_o4_step` (`packages/ice/legoesm/ice/si3_orca2_state.py:3807-3960`, `:4528-4812`), `si3_sbc_Final` (`packages/ice/legoesm/ice/si3_surface.py`) | **No** | **No** (`ln_icethd = False` in `O1`–`O7`) | `step_sea_ice` for `thermo_scheme in ('si3_orca2_o1', 'si3_orca2_o2')` (`packages/ice/legoesm/ice/sea_ice.py:363-366`, `:591-611`, `:620-654`) validates `ln_icethd=False` and stops at the coupler `TileResponse` boundary. |
| **ORCA2 SAS `O8`–`O12` master record** (`100` steps, `jpl=5, nlay_i=10, nlay_s=5, nn_icesal=4`) | `score_o8_to_o12` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:977-1698`) | **No** | **Yes** (`si3_ice_thd_frazil`, `si3_ice_thd_zdf_bl99`, `si3_ice_thd_dh`, `si3_ice_thd_temp`, `si3_ice_thd_category_stages`, `si3_ice_thd_sal`, `si3_ice_thd_o10_category_stages`, `si3_ice_thd_da`, `si3_ice_thd_pnd`) | Scored directly by the round-21 harness; not yet wired into `si3_column_step_arrays` or `si3_orca2_state.py`. |
| **Default (non-SI3) sea-ice runs** (`thermo_scheme='zero_layer'`, `_step_slab` & `_step_dynamic_v2` with `ponds.scheme='cesm'` or `'si3_lev'`) | `step_sea_ice` (`packages/ice/legoesm/ice/sea_ice.py:420-433,475,1737-1772,2388-2413,2469-2535,2820-2845,2982-3062,3196-3250,3295-3297,3414-3432`) -> `step_ponds` (`packages/ice/legoesm/ice/ponds.py:34-311`) | **No** | **No** for `cesm` ponds; calls `si3_pnd_lev` -> `si3_ice_perm_eff` (bypassing `si3_ice_thd_pnd`) when `ponds.scheme == 'si3_lev'` | 0-layer Semtner thermodynamics (`_thermo_v2`); never touches `bitz_lipscomb.py` O8–O11. |

---

## 6. Scratch Rewiring Impact Measurements (Review Finding (a) Item 2)

To measure the exact consequences of rewiring the production path to the certified literal O8–O12 kernels (with one implementation and duplicate helpers removed) **without changing model code in the repository**, a standalone measurement harness was executed in scratch on CPU (`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, `PrecisionPolicy.fp64(transcendentals="libm")`), writing `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round21/rewiring_impact_measurement.json` (`16,445` bytes, SHA-256 `31f6262f981eb7ce57822e1dddc0efb2de109e32d86584a2fd50ac04f77464e6`).

Because `C1D_OMIP_L3` (D5) is a 3-layer (`nlay_i=3, nlay_s=1`) column configuration with `nn_icesal = 2` (`scripts/validate/ocean_fidelity/testcases/configs/c1d_omip_l3_namelist_ice_cfg`), whereas `ORCA2_SAS_ICE` (O8–O12) is a 5-category, 10-layer (`nlay_i=10, nlay_s=5`) configuration with `nn_icesal = 4` (`tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref`), two rewiring designs for `si3_column_step_arrays` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1811-1998`) were measured side by side:

- **Option A (Recommended — Unified namelist-driven `nn_icesal ∈ {2, 4}` literal O8–O12 kernels)**:
  1. Remove `_si3_zdf_bl99_step` (`packages/ice/legoesm/ice/bitz_lipscomb.py:545-804`) and the `595`-line duplicate `_dh_step` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1015-1269`); route `si3_column_step_arrays` directly through the certified `si3_ice_thd_zdf_bl99` (`packages/ice/legoesm/ice/bitz_lipscomb.py:834-919`), `si3_ice_thd_dh` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1288-1311`), `si3_ice_thd_temp` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1272-1277`), `si3_ice_thd_sal` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1705-1782`), and `si3_ice_thd_da` (`packages/ice/legoesm/ice/si3_thd_da.py:65-185`).
  2. In `si3_ice_thd_sal` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1705-1782`), dispatch on `namelist["nn_icesal"]` matching NEMO `icethd_sal.f90` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:311-683`): `CASE (2)` runs the certified `nn_icesal=2` bulk relaxation + `option2_salinity_profile` (`packages/ice/legoesm/ice/bitz_lipscomb.py:163-211`, `:1750-1755`), and `CASE (4)` runs `_si3_ice_thd_sal4_core` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1491-1594`, `:1597-1668`) with general `nlay_i = sz_i.shape[-1]`.
  3. In `step_ponds(scheme='si3_lev')` (`packages/ice/legoesm/ice/ponds.py:34-311`), route 5-category, 10-layer `(jpi, jpj, 5)` states through the certified `si3_ice_thd_pnd` (`packages/ice/legoesm/ice/si3_thd_pnd.py:575-781`).
- **Option B (Forced `nn_icesal = 4` in `si3_ice_thd_sal` on all callers including 3-layer C1D)**:
  - **Option B1**: Keep `nn_icesal = 2` in `si3_ice_thd_dh` but force `si3_ice_thd_sal` (`nn_icesal = 4` gravity drainage + flushing generalized to `nlay_i = 3`) in O10 on C1D.
  - **Option B2**: Force `nn_icesal = 4` in both `si3_ice_thd_dh` and `si3_ice_thd_sal` on C1D.

### 6.1 D5 `C1D_OMIP_L3` 1-year (`8,760` steps) measurements

Measured across all `8,760` steps (`8` stages $\times 8,760$ steps = `621,960` stage-field rows, `1,042,440` total scalar/array elements; `scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:806-1883`):

| Rewiring Option | Full-year SHA-256 (`pinned = b7a7329b...`) | Changed rows / `621,960` | Changed elements / `1,042,440` | Affected stage.field outputs (`changed / total` elements, `max \|diff\|`, `max ULP`) | `jax.grad` finite across all 9 inputs? |
|---|---|---:|---:|---|---|
| **Baseline (`HEAD` `e79f43e5f994`)** | `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2` | `0 / 621,960` | `0 / 1,042,440` | None (`0` ULP) | `True` |
| **Option A (Unified `nn_icesal ∈ {2, 4}` literal O8–O11)** | **`b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`** (`MATCHES_PINNED`) | **`0 / 621,960`** | **`0 / 1,042,440`** | **None** — all 8 per-stage digests (`ENTRY`, `POST_ZDF`, `POST_DH`, `POST_TEMP1`, `POST_SAL`, `POST_TEMP2`, `POST_DO`, `EXIT`) match `HEAD` bit-for-bit (`max \|diff\| = 0.0`, `max ULP = 0`) | **`True`** (`max \|grad\|`: `h_ice=0.9868`, `h_snow=4.8612`, `T_surface=0.7303`, `S_bulk=3.9911`, `S_layers=3.560e-3`, `e_ice=1.756e-9`, `e_snow=2.749e-7`, `qsr_ice=0.03282`, `qns_ice=0.03282`) |
| **Option B1 (Force `nn_icesal=4` in O10 `si3_ice_thd_sal` only)** | `78176c01cce0f3a9281f012940473e9a87dee0ff6c749d4765e4009baf159763` (**BREAKS D5**) | `50,084 / 621,960` | `115,724 / 1,042,440` | `POST_SAL.sz_i`: `24,615 / 26,280` (`max \|diff\| = 0.122065`, `136,614,584,263,040` ULP)<br>`POST_TEMP2.sz_i`: `24,615 / 26,280` (`max \|diff\| = 0.122065`, `136,614,584,263,040` ULP)<br>`POST_DO.sv_i`: `8,632 / 8,760` (`max \|diff\| = 0.045924`, `51,705,710,879,346` ULP)<br>`POST_DO.szv_i`: `24,615 / 26,280` (`max \|diff\| = 0.029919`, `194,557,841,340,964` ULP)<br>`EXIT.sv_i`: `8,632 / 8,760` (`max \|diff\| = 0.045924`, `51,705,710,879,346` ULP)<br>`EXIT.szv_i`: `24,615 / 26,280` (`max \|diff\| = 0.029919`, `194,557,841,340,964` ULP) | `True` |
| **Option B2 (Force `nn_icesal=4` in both O9 `si3_ice_thd_dh` & O10 `si3_ice_thd_sal`)** | `defb511de38256a154c0529167e3524d0eddbae14a7161e5cbf65832aa57933d` (**BREAKS D5**) | `69,520 / 621,960` | `158,452 / 1,042,440` | `POST_DH.sz_i` & `POST_TEMP1.sz_i`: `18,875 / 26,280` each (`max \|diff\| = 0.049684`, `85,777,553,489,348` ULP)<br>`POST_SAL.sz_i` & `POST_TEMP2.sz_i`: `25,884 / 26,280` each (`max \|diff\| = 0.121774`, `132,185,672,548,278` ULP)<br>`POST_DO.sv_i` & `EXIT.sv_i`: `8,603 / 8,760` each (`max \|diff\| = 0.040948`, `46,103,889,525,713` ULP)<br>`POST_DO.szv_i` & `EXIT.szv_i`: `25,864 / 26,280` each (`max \|diff\| = 0.029849`, `186,916,963,695,079` ULP) | `True` (note: `grad(S_bulk)` becomes `0.0` because `nn_icesal=4` overwrites `s_i` from `sz_i`) |

### 6.2 Production `step_sea_ice(thermo_scheme='si3_bl99')`, Lane-3 certified cards, O1–O7 admissions, and default (non-SI3) sea-ice runs

| Target / Card | Rewiring Option | `unequal / total` elements | `max \|diff\|` | `max ULP` | Per-output breakdown | `jax.grad` finite? |
|---|---|---:|---:|---:|---|---|
| **Production `step_sea_ice(thermo_scheme='si3_bl99')`** (`SI3ColumnState` 9 fields + `TileResponse` 4 fields) | **Option A** | `0 / 19` | `0.0` | `0` | All 9 `SI3ColumnState` fields (`h_ice`, `h_snow`, `T_surface`, `concentration`, `S_bulk`, `S_layers`, `age_volume`, `e_ice`, `e_snow`) and all 4 `TileResponse` fields (`T_sfc`, `lhflx`, `lhflx_exchange`, `surface_mass_flux`) match `HEAD` at `0` ULP | `True` |
| **Production `step_sea_ice(thermo_scheme='si3_bl99')`** (`SI3ColumnState` 9 fields + `TileResponse` 4 fields) | **Option B** | `4 / 19` | `1.322648e-2` | `14,891,695,112,905` | `S_bulk`: `1 / 1` (`max \|diff\| = 4.630031e-3`, `5,212,950,989,271` ULP); `S_layers`: `3 / 3` (`max \|diff\| = 1.322648e-2`, `14,891,695,112,905` ULP); other 11 outputs `0` ULP | `True` |
| **Lane-3 `ICE_ADV1D`** (Prather 1D) | **Option A or B** | `0 / 43,659` | `0.0` | `0` | `0` ULP across all 63 Prather moment fields (`693` cells each) | `True` |
| **Lane-3 `ICE_ADV2D`** (Prather 2D) | **Option A or B** | `0 / 169,744` | `0.0` | `0` | `0` ULP across all 68 Prather moment fields (`50 x 50` grid) | `True` |
| **Lane-3 `ICE_ADV2D_RHG`** (EVP + Prather 2D) | **Option A or B** | `0 / 10,609` | `0.0` | `0` | `0` ULP across `u_ice`, `v_ice`, and advected state (`103 x 103` grid) | `True` |
| **ORCA2 SAS `O1`, `O2`, `O3`, `O4`, `O5`, `O7` admissions** | **Option A or B** | `0` unequal | `0.0` | `0` | Do not call `si3_column_step_arrays` or O8–O12 (`ln_icethd=False`); every admitted digest and gate output is unchanged at `0` ULP | `True` |
| **Default `_step_slab`** (`thermo_scheme='zero_layer'`) | **Option A or B** | `0 / 12` | `0.0` | `0` | `h_ice` (`0/4`), `h_snow` (`0/4`), `T_surface` (`0/4`), `TileResponse` (`0` ULP) | `True` |
| **Default `_step_dynamic_v2` 1-cat (`ponds.scheme='cesm'`)** | **Option A or B** | `0 / 4` | `0.0` | `0` | All `DynamicSeaIceState` and `TileResponse` fields unchanged (`0` ULP) | `True` |
| **Default `_step_dynamic_v2` 5-cat (`ponds.scheme='cesm'`)** | **Option A or B** | `0 / 20` | `0.0` | `0` | All `DynamicSeaIceState` and `TileResponse` fields unchanged (`0` ULP) | `True` |
| **`step_ponds(scheme='si3_lev')` vs certified `si3_ice_thd_pnd` on active ice** (`2x2x5`, `a_i=0.16, h_i=1.4`) | **Rewired to `si3_ice_thd_pnd`** | `0 / 64` | `0.0` | `0` | `pond_area`: `0 / 20`, `h_ip`: `0 / 20`, `h_il`: `0 / 20`, `wfx_pnd`: `0 / 4` (`0` ULP across all outputs) | `True` |
| **`step_ponds(scheme='si3_lev')` vs certified `si3_ice_thd_pnd` on vanished-ice cell** (`a_i[0,0,:]=0, h_i[0,0,:]=0` with pre-melt `v_ip=2.88e-3, v_il=2.4e-4`) | **Rewired to `si3_ice_thd_pnd`** | `1 / 64` | `4.446000e-3` | `4,571,775,911,163,164,986` | `pond_area`: `0 / 20`, `h_ip`: `0 / 20`, `h_il`: `0 / 20`; **`wfx_pnd`: `1 / 4` cells differ** (`0.004446 kg m^-2 s^-1` drained into `wfx_pnd[0,0]` by `si3_ice_thd_pnd` vs `0.0` in `step_ponds`) | `True` |

---

## 7. Concrete Rewiring Proposal (M4 Closure, Review Finding (a) Item 3)

### 7.1 Proposed single-implementation architecture (Option A — Recommended)

1. **Unify O9 ZDF, DH, and TEMP1 (`packages/ice/legoesm/ice/bitz_lipscomb.py`)**:
   - Delete the duplicate wrapper `_si3_zdf_bl99_step` (`packages/ice/legoesm/ice/bitz_lipscomb.py:545-804`) and the `595`-line duplicate `_dh_step` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1015-1269`, together with `_c1d_dh_via_core`).
   - Add `t_i: jax.Array | None = None` to `SI3ColumnArrays` and `SI3ColumnState` (`packages/ice/legoesm/ice/state.py:33-75`) so the diagnostic layer temperature produced by `si3_ice_thd_temp` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1272-1277`) is carried through `POST_TEMP1` and `POST_TEMP2` rather than aliasing `post_temp1 = post_dh` and `post_temp2 = post_sal`.
   - Have `si3_column_step_arrays` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1811-1998`) call the certified `si3_ice_thd_zdf_bl99` (`packages/ice/legoesm/ice/bitz_lipscomb.py:834-919`), `si3_ice_thd_dh` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1288-1311`), and `si3_ice_thd_temp` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1272-1277`) (via `si3_ice_thd_category_stages`, `packages/ice/legoesm/ice/bitz_lipscomb.py:1344-1382`), reading `rn_hinew`, `ln_frazil`, `rn_snwblow`, `rn_sinew`, and `nn_icesal` from `SI3ThermoConfig` (`packages/ice/legoesm/ice/config.py:291-338`).
2. **Unify O10 SAL + TEMP2 (`packages/ice/legoesm/ice/bitz_lipscomb.py:1705-1782`, `:1785-1808`)**:
   - In `si3_ice_thd_sal` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1705-1782`), support both NEMO `SELECT CASE (nn_icesal)` branches (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:311-683`):
     - `nn_icesal == 4`: call `_si3_ice_thd_sal4_core` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1491-1594`, `:1597-1668`), generalized from hardcoded `10` layers to `nlay_i = sz_i.shape[-1]`.
     - `nn_icesal == 2`: run the certified C1D bulk-salinity relaxation + `option2_salinity_profile` (`packages/ice/legoesm/ice/bitz_lipscomb.py:163-211`, `:1750-1755`), returning `SI3SalOutputs`.
   - Have `si3_column_step_arrays` call `si3_ice_thd_o10_category_stages` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1785-1808`) -> `si3_ice_thd_sal` + 2nd `si3_ice_thd_temp`, removing the inline O10 block at `packages/ice/legoesm/ice/bitz_lipscomb.py:1750-1755`.
3. **Wire O12 `si3_ice_thd_pnd` (`packages/ice/legoesm/ice/si3_thd_pnd.py:575-781` and `packages/ice/legoesm/ice/ponds.py:34-311`)**:
   - In `step_ponds(scheme='si3_lev')` (`packages/ice/legoesm/ice/ponds.py:34-311`), accept optional pre-melt `a_ip_prev`, `v_ip_prev`, `v_il_prev` and, when invoked on 5-category, 10-layer SI3 state (`jpl=5, nlay_i=10`), delegate directly to `si3_ice_thd_pnd` (`packages/ice/legoesm/ice/si3_thd_pnd.py:575-781`) so vanished-ice pond drainage (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:162-264`) is executed by the certified O12 function; for 1-category / 0-layer `_thermo_v2` calls, apply the same vanished-ice `wfx_pnd` drainage before `si3_pnd_lev`.

### 7.2 Summary of what every certified result becomes under Option A vs Option B

| Certified Result / Gate | Current (`HEAD` `e79f43e5f994`) | Under **Option A (Recommended)** | Under **Option B (Forced `nn_icesal=4` on C1D)** |
|---|---|---|---|
| **O8–O12 100-step `ORCA2_SAS_ICE` master record** (`nemo_si3_o8_o12_master_scorer.py`) | `PASS_ALL_O8_O12_0_ULP` (`0` ULP across all 5 rungs) | **`PASS_ALL_O8_O12_0_ULP` (`0` ULP, unchanged)** — and the scored functions are now the single production implementation (M4 satisfied) | `PASS_ALL_O8_O12_0_ULP` (`0` ULP) |
| **D5 `C1D_OMIP_L3` 1-year (`8,760` steps) trace** (`nemo_si3thd_d5_platform_probe.py`) | SHA-256 `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2` (`0` ULP vs NEMO `C1D_OMIP_L3` oracle) | **SHA-256 `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2` (`0 / 1,042,440` changed, `0` ULP, unchanged)** | **BREAKS**: `78176c01...` (`115,724 / 1,042,440` elements changed in B1) or `defb511d...` (`158,452 / 1,042,440` elements changed in B2) |
| **Lane-3 certified cards (`ICE_ADV1D`, `ICE_ADV2D`, `ICE_ADV2D_RHG`)** | `0` ULP | **`0` ULP (`0` elements changed)** | `0` ULP (`0` elements changed) |
| **ORCA2 SAS `O1`, `O2`, `O3`, `O4`, `O5`, `O7` admissions** | `ADMITTED` (`0` ULP) | **`ADMITTED` (`0` ULP, `0` elements changed)** | `ADMITTED` (`0` ULP, `0` elements changed) |
| **Default (non-SI3) sea-ice runs (`_step_slab`, `_step_dynamic_v2` 1-cat & 5-cat `cesm`)** | Baseline (`0` ULP) | **`0 / 36` elements changed (`0` ULP), all gradients finite** | `0 / 36` elements changed (`0` ULP), all gradients finite |
| **`step_ponds(scheme='si3_lev')` 5-cat** | Active ice matches `si3_ice_thd_pnd` at `0` ULP; vanished-ice cell drops pre-melt pond drainage (`wfx_pnd = 0.0`) | **Active ice `0 / 64` changed (`0` ULP); vanished-ice cell drains `(v_ip + v_il) * rhow * r1_Dt_ice` into `wfx_pnd` (`1 / 4` cells, `max \|diff\| = 0.004446 kg m^-2 s^-1`) matching NEMO `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:162-264`** | Same as Option A |

---

## 8. DECISION 186 — Option A Implementation, Production-Path Scoring, and Final-Code Re-Measurement

Following **`DECISION 186`** (User approval: *"Yes"*, 2026-10-27), Option A was implemented in the repository (`ae10c0c527dfe3249767c29b61f3504fdf71e90f`) and every required measurement was re-executed on the final committed code.

### 8.1 Single-implementation production rewiring (Model-Code Checklist M4)

1. **Duplicate implementations deleted (`packages/ice/legoesm/ice/bitz_lipscomb.py:1811-1998`)**:
   - Deleted `_si3_zdf_bl99_step`, `_piecewise_remap`, `_nemo_snow_enthalpy_remap`, `_nemo_ice_enthalpy_remap`, `_c1d_dh_via_core`, and the `595`-line duplicate `_dh_step` from `packages/ice/legoesm/ice/bitz_lipscomb.py` (`-998` lines).
   - Added `nn_icesal == 2` support inside the certified `si3_ice_thd_sal` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1705-1782`) and `si3_ice_thd_o10_category_stages` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1785-1808`), dispatching on `namelist["nn_icesal"]` (`2` for `C1D_OMIP_L3`, `4` for `ORCA2_SAS_ICE`) matching NEMO `SELECT CASE (nn_icesal)` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:311-683`).
   - Rewired `si3_column_step_arrays` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1811-1998`, called by `step_sea_ice(thermo_scheme='si3_bl99')` at `packages/ice/legoesm/ice/sea_ice.py:286-298,500-517`) to call `si3_ice_thd_o10_category_stages` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1785-1808`) -> `si3_ice_thd_category_stages` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1344-1382`, executing `si3_ice_thd_zdf_bl99` at `:834-919`, `si3_ice_thd_dh` at `:1288-1311`, and 1st `si3_ice_thd_temp` at `:1272-1277`) -> `si3_ice_thd_sal` (`:1705-1782`) + 2nd `si3_ice_thd_temp` (`:1272-1277`) -> `_si3_column_thd_da` (`:2077`, calling `si3_ice_thd_da` at `packages/ice/legoesm/ice/si3_thd_da.py:65-185`) -> `si3_ice_thd_frazil` (`packages/ice/legoesm/ice/bitz_lipscomb.py:2024-2042`).
2. **O12 production pond entry point wired (`packages/ice/legoesm/ice/ponds.py:34-311` and `packages/ice/legoesm/ice/si3_thd_pnd.py:575-781`)**:
   - Rewired `step_ponds(scheme="si3_lev")` (`packages/ice/legoesm/ice/ponds.py:34-311`) to construct the O12 entry dictionary and call the certified `si3_ice_thd_pnd` (`packages/ice/legoesm/ice/si3_thd_pnd.py:575-781`), supporting both 5-category 10-layer `ORCA2_SAS_ICE` states (`e_i` shape `(..., 10, 5)`, `sz_i` shape `(..., 10, 5)`) and 1-category / 5-category `_thermo_v2` states (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:162-264`).

### 8.2 100-step O8–O12 master-record score through the production entry points (`o8_o12_master_score.json`)

Re-running `scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1-2147` across all `100` steps (`kt = 1..100`) on the final committed code (`ae10c0c527dfe3249767c29b61f3504fdf71e90f`) produced `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round21/o8_o12_master_score.json` (`137,492` bytes, SHA-256 `2ffd0cd63ac8503cd0abe6a2751834fc6bb19f2f76ec2b6e92248f220c8b7bd1`). Alongside the unchanged `0`-ULP stage-kernel scores in Sections 2–3, the scorer now evaluates the production entry points directly on every step and category:

| Production Entry Point Scored | Scoring Block | Categories & Steps | Fields Scored | Total `unequal / total` elements | `max \|diff\|` | `max ULP` | First non-bit `(kt, category, cell, field)` | Verdict |
|---|---|---|---:|---:|---:|---:|---|---|
| `si3_column_step_arrays` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1811-1998`, `nn_icesal=4` from `namelist_ice`) | `O11.production_column_step_arrays_o9_o11` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1471-1547`) | `kt = 1..100, jl = 1..5` (`500` category frames, `1,917,071` active cells) | `72` (`14` Kind `4` ZDF + `44` Kind `5` DH + `1` Kind `6` TEMP1 + `3` Kind `7` SAL + `1` Kind `8` TEMP2 + `6` Kind `9` DA + `3` `SI3StepTrace` `POST_TEMP1`/`POST_SAL`/`POST_TEMP2` boundary arrays) | **`0 / 306,731,360`** | **`0.0`** | **`0`** | **`None`** | **`PASS_0_ULP`** |
| `step_ponds(scheme="si3_lev")` (`packages/ice/legoesm/ice/ponds.py:34-311` -> `si3_ice_thd_pnd` at `packages/ice/legoesm/ice/si3_thd_pnd.py:575-781`) | `O12.production_step_ponds_si3_lev` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1549-1581`) | `kt = 1..100, jl = 0` (`100` full-grid records, `180 x 148 x 5`) | `96` (all `8` updated pond/total/flux fields `at_i`, `a_ip`, `v_ip`, `h_ip`, `v_il`, `h_il`, `a_ip_frac`, `wfx_pnd` + `88` pass-through Kind `11` fields) | **`0 / 1,145,653,900`** | **`0.0`** | **`0`** | **`None`** | **`PASS_0_ULP`** |

### 8.3 Final-code re-measurement across all certified cards, admissions, and default runs (`r21_option_a_final_remeasurement.json`)

All certified targets were re-measured on the final committed code (`ae10c0c527dfe3249767c29b61f3504fdf71e90f`) on CPU (`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, `PrecisionPolicy.fp64(transcendentals="libm")`), recorded in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round21/r21_option_a_final_remeasurement.json` (`4,979` bytes, SHA-256 `3b886af81161f2f0cf56c37ce9df39e67a87b634c3cfb731a9fe2e4697884ebb`):

| Re-Measured Target (Final Committed Code `ae10c0c527df`) | Result / Digest | `unequal / total` vs pinned / parent `3d5446cca2b4` | `max \|diff\|` | `max ULP` | `jax.grad` finite? | Status |
|---|---|---:|---:|---:|---|---|
| **D5 `C1D_OMIP_L3` 1-year (`8,760` steps) trace** (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:806-1883`) | SHA-256 `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2` (`MATCHES_PINNED`, all 8 stage digests bit-exact) | `0 / 621,960` rows (`0 / 1,042,440` elements) | `0.0` | `0` | `True` (all 9 state/forcing inputs) | **`MATCHES_PINNED` (`0` ULP)** |
| **Lane-3 `ICE_ADV1D` certified card** (`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2091-2107`) | All 63 Prather moment fields (`693` cells each) | `0 / 43,659` | `0.0` | `0` | `True` | **`PASS` (`0` ULP)** |
| **Lane-3 `ICE_ADV2D` certified card** (`packages/ice/legoesm/ice/fidelity/nemo_adv2d_testcase_recipe.py:388-613`) | All 68 Prather moment fields (`50 x 50` grid) | `0 / 169,744` | `0.0` | `0` | `True` | **`PASS` (`0` ULP)** |
| **Lane-3 `ICE_ADV2D_RHG` certified card** (`packages/ice/legoesm/ice/fidelity/nemo_adv2d_rhg_testcase_recipe.py:152`) | EVP + Prather 2D (`103 x 103` grid) | `0 / 10,609` | `0.0` | `0` | `True` | **`PASS` (`0` ULP)** |
| **ORCA2 SAS `O1`, `O2`, `O3`, `O4`, `O5`, `O7` admissions** (`packages/ice/legoesm/ice/si3_orca2_state.py:3807-3960`, `:4528-4812`) | `O1`: `PASS` (`0` ULP); `O2`: `PASS` (`0` ULP); `O3`: `PASS` (`0` ULP, `120` records); `O4`: `PASS` (`0` ULP); `O5`: `PASS` (`0` ULP); `O7`: `PASS` (`0` ULP, `25` records) | `0` unequal across all admitted records | `0.0` | `0` | `True` | **`PASS` (`0` ULP)** |
| **Production `step_sea_ice(thermo_scheme='si3_bl99')`** (`packages/ice/legoesm/ice/sea_ice.py:286-298,500-517`) | 9 `SI3ColumnState` fields + 4 `TileResponse` fields vs parent `3d5446cca2b4` | `0 / 19` | `0.0` | `0` | `True` (`grad_h_ice = 0.996401`, `grad_h_snow = 0.146247`) | **`PASS` (`0` ULP)** |
| **Default `_step_slab` (`thermo_scheme='zero_layer'`)** | Slab state (`h_ice`, `h_snow`, `T_surface`) + `TileResponse` vs parent `3d5446cca2b4` | `0 / 12` | `0.0` | `0` | `True` (`grad_h_ice = 1.999999`) | **`PASS` (`0` ULP)** |
| **Default `_step_dynamic_v2` 1-cat (`ponds.scheme='cesm'`)** | `DynamicSeaIceState` + `TileResponse` vs parent `3d5446cca2b4` | `0 / 4` | `0.0` | `0` | `True` (`grad_h_ice = 3.980614`) | **`PASS` (`0` ULP)** |
| **Default `_step_dynamic_v2` 5-cat (`ponds.scheme='cesm'`)** | 5-cat `DynamicSeaIceState` + `TileResponse` vs parent `3d5446cca2b4` | `0 / 20` | `0.0` | `0` | `True` (`grad_h_ice = 19.726761`) | **`PASS` (`0` ULP)** |
| **`step_ponds(scheme='si3_lev')` vs certified `si3_ice_thd_pnd` on active ice & vanished-ice cell** (`packages/ice/legoesm/ice/ponds.py:34-311`) | Active ice (`a_i=0.16, h_i=1.4`) and vanished-ice cell (`a_i[0,0,:]=0, h_i[0,0,:]=0`, `wfx_pnd[0,0] = 0.004446 kg m^-2 s^-1`) | Active ice: **`0 / 64`**<br>Vanished-ice cell: **`0 / 64`** | `0.0` | `0` | `True` (`grad_h_ip = 4.351162`, `grad_h_ip_vanished = 104.240741`) | **`PASS` (`0` ULP on both active and vanished ice)** |

---

## 9. Resume 3 & 4 — Explicit `SI3ThermoConfig` Enforcement, Phase-2 Gate Routing, D5 Year Gate Fix, and Detached Re-Verification (`e349abaffc80`, `f81fd9c83d66`)

1. **No silent default in `si3_column_step_arrays` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1811-1998`)**:
   - Removed the fallback `nn_icesal = 2 if si3 is None else int(si3.salinity_scheme)`; `si3_column_step_arrays` now raises `ValueError("si3_column_step_arrays requires an explicit SI3ThermoConfig (si3); no default")` if `si3` or `si3.salinity_scheme` is `None`.
   - `C1D_OMIP_L3` (`build_c1d_omip_l3_card`) parses `nn_icesal = 2`, `rn_sinew = 0.5`, `rn_snwblow = 0.5`, `rn_hinew = 0.05`, and `ln_frazil = False` explicitly from `c1d_omip_l3_namelist_ice_cfg` and passes `si3=card.config.si3` at every call site (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:220-858`).
2. **Phase-2 gate routed through production kernels & AST duplicate-arithmetic guard (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:220-858`)**:
   - Routed `nemo_si3thd_phase2_gate.py` through the production `si3_ice_thd_zdf_bl99`, `si3_ice_thd_dh`, `si3_ice_thd_sal`, `si3_ice_thd_frazil`, and `si3_column_step_arrays` kernels (`-591` duplicate lines removed) and added `_assert_no_duplicate_o8_o12_arithmetic()` with an AST-level plant (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:220-858`).
   - Updated `nemo_si3thd_phase2b_year_gate.py` ablation steps to call `phase2._si3_column_step_arrays_ablation(..., si3=card.config.si3, ...)`.
3. **Detached re-verification results (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/si3_r21/final_checks.log`)**:
   - **D5 `C1D_OMIP_L3` year gate (`nemo_si3thd_phase2b_year_gate.py`)**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round21/d5_phase2b_year_gate_resume3.json` (`13,016,617` bytes, SHA-256 `eedbfd9862eb574d863dbca2e35655a206686f6d092180f9071830dacb797ce1`), `status = "DEBT"` (`629` over-bar rows confined to the `kt = 5842..5867` subnormal snow window; `0` normal-step over-bar rows).
   - **D5 `C1D_OMIP_L3` platform probe & full-year hash (`scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:806-1883`)**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round21/d5_platform_probe_resume3.json` (SHA-256 `53c7aaa90d16cdc4f3dcd873497a499387ed804706abe4470428bb64744a0ffe`) and `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round21/r21_resume4_final_remeasurement.json` (SHA-256 `8e1629cd2cc7fb0d5f1074ac554a74b34506e203858ffb106b15ffe36402cd0d`):
     ```text
     24125:        "full_year_output_sha256": "b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2",
     18:  "platform_disposition": "D5_PLATFORM_FLOOR",
     ```
   - **O8–O12 100-step master-record production-entry score (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1471-1547`, `scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1549-1581`)**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round21/o8_o12_master_score.json` (`137,492` bytes, SHA-256 `2ffd0cd63ac8503cd0abe6a2751834fc6bb19f2f76ec2b6e92248f220c8b7bd1`), `overall_verdict = "PASS_ALL_O8_O12_0_ULP"` (`O11.production_column_step_arrays_o9_o11`: `0 / 306,731,360` unequal, `0` ULP; `O12.production_step_ponds_si3_lev`: `0 / 1,145,653,900` unequal, `0` ULP).
   - **Lane-3 (`ICE_ADV1D`, `ICE_ADV2D`, `ICE_ADV2D_RHG`) and `O1`–`O7` results**: `ICE_ADV1D` (`0 / 43,659` unequal, `0` ULP, finite gradients), `ICE_ADV2D` (`0 / 169,744` unequal, `0` ULP, finite gradients), `ICE_ADV2D_RHG` (`0 / 10,609` unequal, `0` ULP, finite gradients), and `O1`, `O2`, `O3`, `O4`, `O5`, `O7` (`PASS` / `PASS_SELF_TEST`, `0` ULP across all admitted records).
   - **independent review: operator**
