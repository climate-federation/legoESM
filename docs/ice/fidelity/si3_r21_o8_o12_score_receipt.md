# SI3 Ladder Round 21 (`si3_r21`) — O8, O9, O10, O11, O12 Master-Record Scoring Receipt

- **Date**: 2026-10-27
- **Lane / Round**: `si3` / Round 21 (`si3_r21`), authored by Argon
- **Branch**: `fidelity/nemo-testcases-si3-ladder` (parent `3d5446cca2b4`)
- **Preregistration**: `docs/ice/fidelity/si3_r21_o8_o12_score_preregister.md:1-51` (committed at `9b814d537d` before reading any binary record)
- **Master record scored**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/oracle_o8_o16_master` (`docs/ice/fidelity/si3_r18_o8_o16_receipt.md:154-180`, `DECISION 173`)
- **Scoring driver**: `scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1-1501`
- **Direct test suite**: `tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:1-154` (`9 passed in 421.14s`)
- **Full 100-step JSON score report**: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round21/o8_o12_master_score.json` (`105,409` bytes, SHA-256 `cfca3da8a142b2ec670e856450b76a776d7e2b7dc7028e08361460a8504ca2a9`)
- **Verdict**: **`PASS_ALL_O8_O12_0_ULP`** — every EXIT field of rungs **O8, O9, O10, O11, and O12** matches the admitted 100-step NEMO 5.0.2 `ORCA2_SAS_ICE` master record at **`0` ULP** (`unequal == 0`, `max |diff| == 0.0`, `first_non_bit == None`, including IEEE-754 signed-zero bit 63) across all `100` steps (`kt = 1..100`) and all categories (`jl = 1..5` and `jl = 0`). No model code or configuration was changed.
- **independent review: operator**

---

## 1. Master-record admission and resolved-namelist verification

Before scoring any rung, `verify_master_admission_manifests` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:296-351`) and `parse_resolved_ice_namelist` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:255-293`) verify the admitted master record at `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/oracle_o8_o16_master` (`docs/ice/fidelity/si3_r18_o8_o16_receipt.md:154-180`, `scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o16_master_gate.py:40-55`):

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

All 60 resolved namelist parameters parsed from `record_100step/output.namelist.ice` and `record_100step/output.namelist.dyn` (`ln_icedH=True`, `ln_icedA=True`, `ln_icedO=True`, `ln_leadhfx=True`, `rn_hinew=0.1`, `ln_frazil=False`, `rn_maxfraz=1.0`, `rn_vfraz=0.417`, `rn_Cfraz=5.0`, `rn_himin=0.1`, `rn_kappa_i=1.0`, `rn_kappa_s=10.0`, `ln_zdf_BL99=True`, `ln_cndi_U64=False`, `ln_cndi_P07=True`, `rn_cnd_s=0.31`, `nn_qtrice=0`, `ln_zdf_chkcvg=False`, `ln_cndflx=False`, `ln_cndemulate=False`, `nn_snwfra=2`, `rn_beta=1.0`, `rn_dmin=8.0`, `rn_snwblow=0.66`, `nn_icesal=4`, `rn_icesal=4.0`, `rn_sinew=0.75`, `rn_simin=0.1`, `ln_drainage=True`, `ln_flushing=True`, `nn_sal_scheme=1`, `nn_drainage=50`, `nn_flushing=5`, `rn_flushrate=0.3`, `rn_vbrc=0.05`, `nn_liquidus=2`, `rn_sal_himin=0.1`, `rn_alpha_RJW=0.03`, `rn_Rc_RJW=8.0`, `rn_alpha_GN=0.00046`, `rn_Rc_GN=8.1`, `rn_alpha_CW=5e-07`, `ln_sal_chk=False`, `ln_pnd=True`, `ln_pnd_LEV=True`, `rn_apnd_min=0.15`, `rn_apnd_max=0.8`, `rn_pnd_flush=0.1`, `ln_pnd_lids=True`, `ln_pnd_rain=True`, `ln_pnd_CST=False`, `rn_apnd=0.2`, `rn_hpnd=0.05`, `ln_pnd_TOPO=False`, `ln_pnd_alb=True`, `rn_pnd_hl_min=0.005`, `rn_pnd_hl_max=0.015`, `ln_virtual_itd=False`, `ln_icediachk=False`, `ln_drgice_imp=False`) agree bit-for-bit with the 59-element binary scalar header of every scored record (`_verify_record_scalars_against_namelist` in `scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:455-512`), alongside `rDt_ice = 21600.0`, `r1_Dt_ice = 1.0 / 21600.0`, and `r1_nlay_i = 0.1`.

---

## 2. Per-rung 100-step 0-ULP summary table (O8, O9, O10, O11, O12)

All evaluations ran on CPU (`jax.default_backend() == "cpu"`), `fp64` (`JAX_ENABLE_X64=1`), `PrecisionPolicy.fp64(transcendentals="libm")`, under `jax.jit` (`_build_compiled_kernels` in `scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:629-808`) with dynamic array inputs passed as runtime JIT arguments:

| Rung | Stage & NEMO `ppsrc` reference | Production function(s) (`jax.jit`) | Entry -> Exit kinds (`kt`, `jl`) | Exit fields scored | Total `unequal / total` elements | `max \|diff\|` | `max ULP` | First non-bit `(kt, category, cell, field)` | Verdict |
|---|---|---|---|---:|---:|---:|---:|---|---|
| **O8** | Frazil collection thickness (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:393-477`; `docs/ice/fidelity/si3_r15_o8_receipt.md:1-195`) | `si3_ice_thd_frazil` (`packages/ice/legoesm/ice/bitz_lipscomb.py:2822`) | Kind `1` -> Kind `2` (`kt=1..100, jl=0`) | `2` (`ht_i_new`, `fraz_frac`) | `0 / 5,328,000` | `0.0` | `0` | `None` | `PASS_0_ULP` |
| **O9** | Vertical conduction + thickness growth/melt + enthalpy remap + 1st diagnostic `T_i` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:201-205`; `docs/ice/fidelity/si3_r16_o9_receipt.md:1-336`) | `si3_ice_thd_zdf_bl99` (`packages/ice/legoesm/ice/bitz_lipscomb.py:936`), `si3_ice_thd_dh` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1537`), `si3_ice_thd_temp` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1521`), `si3_ice_thd_category_stages` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1593`) | Kind `3` (+ Kind `21`) -> Kinds `4`, `5`, `6` (`kt=1..100, jl=1..5`) + Kind `17` (`kt in (1,2,100), jl=1..5`) | Kind `4` (`16`), Kind `5` (`67`), Kind `6` (`67`), Composed (`59`), Kind `17` (`12` + `iconv`), Gather (`9`) | Kind `4`: `0 / 80,516,982`<br>Kind `5`: `0 / 195,541,242`<br>Kind `6`: `0 / 195,541,242`<br>Composed: `0 / 239,633,875`<br>Kind `17`: `0 / 3,326,220` (`15/15 iconv`)<br>Gather: `0 / 59,429,201` | `0.0` | `0` | `None` | `PASS_0_ULP` |
| **O10** | Gravity drainage + flushing (`nn_icesal=4`) + 2nd diagnostic `T_i` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:311-683`; `docs/ice/fidelity/si3_r19_o10_receipt.md:1-245`) | `si3_ice_thd_sal` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1949-1986`), `si3_ice_thd_temp` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1521`), `si3_ice_thd_o10_category_stages` (`packages/ice/legoesm/ice/bitz_lipscomb.py:1997-2020`) | Kind `6` -> Kind `7` -> Kind `8` (`kt=1..100, jl=1..5`) | Kind `7` (`67`), Kind `8` (`67`), Unpadded composed spot-check (`4`) | Kind `7`: `0 / 195,541,242`<br>Kind `8`: `0 / 195,541,242`<br>Unpadded `kt=1`: `0 / 413,270` | `0.0` | `0` | `None` | `PASS_0_ULP` |
| **O11** | Lateral melting `ice_thd_da` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:167-170`; `docs/ice/fidelity/si3_r17_o11_receipt.md:1-251`) | `si3_ice_thd_da` (`packages/ice/legoesm/ice/si3_thd_da.py:65`) | Kind `8` -> Kind `9` (`kt=1..100, jl=1..5`) | Kind `9` (`67`: `6` updated + `61` pass-through) | `0 / 195,541,242` | `0.0` | `0` | `None` | `PASS_0_ULP` |
| **O12** | Level-ice melt ponds `ice_thd_pnd` / `pnd_LEV` / `ice_perm_eff` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:162-264`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:329-582`; `docs/ice/fidelity/si3_r20_o12_receipt.md:1-188`) | `si3_ice_thd_pnd` (`packages/ice/legoesm/ice/si3_thd_pnd.py:575-763`) / `step_ponds` (`packages/ice/legoesm/ice/ponds.py:34-212`) | Kind `10` -> Kind `11` (`kt=1..100, jl=0`) | Kind `11` (`96`: `8` updated + `88` pass-through) | `0 / 1,105,693,900` | `0.0` | `0` | `None` | `PASS_0_ULP` |

---

## 3. Per-rung field-by-field 0-ULP tables

### 3.1 Rung O8 (`Kind 1 -> Kind 2`, `si3_ice_thd_frazil`, `kt = 1..100, jl = 0`)

Scored across all `100` steps (`180 x 148 = 26,640` interior cells per step; `scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:859-876`):

| Exit field (Kind `2`) | Extent / shape per record | Role | `unequal / total` | `max \|diff\|` | `max ULP` | First non-bit `(kt, category, cell, field)` |
|---|---|---|---:|---:|---:|---|
| `ht_i_new` | `I2` `(180, 148)` | Updated (`si3_ice_thd_frazil.ht_i_new`) | `0 / 2,664,000` | `0.0` | `0` | `None` |
| `fraz_frac` | `I2` `(180, 148)` | Updated (`si3_ice_thd_frazil.fraz_frac`) | `0 / 2,664,000` | `0.0` | `0` | `None` |

### 3.2 Rung O9 (`Kind 3 -> Kind 4 -> Kind 5 -> Kind 6` + `Kind 17` + `Kind 21 -> Kind 3` gather)

Across `kt = 1..100` and `jl = 1..5` (`500` category records), the sum of gathered active cells is $\sum_{kt=1}^{100}\sum_{jl=1}^{5} \mathrm{npti}(kt, jl) = 1,917,071$ (`G1`), giving $19,170,710$ ice-layer cells (`GI`, `10` layers) and $9,585,355$ snow-layer cells (`GS`, `5` layers) (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1031-1254`).

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
| **Composed `si3_ice_thd_category_stages` (all 59 updated outputs: `14` ZDF + `44` DH + `1` TEMP1)** | `G1`, `GI`, `GS` | End-to-end Kind `3` -> Kinds `4, 5, 6` under single `jax.jit` | **`0 / 239,633,875`** | **`0.0`** | **`0`** | **`None`** |
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

### 3.3 Rung O10 (`Kind 6 -> Kind 7 -> Kind 8`, `si3_ice_thd_sal`, `si3_ice_thd_temp`, `si3_ice_thd_o10_category_stages`, `kt = 1..100, jl = 1..5`)

Scored across all `500` `(kt, jl)` category frames (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1143-1168`, `scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1256-1276`):

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

Scored across all `500` `(kt, jl)` category frames (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1278-1289`):

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

Scored across all `100` steps (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1291-1313`):

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
| `STATE_I3` pass-through (`a_i`, `v_i`, `h_i`, `v_s`, `h_s`, `t_su`, `s_i`, `sv_i`, `o_i`, `oa_i`, `t_si`: 11 fields) | `I3` `(180, 148, 5)` | Pass-through from Kind `10` | `0 / 13,320,000` | `0.0` | `0` | `None` |
| `STATE_II4` pass-through (`t_i`, `e_i`, `sz_i`, `szv_i`: 4 fields) | `II4` `(180, 148, 10, 5)` | Pass-through from Kind `10` | `0 / 133,200,000` | `0.0` | `0` | `None` |
| `STATE_IS4` pass-through (`t_s`, `e_s`: 2 fields) | `IS4` `(180, 148, 5, 5)` | Pass-through from Kind `10` | `0 / 66,600,000` | `0.0` | `0` | `None` |
| `PRE_AGG_TOTALS` pass-through (`vt_i`, `vt_s`, `st_i`, `at_ip`, `vt_ip`, `vt_il`, `tm_i`, `tm_s`, `tm_si`, `sm_i`, `tm_su`: 11 fields; `at_i` updated above) | `I2` `(180, 148)` | Pass-through from Kind `10` | `0 / 2,664,000` | `0.0` | `0` | `None` |
| `POND` pass-through (`a_ip_eff`, `dh_i_sum_2d`, `dh_s_sum_2d`: 3 fields) | `I3` `(180, 148, 5)` | Pass-through from Kind `10` | `0 / 13,320,000` | `0.0` | `0` | `None` |
| `POND_FORCING` pass-through (`tprecip`, `sprecip`: 2 fields) | `I2` `(180, 148)` | Pass-through from Kind `10` | `0 / 2,664,000` | `0.0` | `0` | `None` |
| `GLOBAL_AUX` pass-through (`tmask`, `ssmask`, `sss_m`, `sst_m`, `frld`, `t_bo`: 6 `I2` fields; `e1e2t`, `wmask_srf`, `umask_srf`, `vmask_srf`: 4 `F2` fields; `hi_max`: 1 `C0` field; ` hi_mean`: 1 `C1` field) | `I2`, `F2`, `C0`, `C1` | Pass-through from Kind `10` | `I2`: `0 / 2,664,000`<br>`F2`: `0 / 2,796,800`<br>`C0`: `0 / 600`<br>`C1`: `0 / 500` | `0.0` | `0` | `None` |
| `FLUX` pass-through (43 `I2` fields; `wfx_pnd` updated above) | `I2` `(180, 148)` | Pass-through from Kind `10` | `0 / 2,664,000` | `0.0` | `0` | `None` |
| **Kind `11` total (96 fields)** | — | — | **`0 / 1,105,693,900`** | **`0.0`** | **`0`** | **`None`** |

---

## 4. Direct test and planted-violation verification

`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:1-154` exercises `scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1-1501` across 9 test cases (log saved at `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round21/pytest_o8_o12_master_scorer.log`, `1,990` bytes, SHA-256 `081f751b3b9c7b8adfb092e5b0e1658944d55b2632ec63a15a6d54eb17590078`):

| Test / Plant | Target | Expected fail-closed behavior | Observed result |
|---|---|---|---|
| `test_field_score_accumulator_catches_1ulp_and_signed_zero` (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:21-53`) | `FieldScoreAccumulator` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:522-612`) | Catches 1-ULP normal perturbation (`max_ulp == 1`, `first_non_bit` recorded) and `-0.0` vs `+0.0` signed-zero bit difference | `PASS` |
| `test_master_admission_manifests_and_plants` (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:56-83`) | `verify_master_admission_manifests` + `manifest-digest` & `namelist-mismatch` plants | All 8 admission & projection digests match; `manifest-digest` and `namelist-mismatch` (`rn_sinew + 0.05`) raise `STATUS PLANT-FIRED` | `PASS` |
| `test_score_o8_to_o12_representative_steps_0_ulp` (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:86-128`) | `score_o8_to_o12` on `steps=(1, 2, 100)` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:811-1415`) | `PASS_ALL_O8_O12_0_ULP` across `O8`, `O9` (including all 15 Kind 17 Picard frames), `O10`, `O11`, and `O12` | `PASS` |
| `--plant o8-1ulp` (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:131-154`) | 1-ULP flip in O8 exit `ht_i_new[0, 0]` (`kt=1`) | Raises `STATUS PLANT-FIRED: O8 non-bit-exact: unequal=1/53280 ... 'field': 'ht_i_new'` | `PASS` |
| `--plant o9-1ulp` (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:131-154`) | 1-ULP flip in O9 Kind `4` exit `t_su[0]` (`kt=1, jl=1`) | Raises `STATUS PLANT-FIRED: O9 non-bit-exact ... 'field': 't_su'` | `PASS` |
| `--plant o10-1ulp` (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:131-154`) | 1-ULP flip in O10 Kind `7` exit `sz_i[0, 0]` (`kt=1, jl=1`) | Raises `STATUS PLANT-FIRED: O10 non-bit-exact ... 'field': 'sz_i'` | `PASS` |
| `--plant o11-1ulp` (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:131-154`) | 1-ULP flip in O11 Kind `9` exit `a_i[0]` (`kt=1, jl=1`) | Raises `STATUS PLANT-FIRED: O11 non-bit-exact ... 'field': 'a_i'` | `PASS` |
| `--plant o12-1ulp` (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:131-154`) | 1-ULP flip in O12 Kind `11` exit `a_ip[0, 0, 0]` (`kt=1, jl=0`) | Raises `STATUS PLANT-FIRED: O12 non-bit-exact ... 'field': 'a_ip'` | `PASS` |
| `--plant swapped-exit` (`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:131-154`) | Swap O8 exit fields `ht_i_new` <-> `fraz_frac` (`kt=1`) | Raises `STATUS PLANT-FIRED: O8 non-bit-exact` | `PASS` |
