# SI3 Rung O12 (`ice_thd_pnd` / `pnd_LEV` / `ice_perm_eff`) — Model-Preparation Preregistration (`si3_r20`)

Round: `si3_r20` (model preparation for rung O12 of `docs/ice/fidelity/si3_ladder_design.md:232`, without acquisition).
Branch: `fidelity/nemo-testcases-si3-ladder` at parent `e31236c2f6ad`.

## 1. Scope and oracle source card

Rung O12 owns the production melt-pond stage of `ice_thd` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:222-223`, `IF ( ln_pnd .AND. ln_icedH ) CALL ice_thd_pnd( kt )`) at `jpl = 5`, `nlay_i = 10`, `nlay_s = 5` under the `ORCA2_SAS_ICE` production pond selectors (`tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:241-257`, `cfgs/ORCA2_SAS_ICE/EXPREF/namelist_ice_cfg:77-84`):

- `ln_pnd = .true.`, `ln_pnd_LEV = .true.`, `ln_pnd_CST = .false.`, `ln_pnd_TOPO = .false.` (`nice_pnd = np_pndLEV = 2`, `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:52,1471-1477`),
- `rn_apnd_min = 0.15_wp`, `rn_apnd_max = 0.80_wp`, `rn_pnd_flush = 0.1_wp` (`namelist_ice_cfg:80-81`, `namelist_ice_ref:245`),
- `rn_apnd = 0.2_wp`, `rn_hpnd = 0.05_wp` (`namelist_ice_ref:249-250`, read in `ice_thd_pnd_init` at `icethd_pnd.f90:1439` and unused on the `np_pndLEV` branch),
- `ln_pnd_lids = .true.`, `ln_pnd_rain = .true.`, `ln_pnd_alb = .true.`, `rn_pnd_hl_min = 0.005_wp`, `rn_pnd_hl_max = 0.015_wp` (`namelist_ice_ref:246-247,255-257`, `namelist_ice_cfg:82-83`; `ln_pnd_rain` is read in `namthd_pnd` at `icethd_pnd.f90:1441` and consumed only inside `pnd_TOPO` at `icethd_pnd.f90:702`, never in `pnd_LEV`; `ln_pnd_alb`, `rn_pnd_hl_min`, `rn_pnd_hl_max` govern the pond effective area `a_ip_eff` in `ice_var_glo2eqv` at `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:320-353` and `ice_alb` at `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icealb.f90:200-235`),
- `rn_himin = 0.1_wp` (`namelist_ice_ref:44`),
- `np_perm_eff = 2` (`icethd_pnd.f90:57`, harmonic mean in `ice_perm_eff`, `icethd_pnd.f90:1408-1420`).

The compiled reference build is `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_SAS_ICE_O7_REF`:
- `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90` (SHA-256 `11e5de4e7fdd2c3ef6cf6ef75247833c306b9cc75c79a0dc1684f3a4e7bb5c23`, byte-identical across `ORCA2_SAS_ICE_O1_REF`, `O2_REF`, `O5_REF`, `O7_REF`, and `O8O16_REF`),
- `ORCA2_SAS_ICE_O7_REF/BLD/obj/icethd_pnd.o` (SHA-256 `d1a09537f9cb55c6da4cbf44c73729f1ddd3ee107ed3602ef628f9547fa8fec0`, compiled with `%FCFLAGS -fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize`, `ORCA2_SAS_ICE_O7_REF/BLD/arch_nemo.fcm:27`).

## 2. Registered master-record projection (`DECISION 173`, GIVEN-NEMO-ENTRY only)

Per `DECISION 173` and `docs/ice/fidelity/si3_ladder_design.md:205-212,232`, rungs O8–O16 are admitted from ONE shared 100-step production `ORCA2_SAS_ICE` master record (`si3_r18`). No acquisition tool is built in `si3_r20`.

O12's projection of the shared master record is the global pair `(kind 10, kind 11)` around `IF ( ln_pnd .AND. ln_icedH ) CALL ice_thd_pnd( kt )` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:222-223`):

1. **Stage entry (`kind = 10`, `pond_entry_category_done`)**: recorded immediately after the category loop `DO jl = 1, jpl` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:176-217`, which ends with `CALL ice_thd_1d2d( jl, 2 )` at `:212` scattering `a_i`, `h_i`, `h_s`, `t_su`, `s_i`, `t_s`, `e_s`, `t_i`, `e_i`, `sz_i`, `szv_i`, `v_i`, `v_s`, `sv_i`, `oa_i`, `dh_i_sum_2d`, `dh_s_sum_2d` and category flux accumulators back to the 2D/3D/4D arrays) and immediately before `IF ( ln_pnd .AND. ln_icedH ) CALL ice_thd_pnd( kt )` (`:222-223`).
2. **Stage exit (`kind = 11`, `pond_exit_itd_entry`)**: recorded immediately after `CALL ice_thd_pnd( kt )` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:223`) and before `IF( jpl > 1 ) CALL ice_itd_rem( kt )` (`:225`).

### Arrays read at O12 entry (`kind = 10`)

| Array | Shape (`ORCA2`) | Fortran declaration | Role in `ice_thd_pnd` / `pnd_LEV` / `ice_perm_eff` |
|---|---|---|---|
| `a_i` | `(jpi, jpj, jpl)` | `ice.f90:157` | Summed across categories in `at_i(:,:) = SUM( a_i, dim=3 )` (`icethd_pnd.f90:203`); gathered in `pnd_LEV` (`:415`) for thin/tiny-ice test (`:437`), meltwater volume (`:454`), max pond volume caps (`:464,471`), pond area/depth recovery (`:508-509,527-528`), and hydraulic head `zhp` (`:516`). |
| `v_i` | `(jpi, jpj, jpl)` | `ice.f90:158` | Tested in `IF( v_i(ji,jj,jl) < epsi10 .OR. at_i(ji,jj) < epsi10 )` (`icethd_pnd.f90:207`) on `DO_2D(0,0,0,0)` interior cells (`ntsi:ntei, ntsj:ntej`). |
| `h_i` | `(jpi, jpj, jpl)` | `ice.f90:159` | Gathered in `pnd_LEV` (`icethd_pnd.f90:416`); tested against `rn_himin` (`:437`), depth-driven overflow cap (`:471`), hydraulic head `zhp` (`:516`), and Darcy flushing thickness divisor (`:522`). |
| `t_su` | `(jpi, jpj, jpl)` | `ice.f90:167` | Gathered in `pnd_LEV` (`icethd_pnd.f90:417`); freezing degree-temperature `zdT = MAX( zTp+rt0 - t_su_1d(ii), 0._wp )` (`:484`). |
| `a_ip` | `(jpi, jpj, jpl)` | `ice.f90:175` | Zeroed on vanished-ice interior cells (`icethd_pnd.f90:209`); gathered (`:418`), read in initial `zdv_pnd` (`:435`), volume recovery (`:446-447`), lid growth radicand (`:491`), updated at `:440,508,527,538`, read in `zhp` (`:516`), Darcy flushing (`:522`), lid thickness recovery (`:533`), final `v_ip`/`v_il`/`zdv_pnd` (`:554-557`), and scattered (`:565`). |
| `v_ip` | `(jpi, jpj, jpl)` | `ice.f90:176` | Flushed to `wfx_pnd` and zeroed on vanished-ice interior cells (`icethd_pnd.f90:208,210`); recomputed from `h_ip * a_ip` (`:446,554`), updated by melt (`:475`), refreezing (`:497,502`), and Darcy flushing (`:524`), and scattered (`:568`). |
| `h_ip` | `(jpi, jpj, jpl)` | `ice.f90:177` | Zeroed on vanished-ice interior cells (`icethd_pnd.f90:212`); gathered (`:419`), read in initial `zdv_pnd` (`:435`) and `v_ip_1d` (`:446`), updated at `:441,509,528,539`, read in `zhp` (`:516`), lid-to-pond ratio check (`:537`), final `v_ip_1d`/`zdv_pnd` (`:554,557`), and scattered (`:566`). |
| `v_il` | `(jpi, jpj, jpl)` | `ice.f90:179` | Flushed to `wfx_pnd` and zeroed on vanished-ice interior cells (`icethd_pnd.f90:208,211`); recomputed from `h_il * a_ip` (`:447,555`), updated by lid melting (`:478`) and lid growth (`:494`), read in lid thickness recovery (`:533`), and scattered (`:569`). |
| `h_il` | `(jpi, jpj, jpl)` | `ice.f90:180` | Zeroed on vanished-ice interior cells (`icethd_pnd.f90:213`); gathered (`:420`), read in initial `zdv_pnd` (`:435`) and `v_il_1d` (`:447`), updated at `:442,533-534,540`, read in lid-to-pond ratio check (`:537`), final `v_il_1d`/`zdv_pnd` (`:555,557`), and scattered (`:567`). |
| `a_ip_frac` | `(jpi, jpj, jpl)` | `ice.f90:178` | Zeroed on vanished-ice interior cells (`icethd_pnd.f90:214`); unchanged elsewhere during `ice_thd_pnd` (recomputed in post-thermo `ice_var_glo2eqv(2)` at `icestp.f90:254`, `icevar.f90:334-336`). |
| `dh_i_sum_2d` | `(jpi, jpj, jpl)` | `ice.f90:217` | Gathered in `pnd_LEV` (`icethd_pnd.f90:422`); surface ice melt contribution `dh_i_sum(ii) * rhoi` to `zdv_avail` (`:454`). |
| `dh_s_sum_2d` | `(jpi, jpj, jpl)` | `ice.f90:218` | Gathered in `pnd_LEV` (`icethd_pnd.f90:423`); surface snow melt contribution `dh_s_sum(ii) * rhos` to `zdv_avail` (`:454`). |
| `t_i` | `(jpi, jpj, nlay_i, jpl)` | `ice.f90:169` | Gathered in `pnd_LEV` (`icethd_pnd.f90:427`); layer temperatures passed to `ice_perm_eff` (`:519,1389-1401`). |
| `sz_i` | `(jpi, jpj, nlay_i, jpl)` | `ice.f90:165` | Gathered in `pnd_LEV` (`icethd_pnd.f90:426`); layer salinities passed to `ice_perm_eff` (`:519,1389-1401`). |
| `wfx_pnd` | `(jpi, jpj)` | `ice.f90:122` | Incremented on vanished-ice interior cells (`icethd_pnd.f90:208`), gathered before `DO jl = 1, jpl` (`:406`), decremented by `zdv_pnd * rhow * r1_Dt_ice` per category (`:558`), and scattered after `END DO` (`:573`). |

### Arrays written at O12 exit (`kind = 11`)

- `at_i(jpi, jpj)`: overwritten across the full `(jpi, jpj)` array by `SUM( a_i, dim=3 )` (`icethd_pnd.f90:203`).
- `a_ip(jpi, jpj, jpl)`, `v_ip(jpi, jpj, jpl)`, `h_ip(jpi, jpj, jpl)`, `v_il(jpi, jpj, jpl)`, `h_il(jpi, jpj, jpl)`: updated on interior cells (`ntsi:ntei, ntsj:ntej`); halo cells are untouched in `ice_thd_pnd` (`icethd_pnd.f90:247`) and updated later by `icethd`'s `lbc_lnk` call (`icethd.f90:234-235`).
- `a_ip_frac(jpi, jpj, jpl)`: zeroed on interior `(ji, jj, jl)` where `v_i(ji,jj,jl) < epsi10 .OR. at_i(ji,jj) < epsi10` (`icethd_pnd.f90:207,214`); unchanged elsewhere.
- `wfx_pnd(jpi, jpj)`: updated on interior cells (`ntsi:ntei, ntsj:ntej`) by vanished-ice pond drainage (`:208`) and net pond+lid volume change across `jl = 1..jpl` (`:558,573`).
- Every other field in the master record's global packet at `kind = 10` (`a_i`, `v_i`, `v_s`, `h_i`, `h_s`, `s_i`, `sv_i`, `o_i`, `oa_i`, `t_su`, `a_ip_eff`, `t_i`, `e_i`, `sz_i`, `szv_i`, `t_s`, `e_s`, `dh_i_sum_2d`, `dh_s_sum_2d`, and non-`wfx_pnd` fluxes) is bitwise unchanged between `kind = 10` and `kind = 11`.

### Run scalars and constants required (no defaults)

- Timestep & grid scalars: `rDt_ice`, `r1_Dt_ice` (`icestp.f90:418-419`), `nn_hls` (defining interior `ntsi:ntei, ntsj:ntej`).
- Namelist scalars (`&namitd`, `&namthd`, `&namthd_pnd`): `ln_icedH`, `rn_himin`, `ln_pnd`, `ln_pnd_LEV`, `ln_pnd_CST`, `ln_pnd_TOPO`, `ln_pnd_lids`, `ln_pnd_rain`, `ln_pnd_alb`, `rn_apnd_min`, `rn_apnd_max`, `rn_pnd_flush`, `rn_apnd`, `rn_hpnd`, `rn_pnd_hl_min`, `rn_pnd_hl_max`.
- Physical constants (`IceConstantsConfig` / `phycst.f90` / `par_ice.f90`): `rhow` (`rho_fw`), `rhoi` (`rho_ice`), `rhos` (`rho_snow`), `rho0` (`rho_sw`), `r1_rho0` (`1._wp / rho0`), `grav`, `rcnd_i` (`k_ice`), `rLfus` (`latent_fusion`), `rt0` (`T0`), `epsi06 = 1.0e-6`, `epsi10 = 1.0e-10`.

## 3. Pre-registered predictions and falsifiers

- **P1 (Object-code MIN/MAX/SQRT/comparison audit)**: Disassembly of `ORCA2_SAS_ICE_O7_REF/BLD/obj/icethd_pnd.o` maps every `minsd`, `maxsd`, `sqrtsd`, and `comisd`/`ucomisd` in `ice_thd_pnd`, `pnd_LEV`, and `ice_perm_eff`. In particular, gfortran `-O3` fuses lines `522-524` (`zdv_flush = -zperm * ...`, `zdv_flush = MAX( zdv_flush, -v_ip_1d(ii) )`, `v_ip_1d(ii) = v_ip_1d(ii) + zdv_flush`) into positive `pos_flush = +zperm * ...`, `capped = minsd(dst=pos_flush, src=v_ip_1d)`, `v_ip_1d = v_ip_1d - capped` (`instructions_without_token = [522]`, `token_statements_without_instruction = [523]`), and lines `465` and `472` have opposite `minsd` `dst`/`src` operand roles (`465`: `dst = zv_ip_max - v_ip_1d, src = zdv_mlt`; `472`: `dst = zdv_mlt, src = zv_ip_max - v_ip_1d`).
  - **CONFIRM**: Committed audit script and JSON pin every site in `icethd_pnd.o` with zero unresolved instructions, and role-swap plants on reachable `FOUND` sites fail.
  - **REFUTE**: Any unmapped `minsd`/`maxsd`/`sqrtsd` instruction or any `FOUND` site without a bit-changing test.
- **P2 (Independent scalar oracle & real eager == jit)**: `si3_ice_perm_eff`, `si3_pnd_lev`, and `si3_ice_thd_pnd` match an independent pure-Python/glibc scalar transcription (sharing no model code) at `0` ULP (`max_ulp == 0`, identical `uint64` bit patterns including signed zeros) under both `jax.jit` and `with jax.disable_jit():` across all branch classes at `jpl = 5, nlay_i = 10, nlay_s = 5`.
  - **CONFIRM**: `0` ULP across all outputs (`at_i`, `a_ip`, `v_ip`, `h_ip`, `v_il`, `h_il`, `a_ip_frac`, `wfx_pnd`, `zperm`) in both `jit` and `disable_jit`, and every production-path/barrier plant fails when activated.
  - **REFUTE**: Any `>= 1` ULP discrepancy or any plant that does not fail.
- **P3 (Differentiability & central finite differences)**: `jax.grad` and `jax.jvp` through `si3_ice_perm_eff`, `si3_pnd_lev`, and `si3_ice_thd_pnd` are finite (`NaN` count `0`) across all cells (including dry cells, vanished-ice cells, thin-ice cells, zero-pond cells, and impermeable cells), nonzero on active pond cells, and agree with central finite differences within `rtol = 1e-5, atol = 1e-7`.
  - **CONFIRM**: All gradients finite, nonzero, matching central FD, and `stop_gradient` plant fires.
  - **REFUTE**: Any `NaN`/`Inf` gradient or FD mismatch.
- **P4 (Parent-vs-current invariance)**: The D5 C1D full-year hash under `jit` (`b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`) and all 18 D5 arm hashes, plus all existing SI3 rungs (`O1`, `O2`, `O3`, `O4`, `O5`, `O7`, `O8`, `O9`, `O10`, `O11`), remain `0`-ULP / byte-identical to parent `e31236c2f6ad`.
  - **CONFIRM**: All 18 D5 arm hashes and existing SI3 tests match parent bit-for-bit.
  - **REFUTE**: Any hash or bit change in any existing caller.
