# SI3 Ladder Round 20 (`si3_r20`) — O12 model-preparation receipt

Date: 2026-10-26. Parent: `e31236c2f6ad`. Scope: rung **O12** (`docs/ice/fidelity/si3_ladder_design.md:232`) — the production melt-pond stage (`ice_thd_pnd` / `pnd_LEV` / `ice_perm_eff`) with the deck's pond selectors (`ln_pnd = .true.`, `ln_pnd_LEV = .true.`, `ln_pnd_CST = .false.`, `ln_pnd_TOPO = .false.`, `ln_pnd_lids = .true.`, `ln_pnd_rain = .true.`, `ln_pnd_alb = .true.`, `rn_apnd_min = 0.15`, `rn_apnd_max = 0.80`, `rn_pnd_flush = 0.1`, `rn_himin = 0.1`) at `jpl = 5`, `nlay_i = 10`, `nlay_s = 5`, **WITHOUT acquisition**. Preregistration commit: `8f43a8b0a4` (`docs/ice/fidelity/si3_r20_o12_preregister.md:1-77`).

**Verdict: MODEL PREP COMPLETE; O12 remains UNMEASURED against NEMO.** No master-record array was read and no acquisition tool was built. Under DECISION 173 (`docs/ice/fidelity/si3_ladder_design.md:205-209`), O8–O16 are admitted from one shared 100-step production `ORCA2_SAS_ICE` master record (`round18_o8_o16`).

independent review: operator

---

## 1. Search before building

Searched `packages/`, `src/`, `scripts/validate/`, and `tests/` for `icethd_pnd`, `ice_thd_pnd`, `pnd_LEV`, `pnd_CST`, `pnd_TOPO`, `ice_perm_eff`, `rn_apnd_min`, `rn_apnd_max`, `rn_pnd_flush`, `ln_pnd_lids`, `wfx_pnd`, `a_ip_frac`, `zaspect`, and `zvisc`:

- **D5 C1D column path**: does **not** transcribe `icethd_pnd.F90`. The C1D card disables melt ponds (`scripts/validate/ocean_fidelity/testcases/configs/c1d_omip_l3_namelist_ice_cfg:132`: `ln_pnd           = .false.    ! ORCA1 ice cfg:151`), and `si3_column_step_arrays` in `packages/ice/legoesm/ice/bitz_lipscomb.py:2678-2796` contains no pond stage.
- **O7 surface albedo and `glo2eqv` consumers**: `packages/ice/legoesm/ice/si3_surface.py:300-347` (`si3_ice_alb`) consumes `(t_su, h_i, h_s, a_ip_eff, h_ip, h_il)`, and `packages/ice/legoesm/ice/si3_orca2_state.py:941-965` (`si3_var_glo2eqv`) computes `(h_ip, h_il, a_ip_frac, a_ip_eff)` when `ln_pnd = .true.`.
- **O4 Prather horizontal advection consumer**: `packages/ice/legoesm/ice/si3_orca2_state.py:3450-3500` recovers pre-advection `(h_ip, h_il)` and advects `(a_ip, v_ip, v_il)` in `si3_dyn_adv_pra_orca2`.
- **Thermodynamic pond stage (`ice_thd_pnd` / `pnd_LEV` / `ice_perm_eff`)**: had **zero** existing transcription in `packages/` or `src/`. Reused shared helpers (`nemo_source_round`, `transcendentals.exp`, `fortran_sum0`, `ieee_lt_zero`, `si3_ice_thd_o10_category_stages`, `si3_ice_thd_da`) without duplication.

---

## 2. Record build and verbatim source citations

The compiled production record build is `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_SAS_ICE_O7_REF`:

- Compiler flags at `ORCA2_SAS_ICE_O7_REF/BLD/arch_nemo.fcm:27`:
  `%FCFLAGS             -fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize`
- Preprocessed source `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90` (SHA-256 `11e5de4e7fdd2c3ef6cf6ef75247833c306b9cc75c79a0dc1684f3a4e7bb5c23`) is byte-identical across `ORCA2_SAS_ICE_{O1,O2,O5,O7}_{REF,REC}`, and object file `ORCA2_SAS_ICE_O7_REF/BLD/obj/icethd_pnd.o` (SHA-256 `d1a09537f9cb55c6da4cbf44c73729f1ddd3ee107ed3602ef628f9547fa8fec0`) is instruction-identical across those builds.
- Production deck selectors in `tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:45` (`rn_himin = 0.1`), `:157` (`ln_icedH = .true.`), and `:241-257`:
  - `:241`: `ln_pnd            = .true.`
  - `:243`: `rn_apnd_min    =  0.15`
  - `:244`: `rn_apnd_max    =  0.80`
  - `:246`: `ln_pnd_TOPO       = .false.`
  - `:247`: `ln_pnd_rain    = .true.`
  - `:248`: `ln_pnd_LEV        = .true.`
  - `:249`: `rn_pnd_flush   =  0.1`
  - `:250`: `ln_pnd_lids    = .true.`
  - `:251`: `ln_pnd_CST        = .false.`
  - `:255`: `ln_pnd_alb        = .true.`
  Selector resolution in `ice_thd_pnd_init` at `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1424-1487` sets `nice_pnd = np_pndLEV = 2` (`:52,1474`).
- Call point in `ice_thd` after the `DO jl = 1, jpl` category loop at `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:176-223`:
  - `:212`: `CALL ice_thd_1d2d( jl, 2 )` scatters `dh_i_sum(1:npti)` -> `dh_i_sum_2d(ji,jj,jl)` and `dh_s_sum(1:npti)` -> `dh_s_sum_2d(ji,jj,jl)` and converts 1D intensive fields back to 2D extensive volumes/contents at `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:484-588` (`:488-501`).
  - `:222-223`: `IF ( ln_pnd .AND. ln_icedH ) CALL ice_thd_pnd( kt )`.
- Rain switch (`ln_pnd_rain = .true.`): read in `namthd_pnd` at `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1441` and consumed only inside `pnd_TOPO` at `:702`; neither `ice_thd_pnd` nor `pnd_LEV` reads `ln_pnd_rain`.

### Executing statements in `ice_thd_pnd`, `pnd_LEV`, and `ice_perm_eff`

1. `ice_thd_pnd` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:162-264`, model `packages/ice/legoesm/ice/si3_thd_pnd.py:575-763`):
   - `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:203`: `at_i(:,:) = SUM( a_i, dim=3 )` (`packages/ice/legoesm/ice/si3_thd_pnd.py:683`).
   - `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:205-217`: vanished-ice pond removal across `DO jl = 1, jpl` and `DO_2D( 0, 0, 0, 0 )`:
     `IF( v_i(ji,jj,jl) < epsi10 .OR. at_i(ji,jj) < epsi10 ) THEN`
     `wfx_pnd(ji,jj) = wfx_pnd(ji,jj) + ( v_ip(ji,jj,jl) + v_il(ji,jj,jl) ) * rhow * r1_Dt_ice` (`:208`)
     and zeroing of `a_ip, v_ip, v_il, h_ip, h_il, a_ip_frac` (`:209-214`, model `packages/ice/legoesm/ice/si3_thd_pnd.py:699-717`).
   - `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:222-240`: active cell gather (`IF( at_i(ji,jj) >= epsi10 )` over `DO_2D( 0, 0, 0, 0 )`) and `CASE (np_pndLEV) ; CALL pnd_LEV` (`packages/ice/legoesm/ice/si3_thd_pnd.py:720-763`).

2. `pnd_LEV` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:329-582`, model `packages/ice/legoesm/ice/si3_thd_pnd.py:235-572`):
   - `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:385-387` and `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/par_ice.f90:134-135`: `zaspect = 0.8_wp`, `zTp = -0.15_wp`, `zvisc = 1.79e-3_wp`, `epsi06 = 1.e-06_wp`, `epsi10 = 1.e-10_wp` (`packages/ice/legoesm/ice/si3_thd_pnd.py:32-46`).
   - `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:401-403`: `z1_rhow = 1._wp / rhow`, `z1_aspect = 1._wp / zaspect`, `z1_Tp = 1._wp / zTp` (`packages/ice/legoesm/ice/si3_thd_pnd.py:493-495`).
   - `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:412-435`: `DO jl = 1, jpl` category gather and initial `zdv_pnd = ( h_ip_1d(ii) + h_il_1d(ii) ) * a_ip_1d(ii)` (`packages/ice/legoesm/ice/si3_thd_pnd.py:286,530-563`).
   - `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:437-442`: thin-ice / tiny-area pond removal `IF( h_i_1d(ii) < rn_himin .OR. a_i_1d(ii) < 0.01_wp ) THEN` zeroing `a_ip_1d, h_ip_1d, h_il_1d` (`packages/ice/legoesm/ice/si3_thd_pnd.py:289,406-408`).
   - `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:446-478`: volume retrieval, available meltwater `zdv_avail = -( dh_i_sum(ii)*rhoi + dh_s_sum(ii)*rhos ) * z1_rhow * a_i_1d(ii)`, melt fraction `zfr_mlt = rn_apnd_min + ( rn_apnd_max - rn_apnd_min ) * at_i_1d(ii)`, `zdv_mlt = MAX( 0._wp, zfr_mlt * zdv_avail )`, area-driven overflow cap (`:464-465`), depth-driven overflow cap (`:471-472`), pond growth `v_ip_1d(ii) = v_ip_1d(ii) + zdv_mlt` (`:475`), and lid melt `IF( ln_pnd_lids ) v_il_1d(ii) = MAX( 0._wp, v_il_1d(ii) - zdv_mlt )` (`:478`, model `packages/ice/legoesm/ice/si3_thd_pnd.py:294-323`).
   - `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:484-503`: freezing contraction `zdT = MAX( zTp+rt0 - t_su_1d(ii), 0._wp )`, `IF( ln_pnd_lids )` Flocco et al. (2010) quadratic lid growth and pond shrinking (`:490-497`), and `ELSE` Holland et al. (2012) `EXP` pond shrinking (`:500-502`, model `packages/ice/legoesm/ice/si3_thd_pnd.py:326-354`).
   - `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:508-528`: pre-flush `a_ip_1d`/`h_ip_1d` update (`:508-509`), hydraulic head `zhp` (`:516` with `r1_rho0 = 1._wp / rho0` from `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/phycst.f90:52-53`), `zperm = ice_perm_eff( t_i_1d(ii,:), sz_i_1d(ii,:) )` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:519`), Darcy flushing (`:522-524`), and post-flush `a_ip_1d`/`h_ip_1d` update (`:527-528`, model `packages/ice/legoesm/ice/si3_thd_pnd.py:358-387`).
   - `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:531-559`: lid thickness recovery `IF( a_ip_1d(ii) > 0.01_wp ) h_il_1d(ii) = v_il_1d(ii) / a_ip_1d(ii) ELSE 0._wp` (`:533-535`), thick-lid removal `IF ( h_il_1d(ii) > h_ip_1d(ii) * 10._wp )` (`:537-541`), final `v_ip_1d`/`v_il_1d` recovery (`:554-555`), and `wfx_pnd_1d` update (`:557-558`, model `packages/ice/legoesm/ice/si3_thd_pnd.py:390-414`).

3. `ice_perm_eff` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1372-1422`, model `packages/ice/legoesm/ice/si3_thd_pnd.py:168-232`):
   - `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1389-1401`: VC19 cubic brine salinity (`np_liquidus == 2` at `:1393`, `zs_br(jk) = -18.7_wp * zt1 - 0.519_wp * zt2 - 0.00535_wp * zt3`), `zs_br(jk) = MAX( zs_br(jk), epsi10 )` (`:1398`), and `zv_br(jk) = sz_i(jk) / zs_br(jk)` when `zt1 < -epsi06` else `0._wp` (`:1399`, model `packages/ice/legoesm/ice/si3_thd_pnd.py:200-211`).
   - `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1408-1420`: harmonic mean (`np_perm_eff == 2`) with `IF( zv_br(jk) < epsi06 ) THEN ztmp = 0._wp ; EXIT` (`:1412`), `zperm = 3.e-8_wp * zv_br(jk)**3` (`:1416`), `ztmp = ztmp + 1._wp / zperm` (`:1417`), and `IF( ztmp > 0._wp ) ice_perm_eff = REAL( nlay_i, wp ) / ztmp` (`:1419`, model `packages/ice/legoesm/ice/si3_thd_pnd.py:214-232`).

4. Category-to-pond composer `si3_ice_thd_o12_stages` and production pond wiring:
   - `packages/ice/legoesm/ice/si3_thd_pnd.py:766-954`, exported in `packages/ice/legoesm/ice/__init__.py:290-314`.
   - Selectable production pond branch (`scheme = "si3_lev"`, `namthd_pnd = SI3PondConfig(...)`) in `packages/ice/legoesm/ice/config.py:267-288,311-315,652-722`, carrying prognostic lid thickness `h_il` in `DynamicSeaIceState` **only** when `pond_scheme = "si3_lev"` (`h_il = None` for default `pond_scheme = "cesm"` per DECISION 184, set explicitly from the run's selected scheme in `packages/ice/legoesm/ice/state.py:52-76,89-177,213-258,315-335` and `packages/ocean/legoesm/ocean/restart.py:322-329,382-392,859-861`), dispatched through `packages/ice/legoesm/ice/ponds.py:34-212` (`step_ponds`), and called from `packages/ice/legoesm/ice/sea_ice.py:420-433,475,1737-1772,2388-2413,2469-2535,2820-2845,2982-3062,3196-3250,3295-3297,3414-3432` (`step_sea_ice`, `_thermo_v2`, and `_step_dynamic_v2`, computing `at_i` over the category axis via `fortran_sum_categories` as in NEMO's `ice_var_agg`), while keeping default (`scheme = "cesm"`) state pytree structure/leaves and restarts identical to parent `e31236c2f6ad`.
   - **No-silent-physics configuration-time guard for stages that do not yet carry `h_il`/`v_il` (`packages/ice/legoesm/ice/config.py:311-315,662-722` and `packages/ice/legoesm/ice/sea_ice.py:420-433,1737-1772,2820-2845,2982-3062,3196-3250,3295-3297`)**: when `MeltPondConfig(scheme="si3_lev")` is combined with any stage in `step_sea_ice` / `_step_dynamic_v2` that does not yet carry lid volume/thickness (`v_il`/`h_il`) as NEMO does — advection (`dynamics != "none"` or `transport != "none"`), the multi-category concentration cap (`_cap_multicat_concentration`), ITD remap (`itd_remap != "none"` at `n_categories > 1` or `itd_remap == "lipscomb2001"` at `n_categories == 1`), or ridging (`ridging.enabled = True`) — `validate_si3_thermo_config`, `step_sea_ice`, and `_step_dynamic_v2` raise a `ValueError` at configuration time naming the missing stage(s) and never run with a stale `h_il`. Thermodynamics-only O12 composition (`dynamics = "none"`, `transport = "none"`, `ridging.enabled = False`, and `itd_remap = "none"` when `n_categories > 1`) runs cleanly. Rungs **O4** (`ice_dyn_adv_pra` Prather advection), **O6** (`ice_dyn_rdgrft` ridging/rafting and `ice_cor`), and **O13** (`ice_itd_rem` ITD remapping) must carry `v_il` before the combination is allowed.

---

## 3. Exhaustive object-code `MIN/MAX/SIGN/SQRT` audit

The committed audit tool `scripts/validate/ocean_fidelity/testcases/nemo_si3_o12_minmax_audit.py:1-193` (using `scripts/validate/ocean_fidelity/testcases/nemo_si3_o9_minmax_audit.py:644-656` with `cwd=build.resolve()` so shadow `-g` compilation succeeds regardless of caller working directory) disassembles `ORCA2_SAS_ICE_O7_REF/BLD/obj/icethd_pnd.o` across `ice_thd_pnd`, `pnd_LEV`, and `ice_perm_eff`, and pins its report in `scripts/validate/ocean_fidelity/testcases/nemo_si3_o12_minmax_audit.json` (tested by `tests/ocean/fidelity/test_nemo_si3_o12_minmax_audit.py:1-161`):

- Zero FMA (`vfmadd*`/`vfmsub*`), zero fast-reciprocal (`rcpss`/`rcpps`/`rsqrtss`/`rsqrtps`), and zero `SIGN` instructions exist in `icethd_pnd.o`.
- One `call exp` exists in `pnd_LEV` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:500`), inside the `ELSE` of `IF( ln_pnd_lids )`.
- All 13 floating `minsd`/`maxsd`/`sqrtsd` source sites are classified:
  - `:456` (`zdv_mlt = MAX( 0._wp, zfr_mlt * zdv_avail )`): `maxsd(dst=VAR, src=ZERO)` -> `MASKED` (when `dh_i_sum = dh_s_sum = +0.0`, `zdv_avail = -0.0` ties with `+0.0`; masked by `:465` and `:472` outer `maxsd` with `src=ZERO`).
  - `:465` (`zdv_mlt = MAX( 0._wp, MIN( zv_ip_max - v_ip_1d, zdv_mlt ) )`): `minsd(dst=zv_ip_max-v_ip_1d, src=zdv_mlt)` then `maxsd(dst=min, src=ZERO)` -> `NO_TIE`.
  - `:472` (`zdv_mlt = MAX( 0._wp, MIN( zv_ip_max - v_ip_1d, zdv_mlt ) )`): `minsd(dst=zdv_mlt, src=zv_ip_max-v_ip_1d)` then `maxsd(dst=min, src=ZERO)` -> `NO_TIE` (opposite `minsd` operand roles from `:465`, preserved at `packages/ice/legoesm/ice/si3_thd_pnd.py:311,316`).
  - `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:478` (`v_il_1d = MAX( 0._wp, v_il_1d - zdv_mlt )`): `maxsd(dst=VAR, src=ZERO)` -> `MASKED` (masked by `:490` square and `:494` `maxsd` with `src=ZERO`).
  - `:484` (`zdT = MAX( zTp+rt0 - t_su_1d, 0._wp )`): `maxsd(dst=VAR, src=ZERO)` -> `NO_TIE`.
  - `:490` (`zdv_frz = -0.5_wp * MAX( 0._wp, -v_il_1d + SQRT(...) )`): `sqrtsd` + `maxsd(dst=VAR, src=ZERO)` -> `NO_TIE`.
  - `:494` (`v_il_1d = MAX( 0._wp, v_il_1d - zdv_frz )`): gfortran emits `v_il_1d + pos_frz` where `pos_frz = 0.5 * maxsd@490 >= +0.0`, then `maxsd(dst=VAR, src=ZERO)` -> `NO_TIE`.
  - `:497` (`v_ip_1d = MAX( 0._wp, v_ip_1d + zdv_frz )`): gfortran emits `v_ip_1d - pos_frz`, then `maxsd(dst=VAR, src=ZERO)` -> `NO_TIE`.
  - `:502` (`v_ip_1d = MAX( 0._wp, v_ip_1d + zdv_frz )`): `maxsd(dst=VAR, src=ZERO)` -> `NOT_ON_DECK` (`ELSE` of `IF( ln_pnd_lids )`).
  - `:508` and `:527` (`a_ip_1d = MIN( a_i_1d, SQRT( v_ip_1d * z1_aspect * a_i_1d ) )`): `sqrtsd` + `minsd(dst=sqrtsd, src=a_i_1d)` -> `NO_TIE`.
  - `:522` (source `:522-524`): gfortran fuses `zdv_flush = -zperm * ...`, `zdv_flush = MAX( zdv_flush, -v_ip_1d )`, and `v_ip_1d = v_ip_1d + zdv_flush` into positive `pos_flush = +zperm * ...`, `capped = minsd(dst=pos_flush, src=v_ip_1d)` (attributed to DWARF line `522`, leaving token line `523` without a separate instruction), and `v_ip_1d = v_ip_1d - capped` (`packages/ice/legoesm/ice/si3_thd_pnd.py:373-382`) -> `NO_TIE`.
  - `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1398` (`zs_br(jk) = MAX( zs_br(jk), epsi10 )`): `maxsd(dst=zs_br, src=CONST:1e-10)` -> `NO_TIE`.

---

## 4. Independent scalar oracle, barriers, plants, gradients, and caller controls

### Unit and object-audit suites (`89 passed in 231.58s`)

Foreground run of `tests/ice/unit/test_si3_o12_pnd.py` and `tests/ocean/fidelity/test_nemo_si3_o12_minmax_audit.py` on CPU (`JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`, with explicit CPU scoping at `tests/ice/unit/test_si3_o12_pnd.py:35-54` for `transcendentals="libm"`, saved in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round20/pytest_o12_pnd.log`):
`======================== 89 passed in 231.58s (0:03:51) ========================`

- **Independent scalar transcription with independent constants and namelist parser (`0` ULP under `jit` and `jax.disable_jit`)**:
  - Physical constants and `&namthd`/`&namthd_pnd` parameters are transcribed directly from `phycst.f90`, `eosbn2.f90`, `par_ice.f90`, and `namelist_ice_ref`/`namelist_ice_cfg` at `tests/ice/unit/test_si3_o12_pnd.py:62-123` without importing `legoesm.constants` or `NEMO_SI3_CONSTANTS_CONFIG` into any `_oracle_*` function; six planted constant defects in the model's `IceConstantsConfig` all fail against the independent oracle (`tests/ice/unit/test_si3_o12_pnd.py:1870-1911`).
  - `_oracle_ice_perm_eff` (`tests/ice/unit/test_si3_o12_pnd.py:140-179`) vs `si3_ice_perm_eff` at `tests/ice/unit/test_si3_o12_pnd.py:747-793`.
  - `_oracle_pnd_lev_category` / `_oracle_pnd_lev` (`tests/ice/unit/test_si3_o12_pnd.py:182-421`) vs `si3_pnd_lev` for both `ln_pnd_lids = True` and `ln_pnd_lids = False` (glibc `exp`) at `tests/ice/unit/test_si3_o12_pnd.py:796-917`.
  - `_oracle_ice_thd_pnd` (`tests/ice/unit/test_si3_o12_pnd.py:424-520`) vs `si3_ice_thd_pnd` on the 2D/3D `(6, 7, 5)` domain with halo check (`nn_hls = 2`) at `tests/ice/unit/test_si3_o12_pnd.py:920-976`.
- **Inexact-value hex pins and output digest**: pinned at `tests/ice/unit/test_si3_o12_pnd.py:979-1003` (output SHA-256 `5332d71c74a660ea60985d1046bea4f2c24cc60895d711f9dd5039e42b203e89`).
- **Missing-barrier and production-code-path plants (all JIT tests passing inputs as runtime JIT arguments)**:
  - Stripping `nemo_source_round` from 15 individual arithmetic sites/groups across `si3_ice_perm_eff`, `_pnd_lev_single_category`, and `si3_ice_thd_pnd` changes compiled `jit` output bits when arrays are passed as runtime JIT arguments while the protected functions match `jax.disable_jit` bit-for-bit (`tests/ice/unit/test_si3_o12_pnd.py:1006-1239`).
  - Eight source-compiled mutants on `si3_ice_perm_eff`, `_pnd_lev_single_category`, and `si3_ice_thd_pnd` fire through `si3_ice_thd_pnd` (`tests/ice/unit/test_si3_o12_pnd.py:1242-1301`), plus the `epsi06` early-exit boundary plant (`tests/ice/unit/test_si3_o12_pnd.py:1304-1323`).
  - Active `_maxsd`/`_minsd` signed-zero tie-survivor test with role-swap plants (`+0.0` vs `-0.0` changing bit 63) plus pre-mask sign-bit flip and checked downstream masking verification at `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:456` and `:478` (`tests/ice/unit/test_si3_o12_pnd.py:1326-1421`).
- **Five-category composition, call-sequence spy, and production `step_sea_ice`/`step_ponds`/restart wiring**:
  - `O9 -> O10 -> O11 -> ice_thd_1d2d(jl, 2) -> O12` via `si3_ice_thd_o12_stages` matches the scalar oracle bit-for-bit across all 5 categories (`tests/ice/unit/test_si3_o12_pnd.py:1424-1550`).
  - Call-sequence spy verifies `o10` -> `da` (5 categories) -> `ice_thd_1d2d(jl, 2)` volume recovery -> `pnd` (`tests/ice/unit/test_si3_o12_pnd.py:1553-1648`).
  - Production `step_sea_ice` (`packages/ice/legoesm/ice/sea_ice.py:420-433,475,1737-1772,2388-2413,2469-2535,2820-2845,2982-3062,3196-3250,3295-3297,3414-3432`) and `step_ponds` (`packages/ice/legoesm/ice/ponds.py:34-212`) with `MeltPondConfig(enabled=True, scheme="si3_lev", namthd_pnd=...)` match `_oracle_pnd_lev_category` to `0` ULP under both eager and `jax.jit`, fail on planted mutants that remove the `step_ponds` call or force `scheme="cesm"`, and preserve `scheme="cesm"` bit-for-bit (`tests/ice/unit/test_si3_o12_pnd.py:1914-2183`).
  - Prognostic lid thickness `h_il` (`packages/ice/legoesm/ice/state.py:52-76,89-177,213-258,315-335` and `packages/ocean/legoesm/ocean/restart.py:322-329,382-392,859-861`) is allocated (`Field`) and persisted only when `pond_scheme = "si3_lev"` (staying `None` for default `pond_scheme = "cesm"` per DECISION 184), persists across multi-step integration and `save_run_restart`/`load_run_restart`, matches the sequential scalar oracle to `0` ULP, and fails on the planted `h_il = 0` reset mutant (`tests/ice/unit/test_si3_o12_pnd.py:2186-2407`).
  - Multi-category (`jpl = 5`) `step_sea_ice` aggregates `at_i` over the category axis via `fortran_sum_categories` (`packages/ice/legoesm/ice/sea_ice.py:2994-3033`) exactly as NEMO's `ice_var_agg` does, matches the 5-category scalar oracle to `0` ULP across all 5 categories with distinct per-category and spatially varying concentrations, and fails on the planted spatial-axis `at_i` sum mutant (`tests/ice/unit/test_si3_o12_pnd.py:2410-2610`).
  - DECISION 184 tests (`tests/ice/unit/test_si3_o12_pnd.py:2863-3027`, `:3030-3200`, `:3203-3333`):
    - **(a)** A default (`pond_scheme = "cesm"`) run's `DynamicSeaIceState` pytree has `h_il is None` and exact parent (`e31236c2f6`) leaves (`12` leaves) and `Field`-level `jax.tree_util.tree_structure` across `init_dynamic_ice_state`, `slab_to_dynamic`, `distribute_dynamic_state_to_categories`, and `step_sea_ice` (1-category and 5-category), with a planted 13-leaf `h_il = Field(zeros)` control that fails (`tests/ice/unit/test_si3_o12_pnd.py:2863-3027`).
    - **(b)** Restarts (`save_run_restart` and `save_restart`) of a default (`pond_scheme = "cesm"`) run are byte-for-byte identical (`read_bytes() == read_bytes()`) to parent `e31236c2f6`'s restarts across 1-category and 5-category states and cross-load cleanly in both directions (`tests/ice/unit/test_si3_o12_pnd.py:3030-3200`).
    - **(c)** The `si3_lev` run (`pond_scheme = "si3_lev"`) carries and restarts `h_il` across single-category and 5-category (`jpl = 5`) runs via `save_run_restart`/`load_run_restart` and `save_restart`/`load_restart`, and refuses mismatched `cesm` vs `si3_lev` restart/template pairs (`tests/ice/unit/test_si3_o12_pnd.py:3203-3333`).
  - **No-silent-physics configuration-time refusal of stages that do not yet carry `h_il`/`v_il` (`tests/ice/unit/test_si3_o12_pnd.py:3336-3544`)**:
    - Combining `MeltPondConfig(enabled=True, scheme="si3_lev", namthd_pnd=SI3PondConfig(..., ln_pnd_lids=True))` with advection (`dynamics` in `{"free_drift", "evp", "mevp"}` or `transport = "advect"`), the multi-category concentration cap (`_cap_multicat_concentration`), ITD remap (`itd_remap` in `{"lipscomb2001", "simple"}`), or ridging (`ridging.enabled = True`) raises a `ValueError` at configuration time (`validate_si3_thermo_config`, `step_sea_ice`, `_step_dynamic_v2`) naming the missing stage(s) and the corresponding NEMO rungs (**O4** Prather advection `ice_dyn_adv_pra`, **O6** ridging/rafting `ice_dyn_rdgrft` / `ice_cor`, **O13** ITD remapping `ice_itd_rem`), while 1-category and 5-category thermodynamics-only O12 compositions (`dynamics = "none"`, `transport = "none"`, `ridging.enabled = False`, and `itd_remap = "none"` when `n_categories > 1`) run and evolve `h_il` (`tests/ice/unit/test_si3_o12_pnd.py:3336-3544`).
- **Differentiability (`jax.grad`, `jax.jvp`, central finite difference, and boundary safety)**:
  - All 10 continuous physical array inputs have finite, nonzero reverse-mode gradients matching `jax.jvp` (`rtol=1e-12`) and central finite differences (`rtol=1e-5, atol=1e-7`), and a `stop_gradient` plant breaks the `sz_i`/`t_i` gradients (`tests/ice/unit/test_si3_o12_pnd.py:1651-1785`).
  - Custom JVP on `_nemo_sqrt` (`packages/ice/legoesm/ice/si3_thd_pnd.py:90-104`) and safe divisors ensure zero-pond (`rad = 0, disc = 0`), thin-ice, vanished-ice, and impermeable-ice states have finite `jax.grad` with zero forward-bit change (`tests/ice/unit/test_si3_o12_pnd.py:1788-1807`).
- **No defaults and non-production selector refusals**:
  - Every physical array, scalar, `constants`, `namelist`, and `nn_hls` is required without defaults (`tests/ice/unit/test_si3_o12_pnd.py:1810-1837`), and every non-production pond selector is refused (`tests/ice/unit/test_si3_o12_pnd.py:1840-1867`).

### Parent-vs-current controls on shared code callers (M6)

Shared files touched are `packages/ice/legoesm/ice/config.py:267-288,311-315,652-722` (keeping line numbers `291..649` unchanged so zero historical receipt citations shifted), `packages/ice/legoesm/ice/state.py:52-76,89-177,213-258,315-335`, `packages/ocean/legoesm/ocean/restart.py:322-329,382-392,859-861`, `packages/ice/legoesm/ice/ponds.py:34-212`, `packages/ice/legoesm/ice/sea_ice.py:420-433,475,1737-1772,2388-2413,2469-2535,2820-2845,2982-3062,3196-3250,3295-3297,3414-3432`, and `packages/ice/legoesm/ice/__init__.py:290-314`.

1. **Direct parent (`e31236c2f6`) vs current (`HEAD`) comparison across every output of every other caller (`tests/ice/unit/test_si3_o12_pnd.py:2613-2860`)**:
   - Loads `packages/ice/legoesm/ice/state.py`, `packages/ice/legoesm/ice/ponds.py`, and `packages/ice/legoesm/ice/sea_ice.py` directly from parent commit `e31236c2f6` via `git show` and compares every output array bit-for-bit against current `HEAD` across all 6 non-`si3_lev` caller paths (direct `step_ponds` default CESM scheme; state builders `init_dynamic_ice_state`, `slab_to_dynamic`, and `distribute_dynamic_state_to_categories`; `step_sea_ice` slab mode; `step_sea_ice` legacy dynamic `free_drift` mode; `step_sea_ice` single-category v2 mode with CESM ponds + snow + brine + delta-Eddington; and `step_sea_ice` 5-category v2 mode with EVP + advect + Lipscomb2001 + ridging + CESM ponds + snow + brine): **`161` output arrays, `2604` scalar elements compared, `0` unequal** (`0` ULP difference), with a planted 1-ULP perturbation control that fires.
2. **D5 C1D full-year platform probe (`nemo_si3thd_d5_platform_probe.py`)**:
   - Default CPU (`XLA_FLAGS=""`, `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round20/d5_platform_probe_current.json`): all 18 arms are leaf-for-leaf identical to parent (`round17/resume1/d5_wired_cpu.json` except `worktree`); shipped arm `r6_step1_plus_step2_snow_update` `full_year_output_sha256` = `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`.
   - AVX (`XLA_FLAGS="--xla_cpu_max_isa=AVX"`, `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round20/d5_platform_probe_current_avx.json`): all 18 arms are leaf-for-leaf identical to parent (`round17/resume1/d5_wired_avx.json` except `worktree`); shipped arm `full_year_output_sha256` = `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`.
3. **Existing SI3 thermodynamic, pond, restart, and audit suites**:
   - `tests/ice/unit/test_si3_o10_salinity.py` (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round20/pytest_o10_salinity_control.log`): `======================== 44 passed in 682.84s (0:11:22) ========================`.
   - `tests/ice/unit/test_si3_thd_da.py`, `tests/ice/unit/test_si3_o9_zdf_dh_temp.py`, `tests/ocean/fidelity/test_nemo_si3_o9_minmax_audit.py`, `tests/ocean/fidelity/test_nemo_si3_o10_minmax_audit.py` (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round20/pytest_si3_callers_control.log`): `67 passed` (all pass, identical to parent `e31236c2f6ad`).
   - `tests/ocean/unit/test_ocean_run_restart.py` (`74 passed`) and `tests/unit/test_run_omip_core2_restart.py` (`17 passed`).
4. **Private-import and parameter/constant ratchets (`tests/test_no_private_cross_imports.py`, `tests/test_param_specs.py`, `tests/test_no_inline_physics_coeffs.py`, `tests/test_no_hardcoded_constants.py`)**: all pass.

---

## 5. Findings outside this rung (not acted on) and open cross-rung items

- **Open item for rungs O4, O6, and O13**: In NEMO 5.0.2, lid volume `v_il` is carried through Prather horizontal advection (`ice_dyn_adv_pra`, rung **O4**), ridging/rafting and concentration correction (`ice_dyn_rdgrft` / `ice_cor`, rung **O6**), and ITD remapping (`ice_itd_rem`, rung **O13**). In `step_sea_ice` / `_step_dynamic_v2`, those stages do not yet carry `h_il`/`v_il` as NEMO does, so `validate_si3_thermo_config` (`packages/ice/legoesm/ice/config.py:311-315,662-722`) and `step_sea_ice` / `_step_dynamic_v2` (`packages/ice/legoesm/ice/sea_ice.py:420-433,1737-1772,2820-2845,2982-3062,3196-3250,3295-3297`) refuse combining `MeltPondConfig(scheme="si3_lev")` with advection, the concentration cap, ITD remap, or ridging at configuration time naming the missing stage(s). Rungs **O4**, **O6**, and **O13** must carry `v_il` before that combination is allowed.
- In `tests/ice/unit/test_si3_surface.py:514`, `test_production_path_plants_fire[sum_first_category]` fails identically on parent `e31236c2f6ad` and current `HEAD` because commit `e96c1875f8` changed `qsb` tie handling in `packages/ice/legoesm/ice/si3_surface.py:510`; not touched here.

---

## 6. CHOICES (`ASKED` / `UNASKED`)

- **New module `packages/ice/legoesm/ice/si3_thd_pnd.py` with exports appended at the end of `packages/ice/legoesm/ice/__init__.py:290-314`, prognostic `h_il` carried on `DynamicSeaIceState` only when `pond_scheme = "si3_lev"` (`packages/ice/legoesm/ice/state.py:52-76,89-177,213-258,315-335` and `packages/ocean/legoesm/ocean/restart.py:322-329,382-392,859-861`), selectable `"si3_lev"` branch wired into `packages/ice/legoesm/ice/ponds.py:34-212` and `packages/ice/legoesm/ice/sea_ice.py:420-433,475,1737-1772,2388-2413,2469-2535,2820-2845,2982-3062,3196-3250,3295-3297,3414-3432`, and configuration-time refusal when `"si3_lev"` is combined with advection, the concentration cap, ITD remap, or ridging (`packages/ice/legoesm/ice/config.py:267-288,311-315,652-722`)** (`ASKED`): wires the SI3 level-pond scheme (including prognostic lid thickness `h_il` and category-summed `at_i`) into the production sea-ice step and restart, refuses combining `scheme = "si3_lev"` with any stage that does not yet carry `h_il`/`v_il` as NEMO does (naming advection/O4, concentration cap, ITD remap/O13, and ridging/O6), and keeps `bitz_lipscomb.py`, `si3_surface.py`, and `si3_thd_da.py` byte-identical to parent `e31236c2f6ad` and default (`scheme = "cesm"`) state pytree structure/leaves, restarts, and step outputs identical (`161` arrays, `2604` elements, `0` unequal).
- **`_nemo_sqrt` custom JVP with zero-radicand tangent `0.0` (`packages/ice/legoesm/ice/si3_thd_pnd.py:90-104`) and safe divisors on untaken branches** (`UNASKED`; offered for revert): preserves 100% of forward binary64 bits (including `+0.0` and `-0.0`) while preventing `0.0 * inf = NaN` in `jax.grad` on zero-pond (`rad = 0, disc = 0`), thin-ice, and vanished-ice cells.
- **Both `ln_pnd_lids = True` and `ln_pnd_lids = False` (glibc `EXP`) arms transcribed inside `si3_pnd_lev`, while `si3_ice_thd_pnd` enforces the production deck's `ln_pnd_lids = True`** (`UNASKED`; offered for revert): allows direct unit testing of the glibc `exp` arm in `si3_pnd_lev` while keeping `si3_ice_thd_pnd` fail-closed on the production deck selectors.
- **No namelist or acquisition tool changes** (`ASKED`).

---

## 7. `CONFIRMED` vs `PLAUSIBLE`

- **CONFIRMED**:
  - `si3_ice_perm_eff`, `si3_pnd_lev`, `si3_ice_thd_pnd`, `si3_ice_thd_o12_stages`, and the production `step_sea_ice`/`step_ponds` `scheme="si3_lev"` path match the independent pure-Python/ctypes-glibc scalar transcription bit-for-bit (`0` ULP including signed zeros) under both `jax.jit` and `jax.disable_jit()`.
  - All 13 floating `minsd`/`maxsd`/`sqrtsd` sites in `ORCA2_SAS_ICE_O7_REF/BLD/obj/icethd_pnd.o` (including the opposite `minsd` operand orders at `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:465` vs `:472` and the fused Darcy-flushing `minsd` at `:522-524`) are audited and pinned in `nemo_si3_o12_minmax_audit.json`.
  - All missing-barrier plants, source-mutant plants, constant-defect plants, production-wiring plants, call-order spies, finite-difference gradient checks, edge-case gradient finiteness checks, no-default checks, D5 C1D year hashes (`b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`), private-import/constant ratchets, and receipt citation gates pass.
- **PLAUSIBLE (untested until master-record admission)**:
  - Bit-for-bit (`0` ULP) agreement of `si3_ice_thd_pnd` against NEMO's recorded Kind 10 -> Kind 11 transition across all 100 steps of the shared `ORCA2_SAS_ICE` master record (`docs/ice/fidelity/si3_r20_o12_preregister.md:66-74`, predictions `P1`–`P3`).
