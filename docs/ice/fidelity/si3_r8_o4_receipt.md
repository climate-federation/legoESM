# SI3 Ladder Round 8 (`si3_r8`) Receipt — Rung O4 Model Preparation (ORCA2 Prather Advection Across the Cyclic Seam and North Fold, No Acquisition)

* **Round:** `si3_r8`, resume 4 (Claude taking over from the agy author), model preparation only: no NEMO build or run, no acquisition.
* **Branch / parent:** `fidelity/nemo-testcases-si3-ladder`; lane-3 bit-identity is controlled against `042c7bd74c2944dab8c39318687ab0cef54f9d2b` (O1 + D5 landed, before any round-8 model edit).
* **Rung:** O4, design row `docs/ice/fidelity/si3_ladder_design.md:224` ("ORCA2_SAS_ICE with Prather only; dynamics/thermo otherwise off"). The operator brief called this O3; O3 is the initial HFN thickness fit (`docs/ice/fidelity/si3_ladder_design.md:223`). Every function, config builder, test, receipt, preregistration and citation-gate key of this round is renamed O3 -> O4.
* **Preregistration:** `docs/ice/fidelity/si3_r8_o4_preregister.md` (section 4 lists the amendments made in this resume).
* **Decisions:** 153, 158, 159 (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_si3/standing_note.md`).

## 1. Outcome

The two independent reviews (Claude, codex) that HELD the model confirmed the sweep arithmetic, limits, sweep order, Hbig/Hsnow/zapneg and the fold signs bit for bit; they held it on the seven items below. Each is fixed and pinned by a committed test that fails when the fix is reverted (section 3). Labels: CONFIRMED = shown by a committed test or a line of NEMO source quoted here; PLAUSIBLE = inferred.

## 2. The seven review items

### 2.1 Delayed CFL (CONFIRMED)

NEMO takes the maximum CFL through `CALL mpp_max( 'icedyn_adv_pra', zcfl, cdelay = 'cflice' )` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:136`). The deck's resolved value is printed in the admitted O1 record as `ln_mppdelay =  T` (`si3_rounds/round4/refused/oracle_o1_orca2_sas_ice_admission_refused_20261024T161410Z_1010784/ref_2step/ocean.output:43`), so `llisdelay = ln_mppdelay .AND. PRESENT(cdelay)` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lib_mpp.f90:789`) is true. On the first call without a restart the buffer is filled from the current value (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lib_mpp.f90:823-828`); on every later call `todelay(idvar)%zdpbuffin(:) = ptab(:)` stores the current value and `ptab(:) = todelay(idvar)%zdpbuffout(:)` returns the previous call's (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lib_mpp.f90:834-840`). A restart carries the buffer as `DELAY__r8_cflice` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/iom.f90:1396-1400`, written at `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icerst.f90:138`, read at `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icerst.f90:353`). The cycle count is `IF    ( zcfl > 1.5 ) THEN   ;   icycle = 3` / `ELSEIF( zcfl >  .5 ) THEN   ;   icycle = 2` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:138-141`).

The model now carries `cflice_delayed` on the state (`packages/ice/legoesm/ice/si3_orca2_state.py:367`), uses it as `zcfl` when present and stores the current value for the next call (`packages/ice/legoesm/ice/si3_orca2_state.py:3244-3251`), and round-trips it through the restart (`packages/ice/legoesm/ice/si3_orca2_state.py:1684-1686`). The host-side conversion of `zcfl` and the subcycle override argument are gone: `icycle` is traced, all three cycle counts are traced and the selected one is kept (`packages/ice/legoesm/ice/si3_orca2_state.py:3055-3125`). NEMO does not make the count static; it is runtime state bounded to {1, 2, 3} by the lines quoted above, which is how the model bounds it. The test that codified the current-step version now pins the one-call lag in both directions (call 2 uses call 1's value, call 3 uses call 2's) and a plant that ignores the lag fails (`tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:821-1122`); one compiled step also lets a traced `cflice_delayed` select each of the three counts (`tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:2150-2241`).

### 2.2 Hidden defaults (CONFIRMED)

Missing `sss_m`, `sst_m`, `umask`, `vmask` or ORCA2 config now raise instead of becoming the reference salinity, 0, `tmask` or a default config. The O4 builder takes one required argument and validates it (`packages/ice/legoesm/ice/config.py:628-642`). The step, dynamics and advection drivers have no defaulted inputs: step index, config, velocity selector, `cflice_delayed`, the top-category bound and all six accumulators are required (`packages/ice/legoesm/ice/si3_orca2_state.py:3140-3317`, `packages/ice/legoesm/ice/si3_orca2_state.py:3374-3804`, `packages/ice/legoesm/ice/si3_orca2_state.py:3807-3960`). The accumulators are zeroed by the step exactly where NEMO's `diag_set0` zeroes them (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:485`, `:491`, `:502`, `:521-523`). The shared kernels lost every defaulted namelist value and density (`nn_hls`, `nn_icesal`, `ln_pnd_LEV`, `ln_pnd_TOPO`, `rn_simin`, `rn_sinew`, `rt0`, `rho0`, `rhoi`, `rhos`, `rhow`, top-category bound); only private plant hooks keep defaults. The refusals are pinned in `tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:1125-1468` and `tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:1471-1671`.

### 2.3 Duplication (CONFIRMED)

Searched `packages/ice/legoesm/ice/` for every Prather sweep, limiter, receiver merge, Hbig, Hsnow, zapneg, zapsmall and 9-point maximum: the lane-3 sweep kernels were the only implementations, plus the ORCA2 copies. The ORCA2 `adv_x`/`adv_y` copies are deleted; the lane-3 kernels take the loop domains, `pcrh`, `ihls` and `psm` that ORCA2 needs as arguments (`packages/ice/legoesm/ice/transport.py:858-1348`, `packages/ice/legoesm/ice/transport.py:1351-1826`), and the ORCA2 entry points are thin wrappers (`packages/ice/legoesm/ice/transport.py:2233-2275`, `packages/ice/legoesm/ice/transport.py:2278-2318`). The lane-3 correction ledger now calls the one Hbig/Hsnow/zapneg/zapsmall/9-point-maximum implementation (`packages/ice/legoesm/ice/fidelity/nemo_adv2d_testcase_recipe.py:388-613`), with its own historical arithmetic order selected by explicit arguments (`multiply_before_divide`, `use_reciprocal_rhos`, `apply_hi_cap`), so ORCA2 and lane 3 share code but not rounding.

Lane 3 is bit-identical to the parent. A clean `git archive` of the parent and the current tree each run the committed dump (`scripts/validate/ocean_fidelity/testcases/nemo_si3_lane3_prather_dump.py:59-150`) in a fresh interpreter; all 94 arrays (ICE_ADV1D and ICE_ADV2D cards over several steps, both correction orders, the shared ledger in both calling conventions with Hsnow/Hbig/zapneg changing 14 of 16 tracers, both sweep kernels, and the production `step_sea_ice`) agree at 0 ULP, and flipping Hbig's divide/multiply order inside the lane-3 ledger makes the ledger arrays differ (`tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:1713-1744`). The ICE_RHEO card needs its oracle record, so it is covered by its own record-bound tests (section 3), not by the dump.

### 2.4 No stub selector (CONFIRMED)

The `si3_orca2_o3` branch of `step_sea_ice` whose only body raised `NotImplementedError` is removed, and the unrelated reformat of the ridging guard is reverted: `packages/ice/legoesm/ice/sea_ice.py` is byte-identical to the parent. The O4 path is reached through its own entry point (`packages/ice/legoesm/ice/si3_orca2_state.py:3807-3960`); `step_sea_ice` refuses `si3_orca2_o4` as an unknown scheme until rung O16 supplies the ocean-flux response.

### 2.5 Fold test (CONFIRMED)

The test carries an independent scalar-loop replica of `lbc_lnk` for T points (`tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:536-614`): seam first, then the T-point fold transcribed from `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lbcnfd.f90:96-150`, then the closed south rows: the order of the single-process exchange, east-west (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lbclnk.f90:1987`), north fold (`:2105`), south and north (`:2118`). Signs are the ones NEMO passes: `CALL lbc_lnk( 'icedyn_adv_pra', pv_i  , 'T', 1._wp, sxice , 'T', -1._wp, syice , 'T', -1._wp  & ! ice volume` and `&                          , sxxice, 'T', 1._wp, syyice, 'T',  1._wp, sxyice, 'T',  1._wp  &` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:451-452`). After a two-cycle step every extensive and every moment of all 11 families matches the replica at every fold row and seam column, each compared field is checked non-zero there, and a wrong sign planted on each of `sx`, `sy`, `sxx`, `syy`, `sxy` separately fails. The conservation test keeps ice at and next to the fold (rows 148, 149) with flow across both.

### 2.6 Gradient test (CONFIRMED on a reduced roster; FINDING on the full one)

Besides forward = reverse (relative `<= 1e-11`), the check takes central finite differences along a smooth `u_ice` direction and along `v_i` with step `1e-6` and requires the ratio to the AD derivative within the preregistered `1e-6` of 1 at float64 (`tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:2244-2438`, run by `tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:2441-2466`). The measured ratios are in section 3.

FINDING: the reverse-mode program of the full step does not compile in practice on the CPU. On a 24 x 20 cyclic + T-fold grid, XLA at its default backend level exceeded 40 GB compiling the advection reverse program alone (its jaxpr is 359 equations); with backend optimisation off, the full 5-category 10+5-layer step ran more than 2 h at 58 GB without finishing. The agy sessions' timeouts were this. The committed check therefore runs in a fresh interpreter with backend optimisation off, on 1 category and 1+1 layers with all 11 moment families (one reverse program then compiles in about 4 minutes at 15 GB). Forward bits are unaffected: the primal and jit checks run in-process at production settings. Online training through the full O4 step needs this compile cost fixed first.

### 2.7 `t_si`, quotes, manifest, CHOICES (CONFIRMED)

`t_si` has no halo cells in NEMO: it is allocated on the interior only, `ALLOCATE( t_si(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpl) , tm_si(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) , qcn_ice_bot(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpl) , qcn_ice_top(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpl) , STAT = ierr(ii) )` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/ice.f90:465`), and `diag_set0` sets `t_si       (ji,jj,jl) = rt0     ! temp at the ice-snow interface` on that interior (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:529`). The padded `(184, 152, jpl)` array therefore has nothing to fill outside the interior; the zero padding is the convention the O1 record writer uses, `pdst(:,:,:) = 0.0_real64` before copying the interior (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/o1_si3_state_writer.F90:35-40`). Every Fortran quote in the preregistration was rechecked against the ppsrc by a whitespace-normalised substring match; wildcarded calls, one-line renderings of continued statements and composite `WHERE` quotes were replaced by the verbatim lines. The evidence manifest digest is recomputed in section 4.

`ice_stp` calls `ice_var_glo2eqv(2)` twice with `ice_var_agg(1)` between (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:226-227`, `:254-255`). With thermodynamics off nothing changes the state in between, so the step evaluates the conversion once and `ice_var_agg(2)`; the gradient test asserts that a second conversion of the step output changes nothing at 0 ULP. `ice_sbc_flx` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:245`) belongs to O7 and the rest of `ice_update_flx` to O16.

## 3. Verification (CPU, float64, `JAX_ENABLE_X64=1`)

Model code is identical at `10689480ac` and `5571978732` (the second commit changes only tests and documents), so each run is quoted with the commit it ran on.

| Run | Commit | Decisive line |
|---|---|---|
| O4 test file (8 tests) | `5571978732` | `8 passed in 8160.67s (2:16:00)`; AD line `O4_AD_CHECKS_PASS fd_u_ratio=1.000000346896843 fd_vi_ratio=0.9999999995498314 rel_ad_diff=8.10341647527711e-16` |
| Lane-3 ice fidelity suite (`tests/ice/fidelity/`, includes the ICE_ADV1D, ICE_ADV2D, ICE_RHEO record-bound gates) | `10689480ac` | `111 passed in 1220.22s (0:20:20)` |
| Default gate (the landing set, includes the citation-gate test and private-import ratchet) | `10689480ac` | `145 passed in 1338.78s (0:22:18)` |
| Private-import ratchet alone | `10689480ac` | `6 passed in 5.67s` |
| Per-receipt citation gate, all six SI3 receipts, full-map audit | `10689480ac` | each `PASS`, unmapped 0, failures 0, map entries failing audit 0, self-test 9/9 |

The lane-3 suite run on the uncommitted tree had failed 19 tests only because the provenance stamp refuses a dirty tree; on the committed tree all pass. Compile-cost numbers in 2.6: the 14.6 GB / 257 s reverse compile is in `prof_d.log`; the > 40 GB and > 2 h / 58 GB figures were read by a memory watchdog that stopped the runs and are not archived.

## 4. Record contract for the O4 acquisition round, and evidence

The acquisition round must record, for two consecutive ice steps (`kt = 1`, x then y; `kt = 2`, y then x) of `ORCA2_SAS_ICE` with dynamics reduced to Prather advection and thermodynamics off: the horizontal metrics and the three surface masks; `u_ice`, `v_ice`, `sss_m`, `sst_m` at advection entry; BOTH the current and the delayed `zcfl` and the resulting `icycle` at every step, plus `DELAY__r8_cflice` in every restart; the full state and all 55 moment arrays before advection, after advection, after `ice_var_zapsmall` plus the end-of-dynamics halo exchange, and at the end of the step; the six advection diagnostics and five transport diagnostics. Which dynamics overlay produces "Prather only" is a deck choice not yet made (section 5).

Evidence directory `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round8/resume4_claude` holds the logs above (`final2_o4.log`, `final_lane3.log`, `final_default.log`, `final_ratchet.log`, `d_ad.log`, `prof_d.log`), the six gate JSONs and the codex prompt and log. The round-8 manifest `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round8/SHA256SUMS` was rebuilt over every file in the round-8 directory; its own SHA-256 is `2071c56f2fbaf6b034bbf0fecb6fae4f7219bd182f9effe74cc1ddf6dcd7d55a`. The earlier receipt's `66a656...` value was wrong; the manifest it described hashed to `89238530c24655ec25a6c2613905dc0e05f213dd3f3135c07bd498543cb00c8d`, as codex found.

## 5. CHOICES and review

* `ASKED` (operator brief): rename O3 -> O4 and cite the O4 design row; transcribe the delayed CFL; refuse missing inputs and remove builder defaults; generalise the lane-3 kernels and delete the copies; remove the stub selector and revert the ridging reformat; independent fold replica with per-moment sign plants and ice at the fold; central finite-difference check; fill or justify the `t_si` halo; re-quote sources; recompute the manifest digest; list choices honestly.
* `ASKED` (design row `docs/ice/fidelity/si3_ladder_design.md:224`): the advection-only mode, with dynamics and thermodynamics otherwise off, is what O4 admits.
* `UNASKED` (offered for revert): the new config field `ln_mppdelay` defaults to true, the NEMO reference value and the deck's resolved value (`ORCA2_SAS_ICE_O1_REF/EXP00/namelist_ref:1515` and the record line quoted in 2.1).
* `UNASKED` (offered for revert): `rn_uice`/`rn_vice` both `None` selects replay of the state's velocities as exact input; both given selects NEMO's prescribed-velocity mode. The caller must state which.
* `UNASKED` (offered for revert): the lane-3 control's parent is `042c7bd74c2944dab8c39318687ab0cef54f9d2b` (the commit the preregistration names), and its dump covers several card steps rather than a full trajectory.
* `UNASKED` (offered for revert): the dump script lives in `scripts/validate/ocean_fidelity/testcases/` and is run by the O4 test, not by a separate test file.
* `UNASKED` (offered for revert): the AD check's grid (24 x 20), roster (1 category, 1+1 layers) and XLA backend optimisation level (off, subprocess only); the in-process primal check's grid (24 x 20, full roster).
* Review: codex (independent, read-only, on the uncommitted diff before `10689480ac`): no defect in the seven areas; HOLD on the then-unmigrated citation gate and stale receipt, both fixed. The later test-d rework (`5571978732`) is UNREVIEWED. GLM: NOT RUN (no GLM tool in this session), so the dual-review rule is not met.

## 6. Resume-5 rebase onto the current SI3 lane

Resume 5 rebased the seven round-8 commits from `042c7bd74c` onto the current
lane tip `e795e17482` (O2 `62a28e1324`, O3 `1bb17df9dd`, O5
`f76638f1a7`, and O7 `e795e17482`). The rebase produced the following textual
conflicts; files not listed here merged automatically.

1. Applying original O4 implementation `7b85ef1ca2`,
   `packages/ice/legoesm/ice/config.py`: the lane side supplied the landed O2
   builder; round 8 supplied the then-named O3 Prather validator and builder.
   Resolution: retained the complete O2 builder and added the complete Prather
   validator/builder after it. The later review-fix commit renamed only the
   round-8 path to O4.
2. Applying `7b85ef1ca2`, `packages/ice/legoesm/ice/sea_ice.py`: the lane side
   admitted and dispatched O2; round 8 admitted and dispatched the then-named
   O3 path and its temporary refusal stub. Resolution: retained both selector
   lists and both dispatch branches at this commit, without changing the
   landed ridging code. The later review-fix commit removed the round-8 stub as
   intended while leaving O2 intact.
3. Applying receipt/map commit `bfffae32af`,
   `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py`:
   the lane side contained all landed O2/O3/O5/O7 file and citation entries;
   round 8 supplied its preregistration, receipt, test, NEMO-source entries and
   receipt-name parser alternatives. Resolution: took the union of both
   `FILES` and `CITATION_MAP` blocks and the union of receipt-name alternatives;
   no landed map entry was removed.
4. Applying review-fix/rename commit `10689480ac`,
   `packages/ice/legoesm/ice/config.py`: the rebased side contained O2 plus the
   temporary round-8 O3 builder; the commit renamed and hardened that builder
   as O4 with an explicit required `SI3ORCA2StateConfig`. Resolution: retained
   O2 unchanged and applied the O3-to-O4 replacement and explicit-config
   validation only to round 8.
5. Applying `10689480ac`, `packages/ice/legoesm/ice/sea_ice.py`: the rebased
   side contained landed O2 plus the temporary round-8 O3 selector, dispatch,
   and `NotImplementedError` stub; the commit removed the round-8 selector and
   stub because O4 is exposed only through its dedicated entry point.
   Resolution: retained O2 in the liquidus refusal, allowed-selector list and
   dispatcher, and removed only the obsolete round-8 O3/O4 dispatcher/stub.
6. Applying `10689480ac`, the citation map: the rebased side contained every
   landed rung block plus round 8's old O3 names; the commit supplied O4 file
   names, citations and parser alternatives. Resolution: retained every landed
   O2/O3/O5/O7 entry, replaced only round 8's old O3 block with its O4 block,
   and kept all landed plus O4 receipt-name alternatives. The duplicate-key
   gate remains authoritative for the merged literals.

`packages/ice/legoesm/ice/si3_orca2_state.py`,
`packages/ice/legoesm/ice/__init__.py`, both round-8 receipt/preregistration
renames, and the O4 test rename merged without textual conflicts. Their landed
and round-8 symbols remain subject to the post-rebase rung suites and
parent-vs-current controls below. No O4 acquisition tool was run.
