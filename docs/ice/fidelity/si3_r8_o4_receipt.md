# SI3 Ladder Round 8 (`si3_r8`) Receipt — Rung O4 Model Preparation (ORCA2 Prather Advection Across the Cyclic Seam and North Fold, No Acquisition)

* **Round:** `si3_r8`, resume 4 (Claude taking over from the agy author), model preparation only: no NEMO build or run, no acquisition.
* **Branch / parent:** `fidelity/nemo-testcases-si3-ladder`; lane-3 bit-identity is controlled against the round's real parent, the lane tip `d2402fd17975be7905d9ecc990920a2e334004b6` (O11 landed, before any round-8 commit; section 8). Earlier resumes controlled against `042c7bd74c2944dab8c39318687ab0cef54f9d2b`.
* **Rung:** O4, design row `docs/ice/fidelity/si3_ladder_design.md:224` ("ORCA2_SAS_ICE with Prather only; dynamics/thermo otherwise off"). The operator brief called this O3; O3 is the initial HFN thickness fit (`docs/ice/fidelity/si3_ladder_design.md:223`). Every function, config builder, test, receipt, preregistration and citation-gate key of this round is renamed O3 -> O4.
* **Preregistration:** `docs/ice/fidelity/si3_r8_o4_preregister.md` (section 4 lists the amendments made in this resume).
* **Decisions:** 153, 158, 159, 170 (O4 acquisition overlay) (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_si3/standing_note.md`).

## 1. Outcome

The two independent reviews (Claude, codex) that HELD the model confirmed the sweep arithmetic, limits, sweep order, Hbig/Hsnow/zapneg and the fold signs bit for bit; they held it on the seven items below. Each is fixed and pinned by a committed test that fails when the fix is reverted (section 3). Labels: CONFIRMED = shown by a committed test or a line of NEMO source quoted here; PLAUSIBLE = inferred.

## 2. The seven review items

### 2.1 Delayed CFL (CONFIRMED)

NEMO takes the maximum CFL through `CALL mpp_max( 'icedyn_adv_pra', zcfl, cdelay = 'cflice' )` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:136`). The deck's resolved value is printed in the admitted O1 record as `ln_mppdelay =  T` (`si3_rounds/round4/refused/oracle_o1_orca2_sas_ice_admission_refused_20261024T161410Z_1010784/ref_2step/ocean.output:43`), so `llisdelay = ln_mppdelay .AND. PRESENT(cdelay)` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lib_mpp.f90:789`) is true. On the first call without a restart the buffer is filled from the current value (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lib_mpp.f90:823-828`); on every later call `todelay(idvar)%zdpbuffin(:) = ptab(:)` stores the current value and `ptab(:) = todelay(idvar)%zdpbuffout(:)` returns the previous call's (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lib_mpp.f90:834-840`). A restart carries the buffer as `DELAY__r8_cflice` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/iom.f90:1396-1400`, written at `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icerst.f90:138`, read at `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icerst.f90:353`). The cycle count is `IF    ( zcfl > 1.5 ) THEN   ;   icycle = 3` / `ELSEIF( zcfl >  .5 ) THEN   ;   icycle = 2` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:138-141`).

The model now carries `cflice_delayed` on the state (`packages/ice/legoesm/ice/si3_orca2_state.py:367`), uses it as `zcfl` when present and stores the current value for the next call (`packages/ice/legoesm/ice/si3_orca2_state.py:3244-3251`), and round-trips it through the restart (`packages/ice/legoesm/ice/si3_orca2_state.py:1684-1686`). The host-side conversion of `zcfl` and the subcycle override argument are gone: `icycle` is traced, all three cycle counts are traced and the selected one is kept (`packages/ice/legoesm/ice/si3_orca2_state.py:3055-3125`). NEMO does not make the count static; it is runtime state bounded to {1, 2, 3} by the lines quoted above, which is how the model bounds it. The test that codified the current-step version now pins the one-call lag in both directions (call 2 uses call 1's value, call 3 uses call 2's) and a plant that ignores the lag fails (`tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:834-1135`); one compiled step also lets a traced `cflice_delayed` select each of the three counts (`tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:2214-2312`).

### 2.2 Hidden defaults (CONFIRMED)

Missing `sss_m`, `sst_m`, `umask`, `vmask` or ORCA2 config now raise instead of becoming the reference salinity, 0, `tmask` or a default config. The O4 builder takes one required argument and validates it (`packages/ice/legoesm/ice/config.py:641-655`). The step, dynamics and advection drivers have no defaulted inputs: step index, config, velocity selector, `cflice_delayed`, the top-category bound and all six accumulators are required (`packages/ice/legoesm/ice/si3_orca2_state.py:3140-3317`, `packages/ice/legoesm/ice/si3_orca2_state.py:3374-3804`, `packages/ice/legoesm/ice/si3_orca2_state.py:3807-3960`). The accumulators are zeroed by the step exactly where NEMO's `diag_set0` zeroes them (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:485`, `:491`, `:502`, `:521-523`). The shared kernels lost every defaulted namelist value and density (`nn_hls`, `nn_icesal`, `ln_pnd_LEV`, `ln_pnd_TOPO`, `rn_simin`, `rn_sinew`, `rt0`, `rho0`, `rhoi`, `rhos`, `rhow`, top-category bound); only private plant hooks keep defaults. The refusals are pinned in `tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:1138-1481` and `tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:1535-1735`.

### 2.3 Duplication (CONFIRMED)

Searched `packages/ice/legoesm/ice/` for every Prather sweep, limiter, receiver merge, Hbig, Hsnow, zapneg, zapsmall and 9-point maximum: the lane-3 sweep kernels were the only implementations, plus the ORCA2 copies. The ORCA2 `adv_x`/`adv_y` copies are deleted; the lane-3 kernels take the loop domains, `pcrh`, `ihls` and `psm` that ORCA2 needs as arguments (`packages/ice/legoesm/ice/transport.py:860-1350`, `packages/ice/legoesm/ice/transport.py:1353-1828`), and the ORCA2 entry points are thin wrappers (`packages/ice/legoesm/ice/transport.py:2235-2277`, `packages/ice/legoesm/ice/transport.py:2280-2320`). The lane-3 correction ledger now calls the one Hbig/Hsnow/zapneg/zapsmall/9-point-maximum implementation (`packages/ice/legoesm/ice/fidelity/nemo_adv2d_testcase_recipe.py:388-613`), with its own historical arithmetic order selected by explicit arguments (`multiply_before_divide`, `use_reciprocal_rhos`, `apply_hi_cap`), so ORCA2 and lane 3 share code but not rounding.

Lane 3 is bit-identical to the parent. A clean `git archive` of the parent and the current tree each run the committed dump (`scripts/validate/ocean_fidelity/testcases/nemo_si3_lane3_prather_dump.py:59-150`) in a fresh interpreter; all 94 arrays (ICE_ADV1D and ICE_ADV2D cards over several steps, both correction orders, the shared ledger in both calling conventions with Hsnow/Hbig/zapneg changing 14 of 16 tracers, both sweep kernels, and the production `step_sea_ice`) agree at 0 ULP, and flipping Hbig's divide/multiply order inside the lane-3 ledger makes the ledger arrays differ (`tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:1777-1808`). The ICE_RHEO card needs its oracle record, so it is covered by its own record-bound tests (section 3), not by the dump.

### 2.4 No stub selector (CONFIRMED)

The `si3_orca2_o3` branch of `step_sea_ice` whose only body raised `NotImplementedError` is removed, and the unrelated reformat of the ridging guard is reverted: `packages/ice/legoesm/ice/sea_ice.py` is byte-identical to the parent. (Resume 5's rebases had reintroduced a reformat of the O2 refusal message; resume 6 removed it, section 8.) The O4 path is reached through its own entry point (`packages/ice/legoesm/ice/si3_orca2_state.py:3807-3960`); `step_sea_ice` refuses `si3_orca2_o4` as an unknown scheme until rung O16 supplies the ocean-flux response.

### 2.5 Fold test (CONFIRMED)

The test carries an independent scalar-loop replica of `lbc_lnk` for T points (`tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:549-627`): seam first, then the T-point fold transcribed from `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lbcnfd.f90:96-150`, then the closed south rows: the order of the single-process exchange, east-west (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lbclnk.f90:1987`), north fold (`:2105`), south and north (`:2118`). Signs are the ones NEMO passes: `CALL lbc_lnk( 'icedyn_adv_pra', pv_i  , 'T', 1._wp, sxice , 'T', -1._wp, syice , 'T', -1._wp  & ! ice volume` and `&                          , sxxice, 'T', 1._wp, syyice, 'T',  1._wp, sxyice, 'T',  1._wp  &` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:451-452`). After a two-cycle step every extensive and every moment of all 11 families matches the replica at every fold row and seam column, each compared field is checked non-zero there, and a wrong sign planted on each of `sx`, `sy`, `sxx`, `syy`, `sxy` separately fails. The conservation test keeps ice at and next to the fold (rows 148, 149) with flow across both.

### 2.6 Gradient test (CONFIRMED on a reduced roster; FINDING on the full one)

Besides forward = reverse (relative `<= 1e-11`), the check takes central finite differences along a smooth `u_ice` direction and along `v_i` with step `1e-6` and requires the ratio to the AD derivative within the preregistered `1e-6` of 1 at float64 (`tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:2315-2509`, run by `tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:2512-2537`). The measured ratios are in section 3.

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

The acquisition round must record, for two consecutive ice steps (`kt = 1`, x then y; `kt = 2`, y then x) of `ORCA2_SAS_ICE` with dynamics reduced to Prather advection and thermodynamics off: the horizontal metrics and the three surface masks; `u_ice`, `v_ice`, `sss_m`, `sst_m` at advection entry; BOTH the current and the delayed `zcfl` and the resulting `icycle` at every step, plus `DELAY__r8_cflice` in every restart; the full state and all 55 moment arrays before advection, after advection, after `ice_var_zapsmall` plus the end-of-dynamics halo exchange, and at the end of the step; the six advection diagnostics and five transport diagnostics. DECISION 170 chose the overlay: `ln_dynADV2D=T` with `namelist_ice_ref` `rn_uice=rn_vice=0.5` m/s. With constant velocities `zcfl` is the same on every call, so this record cannot exercise the one-call CFL lag of 2.1 (section 8 names the record that will).

Evidence directory `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round8/resume4_claude` holds the logs above (`final2_o4.log`, `final_lane3.log`, `final_default.log`, `final_ratchet.log`, `d_ad.log`, `prof_d.log`), the six gate JSONs and the codex prompt and log. The round-8 manifest `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round8/SHA256SUMS` was rebuilt over every file in the round-8 directory; its own SHA-256 is `2071c56f2fbaf6b034bbf0fecb6fae4f7219bd182f9effe74cc1ddf6dcd7d55a`. The earlier receipt's `66a656...` value was wrong; the manifest it described hashed to `89238530c24655ec25a6c2613905dc0e05f213dd3f3135c07bd498543cb00c8d`, as codex found.

## 5. CHOICES and review

* `ASKED` (operator brief): rename O3 -> O4 and cite the O4 design row; transcribe the delayed CFL; refuse missing inputs and remove builder defaults; generalise the lane-3 kernels and delete the copies; remove the stub selector and revert the ridging reformat; independent fold replica with per-moment sign plants and ice at the fold; central finite-difference check; fill or justify the `t_si` halo; re-quote sources; recompute the manifest digest; list choices honestly.
* `ASKED` (design row `docs/ice/fidelity/si3_ladder_design.md:224`): the advection-only mode, with dynamics and thermodynamics otherwise off, is what O4 admits.
* `UNASKED` (offered for revert): the new config field `ln_mppdelay` defaults to true, the NEMO reference value and the deck's resolved value (`ORCA2_SAS_ICE_O1_REF/EXP00/namelist_ref:1515` and the record line quoted in 2.1).
* `UNASKED` (offered for revert): `rn_uice`/`rn_vice` both `None` selects replay of the state's velocities as exact input; both given selects NEMO's prescribed-velocity mode. The caller must state which.
* `ASKED` (operator brief, resume 6): the lane-3 control's parent is the real parent `d2402fd17975be7905d9ecc990920a2e334004b6`. `UNASKED` (offered for revert): its dump covers several card steps rather than a full trajectory.
* `UNASKED` (offered for revert): the dump script lives in `scripts/validate/ocean_fidelity/testcases/` and is run by the O4 test, not by a separate test file.
* `UNASKED` (offered for revert): the AD check's grid (24 x 20), roster (1 category, 1+1 layers) and XLA backend optimisation level (off, subprocess only); the in-process primal check's grid (24 x 20, full roster).
* Review: codex (independent, read-only, on the uncommitted diff before `10689480ac`): no defect in the seven areas; HOLD on the then-unmigrated citation gate and stale receipt, both fixed. The later test-d rework (`5571978732`) is UNREVIEWED. GLM: NOT RUN (no GLM tool in this session), so the dual-review rule is not met.

## 6. Resume-5 rebase onto the current SI3 lane

Resume 5 first rebased the seven round-8 commits from `042c7bd74c` onto lane
tip `e795e17482` (O2 `62a28e1324`, O3 `1bb17df9dd`, O5 `f76638f1a7`, and O7
`e795e17482`). During the long foreground verification the lane advanced, so
the complete round was rebased again onto the then-current tip `d3f2cff704`,
which adds the O5 acquisition result. A final pre-handoff fetch found O7
acquisition and O8 had then landed, so the round was rebased a third time onto
`abd7262c11`. The three rebases produced the following textual conflicts;
files not listed here merged automatically.

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
7. In the second rebase, applying the original round-8 receipt/map commit, the
   citation map's lane side supplied the round-12 O5 admitted-record entries
   while round 8 supplied its old O3-named map block. Resolution: retained the
   two O5 admitted-record entries and the full round-8 block; no landed entry
   was replaced.
8. Applying the O4 review-fix/rename during the second rebase, the citation
   map's lane side contained the round-12 O5 entries followed by round 8's old
   O3 block, while the replayed commit replaced that old block with O4.
   Resolution: retained the O5 entries, discarded only the superseded
   round-8 O3 block, and retained the complete O4 replacement block.
9. Applying the landed-anchor repair during the second rebase, the citation
   map's lane side supplied the two O5 admitted-record entries while the
   replayed commit supplied the two landed O1 anchors used by the re-anchor
   chain. Resolution: retained both sets in the literal map.
10. Applying the composed-anchor repair during the second rebase, the citation
    map's lane side supplied the round-12 O5 dynamics, O1-tool, and O5-test
    re-anchor chain while the replayed commit supplied the O4/O2/O3 composed
    chain. Resolution: retained both complete chains; their touched files are
    disjoint, and the mandatory full-map audit validates their final targets.
11. Applying the receipt re-anchor commit during the second rebase,
    `docs/ice/fidelity/si3_r10_o5_receipt.md` conflicted between the newly
    landed O5 dynamics shift and O4's state-file shift. Resolution: used the
    O4-composed state anchor at line 1389 and the landed O5 fold-program anchor
    at line 1970, preserving both sides' current locations.
12. In the third rebase, applying the original round-8 receipt/map commit,
    the citation map's lane side supplied the landed O7 retained-record and O8
    frazil/production-wiring entries while round 8 supplied its old O3 block.
    Resolution: retained all O7/O8 entries followed by the full round-8 block.
    The same file's receipt-name-regex hunk kept the lane's generalized SI3
    receipt pattern instead of the older explicit round list, so the new O7/O8
    receipts remain discoverable.
13. Applying the O4 review-fix/rename in the third rebase, the citation-map
    lane side contained O7/O8 followed by the superseded round-8 O3 block,
    while the replayed commit supplied the O4 replacement. Resolution:
    retained O7/O8, removed only the old O3 block, and retained O4. Its second
    regex hunk again kept the lane's generalized receipt pattern.
14. Applying the landed-anchor repair in the third rebase, the citation-map
    lane side supplied O7/O8 while the replayed commit supplied the landed O1
    anchors. Resolution: retained both complete sets. The later composed
    re-anchor chains and mandatory full-map audit then resolved without a
    textual conflict.

`packages/ice/legoesm/ice/si3_orca2_state.py`,
`packages/ice/legoesm/ice/__init__.py`, both round-8 receipt/preregistration
renames, and the O4 test rename merged without textual conflicts. Their landed
and round-8 symbols remain subject to the post-rebase rung suites and
parent-vs-current controls below. No O4 acquisition tool was run.

## 7. Resume-5 post-rebase verification

All commands in this section ran in the foreground with the repository source
roots first on `PYTHONPATH`, CPU JAX, and float64 enabled. O1/O2/O3/O4/O5 and
the three parent controls ran on clean `38a64e9dadaa938f231d53f61f49a71a8992d45d`. Amendment (2026-10-29 provenance audit): this was a pre-rebase commit and is no longer reachable.
The final O7/O8, D5, citation, default, and ratchet runs used clean
`fb76ac2fe4493007a00415892f96d751341c8fd9`; Amendment (2026-10-29 provenance audit): this was a pre-rebase commit and is no longer reachable. A path-limited tree diff reported
`O4_TESTED_CLOSURE_BYTE_IDENTICAL_PASS` for the O4 model, shared kernels,
dump, and tests between those commits. The exact decisive lines were:

| Check | Decisive pass line |
|---|---|
| O1 | `15 passed in 178.41s (0:02:58)` |
| O2 | `12 passed in 244.31s (0:04:04)` |
| O3 | `10 passed in 97.86s (0:01:37)` |
| O4 full file | `8 passed in 8935.89s (2:28:55)` |
| O5 default-marker suite | `21 passed, 1 deselected in 90.60s (0:01:30)` |
| Landed O5 acquisition gate | `17 passed in 26.32s` |
| O7 surface, bulk-flux, round-314 audit, and landed acquisition tests | `61 passed, 1 xfailed in 165.46s (0:02:45)` |
| O8 frazil unit suite | `13 passed in 1.64s` |
| O8 C1D shared-frazil production controls | `4 passed in 22.24s` |
| Lane-3 Prather parent-vs-current control | `1 passed in 73.72s (0:01:13)` |
| Lane-3 RHG parent-vs-current control | `1 passed in 17.67s` |
| Lane-3 RHEO parent-vs-current control, explicitly opted in with `-m slow` | `1 passed in 459.66s (0:07:39)` |
| Citation-gate test suite | `32 passed in 6.95s` |
| Default landing gate | `160 passed in 1287.41s (0:21:27)` |
| Private-import ratchet alone | `6 passed in 4.99s` |

The O5 default marker expression intentionally deselected the slow RHEO
control; the next standalone row ran that exact node with its required
`-m slow` opt-in. The O7 xfail is the suite's declared expected case.

The D5 C1D production-JIT year probe evaluated all 8,760 steps twice. Its
decisive lines were:

* `D5_YEAR_PASS: arm=default execution_mode=jax.jit steps=8760 hash=b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`
* `D5_YEAR_PASS: arm=--xla_cpu_max_isa=AVX execution_mode=jax.jit steps=8760 hash=b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`
* `D5_CPU_AVX_EQUAL_PASS: b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2 == b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2 == b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`

The per-receipt citation gate audited each receipt plus the complete
2,934-entry map. The exact pass lines were:

* `CITATION_RECEIPT_PASS: docs/ice/fidelity/si3_r2_d5_receipt.md status=PASS citations=49 full_map_failures=0`
* `CITATION_RECEIPT_PASS: docs/ice/fidelity/si3_r3_d5_receipt.md status=PASS citations=96 full_map_failures=0`
* `CITATION_RECEIPT_PASS: docs/ice/fidelity/si3_r4_o1_receipt.md status=PASS citations=8 full_map_failures=0`
* `CITATION_RECEIPT_PASS: docs/ice/fidelity/si3_r5_d5_receipt.md status=PASS citations=53 full_map_failures=0`
* `CITATION_RECEIPT_PASS: docs/ice/fidelity/si3_r6_d5_receipt.md status=PASS citations=69 full_map_failures=0`
* `CITATION_RECEIPT_PASS: docs/ice/fidelity/si3_r7_o2_receipt.md status=PASS citations=9 full_map_failures=0`
* `CITATION_RECEIPT_PASS: docs/ice/fidelity/si3_r8_o4_receipt.md status=PASS citations=48 full_map_failures=0`
* `CITATION_RECEIPT_PASS: docs/ice/fidelity/si3_r9_o3_receipt.md status=PASS citations=44 full_map_failures=0`
* `CITATION_RECEIPT_PASS: docs/ice/fidelity/si3_r10_o5_receipt.md status=PASS citations=38 full_map_failures=0`
* `CITATION_RECEIPT_PASS: docs/ice/fidelity/si3_r11_o7_receipt.md status=PASS citations=38 full_map_failures=0`
* `CITATION_RECEIPT_PASS: docs/ice/fidelity/si3_r12_o5_acquisition_receipt.md status=PASS citations=11 full_map_failures=0`
* `CITATION_RECEIPT_PASS: docs/ice/fidelity/si3_r13_o7_acquisition_receipt.md status=PASS citations=8 full_map_failures=0`
* `CITATION_RECEIPT_PASS: docs/ice/fidelity/si3_r13_o7_admission_receipt.md status=PASS citations=7 full_map_failures=0`
* `CITATION_RECEIPT_PASS: docs/ice/fidelity/si3_r15_o8_receipt.md status=PASS citations=23 full_map_failures=0`

The composed map itself reported `MAP_ENTRIES 2934`, `AUDIT_FAILURES 0`, and
`DUPLICATE_LITERALS []`. No O4 acquisition command was started during resume
5. The acquisition overlay has since been decided (DECISION 170).

## 8. Resume 6 (Claude): review HOLD items

Two reviews (Claude, Argon) HELD the rebased O4. Each item below is fixed or
answered; labels as in section 1.

1. **Statement barriers (CONFIRMED).** XLA regroups `(x*c1)*c2` into
   `x*(c1*c2)` even when only one factor is a constant, so an outer
   `nemo_source_round` alone did not hold NEMO's left-to-right order. Every
   intermediate of the Hbig_pra, Hsnow_pra, `ice_var_zapneg` and
   `ice_var_zapsmall` flux statements is now barriered as NEMO evaluates it,
   e.g. `zwfx_res + ( pv_s - pa_i * phs_max ) * rhos * z1_dt` is
   `sr(sr(sr(v_s - target) * rhos) * z1_dt)`. An AST sweep of the O4 kernels
   and the O4 functions of the state module found no other arithmetic with an
   unbarriered arithmetic operand; the remaining hits are the lane-3-only
   `multiply_before_divide` and non-reciprocal Hsnow branches, which O4 does
   not execute. Every O4 divisor is a runtime array or a traced value, so the
   division-by-constant rewrite (also measured: 33,373 of 100,000 quotients)
   cannot occur on the O4 path.
   Measured on the test's 24 x 20 cyclic + T-fold state, kt 1 and 2, each
   `icycle`: production `jax.jit` against the whole call under
   `jax.disable_jit()` differed in 774 cells (wfx_res 87-117 and sfx_res 20-36
   per arm, 1-2 ULP) on the parent and in 0 cells after the fix.
2. **Truly eager test (CONFIRMED).** `test_o4_d` now runs its eager arms under
   `jax.disable_jit()` and compares raw bits (signed zeros included) on every
   floating leaf for all three `icycle` values. On the parent model it fails
   (`jit_primal_diag: 87 cells differ bitwise`); on the fixed model it passes.
   The subcycle comment no longer claims `lax.cond`: all three subcycles are
   computed and subcycle `jt` is kept where `icycle >= jt`.
3. **Signed-zero ties (CONFIRMED).** The object code evaluates
   `MAX( ABS( zs1new ) - zslpmax, zpsxx )` as `maxsd` with `psxx` as the
   source operand (`adv_x.constprop.1.isra.0+0x872`; the same in the second
   loop copy and in `adv_x.constprop.0`), likewise `zpsyy` in `adv_y`
   (`+0x8a6`), so a tie keeps `zpsxx`/`zpsyy`; `Hsnow_pra`'s
   `MIN( pa_ip, pa_i )` is `minsd` with `pa_i` as source (`+0x784`), so a tie
   keeps `pa_i`. `jnp.maximum`/`jnp.minimum` return `+0`/`-0` on any tie. Both
   now use the instruction's selection. The other executing sites cannot
   change bits on a tie: `MAX( 0, zps0 )`, `MAX( 0, ... )` in Hsnow and the
   `pato_i` update keep the literal `+0` (`maxsd zero, reg`), which
   `jnp.maximum` also returns; the slope caps, `epsi20` floors, the 9-point
   maxima and `MIN( phi_max, hi_max )` compare only non-zero values.
   `test_o4_a5` pins the three sites through the public `si3_adv_x`,
   `si3_adv_y` and `si3_hsnow_pra`; reverting any one site alone fails it. The
   scalar replica now uses MAXSD selection too (Python `max` kept the first
   operand). Lane 3 shares the limiter: its dump is bitwise identical to the
   parent on all 94 arrays, so no certified lane-3 change goes to the user.
4. **One category sum (CONFIRMED).** `_seq_sum_axis` is deleted; Hbig, Hsnow
   and zapsmall call `fortran_sum_categories` (function-scope import: the state
   module imports the kernels). FINDING, not acted on: gfortran's `SUM` starts
   from `+0` (`pxor` at `ice_dyn_adv_pra+0x1037a`, then `addsd`), while the shared
   helper starts from the first element, so an all-`-0` sum gives `-0` instead
   of `+0`. The helper is shared by certified rungs; changing it is the user's
   decision.
5. **Parent and stale receipt lines (CONFIRMED).** The lane-3 control pins the
   real parent `d2402fd179` and compares bits. The rebases had reintroduced a
   reformat of the O2 refusal in `sea_ice.py`; the file is byte-identical to
   the parent again, and the five re-anchors and two receipt edits that existed
   only for it are removed. DECISION 170 is recorded in the header and 4.
6. **CFL lag coverage (CONFIRMED by reading the deck; the future record is
   PLAUSIBLE until acquired).** Under DECISION 170 the velocities are the fixed
   `rn_uice = rn_vice = 0.5` m/s, so `zcfl` is the same on every call and the
   delayed value equals the current one: the O4 record cannot exercise the
   one-call lag. The first scored record whose velocities change between
   Prather calls is O6's production-deck acquisition (DECISION 174: rheology
   live, thermodynamics on); its record contract must carry the current and
   delayed `zcfl` and `icycle` at every step, as section 4 lists for O4. O17
   composes it over 100 steps. Until then the lag is pinned only by the
   synthetic tests of 2.1.
7. **Rebase onto the current lane tip.** Rebased all round-8 commits from
   `abd7262c11` onto `d2402fd179` (O11). Conflicts, each resolved keeping both
   sides:
   - Original O4 implementation, `config.py`: the O1/O3 validator call (round
     8) and O11's identity line (lane) are both kept; at end of file O11's
     `SI3LateralMeltConfig` and round 8's validator/builder are both kept.
   - Original receipt/map commit, citation map: O11's entries, then round 8's
     block.
   - O3 -> O4 rename commit, `config.py` (same two hunks) and the citation map:
     lane content kept, only round 8's superseded O3 lines replaced by O4.
   - Landed-anchor repair, citation map: O11's entries and the landed O1
     anchors both kept.
   Check: for every file, rebased tip minus pre-rebase tip equals exactly the
   lane's `abd7262c11..d2402fd179` change (identical changed lines in nine
   files; in `config.py` the same lines with the two blank lines placed before
   the following function).

Review: codex (independent, read-only, on 41626cbc7c..75089274a5) returned HOLD on one claim only: that the unbarriered residual add chains could be regrouped by XLA. Measured: production jit and op-by-op agree on 200,000 cells for a 6-term add chain, the where/subtract chain as written in the kernels, and `(a+b)+c`, and codex's own case `[1e16, -1e16, 1]` gives `1` both ways; the finding is refuted and no guard was added. Codex confirmed both tie pins under jit and eager. GLM: NOT RUN (no GLM tool in this session), so
the dual-review rule is not met.

Verification, CPU, float64, production `jit`, all on clean `75089274a5`
(evidence `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round8/resume6_claude/verify/`).
The O4 file was first run whole; the session's 2 h task limit stopped it during
the gradient test, after the other eight had passed, so both parts were re-run
to their summary lines.

| Check | Decisive pass line |
|---|---|
| O1 tests | `15 passed in 196.15s (0:03:16)` |
| O1 admitted-record rescore | `status PASS`, every `max_ulp` 0 |
| O2 tests | `12 passed in 286.83s (0:04:46)` |
| O2 admitted-record rescore | `status PASS`, every `max_ulp` 0 |
| O3 tests | `10 passed in 125.69s (0:02:05)` |
| O3 score / entry steps | `PASS`, `jit_vs_eager_max_ulp 0` / `PASS` |
| O4, eight non-gradient tests | `8 passed, 1 deselected in 552.53s (0:09:12)` |
| O4 gradient test | `1 passed in 9971.77s (2:46:11)`; `O4_AD_CHECKS_PASS fd_u_ratio=1.000000346896843 fd_vi_ratio=0.9999999995498314 rel_ad_diff=8.10341647527711e-16` |
| O5 unit (default markers) | `21 passed, 1 deselected in 103.11s (0:01:43)` |
| O5 acquisition gate tests | `17 passed in 28.11s` |
| O5 admitted-record rescore | `PASS_O5_ADMISSION` |
| Lane-3 RHEO control (`-m slow`) | `1 passed in 407.55s (0:06:47)` |
| O7 surface, bulk-flux, round-314 audit, acquisition tests | `61 passed, 1 xfailed in 171.12s (0:02:51)` |
| O7 admitted-record rescore | `PASS_O7_ADMISSION` |
| Lane-3 bulk-flux gate | `AT-BAR` |
| O8 frazil unit / C1D production controls | `13 passed in 2.38s` / `4 passed in 30.19s` |
| O11 lateral melt | `32 passed in 120.15s (0:02:00)` |
| Lane-3 ice fidelity suite | `111 passed in 1136.48s (0:18:56)` |
| Default landing gate (7 files) | `161 passed in 1297.74s (0:21:37)` |
| Private-import ratchet | `6 passed in 5.21s` |
| D5 year, default and `--xla_cpu_max_isa=AVX` | `D5_PLATFORM_FLOOR`; shipped arm `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2` in both; all 21 arm hashes equal the reviewed pre-rebase rescore in each arm |

The O4 lane-3 control (inside the eight) compares bits against `d2402fd179`.
The citation gate results for every SI3 receipt are below this table's commit.

## 9. Resume 7 (Claude): DECISION 180, shared category sum from +0

Evidence: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round8/resume7_claude/`.

1. **Rebase (CONFIRMED).** All 18 round-8 commits rebased from `53cd3c732d` onto
   the lane tip `cf45f424e3` (O9). Three conflicts, all in the citation map,
   each resolved keeping both sides: the O3 receipt entries (lane O9 block,
   then round 8's block); the O3 -> O4 rename (lane O9 lines kept, round 8's
   superseded O3 block replaced by its O4 block); the landed O1 anchors (lane
   O9 lines, then the anchors). Check: for all 22 files changed on either
   side, the rebased tip minus `53cd3c732d` equals, line for line, the lane's
   `d2402fd179..cf45f424e3` change (0 mismatches). Parent of the fix:
   `f638172d7b`.
2. **Fix (CONFIRMED).** The shared category sum now starts from a
   barriered `+0` and adds every category left to right, as gfortran's `SUM`
   does. A new production-jit test (all categories `-0` along either axis,
   nested sums, Fortran order) fails 3 of 3 on the parent, fails 3 of 3 with
   an unbarriered `+0` seed (XLA folds it away, as codex measured), and passes
   with the barrier; its eager arm runs under `jax.disable_jit()`.
   For finite inputs the old and new sums differ only where every term is
   `-0`: both become identical at the first term that is not `-0`.
3. **Every caller, parent vs current (CONFIRMED).** A committed probe ran
   each driver in a parent worktree and in the current tree, captured every
   floating output of every production `jax.jit` call, and compared bits; a
   second run counted, per call line, the cells whose summed categories are
   all `-0`. A known-answer driver showed exactly one changed cell (`-0` to
   `+0`). Coverage: all 61 production calls of the sum in the ice package
   were executed; all-`-0` cells: 0 at every call.

   | Driver | jit calls | leaves compared | leaves changed |
   |---|---|---|---|
   | O1 admission | 3 | 333 | 0 |
   | O2 admission | 4 | 443 | 0 |
   | O3 score / entry steps | 1 / 1 | 55 / 224 | 0 / 0 |
   | O5 admission | 36 | 78 | 0 |
   | O7 admission | 21 | 282 | 0 |
   | O1 / O2 / O3 / O5 tests | 20 / 29 / 18 / 2 | 1893 / 2673 / 1540 / 8 | 0 |
   | O4 tests (eight non-gradient) | 48 + 43 | 4608 + 4276 | 0 |

   The lane-3 correction dump (no jit; Hbig/Hsnow ledger, both ADV2D orders,
   ADV1D, production sea-ice step) is bitwise identical on all 94 arrays; the
   95th entry is the dump's own source path, which names each tree. The
   lane-3 RHEO control passed in both trees. D5 year, default and
   `--xla_cpu_max_isa=AVX`: all 166 hashes equal the parent's in each arm;
   shipped arm `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`
   on both.
   So no output of any caller changes on any record or test input, because
   none of them contains a cell whose categories are all `-0`. A future
   input that does will now give `+0`, as NEMO does.
4. **Admissions (CONFIRMED).** Re-run in both trees; every admission JSON is
   byte-identical between parent and current: O1 `c1c2ea1c`, O2 `3748bf86`,
   O3 score `7f468722`, O3 entry steps `beaba85b`, O5 `8f9d911f`, O7
   `67e799da`. No digest changed, so none needs explaining. (Both O3 runs
   stamp the scorer head `f638172d7b`: the fix was not yet committed.)
5. **Review.** Codex (read-only): HOLD three times, all on the probe — it
   could pass without the callers running; the call scan missed aliases,
   attribute calls and subpackages and accepted zero calls; and it counted a
   zero-count record as coverage. Each fixed with a plant in the probe's test;
   the rescan found the same 61 calls. Fourth pass: LAND. Codex confirmed the
   fix gives `+0` under jit and eager and changes only all-`-0` sums. GLM: NOT
   RUN (no GLM tool in this session), so the dual-review rule is not met.
6. **Not acted on.** Two other `+0`-seeded category sums with the same
   arithmetic already exist (the O7 surface module and the O9 thermodynamics
   module); merging them into the shared one is a refactor for the user to
   order.

Verification, CPU, float64, production `jit`:

| Check | Decisive pass line |
|---|---|
| Category-sum test + probe test | `7 passed in 12.31s` |
| Default landing gate (7 files) | `162 passed in 1462.36s (0:24:22)` |
| Lane-3 ice fidelity suite | `104 passed` + the 7 that refused a dirty tree mid-run, re-run clean: `7 passed in 75.58s` |
| Private-import ratchet | `6 passed in 5.55s` |
| Caller coverage | `CATSUM_COVERAGE production_calls=61 uncovered=0 all_negzero_cells=0` |
| O1 / O2 / O3 / O5 tests (each tree) | `15 passed` / `12 passed` / `10 passed` / `21 passed, 1 deselected` |
| O4 tests, eight non-gradient (each tree) | `5 passed` + `3 passed, 6 deselected` |
| O5 / O7 admission (each tree) | `PASS_O5_ADMISSION` / `PASS_O7_ADMISSION` |
| D5 year, default and AVX (each tree) | `D5_PLATFORM_FLOOR`, 166 of 166 hashes equal |

The O4 gradient test was not re-run: the sum change is a `+0` add, and no
input it uses has an all-`-0` sum (census above).
