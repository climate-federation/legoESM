# SI3 round 14 receipt — O6 model preparation: five-category ridging/rafting and `ice_cor(kt,1)`

## Own errors, first

1. My first `ice_cor` divided by the compile-time constant `rn_himin`; XLA
   rewrote it as a multiply by the reciprocal and one pond-area word moved
   by one ULP against the independent transcription. The divisor is now
   materialized, as O7 learned for `rtt0`.
2. My first porosity-ramp fixture cell held about 3 m of ice, not 10-20 m,
   so its plant could not bind. The cell now holds about 13 m and the test
   asserts it is selected.
3. My first full-suite run piped pytest through `grep` and hit the 10-minute
   background cap with no output; the slow lane-3 control was then
   deselected by the default marker. Both rerun, logs below.
4. (resume) This receipt said the parent text of all three touched files is
   an exact prefix of the current file. False for `si3_surface.py`: its
   `__all__` assignment was rewritten to add three names and three alias
   assignments were appended. No executable statement changed.
5. (resume) The MIN/MAX tie order was read at two sites only, NEMO's own
   conservation check was not the one tested, and two divisions in
   `itd_shiftice` were not source-rounded. Fixed below (codex HOLD).
6. (resume) My first lane-3 probe plant (`rn_hstar`) could not bind on that
   card, and my first `ice_cons_hsm` ignored `smask0_i` in its extrema; the
   first was caught by the plant itself, the second by codex.

## Outcome

O6's two exact-input pieces are in the one shared ice package as new,
selectable functions; no existing executable statement changed. Both equal an
independent scalar transcription bit for bit. No ORCA2 NEMO array was read
(the resume reads lane 3's ICE_RHEO record); the
acquisition is preregistered (`docs/ice/fidelity/si3_r14_o6_preregister.md`),
not built.

## Build cited

`ORCA2_SAS_ICE_O7_REC` (the production-deck record build, DECISION 165).
Flags (`ORCA2_SAS_ICE_O7_REC/BLD/arch_nemo.fcm:27`):
`%FCFLAGS             -fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize`.
Its `icedyn_rdgrft.f90`, `icecor.f90`, `iceitd.f90`, `icevar.f90`, `icedyn.f90`
are byte-identical to the `_O1_REF/_O1_REC/_O2_REF/_O2_REC/_O5_REF/_O7_REF`
copies. Recompiling the four files with the build's own `gfortran` and flags
gives object code instruction-identical to the record build's `.o`, with and
without `-g` (evidence `objcode/`), so the `-g` line map is the record's.
No `vfmadd` in any of them; `exp`/`tanh` (ridging) are calls into libm.

## Search before build (RULE 4)

- `git grep -n -i "rdgrft|partf|apartf|closing_gross|itd_reb|zapsmall|var_roundoff|shiftice|hi_hrft"`: lane 3's one-category arm (`apply_si3_jpl1_ridging`), lane 3's one-category `ice_cor` inside its ICE_RHEO recipe, and two lane-3 zapsmall helpers for ADV1D/ADV2D. No five-category ridging, no rebinning.
- Reused: O1's `hi_max`/`hi_mean` (inputs), O7's sequential `SUM` and tie-keeping `MIN`/`MAX` (made public by appended aliases), the shared `nemo_source_round` and policy `exp`/`tanh`.
- Lane 3's one-category ridging arm and `ice_cor`: measured against lane 3's record in the resume section below (item 2).
- Resume: NEMO's `glob_2Dsum` already exists as `si3_glob_2dsum_ddpdd` (reused for `ice_cons_hsm`); no conservation check or MIN/MAX audit tool existed (`git grep -n -i "cons_hsm|rchk_m|rn_icechk|glob_2dmin|minsd|fverbose-asm"`).

## What was transcribed

Call order, `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icedyn.f90:193-194`:
`CALL ice_dyn_rdgrft( kt )` then `CALL ice_cor       ( kt , 1 )`.

Ridging, `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icedyn_rdgrft.f90`:
- `:268` interior `at_i` recompute; `:295` `closing_net(ii) = rn_csrdg * 0.5_wp * ( zdelt(ii) - ABS( zdivu(ii) ) ) - MIN( zdivu(ii), 0._wp )`.
- `:312` the selection `IF( SUM( apartf(ii,1:jpl) ) > 0._wp .AND. closing_gross(ii) > 0._wp ) THEN`; `:341` `DO WHILE( iterate_ridging > 0 .AND. iter < jp_itermax )`, one flag over the rank's list.
- `:481` `zGsum(1:npti,jl) = ( pato_i(1:npti) + SUM( pa_i(1:npti,1:jl), dim=2 ) ) * z1_asum(1:npti)`; `:504` `zGsum(ii,jl) = EXP( -zGsum(ii,jl) * z1_astar ) * zfac`.
- `:519` `paridge(ii,jl) = ( 1._wp + TANH ( rn_craft * ( zhi(ii,jl) - rn_hraft ) ) ) * 0.5_wp * papartf(ii,jl)`.
- `:617-618` the normalisation, left to right: `( zaksum + ridge term ) + raft term`.
- `:713` `ato_i_1d(ii) = MAX( 0._wp, ato_i_1d(ii) + ( popning(ii) - papartf(ii,0) * pclosing_gross(ii) ) * rDt_ice )`, every pass, every listed cell.
- `:749-751` porosity by total volume, with the 10-20 m ramp; `:910-911` the last-category conservation fallback; `:944` roundoff.
- `:1273-1276` the `ln_icethd = F` override, as `si3_rdgrft_config_after_init`.
- Model: `packages/ice/legoesm/ice/ridging.py:1245` (`def si3_ice_dyn_rdgrft(`), `packages/ice/legoesm/ice/ridging.py:1015` (`def _rdgrft_prep(`), `packages/ice/legoesm/ice/ridging.py:1088` (`def _rdgrft_shift(`).

Correction, `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icecor.f90`:
- `:87` thin-ice area scaling; `:94` `amax` clamp; `:99` `IF( jpl > 1 )   CALL ice_itd_reb( kt )`; `:117-118` salt bounds; `:128` `CALL ice_var_zapsmall`.
- Rebinning `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/iceitd.f90:632` (too thick, half moved, `:651`) and `:668` (too thin, all moved); every packed cell's `t_su` is rewritten `:577` `t_su_2d(1:npti,:)  =  zaTsfn(1:npti,:) / a_i_2d(1:npti,:)`.
- Zapsmall `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icevar.f90:735`; roundoff's lid window `:885`.
- Model: `packages/ice/legoesm/ice/itd.py:993` (`def si3_ice_cor_dyn(`), `packages/ice/legoesm/ice/itd.py:932` (`def si3_ice_itd_reb(`), `packages/ice/legoesm/ice/itd.py:950` (`def si3_var_zapsmall(`).

MIN/MAX on a signed-zero tie or a NaN: every executing site, every compiled
copy, read from the record build's object code (resume section, item 1).

Not transcribed, stated: the ridging diagnostics (`ll_diag_rdg`, `iom_put`
only); `ln_partf_lin`, EAP closing and `nn_icesal /= 4` are refused.

## Exact-input structure

(a) `si3_ice_dyn_rdgrft(state, divu_i, delta_i, sst_m, sss_m, hi_max, nn_hls, config)`: NEMO's arrays at `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icedyn.f90:193` -> the ridged state, `at_i`, `ato_i`, the five dynamics fluxes, the selected cells and the pass count.
(b) `si3_ice_cor_dyn(state, rn_amax_2d, sss_m, sst_m, hi_max, hi_mean, nn_hls, config)`: NEMO's arrays at `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icedyn.f90:194` -> the corrected state and the three residual fluxes.
Every physical input is required; `None` raises; configs have no defaults.

## Tests (`tests/ice/unit/test_si3_o6_rdgrft_cor.py`)

- Ridging equals the scalar transcription, every word, both `ln_icethd` arms; the fixture runs at least two passes, the porosity ramp and the fluxes (`tests/ice/unit/test_si3_o6_rdgrft_cor.py:457`).
- `ice_cor` equals its scalar transcription, every word, with thin-ice, `amax`, both rebinning directions, salt bounds and zapsmall reached (`tests/ice/unit/test_si3_o6_rdgrft_cor.py:480`).
- Superseded in the resume: NEMO's own `ice_cons_hsm` replaces the in-house budget check (item 5).
- Plants in the production path (ridge minimum multiplier, porosity slope, raft area ratio, reversed `SUM`, rebinning share) each turn the comparison red.
- Eager (`jax.disable_jit`) equals jit, both pieces; central finite difference on one `v_i` cell agrees with the jit gradient; the full `delta_i` gradient is finite.
- No defaults; halos untouched; value pins.
- Result: `18 passed in 1502.99s (0:25:02)` (evidence `o6_unit.log`). Existing consumers of the three touched modules (ridging, ITD, O3 fit, O7 surface suites): `92 passed, 1 xfailed in 119.13s` (`touched_consumers.log`).

## Parent-vs-current controls (parent `e8ee89ece8cb`)

- The parent text of `ridging.py` and `itd.py` is an exact prefix of the current file; `si3_surface.py` differs in its rewritten `__all__` assignment and three appended aliases (resume correction 4).
- Lane 3's one-category ridging arm and the generic `apply_ridging`, parent module vs current: every output word equal (`parent_controls.log`).
- Lane 3's ICE_RHEO card, one full step with the parent's ridging, ITD and surface modules, every state leaf equal: `1 passed, 17 deselected in 658.10s` (`parent_control_rheo.log`; run before the last edits, which touched only the new functions and the test file).

Evidence directory: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round14/`.

## Gates

- Citation gate, all 13 SI3 receipts, full map: every receipt PASS, 0 unmapped, 0 failing, 0 map entries failing audit (`citation_gate_all_si3.log`); a shifted-line plant on this receipt fires (exit 1).
- Default citation gate: `31 passed in 6.90s` (`default_gate.log`).
- (Original round; the resume's gates are below.) Private-import ratchet: `6 passed in 4.94s`. Inline-coefficient, hardcoded-constant, dispatch, parameter-spec and saturation ratchets: 11 reds, none in a touched file, the identical 11 at the parent (`ratchets_current.log`, `ratchets_parent.log`).
- Review: single review (codex): no transcription defect on the executed path; three confirmed test/API hygiene findings (oracle read global `rhow`/`hi_max`, a `1000.0` literal, aliases missing from `__all__`) fixed, plus its suggested `hi_max` and lid input plants added. GLM: no route available in this session, so not run (honour-system rule unmet, stated).
- GPU spot check: not run (no certification this round).

## Resume: codex HOLD items (Argon LAND, codex HOLD)

Evidence directory: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round14/r2/`.

### 1. MIN/MAX tie and NaN order, every executing site

x86 `minsd/maxsd SRC,DST` return SRC on a `+0/-0` tie and on any NaN; gfortran
lowers MIN/MAX with unspecified order, and versioned or unrolled loops give
one line several copies. A committed tool,
`scripts/validate/ocean_fidelity/testcases/nemo_si3_minmax_tie_audit.py`,
recompiles the record ppsrc with the build's own gfortran and flags, refuses
unless the object is instruction-identical to the record's `BLD/obj`, and
names each copy's surviving operand through `-fverbose-asm` and the GIMPLE
dumps. All 27 MIN/MAX lines of `ice_dyn_rdgrft`, `rdgrft_prep`, `rdgrft_shift`,
`ice_cor` and `ice_var_zapsmall` (`ice_itd_reb`, `itd_shiftice` and
`ice_var_roundoff` have none): 180 compiled copies, all matched (174 by the
result name, 6 by both operands), all equal to the committed table
`nemo_si3_o6_minmax_sites.json`. Completeness is checked both ways: every
min/max instruction inside those compiled functions is on a table line or a
declared idle arm, and every MIN/MAX/SIGN source line of those routines is in
the table. The routine set is pinned to NEMO's call graph: following calls
through the whole build's ppsrc from the two calls parsed at
`ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icedyn.f90:193-194` reaches the nine
audited routines (with `ice_dyn_1d2d`, which has none) and six `icetab` array
copies with no MIN/MAX/SIGN; calls behind the check, print and timing
switches (`ln_icediachk`, `ln_icectl`, `sn_cfctl%l_prtctl`, `ln_timing`) and
`iom_put` are not followed. Each
idle line is checked to sit in an `nn_icesal` arm the deck does not take
(icecor line 108 in `nn_icesal == 2`, icevar line 650 in the `ELSE` of
`nn_icesal == 4`; the deck resolves `nn_icesal = 4`). It agrees with my hand reads
of icedyn_rdgrft lines 295, 298, 375-380, 604-605, 713, 856-857 and 872-889,
icecor line 118 and icevar line 636.

- Model changed to the record's survivor at `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icedyn_rdgrft.f90:298` (keeps `-zdivu` in both EVP copies), `:610`/`:613`/`:878`/`:889` (keep `epsi20`), `:874`/`:887` (keep `phrexp`), `:857` (keeps `hi_max(jl2)`), `:751` (the record has no MAX instruction; it keeps `2 - 0.1*zvti`), `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icecor.f90:117-118` (MAX keeps `szv_i`; in the `kn /= 0` copy MIN keeps the MAX result), and `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icevar.f90:636` (compiled `MIN(h_i, MIN(v_i, a_i))`). The NaN-propagating `jnp.minimum/maximum` are gone from the rung.
- Copy-dependent: `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icedyn_rdgrft.f90:856` keeps `phrmin` in one copy and `hi_max(jl2-1)` in the other. Running the shift as written and with that call's operands swapped, `phrmin = -0`, branch forced open: every output bit equal.
- The scalar oracle uses the same survivors through its own helpers.
- Tests: `tests/ice/unit/test_si3_o6_rdgrft_cor.py:770` (model and oracle calls equal the table; the pre-audit source fails it), `:789` (helpers return what `minsd/maxsd` return, eager and jit), `:806` (a NaN `delta_i` at line 298; the pre-fix model fails it, 5 words), `:817` (the line-856 counterfactual); `tests/ice/test_nemo_si3_minmax_tie_audit.py` (record object equals the table; three table plants, a dropped-line plant and a wrong-object plant fire; the matcher refuses ambiguous rows; source-level completeness; call-graph scope; idle arms): `8 passed in 13.44s` (`audit_tests.log`).

### 2. Lane 3's duplicate kernels, measured on lane 3's record

Instrument: lane 3's own exact-dynamics arm (oracle step-8 rheology, then
ridging, `ice_cor`, transport, scored bitwise on the oracle step-9 entry,
1,000,000 cells), one kernel swapped at a time
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o6_lane3_kernel_probe.py`).

| arm | informative rows bit-exact |
|---|---|
| lane 3 as committed | 36 / 36 |
| O6 ridging | 36 / 36 |
| O6 ridging + O6 `ice_cor` | 36 / 36 |
| O6 ridging, `rn_murdg` x (1 + 1e-10) (plant, same run) | 13 / 36 |

Both kernels are NEMO-exact on this lane-3 record.
Where they differ, measured on a synthetic jpl = 1 input (the cause read off
the code, PLAUSIBLE): lane 3's open-water output,
which its card overwrites with `1 - a_i`, and two branches where lane 3
divides by `dt` while NEMO multiplies by `r1_Dt_ice`
(`ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icedyn_rdgrft.f90:647`, `:659`). Lane 3
reports that those branches never fire on its oracle frames, 719 transitions
(`docs/ocean/fidelity/testcases/nemo_testcases_l3dyn_phase2_receipt.md`,
"excessive-removal scan"). So the O6 kernel is the statement-literal one;
lane 3's is exact on its card's record.

Routing lane 3's card through the O6 kernels, both directions, lane 3's own
committed gates rerun per arm
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o6_lane3_routing_probe.py`;
kernel calls counted per gate; plant `rn_murdg` x (1 + 1e-10)):

| gate (rows) | lane 3 | O6 ridging | O6 ridging + `ice_cor` | rows changed / worsened / improved | plant: rows changed |
|---|---|---|---|---|---|
| kt=1 gate (50) | 46 AT-BAR, 4 UNINF | same | same | 0 / 0 / 0 | 10 |
| first-divergence sweep (6) | 1 AT-BAR, 5 DEBT | same | same | 0 / 0 / 0 | 0 |
| active window, step 8 (275) | 143 AT-BAR, 128 DEBT, 4 UNINF | same | same | 0 / 0 / 0 | 22 |

Every row is identical in every number, not only in status. Not measured:
lane 3's 9-720 walk and restart (hours per arm), and the active window's
oracle-frame ridging projection, which reads lane 3's per-cell iteration
counts and clamp flags that the O6 kernel does not return (that diagnostic
kept lane 3's arm). So routing keeps every lane-3 number measured, but
deleting lane 3's arm would need those diagnostics in the O6 kernel and a
walk re-measurement first. Lane 3's arm stays for now, for that measured
reason; unifying is the decision asked below.

### 3. Source-rounding barriers

`zworkv`/`zworka` (`ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/iceitd.f90:489-492`)
are now `nemo_source_round`ed. An AST sweep over every rung function
(`tests/ice/unit/test_si3_o6_rdgrft_cor.py:899`; binary operators, `**`,
augmented assignment, `jnp` arithmetic calls, `si3_safe_div`) finds no other
unbarriered fp operation; the pre-fix source fails it on those two lines.
Value pins unchanged.

### 4. D5 C1D year, parent `42742c23c465` vs current, CPU jit

`nemo_si3thd_d5_platform_probe.py --skip-step1-reproduction`, all 18
year-hash arms by path: current equals parent on 18/18 at default ISA and
18/18 at `--xla_cpu_max_isa=AVX`. The shipped arm's 256-bit result digest,
written as two 128-bit words, is
`b7a7329bed74c17918fa3cbc11c7613ac:b2074851e2357a08869fcd4661e12d2` in all
four runs (CPU == AVX); disposition `D5_PLATFORM_FLOOR` in all four. Current
was `12fa6f8e0d8a`; rerun at `f10fee86189c` (after the conservation and tooling
commits): 18/18 identical to the parent at both ISAs, same shipped hash
(`d5/d5_parent_vs_current.txt`, `d5/d5_head_*.json`).

### 5. NEMO's own conservation check

`ice_cons_hsm` (`ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icectl.f90:81`) is
transcribed into `packages/ice/legoesm/ice/si3_orca2_state.py:3112` and
`:3170`: area-weighted cell quantities and all 36 flux terms in source order,
NEMO's DDPDD `glob_2Dsum` (existing `si3_glob_2dsum_ddpdd`), the
`smask0_i`-masked global max and minima, the diagnostics
(`ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icectl.f90:176-178`), the threshold
`rchk * rn_icechk_glo * zchk3(9)` (`:182`) with the area row
`SUM(a_i + epsi10) * e1e2t` (`:134`), the negativity checks and the
`amax` rule that skips `icedyn_rdgrft` (`:198`). What differs: NEMO prints
warnings; this returns flags. Not transcribed: the advection-only rows
`zchk3(7:8)` and the per-cell `ice_cons2D`.

Both pieces raise no flag on the fixtures; each plant raises exactly the
flag of the warning NEMO would print: dropped mass, salt or heat flux; +1e-4 m of ice in one cell; a
negative `v_s`; `at_i` above `amax` flagged for `icecor`, not for
`icedyn_rdgrft`; a masked cell hidden from the extrema
(`tests/ice/unit/test_si3_o6_rdgrft_cor.py:505`, `:548`, `:568`).

### Tests and controls (this resume)

- O6 suite without the two slow tests: `35 passed in 194.05s` (`o6_fast_try2.log`); the slow set (FD gradient, both parent-vs-current controls including lane 3's ICE_RHEO card step): `3 passed in 1577.84s` (`o6_slow.log`).
- Consumers of the touched modules (lane-3 jpl=1 ridging, ridging, ITD, O3, O5, surface suites): `121 passed, 1 xfailed in 387.84s`; O2 gate (`si3_orca2_state` consumer): `12 passed in 186.93s`.
- Probe tests (kernel and routing probes): `7 passed in 5.22s` (`probe_tests.log`).

### Gates (this resume)

- Citation gate, all 13 SI3 receipts, full map: every receipt PASS, 0 unmapped, 0 failing, 0 map entries failing audit; this receipt 65 citations; a shifted-line plant fires (exit 1) (`citation_gate_all_si3.log`).
- Default citation gate: `31 passed in 6.75s` (`default_gate.log`). Private-import ratchet: `6 passed in 5.04s` (`private_imports.log`).
- Inline-coefficient, hardcoded-constant, dispatch, parameter-spec and saturation ratchets: `11 failed, 12053 passed, 4 skipped`, the identical 11 at the parent; the one in a touched file (`si3_orca2_state.py`, inline coefficients) lists the same five pre-existing sites, none in the appended code (`ratchets_current.log`, `inline_si3_orca2_current.txt`).
- Reviews (codex, read-only CLI): pass 1 HOLD, five harness/tool defects confirmed and one plausible, all fixed; pass 2 confirmed five fixed and the matcher not, now fixed and tested; pass 3 (claims check) HOLD on wording and completeness, fixed; pass 4 HOLD on self-declared completeness scope, unvalidated idle arms, one count and two unlogged runs, fixed; pass 5 confirmed three of those and held on the unit set and hard-coded roots, now taken from the whole ppsrc and from `icedyn.f90`; pass 6: LAND (`review_codex_r2.log`..`r7`). GLM: no route in this session, not run (honour-system rule unmet, stated).
- GPU spot check: not run (no certification this round).

## CONFIRMED vs PLAUSIBLE

- CONFIRMED: both pieces equal an independent scalar transcription bit for bit; the record build's object code is the one read; the transcribed `ice_cons_hsm` raises no flag on the fixtures; plants fire; eager equals jit; FD agrees; lane 3 unchanged on its one-step parent-vs-current control and on its ICE_RHEO step-8 record.
- PLAUSIBLE: bit-exactness against NEMO's ORCA2 record (no record yet, P1-P6).

## CHOICES

- Deck for the acquisition: production deck vs the design row's `ln_icethd = F`. ASKED (still with the user).
- Resume: model and oracle MIN/MAX orders moved to the record's survivors; zworkv/zworka rounded; NEMO's check transcribed; D5 control. ASKED (codex items 1, 3, 4, 5).
- Resume: at the copy-dependent line 856 the model keeps `phrmin` (one copy's order); shown unobservable. UNASKED; offered for revert.
- Resume: `ice_cons_hsm` placed beside its `glob_2Dsum` in `si3_orca2_state.py`; `ice_cons2D` and the advection rows not transcribed; test fixture `rn_icechk_glo = 1e-4` (namelist_ice_ref) and non-uniform cell areas. UNASKED; offered for revert.
- Resume: lane 3 measured, not unified; the routing probe routes the card's step only. ASKED (DECISION_NEEDED).
- Cited build `ORCA2_SAS_ICE_O7_REC`. UNASKED, mechanical.
- `nn_itend = 2`, one rank, sparse passes {1, 2, last} in the preregistration. UNASKED; offered for revert.
- O7's three helpers made public by appended aliases (the `__all__` line changed). UNASKED; offered for revert.
- Test-local layer counts 3/2 and a 9 x 8 grid. Tests only.
- No configuration default or production config changed.

## Resume 5: Decision 177 merge slowdown

Decision 177 is taken: lane 3 routes through the single O6 ridging and
`ice_cor` implementation.  The first merged full walk was stopped after 37 h,
so a preregistered 10-step parent/current diagnostic separated eager setup,
odd/even compilation, compiled execution, stage timing, callback counts and
scientific rows before another full walk
(`nemo_si3_r14_short_window_diagnosis.py`; artifacts
`round14/r3/diag_{parent_committed,fixed_84e070f19d}_10steps.json`).

CONFIRMED root cause: the static 19-slot transcription executed
`_rdgrft_prep` and `_rdgrft_shift` before masking an inactive result, so all
19 expensive bodies ran after NEMO had converged in one pass.  The unfixed
merged window averaged 225.1791 s/step and made 83 libm callbacks per step;
the parent averaged 41.7903 s/step.  This was execution, not compilation or
retracing: odd/even caches were stable and the precompiled handles were reused.
The landed loop retains NEMO's 19-pass upper bound and native reverse-mode AD,
but wraps the expensive body in the `iterate_ridging` condition.  A direct
eager regression requires `_rdgrft_shift` calls to equal `n_passes` and to
stop below 19 (`test_si3_o6_rdgrft_cor.py`).

The committed fix (`84e070f19d`) measured 43.6059 s/step (34.7941--49.0975),
with odd/even compilation 10.2498/9.6771 s.  The parent measured 41.7903
s/step (41.5101--42.1918), with compilation 23.5981/23.4984 s.  At step 11,
both report one active ridging pass, 996004 selected cells and one maximum
cell shift; fixed ridging is 15.2192 s versus parent 23.4590 s.  The fixed
merged literal kernel performs 11 ridging libm callbacks for that one active
pass, versus 82 ridging callbacks in the unfixed merged arm.  It does not
repeat callbacks after convergence.

The 390 scientific rows (320 AT-BAR) are exactly equal as serialized row
objects; their 256-bit result digest, written as two 128-bit words, is
`96dece7431e96cd71fa4a151c49507315:305f882d7a9cede292d3f6a9ad82cf1`.
Thus the short-window fix changes no lane-3 certified row.  PLAUSIBLE only:
the small timing spread is host scheduling and parallel callback variance;
it is not used for numerical certification.

Both 720-step arms then ran concurrently under detached `setsid`/`nohup`
jobs with pidfiles.  The parent completed in 36475.2 s (29.25 GiB peak RSS)
and fixed current in 36535.3 s (28.70 GiB); both reached the sparse 9, 10,
50, 100, 200, 485 and 720 checkpoints.  Their gate exit is 1 because the
lane remains DEBT, not because either walk failed.

The full JSON artifacts are not byte-identical.  Comparing all 553 row
objects finds 308 numerical differences, all in rows already classified
DEBT: zero parent AT-BAR rows differ and zero statuses change.  By growth
checkpoint, differing rows are 0 at steps 9 and 10, 3 at step 50, and 33 at
each of 100, 200, 485 and 720; none is parent AT-BAR.  The final debt count
is 173 in both arms.  Therefore Decision 177's certified-row condition is
met, while the debt trajectory is explicitly not claimed bit-identical
(`round14/r3/walk_{parent_resume5,fixed_5b0a9b7994}.json`).

## Resume 6 amendment: scope and Decision 177 drift

This amendment preserves the Resume-5 record above and corrects its scope.
The 390 scientific rows, of which 320 are AT-BAR, belong only to the
preregistered 10-step timing window.  Across every serialized row collection
in the full-walk receipts there are 553 row objects: all 134 AT-BAR rows are
identical, 308 of 387 rows already classified DEBT differ, 32 are
UNINFORMATIVE, and no row changes status.  The three first differences at
step 50 are already-DEBT `u_ice`, `stress1_i`, and `stress2_i`; their reported
metrics move at about the `1e-13` absolute scale or below.  At step 485 the
merged-to-parent field-relative ratio is about `1.4` for the inventory rows
(for example `v_i`, `1.3986025227886831`).  This is drift within admitted
DEBT, not a certified-row regression.

The active gate's non-row ridging-projection leaf also moved.  Its common
`minimum_open_water_before` is `0.009999990410122006`;
`minimum_open_water_after` is `0.00999999040071575` for the parent arm and
`0.009999990445945461` for the merged literal O6 arm.  The lane-3 receipt now
contains an appended amendment that supersedes its earlier zero-movement
sentence without rewriting that historical record.  Both values remain
positive, and this diagnostic leaf is not one of the 553 scored row objects.

The drift attribution is **CONFIRMED** at its first seed.  Independent native
parent and merged processes carry bit-identical state through completed step
45.  On the common step-46 input, rheology and transport output hashes remain
identical; ridging is the first differing stage, and `ice_cor` only carries
that changed inventory onward.  An aligned expression probe finds zero
unequal elements in the parent/literal closing-rate and opening statements.
The ordered first-pass preparation trace then remains identical through total
area, inverse area, cumulative open-water fraction, `1/rn_astar`, the constant
exponential and its normalization, and the per-cell exponential argument.
The first different statement is the per-cell
`EXP(-zGsum(ii,jl)*z1_astar)` at
`ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icedyn_rdgrft.f90:504`; `g_zero`,
`participation_open`, and `participation_ice` differ immediately afterward.
The parent one-category arm used native XLA `jnp.exp`; the merged O6 arm uses
the scalar-libm path and NEMO statement barriers, matching the recorded
Fortran kernel.  Therefore the DEBT-only drift is the expected consequence of
making those rows more literal, not a defect in the converged-loop speed fix.
The subsequent nonlinear growth pattern--from the step-50 seed to the roughly
`1.4` step-485 ratio--is **PLAUSIBLE** propagation of that confirmed seed; no
claim assigns each later field's magnitude to a separate source statement.
Evidence is `round14/r4/{parent,merged}_native_step46_prep.json`.

Resume-6 verification: the integer pass counter again carries the rounding
audit's explicit integer-cast exemption, and the complete default O6 unit-file
scope is `27 passed, 1 deselected in 1599.98s`; the short-window diagnostic's
direct controls are `3 passed`.  The per-receipt citation gate finds 65
citations with zero unmapped citations, failures, or map-audit failures; its
11 self-test controls all fire.  The default citation suite is `31 passed`,
and the private-import ratchet is `6 passed`.  The inline-coefficient,
hardcoded-constant, dispatch, parameter-spec, and saturation ratchets report
`11 failed, 12033 passed, 4 skipped`; a clean detached `4fd244ebbdec` worktree
has the same counts and the same 11 failure identities, so the ratchet delta
is empty.

## Resume 7: rebase onto the O8--O16 production lane

### Rebase identity and conflict disposition

This resume used a **rebase**, not a merge.  The 26 reviewed O6 commits from
pre-rebase head `7441c9f79cc86182d71ab2cefde5a680071f2039` were replayed from
their computed merge base `e8ee89ece8cbf611ecc6de0906a72b147565baae` onto
the fetched and independently resolved lane commit
`0d339a87e5c1d718a0b9aa4bb49f8fb8788562be`.  Before the receipt/gate fix in
this section, the replayed head computed as
`ce68f273253e077aa118b775e487cbce9d333c57`; the lane commit is its direct
ancestor.

Git reported exactly two textual conflict hunks:

1. `packages/ice/legoesm/ice/itd.py`.  The lane side adds the O13 compiled
   `MINSD` ordering helper used by the later exact remap.  The O6 side adds the
   reviewed `SI3IceCorConfig`/`SI3IceCorState`, roundoff, rebinning,
   `zapsmall`, and `ice_cor(kt,1)` implementation.  Both blocks were retained
   byte-for-byte in their original order; only Git's three marker lines were
   discarded.  They implement distinct rung owners, so choosing either side
   would silently delete a certified production path.
2. `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py`.
   The lane side adds the O8--O16, O15, O16, provenance-era, and cross-rung
   citation anchors.  The O6 side adds the O6 source, model, test, and
   conservation anchors.  Both maps were retained; only conflict markers were
   discarded.  The duplicate-key and full-map tests, rather than dict
   overwrite semantics, decide whether the union is valid.

One additional **semantic auto-merge conflict** was found by importing the
citation gate.  It is not counted as a Git conflict hunk: the lane had already
re-anchored two O1-era `ice/__init__.py` keys before the later O6 Decision-177
pin loop tried to read their historical names.  The lane's live keys and all
their later re-anchors were kept unchanged.  A full receipt search proved that
neither obsolete historical name is cited, so only those two dead pin-loop
requests were discarded; no live anchor or O6 claim was discarded.  The
remaining O6 Decision-177 citations stay pinned to their existing computed
parent `6c5381338431f87e1ed0895a0118941aa6e7a38a`.  Before this resolution,
importing the gate failed closed with the missing historical key.

The other files named in the rebase brief replayed without textual conflicts:
`packages/ice/legoesm/ice/__init__.py`, `si3_orca2_state.py`,
`si3_surface.py`, `docs/ice/fidelity/si3_ladder_design.md`, and every later
edit of the two conflicted files.  Nothing was hand-selected in those files.
`git range-diff` reports 23 commits patch-equivalent and the other three
differing only at the additive `itd.py`/citation-map context described above.

### Bit-preservation checks before execution

The complete rebased `ridging.py` has the same computed SHA-256 as the reviewed
pre-rebase file.  The O6 unit file also matched before this resume appended the
new three-input parent/gradient control.  AST source extraction also compares
every O6 `itd.py` config/state class and all seven O6 roundoff/rebin/correction
functions byte-for-byte equal to the reviewed tree.  The two O6
`ice_cons_hsm` functions are byte-for-byte equal as well.  Against the O6
parent, the shared generic `apply_ridging` and `participation_weights`
functions are byte-for-byte equal, before the state/gradient execution control
below.

### Post-rebase execution proofs

- The complete reviewed default O6 file scope emitted
  `27 passed, 1 deselected in 1687.36s (0:28:07)`.  This resume then appended
  only the explicit parent/current three-input control; that control emitted
  `1 passed in 4.67s`.  It compares every generic ridging output bit and every
  reverse-mode gradient bit with respect to area, thickness, and closing rate.
  The existing slow lane-3 full-card parent/current control emitted
  `1 passed, 28 deselected in 593.69s (0:09:53)`.
- The rdgrft regression selection was
  `tests/ice/test_nemo_si3_minmax_tie_audit.py`,
  `tests/ice/unit/test_ice_convergence_ridging.py`, and
  `tests/ice/unit/test_ice_ridging_faithful.py`.  It emitted
  `45 passed in 37.75s`.
- The later production exactness selection named
  `test_score_o8_to_o12_representative_steps_0_ulp` (its report status is
  `PASS_ALL_O8_O13_0_ULP`),
  `test_score_o14_representative_steps_0_ulp_and_plants` (status
  `PASS_O14_0_ULP`), and
  `test_score_o15_and_kind20_owned_interior_representative_steps_0_ulp`
  (status `PASS_ALL_O15_O16_0_ULP`), plus all eight tests in
  `tests/ice/unit/test_si3_finalize.py`.  The combined run emitted
  `11 passed in 342.84s (0:05:42)`.  Thus O15 had zero unequal elements, and
  O16 aggregation, flux, and owned kind-20 stress had zero unequal elements,
  at each representative production step, while the independent finalize
  oracle, signed-zero, gradient, shape-refusal, zero-ice, and production-call
  tests all passed.
- The per-receipt citation gate reports `status: PASS`, 65 citations, zero
  unmapped citations, zero citation failures, and zero full-map audit failures.
  Its default regression suite emitted `42 passed in 8.31s`; the prospective
  provenance-SHA gate emitted `1395 passed in 1.32s`.
- With all eight package namespace roots present, the six repository ratchets
  emitted `13 failed, 12126 passed, 4 skipped`.  The exact fetched lane commit
  emitted `13 failed, 12118 passed, 4 skipped`; all 13 failure identities and
  all four skips are identical, while this tree adds eight passes.  The
  ratchet delta is therefore empty, and the private-cross-import component is
  independently visible as six passes.
