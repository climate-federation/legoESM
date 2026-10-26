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
the table. The routine set is pinned to NEMO's call graph (every routine of
these four units reachable from the two calls at `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icedyn.f90:193-194`;
this adds `ice_dyn_1d2d`, which has none; calls into other modules are array
copies, diagnostics behind `ln_icediachk`/`ln_icectl`, I/O and timing). Each
idle line is checked to sit in an `nn_icesal` arm the deck does not take
(icecor line 108 in `nn_icesal == 2`, icevar line 650 in the `ELSE` of
`nn_icesal == 4`; the deck resolves `nn_icesal = 4`). It agrees with my hand reads
of icedyn_rdgrft lines 295, 298, 375-380, 604-605, 713, 856-857 and 872-889,
icecor line 118 and icevar line 636.

- Model changed to the record's survivor at `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icedyn_rdgrft.f90:298` (keeps `-zdivu` in both EVP copies), `:610`/`:613`/`:878`/`:889` (keep `epsi20`), `:874`/`:887` (keep `phrexp`), `:857` (keeps `hi_max(jl2)`), `:751` (the record has no MAX instruction; it keeps `2 - 0.1*zvti`), `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icecor.f90:117-118` (MAX keeps `szv_i`; in the `kn /= 0` copy MIN keeps the MAX result), and `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icevar.f90:636` (compiled `MIN(h_i, MIN(v_i, a_i))`). The NaN-propagating `jnp.minimum/maximum` are gone from the rung.
- Copy-dependent: `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icedyn_rdgrft.f90:856` keeps `phrmin` in one copy and `hi_max(jl2-1)` in the other. Running the shift as written and with that call's operands swapped, `phrmin = -0`, branch forced open: every output bit equal.
- The scalar oracle uses the same survivors through its own helpers.
- Tests: `tests/ice/unit/test_si3_o6_rdgrft_cor.py:770` (model and oracle calls equal the table; the pre-audit source fails it), `:789` (helpers return what `minsd/maxsd` return, eager and jit), `:806` (a NaN `delta_i` at line 298; the pre-fix model fails it, 5 words), `:817` (the line-856 counterfactual); `tests/ice/test_nemo_si3_minmax_tie_audit.py` (record object equals the table; three table plants, a dropped-line plant and a wrong-object plant fire; the matcher refuses ambiguous rows; source-level completeness; call-graph scope; idle arms): `8 passed in 13.50s` (`audit_tests.log`).

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
18/18 at `--xla_cpu_max_isa=AVX`. The shipped arm is
`b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2` in all
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
- Default citation gate: `31 passed in 6.90s` (`default_gate.log`). Private-import ratchet: `6 passed in 5.43s` (`private_imports.log`).
- Inline-coefficient, hardcoded-constant, dispatch, parameter-spec and saturation ratchets: `11 failed, 12053 passed, 4 skipped`, the identical 11 at the parent; the one in a touched file (`si3_orca2_state.py`, inline coefficients) lists the same five pre-existing sites, none in the appended code (`ratchets_current.log`, `inline_si3_orca2_current.txt`).
- Reviews (codex, read-only CLI): pass 1 HOLD, five harness/tool defects confirmed and one plausible, all fixed; pass 2 confirmed five fixed and the matcher not, now fixed and tested; pass 3 (claims check) HOLD on wording and completeness, fixed; pass 4 HOLD on self-declared completeness scope, unvalidated idle arms, one count and two unlogged runs, fixed (`review_codex_r2.log`..`r5`). GLM: no route in this session, not run (honour-system rule unmet, stated).
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
