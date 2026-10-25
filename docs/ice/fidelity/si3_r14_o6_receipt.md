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

## Outcome

O6's two exact-input pieces are in the one shared ice package as new,
selectable functions; no existing statement changed. Both equal an
independent scalar transcription bit for bit. No NEMO array was read; the
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
- Not reused, measured: routing lane 3's arm through the new kernel changes its bits (`tests/ice/unit/test_si3_o6_rdgrft_cor.py:680`). Read off the code (PLAUSIBLE as the cause): lane 3 takes the top cumulative area as exactly 1 and stops updating a cell once it no longer shifts. Lane-3 cards must stay bit-identical, so the arm stays; unifying it is a separate lane-3 decision.

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
- Model: `packages/ice/legoesm/ice/ridging.py:1244` (`def si3_ice_dyn_rdgrft(`), `packages/ice/legoesm/ice/ridging.py:1015` (`def _rdgrft_prep(`), `packages/ice/legoesm/ice/ridging.py:1088` (`def _rdgrft_shift(`).

Correction, `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icecor.f90`:
- `:87` thin-ice area scaling; `:94` `amax` clamp; `:99` `IF( jpl > 1 )   CALL ice_itd_reb( kt )`; `:117-118` salt bounds; `:128` `CALL ice_var_zapsmall`.
- Rebinning `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/iceitd.f90:632` (too thick, half moved, `:651`) and `:668` (too thin, all moved); every packed cell's `t_su` is rewritten `:577` `t_su_2d(1:npti,:)  =  zaTsfn(1:npti,:) / a_i_2d(1:npti,:)`.
- Zapsmall `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icevar.f90:735`; roundoff's lid window `:885`.
- Model: `packages/ice/legoesm/ice/itd.py:991` (`def si3_ice_cor_dyn(`), `packages/ice/legoesm/ice/itd.py:932` (`def si3_ice_itd_reb(`), `packages/ice/legoesm/ice/itd.py:950` (`def si3_var_zapsmall(`).

MIN/MAX on a signed-zero tie, read from the object code: the `MINSD` at
`ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icedyn_rdgrft.f90:295` and the `MAXSD` at `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icedyn_rdgrft.f90:713` both return the constant `0` on a tie; transcribed so. At the
other MIN/MAX sites a tie needs a `-0` operand the fixtures never make
(PLAUSIBLE, stated in the preregistration).

Not transcribed, stated: the ridging diagnostics (`ll_diag_rdg`, `iom_put`
only); `ln_partf_lin`, EAP closing and `nn_icesal /= 4` are refused.

## Exact-input structure

(a) `si3_ice_dyn_rdgrft(state, divu_i, delta_i, sst_m, sss_m, hi_max, nn_hls, config)`: NEMO's arrays at `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icedyn.f90:193` -> the ridged state, `at_i`, `ato_i`, the five dynamics fluxes, the selected cells and the pass count.
(b) `si3_ice_cor_dyn(state, rn_amax_2d, sss_m, sst_m, hi_max, hi_mean, nn_hls, config)`: NEMO's arrays at `ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icedyn.f90:194` -> the corrected state and the three residual fluxes.
Every physical input is required; `None` raises; configs have no defaults.

## Tests (`tests/ice/unit/test_si3_o6_rdgrft_cor.py`)

- Ridging equals the scalar transcription, every word, both `ln_icethd` arms; the fixture runs at least two passes, the porosity ramp and the fluxes (`tests/ice/unit/test_si3_o6_rdgrft_cor.py:457`).
- `ice_cor` equals its scalar transcription, every word, with thin-ice, `amax`, both rebinning directions, salt bounds and zapsmall reached (`tests/ice/unit/test_si3_o6_rdgrft_cor.py:480`).
- NEMO's own budget (`ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icectl.f90:176-178`) closes for mass, salt and heat, to 1e-12 of the inventory, for both pieces; minima stay non-negative; selected cells end with `a_i + ato_i = 1`.
- Plants in the production path (ridge minimum multiplier, porosity slope, raft area ratio, reversed `SUM`, rebinning share) each turn the comparison red.
- Eager (`jax.disable_jit`) equals jit, both pieces; central finite difference on one `v_i` cell agrees with the jit gradient; the full `delta_i` gradient is finite.
- No defaults; halos untouched; value pins.
- Result: RESULT_UNIT (evidence `o6_unit.log`).

## Parent-vs-current controls (parent `e8ee89ece8cb`)

- The parent text of all three touched files is an exact prefix of the current file (zero removed lines): no existing statement changed.
- Lane 3's one-category ridging arm and the generic `apply_ridging`, parent module vs current: every output word equal (`parent_controls.log`).
- Lane 3's ICE_RHEO card, one full step with the parent's ridging, ITD and surface modules: RESULT_RHEO (`parent_control_rheo.log`).

Evidence directory: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round14/`.

## Gates

RESULT_GATES

## CONFIRMED vs PLAUSIBLE

- CONFIRMED: both pieces equal an independent scalar transcription bit for bit; the record build's object code is the one read; NEMO's budget closes; plants fire; eager equals jit; FD agrees; lane 3 unchanged.
- PLAUSIBLE: bit-exactness against NEMO's record (no record yet, P1-P6); signed-zero tie order at the unread MIN/MAX sites; the cause of lane 3's divergence from the kernel.

## CHOICES

- Deck for the acquisition: production deck vs the design row's `ln_icethd = F`. ASKED (DECISION_NEEDED).
- Lane 3's one-category arm left separate (measured bit change if unified). UNASKED; offered for a lane-3 decision.
- Cited build `ORCA2_SAS_ICE_O7_REC`. UNASKED, mechanical.
- `nn_itend = 2`, one rank, sparse passes {1, 2, last} in the preregistration. UNASKED; offered for revert.
- O7's three helpers made public by appended aliases (no line moved). UNASKED; offered for revert.
- Test-local layer counts 3/2 and a 9 x 8 grid. Tests only.
- No configuration default or production config changed.
