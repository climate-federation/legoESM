# SI3 round 24 receipt — O15 and O16 landed

Date: 2026-10-28
Round: `si3_r24`
Parent: `87a1c8843803f0cb6263a2a84f44f8d83183ee96`
Preregistration commits: `cac0b9a415`, `ac2ad34834`, `ccb248ae0d`
Production implementation and scorer commits: `276014c01b`, `254406a53a`
Independent review: operator (Claude)

## Verdict

**LANDED — O15 and O16 are bit exact through their production entry points for
all 100 steps and every owned field.** The O15 NEMO-statement materialization
barriers remain necessary. Decision 187's version-1/kind-15 reader translation
and the separate version-1/kind-19 translation are proved from the committed
writer order and independent fields; both wrong-permutation plants fail.
Production `step_sea_ice` calls O15 correction, O16 flux aggregation, and O16
stress update; every kind-16 output plus kind 20 over exactly its owned interior
is 0 ULP.

## O15 production result

The original stopped score localized the first mismatch to `kt=1`, `a_i` cell
`[8,25,1]`: master bits `0x3f7589a004f0bea2`, production bits
`0x3f7589a004f0bea3`. NEMO performs distinct donor and receiver stores
(`iceitd.f90:496-500`). The production implementation now materializes each
completed transfer statement with the shared source-rounding primitive before
the next category boundary can consume it
(`packages/ice/legoesm/ice/si3_finalize.py:325-384`). It also preserves the
literal correction statements before rebinning (`icecor.f90:81-99` and
`packages/ice/legoesm/ice/si3_finalize.py:536-601`). NEMO gathers rebin results
back only for selected `nptidx` cells (`iceitd.f90:575-595`), so production
leaves `h_i` and `t_su` unchanged outside that write set.

The all-step CPU-fp64 production-JIT score is:

- O15: 0 unequal / 1,006,992,000 elements, maximum 0 ULP, no first non-bit;
- all `kt=1..100`, every recorded owned field, and every category element are
  included;
- artifact:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round24/o15_kind20_score.json`,
  SHA-256 `6df5e4d29176246f490b6c0632242f1c23f502f3f0f1b485e41c71b290913804`.

The private `o15-strip-transfer-barriers` plant selects the same production
arithmetic without statement materialization. It fails at the original
`kt=1`, `a_i[8,25,1]` site with maximum 1 ULP. Thus the control demonstrates
that the barriers, rather than a changed physical formula, close the defect.

## Kind 20 owned-output result

NEMO's final stress stores execute only for `ji=ntsi..ntei` and
`jj=ntsj..ntej` (`iceupdate.f90:410-419`). The scorer therefore derives the
Python slices from each record's bounds and scores only that 180 by 148 owned
interior for full-halo `utau` and `vtau`; the recorded `taum` is already
interior (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:2440-2455`).

The 100-step result is 0 unequal / 7,992,000 elements, maximum 0 ULP, no first
non-bit. A one-ULP `utau` plant at the first owned cell fails. Unwritten halo
bytes are neither recorded as NEMO-owned outputs nor scored.

## O16 versioned kind-15 reader translation

The committed writer fixes the kind-15 tail order as `at_i_b`,
`wfx_snw_dyn`, `qsr`, `sfx`, `cloud_fra`, `tmask`, `smask0`,
`snwice_mass`, `snwice_mass_b` (`o8_o16_master_writer.F90:446-457`). The
admitted version-1 reader instead labels the first five of those as `at_i_b`,
`wfx_snw_dyn`, `cloud_fra`, `qsr`, `sfx`
(`nemo_si3_o8_o16_master_gate.py:171-180`).

The reader therefore translates those three byte positions only when
`version == 1` and `kind == 15`; every other version/kind returns unchanged,
and unsupported master versions already fail closed
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:257-300` and
`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:427-536`).
The evidence checker reads the committed writer calls in source order and, at
steps 1, 2, and 100, obtains these independent checks
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:2159-2232`):

- all 12 state fields also present in kind 18 align bit-for-bit, for 36 checks;
- translated `qsr` equals the independently recorded `qsr_oce` block
  bit-for-bit;
- translated `cloud_fra` is uniformly 0.81, matching NEMO's independently
  defined `pp_cldf` (`sbc_phy.F90:69`);
- a planted wrong reader permutation fails the independent `qsr` check.

The direct test pins the writer/schema orders, version/kind restriction, all 36
cross-kind checks, both independent values, and the failing permutation plant
(`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:205-242`).

Kind 19 has a separate version-1 writer/schema permutation: the writer emits
`tmod_io`, `utau`, `vtau` (`o8_o16_master_writer.F90:476-486`), while the
admitted schema labels those byte blocks `utau`, `vtau`, `tmod_io`. The same
versioned translation layer now owns this mapping. Its proof reads the writer
calls in source order and checks the independently nonnegative magnitude
against both signed stress components at steps 1, 2, and 100. A wrong
kind-19 permutation fails before scoring.

## O16 production result

The scorer reconstructs kind 18, runs the production `si3_o16_step` call site,
which performs `ice_var_glo2eqv`, `ice_var_agg`, and `si3_ice_update_flx`, then
scores every kind-15 state/total field and every kind-16 output field
(`packages/ice/legoesm/ice/si3_finalize.py:894-954` and
`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:2298-2475`).
Across `kt=1..100` under CPU fp64, glibc/libm, and production JIT:

- aggregation into kind 15: 0 unequal / 948,384,000 elements, maximum 0 ULP;
- all 14 kind-16 fields: 0 unequal / 59,139,200 elements, maximum 0 ULP;
- kind-20 owned interior: 0 unequal / 7,992,000 elements, maximum 0 ULP;
- every score has `first_non_bit = None`.

The headless-safe disjoint artifacts cover every step exactly once:

- `001..025`: SHA-256 `ca59f204737a9ee9d92b23aab0b02d1ed85d5c94e465ebb839e003937f4ec294`;
- `026..050`: SHA-256 `f1d8baf0d1c1d7425c2db7365bfe6526d3c3a28c4d307ca525c5b160e0e08751`;
- `051..065`: SHA-256 `0ae03fd37dac18de6d7fc3d549e3110f0567bd849fe1bc30a3fbd4b3631582dd`;
- `066..080`: SHA-256 `f2f56c6dc7b8b375d76e4e134faeb6395780434d06693a85b837ea8fd8256e7f`;
- `081..095`: SHA-256 `84863b41d7cf854c6f2647d974b9108caf45a5766d9bfbec985694a4219c790b`;
- `096..100`: SHA-256 `34b79a739831aad8f9c269587a414a605d5949b47d36f22302b29363cc1e4a69`.

They live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round24/` as
`o15_o16_translated_score_<range>.json`. The monolithic command hit its fixed
10-minute wrapper before producing an artifact; the six completed ranges use
the identical scorer and frozen protocol. A one-ULP `qsr` output plant and an
executing model plant that omits `hfx_res` both fail the O16 flux score.

## Resume 3 — production dispatcher, object-code tie, and rebased score

After rebasing onto lane tip `5f0cf16fbe`, the late stages were wired into the
existing `step_sea_ice(thermo_scheme="si3_orca2_o2")` dispatcher through the
typed `SI3FinalizeForcing` boundary. O15, O16 flux aggregation, and O16 stress
are now called by that production entry. The recorded kind-18 and kind-19
boundaries remain explicit because the intervening age and update stages are
outside this rung; the dispatcher does not invent those unimplemented stages.
An executable removal test bypasses O15 on an input where it changes the state
and observes the production response change.

The `icecor.o` instruction at `0x1dbe` is `minsd %xmm6,%xmm9`: bounded-low is
the source and upper is the destination, so bounded-low survives an ordered
tie. Production now uses the corresponding strict comparison. A reachable
`bounded_low=+0`, `upper=-0` test pins the surviving `+0` bits. Five-category
O15 rebin/shift and O16 aggregation, zero-ice O16, gradients, and explicit
state/config mismatch rejection are covered by small unit inputs.

The fresh 100-step CPU-fp64 production-JIT score ran through `step_sea_ice` and
reported O15 `0 / 1,006,992,000`, O16 aggregation `0 / 948,384,000`, O16 flux
`0 / 59,139,200`, and owned kind-20 stress `0 / 7,992,000`, all at maximum
`0` ULP with no first non-bit. Artifact:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round24/o15_o16_production_score_resume3.json`,
SHA-256 `053cefcc173117e106658ad8975bac29afdbb6f6e971df7192180829095a8d6f`.

Fresh post-rebase all-step compatibility scores also returned
`PASS_ALL_O8_O12_0_ULP` for O8 through O12 and `PASS_O14_0_ULP` for O14. O13
was not present on the fetched lane tip and therefore had no landed scorer to
rerun.

## Resume 4 — O13/O14 lane rebase resolution

Before fetching, branch `backup/si3_r24-resume4-pre-rebase` preserved
`83887b735636a65c640c1cccdc80c5ef2242bb09`. The feature branch was rebased
onto fetched SI3 lane tip `2351cfdda9f5` (O13 plus O14). Conflicts in the
shared master scorer, its direct tests, the receipt citation gate, and the
affected SI3 receipts were resolved additively: the final scorer retains the
O13 conservation scorer and dropped-transfer plant, the O14 scorer and
selector/model-bypass plants, and all O15/O16 scoring, translation, barrier,
model-mutation, flux, and owned-interior stress behavior. Historical receipt
citations were translated to the combined tree without relaxing endpoint or
extent checks. Resolution commit `586d377d8247` is clean and contains no model
or scoring change beyond the conflict resolutions.

The exhaustive CPU-fp64 production scorer ran every admitted step and emitted:

- `O8_O13_STATUS: PASS_ALL_O8_O13_0_ULP`; 100 steps; every O8 through O13
  score block had unequal sum 0 and maximum 0 ULP. JSON SHA-256
  `ebcb768d0a6c6d0ad9f14be203495d2b10018547531fdea08e760d4360a0db8a`.
- `O14_STATUS: PASS_O14_0_ULP`; 100 steps; `O14_SCORE: unequal=0 max_ulp=0
  status=PASS_0_ULP`. JSON SHA-256
  `59d59a1f464928c5ebb0cef883fa1f7b85b69d3febfb085cc79366cc36a3cc10`.
- `O15_O16_STATUS: PASS_ALL_O15_O16_0_ULP`; 100 steps; O15, O16 aggregation,
  O16 flux, and kind-20 owned interior each had unequal 0 and maximum 0 ULP.
  JSON SHA-256
  `0f322ac2098a08c8993faff34b07158b1a3dc2bbc9424d25fff7cfbd7866efa8`.

The clean-tree D5 CPU probe emitted `D5_YEAR_STATUS: PASS`,
`D5_PLATFORM: D5_PLATFORM_FLOOR`, `D5_STEPS: 8760`, and
`D5_SNOW_TEMP_NONZERO: 0`. The shipped
`r6_step1_plus_step2_snow_update` year hash remained
`b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`;
probe JSON SHA-256
`2c07a832729871c4183ce5a753df69a82ee0632ee76eb6ba3e47d196220606d5`.

The exact `land.sh` T4 seven-file set emitted
`169 passed in 1187.19s (0:19:47)`; log SHA-256
`9be609f47d7c9c9188ca8e0db79a949347b7c2f5ba6f40f694839333ba949a01`.
The standalone clean-tree citation-gate run also emitted
`41 passed in 8.22s`.

## Compatibility and controls

- Translation proof, representative `kt=1,2,100` production score, O15/O16
  one-ULP, barrier, stress, and executing heat-term plants: 8 passed, 17
  deselected on the final code. `test_si3_finalize.py`: 8 passed.
- O8–O12 all-step production score: `PASS_ALL_O8_O12_0_ULP`; artifact
  `o8_o12_master_score.json`, SHA-256
  `fa2cc5fc534b7789dc565c491d58472784bf6e568296635397afb2169c4c69bf`.
- Clean-tree D5 CPU production-JIT year probe: 8,760 steps,
  `D5_PLATFORM_FLOOR`; shipped arm `r6_step1_plus_step2_snow_update` retained
  `full_year_output_sha256 =
  b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`.
  Probe artifact SHA-256:
  `ab03c16a4f0a424d6be496733f815153f21c4e09681dd4ccbc09d0571b05c726`.
- Clean-tree per-receipt citation gate: 17 citations, zero failures, zero
  unmapped citations, all planted self-tests fired. All 22 SI3 receipts also
  resolve through the live full citation map. The broad ratchet selection
  reports 6,309 passed and 2 skipped; its remaining 11 failures are unchanged
  from the rebased lane tip and touch no file in this round's diff.

## Choices

### ASKED

- Use the shared source-rounding primitive as a materialization barrier after
  each NEMO transfer statement; demonstrate a barrier-stripped failure.
- Score O15 through production at 0 ULP for every step, field, and category;
  retain the D5 year hash and O8–O12 score.
- Score kind 20 only where NEMO's loop writes the owned interior.
- Per Decision 187, translate only version-1/kind-15 bytes according to the
  writer's actual order; prove it with independent fields and a wrong-
  permutation plant, then score every O16 step and field at 0 ULP.

### UNASKED

- No NEMO configuration or admitted master-byte change.
- No relaxation of the 0-ULP bar.
- No re-acquisition because the proof and all-step O16 score are clean.
- No nested Codex review; independent review is the operator (Claude).

## CONFIRMED versus PLAUSIBLE

CONFIRMED: O15's 0-ULP all-step production score; the barrier-stripped +1-ULP
control; the version-1/kind-15 writer-order translation and failing wrong-
permutation plant; production O16 aggregation, every kind-16 field, and the
kind-20 owned domain at 0 ULP for every step; unchanged O8–O12 and D5 results.

PLAUSIBLE: none within the O15/O16 claim.
