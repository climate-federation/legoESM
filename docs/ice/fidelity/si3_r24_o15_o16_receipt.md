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
is proved from the committed writer order and independent fields; the wrong-
permutation plant fails. Production `ice_update_flx` aggregation and every
kind-16 output, plus kind 20 over exactly its owned interior, are all 0 ULP.

## O15 production result

The original stopped score localized the first mismatch to `kt=1`, `a_i` cell
`[8,25,1]`: master bits `0x3f7589a004f0bea2`, production bits
`0x3f7589a004f0bea3`. NEMO performs distinct donor and receiver stores
(`iceitd.f90:496-500`). The production implementation now materializes each
completed transfer statement with the shared source-rounding primitive before
the next category boundary can consume it
(`packages/ice/legoesm/ice/si3_finalize.py:259-318`). It also preserves the
literal correction statements before rebinning (`icecor.f90:81-99` and
`packages/ice/legoesm/ice/si3_finalize.py:470-534`). NEMO gathers rebin results
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
interior (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:2015-2030`).

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
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:236-270,396-503`).
The evidence checker reads the committed writer calls in source order and, at
steps 1, 2, and 100, obtains these independent checks
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1826-1900`):

- all 12 state fields also present in kind 18 align bit-for-bit, for 36 checks;
- translated `qsr` equals the independently recorded `qsr_oce` block
  bit-for-bit;
- translated `cloud_fra` is uniformly 0.81, matching NEMO's independently
  defined `pp_cldf` (`sbc_phy.F90:69`);
- a planted wrong reader permutation fails the independent `qsr` check.

The direct test pins the writer/schema orders, version/kind restriction, all 36
cross-kind checks, both independent values, and the failing permutation plant
(`tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:170-191`).

Kind 19 has a separate version-1 writer/schema permutation: the writer emits
`tmod_io`, `utau`, `vtau` (`o8_o16_master_writer.F90:476-486`). The kind-20
stress harness continues to use that fixed producer order separately.

## O16 production result

The scorer reconstructs kind 18, runs the production `si3_o16_step` call site,
which performs `ice_var_glo2eqv`, `ice_var_agg`, and `si3_ice_update_flx`, then
scores every kind-15 state/total field and every kind-16 output field
(`packages/ice/legoesm/ice/si3_finalize.py:825-882` and
`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1903-2051`).
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

## Compatibility and controls

- Translation proof plus representative `kt=1,2,100` score: 2 passed, 16
  deselected. O15/O16 one-ULP, barrier, stress, and executing heat-term plants:
  4 passed, 14 deselected. `test_si3_finalize.py`: 4 passed.
- O8–O12 all-step production score: `PASS_ALL_O8_O12_0_ULP`; artifact
  `o8_o12_master_score.json`, SHA-256
  `fa2cc5fc534b7789dc565c491d58472784bf6e568296635397afb2169c4c69bf`.
- Clean-tree D5 CPU production-JIT year probe: 8,760 steps,
  `D5_PLATFORM_FLOOR`; shipped arm `r6_step1_plus_step2_snow_update` retained
  `full_year_output_sha256 =
  b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`.
  Probe artifact SHA-256:
  `ab03c16a4f0a424d6be496733f815153f21c4e09681dd4ccbc09d0571b05c726`.
- Per-receipt citation gate, default citation gate, and focused ratchets are
  recorded after the final receipt anchors are committed.

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
