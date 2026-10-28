# SI3 round 24 receipt — O15 landed; O16 kind 15 held

Date: 2026-10-28
Round: `si3_r24`
Parent: `87a1c8843803f0cb6263a2a84f44f8d83183ee96`
Preregistration commits: `cac0b9a415`, `ac2ad34834`
Production implementation commit: `276014c01b`
Independent review: operator (Claude)

## Verdict

**LANDED — O15 is bit exact through the production entry point for all 100
steps, fields, cells, and category elements.** The NEMO-statement
materialization barriers are necessary: the barrier-stripped control restores
the confirmed +1-ULP result. Kind 20 is also 0 ULP over exactly the owned
interior. O16 kind 15 remains deliberately untranslated and unscored while the
user chooses translation or re-acquisition; only translation-option evidence
was prepared.

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
interior (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1927-1941`).

The 100-step result is 0 unequal / 7,992,000 elements, maximum 0 ULP, no first
non-bit. A one-ULP `utau` plant at the first owned cell fails. Unwritten halo
bytes are neither recorded as NEMO-owned outputs nor scored.

## O16 kind-15 translation evidence — no implementation

The committed writer fixes the kind-15 tail order as `at_i_b`,
`wfx_snw_dyn`, `qsr`, `sfx`, `cloud_fra`, `tmask`, `smask0`,
`snwice_mass`, `snwice_mass_b` (`o8_o16_master_writer.F90:446-457`). The
admitted version-1 reader instead labels the first five of those as `at_i_b`,
`wfx_snw_dyn`, `cloud_fra`, `qsr`, `sfx`
(`nemo_si3_o8_o16_master_gate.py:171-180`).

The evidence-only checker verifies the committed writer sequence and, at
steps 1, 2, and 100, obtains these independent checks
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1778-1845`):

- all 12 state fields also present in kind 18 align bit-for-bit, for 36 checks;
- reader-labelled `cloud_fra` equals the independently recorded `qsr_oce`
  block bit-for-bit;
- reader-labelled `sfx` is uniformly 0.81, matching NEMO's independently
  defined `pp_cldf` (`sbc_phy.F90:69`);
- a planted writer-order permutation fails the evidence check.

The checker reports `translation_applied=false`. Kinds 15, 16, and 18 are not
opened by the resumed scorer
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1849-1875`).
No versioned translation and no re-acquisition was implemented.

Kind 19 has a separate version-1 writer/schema permutation: the writer emits
`tmod_io`, `utau`, `vtau` (`o8_o16_master_writer.F90:476-486`). The kind-20
stress harness uses that fixed producer order only to reconstruct the already
authorized kind-20 input; it does not translate kind 15.

## Compatibility and controls

- Focused O15/kind-20/evidence tests: 8 passed, 12 deselected.
- O8–O12 all-step production score: `PASS_ALL_O8_O12_0_ULP`; artifact
  `o8_o12_master_score.json`, SHA-256
  `fa2cc5fc534b7789dc565c491d58472784bf6e568296635397afb2169c4c69bf`.
- Clean-tree D5 CPU production-JIT year probe: 8,760 steps,
  `D5_PLATFORM_FLOOR`; shipped arm `r6_step1_plus_step2_snow_update` retained
  `full_year_output_sha256 =
  b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2`.
  Probe artifact SHA-256:
  `ab03c16a4f0a424d6be496733f815153f21c4e09681dd4ccbc09d0571b05c726`.
- Receipt citation gate, default gate, and ratchets are recorded after the
  final receipt anchors are committed.

## Choices

### ASKED

- Use the shared source-rounding primitive as a materialization barrier after
  each NEMO transfer statement; demonstrate a barrier-stripped failure.
- Score O15 through production at 0 ULP for every step, field, and category;
  retain the D5 year hash and O8–O12 score.
- Score kind 20 only where NEMO's loop writes the owned interior.
- For kind 15, prepare translation evidence only and wait for the user's
  translation-versus-re-acquisition decision.

### UNASKED

- No NEMO configuration or admitted master-byte change.
- No relaxation of the 0-ULP bar.
- No kind-15 reader translation or re-acquisition.
- No nested Codex review; independent review is the operator (Claude).

## CONFIRMED versus PLAUSIBLE

CONFIRMED: O15's 0-ULP all-step production score; the barrier-stripped +1-ULP
control; kind 20's loop-derived owned domain and 0-ULP score; the writer-fixed
kind-15 permutation and independent-value alignment; unchanged O8–O12 and D5
results.

PLAUSIBLE, not claimed: a future versioned kind-15 translation would make its
remaining O16 outputs admissibly scoreable. That option remains a user
decision.
