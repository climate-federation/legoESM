# Grid ladder rung G1b — round 6 receipt, Decision 127 correction

Status: **LANDED; G1b remains DEBT**.  The rejected per-statement subnormal
preservation has been removed.  Decision 127 now lives once in the shared
trajectory comparison helper used by the ladder: a NEMO nonzero binary64
subnormal paired with card `+0.0` or `-0.0` matches and is counted separately;
every other unequal active cell is scored against the unchanged `1e-15`
normalized bar.

Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g1b_round6.md`
(`b4d2755c5`, before the original round-6 measurement).  Corrected evidence is
under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round35/decision127/`.

## Retraction and implementation

The original round-6 landing claim is **RETRACTED**.  Commit `3f00fe185` scaled
the KE-gradient path by exact powers of two, but failed when the final result
was itself subnormal and applied the recovered increment after NEMO's
barotropic-removal statement rather than before it.  The option, G1b selector,
private cross-module import, production helpers, CLI removal arm, and dedicated
tests are all removed; no dead selector or field remains.

The shared scorer now reports raw identity, exactness modulo Decision 127, and
the unchanged bar status separately.  Its three direct classifier plants
establish:

- NEMO `+/-1.0e-310` versus card `+0.0`/`-0.0`: matching, two
  flushed-subnormal cells;
- NEMO `2.3e-308` versus card zero: not exact modulo D127, one normal-range
  difference, but AT-BAR;
- NEMO `1.0e-310` versus card `2.0e-310`: not exact modulo D127, one
  normal-range difference, but AT-BAR.

The latter two plants are deliberately far below `1e-15`: they prove that D127
does not reclassify those cells as exact while the ordinary bar still scores
their rows AT-BAR.  The existing one-unit wet-state plant remains DEBT.

## Producing statement, first non-exact cell, and first debt

At kt 7, stage 3, U cell `[130,113]`, NEMO's C2 kinetic-energy-gradient
contribution is `-6.279798654e-314` and the card's is zero.  That pair is the
tolerated class at its producing statement, NEMO
`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynkeg.f90:117-130` (enclosing routine
`:61-175`).

The next scored statement is not tolerated: the stage-3 U values are normal
(`-7.767256946156943e-297` in NEMO and
`-7.767256946156942e-297` in the card), one bit and
`1.35807730622e-312` apart at `[130,113]`.  Stages 1 and 2 remain matching;
stage 3 U is therefore the first normal-range trajectory difference under both
whole-step labels and the restart-entry label.  Its normalized error is
`1.35807730622e-312`, so the row is AT-BAR, not DEBT.  NEMO's vector stage
update and barotropic removal are at
`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynzdf.f90:138-160` (enclosing routine
`:67`).  This stage-3 U statement is the next trace target; it was not traced
further in this correction round.

The first row actually over the campaign bar is the INDEPENDENT chain's kt-81
SSH row: cell `[1,49]`, normalized maximum `8.904552442623448e-15`.  GIVEN
through kt 50 and restart-entry through kt 100 have no over-bar row.

Decision 122's full-ORCA2 source carries the same compiled statements at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynkeg.f90:117-130` and
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:138-160`.

## Restored-bar rescore

The GIVEN table excludes only kt-1 T/S land cells, the standing convention;
no wet or scored-field row is excluded.

| label / scope | exact | exact modulo D127 | AT-BAR | DEBT | normal-difference cells | flushed cells | first normal difference | first over-bar row |
|---|---:|---:|---:|---:|---:|---:|---|---|
| INDEPENDENT, kt 1..100 | 170 / 598 | 178 / 598 | 575 / 598 | 23 / 598 | 3,216,400 | 222 | kt 7 U `[130,113]` | kt 81 SSH `[1,49]` |
| GIVEN-NEMO-ENTRY, kt 1..50 | 263 / 348 | 271 / 348 | 348 / 348 | 0 / 348 | 2,045 | 222 | kt 7 U `[130,113]` | none |
| restart entry, kt 3..100 | 305 / 490 | 313 / 490 | 490 / 490 | 0 / 490 | 4,812 | 222 | kt 7 U `[130,113]` | none |

The kt-7 stage-local rescore identifies that whole-step U cell at stage 3;
stages 1 and 2 are exact there.

All three G1b labels have the same nonzero flushed-subnormal tally by field and
step:

| step | ssh | uu_b | vv_b | u | v | T | S |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 5 | 6 | 4 | 1 | 5 | 0 | 0 |
| 9 | 25 | 32 | 34 | 21 | 17 | 0 | 0 |
| 10 | 11 | 9 | 10 | 6 | 6 | 0 | 0 |
| 11 | 7 | 7 | 5 | 7 | 4 | 0 | 0 |

Every field at steps 1–7 and 12–100 in the INDEPENDENT record, at steps 1–7
and 12–50 in the GIVEN record, and at scored restart steps 3–7 and 12–100 has a
zero flushed-subnormal tally.  The table totals 222 D127-matching cells per
label.  Normal-range differences at steps 9 onward retain ordinary bar status.

The certified G1-flat control was re-scored over steps 1–12: all 84 rows are
AT-BAR and none is DEBT (68 exact, 70 exact modulo D127, 159 normal-difference
cells, 126 flushed-subnormal cells).  Its first normal-range differences at kt
9 are about `1e-305`, so G1 keeps its certification under the restored bar.

## Controls, provenance, and choices

- The three Decision-127 plants pass in the shared scorer's direct test; the
  ordinary one-unit wet-state plant still leaves the bar.
- Focused CPU/fp64 tests passed: 11 direct scorer/ladder tests and six G1
  classifier, environment, and 100-step option-plant executions.
- After substantive commit `1e7ee31e1`, the required receipt-citation and
  no-private-cross-import gates passed 23/23 on the clean tree; the citation
  gate's plants fired.  All five final JSON artifacts stamp that commit.
- The G1-flat 12-step control; G1b independent 100-step, GIVEN 50-step, and
  restart-entry 100-step labels; and the kt-7 stage rescore all ran
  synchronously on CPU in fp64.
- The operator's independent Claude review verdict was **HOLD** and caused the
  preservation retraction.  No Codex review was run, as instructed for this
  resume.
- Search-before-build found the shared phase-3 trajectory `score` helper and
  extended it; no second tolerance implementation was added.
- Controlled comparison: record, card, deck, timestep, masks, labels, and
  precision are unchanged.  Only the rejected workaround was removed and the
  user-selected comparison rule changed.
- Choices made: ASKED — restore the `1e-15` bar, retain D127 only as the
  producing-cell exactness reclassification, rescore all three labels, and
  restore the DINO citation.  UNASKED: none.

Artifact SHA-256: `g1_flat_independent12_restored_bar.json`
`429702fceccc73540d9d96d5d807aa0b378881e5de92ba992952b21c64b57d4c`;
`independent100_restored_bar.json`
`2cd04f520535482a7e75a6ae61c9fd49985dcfba3ae27793e560bc663612f1eb`;
`given_entry50_restored_bar.json`
`411b5e7500f0429883a839d8d02e76f10c7a03629e0ce488f245e7bb135bb3e9`;
`restart_entry100_restored_bar.json`
`c64c724f7764d6350aa68ea25b99c0f2608a7b7fa97e4b259c3655ee0ad5fd35`;
`stages_kt7_restored_bar.json`
`bd45d4b845b6d3beb28a607b3c71e4fff37ca76eff27432c94096f64a23cf376`.

UNVERIFIED:
- the full ORCA2 card executes the cited statements but was read, not run;
- the stage-3 U normal-range difference is localized, not traced further;
- G1b remains uncertified.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/gridladder_r35/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: LANDED — Decision 127 is shared; kt-7 stage-3 U is the first non-exact normal-range cell and kt-81 SSH the first over-bar row.
