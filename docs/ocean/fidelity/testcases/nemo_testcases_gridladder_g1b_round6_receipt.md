# Grid ladder rung G1b — round 6 receipt, Decision 127 correction

Status: **LANDED; G1b remains DEBT**.  The rejected per-statement subnormal
preservation has been removed.  Decision 127 now lives once in the shared
trajectory comparison helper used by the ladder: a NEMO nonzero binary64
subnormal paired with card `+0.0` or `-0.0` matches and is counted separately;
every other unequal active cell is debt.

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

The shared scorer now reports raw bit identity separately from Decision-127
matching.  Its three direct plants establish:

- NEMO `+/-1.0e-310` versus card `+0.0`/`-0.0`: matching, two
  flushed-subnormal cells;
- NEMO `2.3e-308` versus card zero: debt, one normal-range difference;
- NEMO `1.0e-310` versus card `2.0e-310`: debt, one normal-range difference.

The normal-range plants are deliberately far below the old `1e-15` normalized
bar: they prove that Decision 127 does not silently turn the rule into a wider
tolerance.

## Producing statement and first remaining debt

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
labels.  NEMO's vector stage update and barotropic removal are at
`ORCA2G1_OMIP_L1_R4REF/BLD/ppsrc/nemo/dynzdf.f90:138-160` (enclosing routine
`:67`).  This stage-3 U statement is the next trace target; it was not traced
further in this correction round.

Decision 122's full-ORCA2 source carries the same compiled statements at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynkeg.f90:117-130` and
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:138-160`.

## Both-label rescore

The GIVEN table excludes only kt-1 T/S land cells, the standing convention;
no wet or scored-field row is excluded.

| label / scope | matching rows | normal-difference cells | flushed-subnormal cells | first normal-range debt |
|---|---:|---:|---:|---|
| INDEPENDENT, kt 1..100 | 178 / 598 | 3,216,400 | 222 | kt 7, stage 3, U `[130,113]` |
| GIVEN-NEMO-ENTRY, kt 1..50 | 271 / 348 | 2,045 | 222 | kt 7, stage 3, U `[130,113]` |

Both labels have the same nonzero flushed-subnormal tally by field and step:

| step | ssh | uu_b | vv_b | u | v | T | S |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 5 | 6 | 4 | 1 | 5 | 0 | 0 |
| 9 | 25 | 32 | 34 | 21 | 17 | 0 | 0 |
| 10 | 11 | 9 | 10 | 6 | 6 | 0 | 0 |
| 11 | 7 | 7 | 5 | 7 | 4 | 0 | 0 |

Every field at steps 1–7 and 12–100 in the INDEPENDENT record, and at steps
1–7 and 12–50 in the GIVEN record, has a zero flushed-subnormal tally.  The
table totals 222 tolerated cells per label.  Normal-range differences at steps
9 onward remain debt even where the same row also contains tolerated cells.

## Controls, provenance, and choices

- The three Decision-127 plants pass in the shared scorer's direct test; the
  ordinary one-unit wet-state plant still leaves the bar.
- Focused CPU/fp64 tests passed: 32 shared-scorer, G1b-card, environment, and
  100-step option-plant tests.  After the substantive commit, the required
  receipt-citation and no-private-cross-import gates passed 23/23 on the clean
  tree; the citation map's shifted-anchor plants all fired.
- The independent 100-step and GIVEN 50-step rescoring ran synchronously on
  CPU in fp64.  The kt-7 stage rescore independently names stage 3 U and the
  same cell.
- The operator's independent Claude review verdict was **HOLD** and caused the
  preservation retraction.  No Codex review was run, as instructed for this
  resume.
- Search-before-build found the shared phase-3 trajectory `score` helper and
  extended it; no second tolerance implementation was added.
- Controlled comparison: record, card, deck, timestep, masks, labels, and
  precision are unchanged.  Only the rejected workaround was removed and the
  user-selected comparison rule changed.
- Choices made: ASKED — remove the preservation, adopt Decision 127, rescore
  both labels, correct the DINO citation.  UNASKED: none.

Artifact SHA-256: `independent100.json`
`d318e4cd669831fb0419a128439fb729d30fab76f09e9f3aff2f4ebc6c268361`;
`given_entry50.json`
`73ba36c246f2864c36e7ed0c2f7fd332e6630d8e9b99c3d860bebb80ac1205c9`;
`stages_kt7.json`
`6a6855f837b5f25018e12bc82370ccfbfda8219628ce578ceb6e707e00696d74`.

UNVERIFIED:
- the full ORCA2 card executes the cited statements but was read, not run;
- the stage-3 U normal-range difference is localized, not traced further;
- G1b remains uncertified.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/gridladder_r35/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: LANDED — Decision 127 is shared and the next debt is kt-7 stage-3 U.
