# ORCA2 round 281 — passive stage-1 V source replay

Date: 2026-10-15

Status: **HELD**.  The operator's round-280 stage record is admitted.  The
round-279 1,440-cell stage-1 V source-row attribution is retracted because its
in-executable observer is not bit-passive.  A source-exact offline replay
instead names the stage-1 vector-form V update on the full active support as
the first trusted valued boundary.  No package, card, deck, physics, runoff,
or sea-ice setting changed.

Every scientific number below is **independent OMT-4, given NEMO's recorded
stage RHS and external-mode V target**.  It is not mixed with a given-entry
trajectory score.

## Provenance and admission

- Lane base: `bfd749704`.
- Frozen preregistration: `PREREG_nemo_testcases_l4_orca2_round281.md`, commit
  `01e079b5d`.
- Operator record:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round280/acquisition/orca2_omt4_stage1_source_10step_np2`.
- Measurement gate: commit `2f0fef730`; evidence
  `orca2_rounds/round281/measurement.json`, SHA256
  `a1055946e5aaa58181e9bbc72164aa3fe3a52210c81f762dad0047770f5a0d72`.
- Admission log SHA256:
  `dc68a0e0bc35eabf82085dcb5a8352fc7ced4753b9af0bbdd671b6963429d74d`.

The first cross-clone admission correctly refused because round 280's
aggregate producer digest included absolute clone paths.  All five pinned
artifact hashes were identical.  The checker now derives a path-independent
content digest and admits the legacy record only after independently matching
the old runner hash, the other four artifact hashes, and the old manifest
digest.  Re-admission reports `PASS_R280_OMT4_STAGE1_ADMISSION`; all 80
inherited stage frames and both kt=10 restarts remain byte-identical.

## Frozen predictions

| Prediction | Verdict | Measurement |
|---|---|---|
| R281-P1: record admits and all inherited streams remain identical | **CONFIRMED** | admission PASS; 80 frames and two restarts identical |
| R281-P2: raw update bit-exact on the full active support and source row | **REFUTED** | active 1,319 unequal, max 0.011299245240990687 m/s; source row has 323 signed-zero-only differences |
| R281-P3: correction first creates the round-279 1,440-cell residual | **REFUTED** | valued debt is already present after the raw update; the round-279 calibration does not reproduce |
| R281-P4: update and correction exact, leaving the post-stage association first | **REFUTED** | raw update is first |
| R281-P5: walk the correction operands if P3 holds | **OPEN_AT_EARLIER_BOUNDARY** | P3 did not hold; walk stops at the vector update |

Failed predictions remain in the ledger.

## Source-ordered replay and retraction

The replay uses the shared pure `rk3_stage_velocity_update` and
`rk3_stage_barotropic_correction` functions, the production D114 resolved
mesh route, NEMO's recorded `rhs_v` and `ext_vb`, and the independent OMT-4
entry.  It does not insert a callback or materialise an intermediate in the
executing model.

| Matched boundary | active unequal / 415,175 | active max abs (m/s) | source row unequal / 5,400 | source-row max abs |
|---|---:|---:|---:|---:|
| entry V | 0 | 0 | 0 | 0 |
| after raw vector update | 1,319 | 0.011299245240990687 | 323 | 0 |
| after barotropic correction | 1,319 | 0.027227136128767482 | 121 | 0 |
| completed V | 1,319 | 0.027227136128767482 | 121 | 0 |

The first trusted valued boundary is therefore NEMO's stage-1 vector-form V
update, which writes `(vv(Kbb) + rDt*vv(Krhs))*vmask` at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:477-480`.  The
active-support argmax after that statement is `[147,133,27]`; the source row
that round 279 claimed as the owner has no valued error in this replay.

The round-279 result was produced through an in-executable observer.  Two
attempts to reconstruct that seam in round 281 moved unobserved carried
fields and were refused before their numbers were read.  The original
1,440/5,400 and 0.0162073695895 m/s source-row claim therefore lacks a
bit-passivity proof and is **RETRACTED**, not reassigned to the post-stage
association.  The old tool now raises a fail-closed retraction instead of
printing `PASS_R279_RUNOFF_AND_TOPV_DIAGNOSIS`; a direct test proves that
withdrawn claim cannot be emitted again.  Round 279's independent runoff
finding is unaffected.

The new gate fires all eight controls: admission, source order, matched
program point, pre/correction/final one-bit changes, source-row registry, and
the standalone one-bit control.  `plants.log` has SHA256
`e803cbdaf692c2e5f72e393c1d997bc697012fc83c53ba31796dbc1778f737e3`.

## Scope and gates

This round changes validation and receipt code only.  It changes no model
file, recipe, selector, or card, so ORCA2, OMT, GYRE, DINO, tank, runoff, and
sea-ice trajectories are ineligible to move.  Decision 125 remains as taken:
the top-V exchange stays off.  Decision 126 remains pending with the user;
the shipped runoff setting is untouched and is not re-raised.

Focused round-281 tests pass 10/10 and both lint invocations pass.  The old
round-279 module retains its pre-existing import-layout lint exclusions; the
new retraction itself is clean.  The citation gate and the one permitted
`tests/ocean/fidelity -n 12` battery are recorded below after execution.

Independent review: **operator**.  Per the current lane instruction, no
nested review is attempted in this sandbox.

## Configuration choices

- **ASKED:** no new choice.  Decision 126 remains pending outside this round.
- **UNASKED:** none.
- Sea ice remains exactly as the ORCA2 card carries it, including its
  `unmeasured_features` tuple.

## OPEN

1. Reconstruct the candidate's live stage-1 completed RHS offline from the
   admitted passive entry/state operands and prove that replay reproduces the
   candidate update at the floor before reading a leaf.
2. Split the trusted 1,319-cell raw-update boundary in NEMO statement order:
   entry V, completed RHS, `rDt`, three-dimensional V mask, then written
   arithmetic order.  The source row is no longer the walk target.
3. No statement lands until its source-exact arm passes the standing ORCA2 /
   OMT, GYRE, DINO, tank, citation, tests, moved-row, and operator-review gates.
