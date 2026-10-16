# PREREGISTRATION — grid ladder rung G1b, round 6 (gridladder_r35)

Frozen before any round-6 measurement.  The round-5 card at
`1405ca63209e1ff422b1936b7f05da0ec6c0df8d` is the immutable before arm.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round35/`.

## Record, labels, and bar

- NEMO record: `round26/g1b_merged`, admitted over steps 2..100.
- GIVEN-NEMO-ENTRY: entry frames and the per-step NEMO restarts.
- INDEPENDENT: the card's own chain kt 1..100.
- Bar: maximum absolute error normalised by NEMO's field maximum, `1e-15`.
- Round-5 result: GIVEN is at the bar over the record.  INDEPENDENT first
  exceeds it at ssh kt 81 and v kt 99; u never exceeds it.  Worst error is
  `1.4407e-14` (ssh).
- Already on file, not a round-6 measurement: the first INDEPENDENT unequal
  step-final value at all is kt 7 U cell `[130,113]`, one unequal cell,
  maximum `1.35807730622e-312`; NEMO and the card values there are about
  `7.767256946156943e-297` and differ by one binary64 bit.  kt 1..6 are
  bit-identical at every scored step-final field.  The restart-entry arm also
  produces that kt-7 cell, so it is inside the GIVEN label's tolerated
  last-bit class, not evidence by itself of unequal carried state.

## Frozen predictions and falsifiers

| id | prediction | falsifier |
|---|---|---|
| R6-F1 | the first independent leave is kt 7, stage 1, U cell `[130,113]`; entry and the external-mode handoff are bit-identical, and no earlier stage field differs | any kt 1..6 stage field differs, or kt-7 entry/handoff differs, or the first kt-7 leave is not stage-1 U at that cell |
| R6-C1 | every restart-held leaf carried into kt 7 after kt 6 is bit-identical: T, S, u, v, eta, uu_b, vv_b and all six barotropic histories; `w` and the optional after-ssh slot are listed separately because NEMO's restart does not hand them to the card | any restart-held leaf differs; that leaf becomes the causal-swap target |
| R6-C2 | kt 7 from the card's carried state and from NEMO's restart produces the same one-cell U leave; replacing all restart-held fields therefore does not remove it | the restart arm is bit-identical, or a single unequal carried leaf owns the leave under a two-way swap |
| R6-S1 | the first non-bit statement is a stage-1 momentum arithmetic statement after the exact external handoff, most likely the vector velocity update in `stp_RK3_stg`; the compiled-order walk names the first unequal boundary and opens its enclosing subroutine before citation | the leave first appears in a tracer, barotropic, later-stage, or unrecorded carried-state boundary |
| R6-L1 | if a source-order transcription is proven, selecting it on G1b makes the kt-7 boundary bit-identical, loses no exact GIVEN row, and leaves no measured ORCA2 card with an earlier first-over-bar row under Decision 122 | the local boundary remains unequal, any exact GIVEN row is lost, or any measured card's first-over-bar row moves earlier |
| R6-L2 | after an eligible landing, both labels are at the bar through kt 100; if any field remains over the bar, G1b remains DEBT rather than CERTIFIED | every field is at the bar (CERTIFIED), or a remaining over-bar field is hidden or waived |

## Measurement and landing rules

The existing G1 ladder is extended rather than duplicated: one carried-state
arm, modelled on the G5 arm, compares every pytree leaf with the kt-6 NEMO
restart and performs controlled swap steps; the stage arm walks kt 7 in
compiled order.  A new arm must have a direct test and a planted violation.

Only a G1b-card-selected statement demonstrated bit-identical from NEMO's own
inputs may land.  Its compiled NEMO line is cited in both the G1b record build
and `ORCA2_OMIP_L4`, after opening the enclosing subroutine.  Defaults and all
other cards remain unchanged unless Decision 122's measured conditions require
the already-authorised full-ORCA2 selection.

## Choices

ASKED: locate the first unequal bit; compare all carried state with the
restart; walk the stage; land only a cited transcription defect; rescore both
labels over the record.  No default, deck, timestep, record, scoring, precision
or carried-state choice is changed.  UNASKED: none.
