# PREREGISTRATION — grid ladder rung G1b, round 5 (gridladder_r29)

Frozen before any round-5 measurement.  The corrected round-4 card at
`e1062dce910a81ec0bc090de463014e7ca13b255` is the immutable before arm.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round29/`.

## Record, labels, and bar

- NEMO record: `round26/g1b_merged`, admitted over steps 2..100.
- GIVEN-NEMO-ENTRY: entry frames kt 1..50 and restarts kt 3..100.
- INDEPENDENT: the card's own chain kt 1..100.
- Bar: maximum absolute error normalised by NEMO's field maximum, `1e-15`.
- Round-4 before values: GIVEN has no field over the bar.  INDEPENDENT first
  exceeds it at ssh/u/v kt 27/63/62; worst ssh/u/v is
  `1.70e-14`/`2.09e-15`/`2.28e-15`, with 46 exact rows and 5,508,585 unequal
  cells over all scored fields.

## Compiled statement and measurement

The G1b record build accumulates HPG, VOR, then KEG in `stp2d.f90:138`,
`:144`, `:161`; KEG subtracts its completed gradient from Krhs at
`dynkeg.f90:128-130`.  The Decision-122 ORCA2 build executes the same lines at
`stp2d.f90:141`, `:147`, `:164` and `dynkeg.f90:128-130`.

Round 4 leaves the kt-2 stage-1 boundary at 4 U / 0 V unequal faces, maximum
`1.65e-24`, after HPG -> VOR -> KEG.  HPG and KEG are individually exact.
For exactly those faces, a traced-input JIT probe will compare (a) the current
optimized accumulator, (b) a one-source-operation-at-a-time NumPy replay,
(c) an explicit fused multiply-add replay, and (d) the same JIT with the
existing `nemo_source_round` materialisation around KEG's completed addend and
the following Krhs add.  Optimized HLO is supporting evidence only; values on
the recorded faces decide the result.

The next compiled statement is the stage-1 vector velocity update at
`stprk3_stg.f90:366-370` in the record build and `:467-471` in ORCA2.  It is
walked after the RHS result, using the existing stage replay.

## Frozen predictions and falsifiers

| id | prediction | falsifier |
|---|---|---|
| R5-M1 | on all four unequal U faces, the current traced JIT equals the fused replay and differs from the one-operation-at-a-time value | any face does not have that relationship |
| R5-M2 | `nemo_source_round` around KEG's final addend and the Krhs add makes the traced JIT equal the one-operation-at-a-time replay on all four faces; V remains exact | any recorded face remains unequal, or a previously exact V face moves |
| R5-C1 | removing either source-round materialisation from the traced test produces at least one unequal value | either planted removal still passes |
| R5-S1 | production selection makes the kt-2 stage-1 RHS 4/0 -> 0/0; the following vector velocity statement remains 0/0 before and after | RHS remains unequal, or the velocity statement is non-bit |
| R5-L1 | GIVEN remains at the bar over both record arms, with no exact row lost | any GIVEN field exceeds the bar or an exact row is lost |
| R5-L2 | INDEPENDENT remains uncertified and its first-over-bar steps and worst scores are unchanged or move only in the registered last-place class | all fields reach the bar (CERTIFIED), or a material trajectory regression appears |

If R5-M1 is refuted, no contraction guard lands; the first non-bit operand is
named from the traced values and cited to the executed NEMO statement.  If it
holds, the change is selected by G1b's existing
`rk3_vector_rhs_nemo_order` path; no new selector or default is introduced.

## Landing and stopping rules

The statement lands only if its recorded boundary becomes bit-identical, the
traced-input test is non-vacuous, and no exact GIVEN row is lost.  G1b is
CERTIFIED only if every scored field is at the bar under both labels through
kt 100.  No NEMO build or run is performed.

## Choices

ASKED: measure the contraction reading; if confirmed, use the existing source
rounding mechanism on the G1b-selected statement; walk the next velocity
statement; rescore both labels.  No default, deck, timestep, record, scoring,
or state choice is changed.
