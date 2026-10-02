# Preregistration — ORCA2 round 111 EEN fraction admission and operand walk

Date: 2026-10-02. Base: `1eb05f36d7cc5654412645cbe069f71f4605c35e`.
Scope is measurement-only on hierarchy rung 0. Every ocean number is
**independent** because rung 0 starts from NEMO's own from-rest state. No model
physics, card field, configuration value, carried state, threshold,
stabilizer, sea-ice selector, or `unmeasured_features` entry may change.

## Admitted boundary

Round 109 proved that the first raw non-bit boundary is NEMO's ordered
three-fraction `zpvo_nw` assignment: 180 magnitude differences, all at global
row `j=0`, level `k=0`. Round 110 committed an additions-only recorder for the
west, center, and south fractions and all operands. The operator reports that
the acquisition and its admission completed with
`PASS_R110_EEN_FRACTION_ADMISSION`. This round has not inspected the admission
JSON or fraction payloads before freezing the predictions below.

The compiled rung-0 source evaluates the west, center, and south fractions in
that order, then performs the two ordered additions
(`ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1241-1245`).

## Frozen predictions and falsifiers

These predictions preserve round 110's frozen R110-P1 through R110-P4 rather
than adapting them to the operator's success marker.

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R111-P1 | The round-110 writer is observationally passive. | Both new records admit; every inherited round-105/107 stream and every rank's kt=1..10 restart shard is byte-identical to the admitted round-108 run; all arithmetic and planted violations pass/fire. | Any moved inherited stream, restart, or replay rejects the record. Repair only the instrument and reacquire before reading the fractions. |
| R111-P2 | West and center fractions are bit-exact; the south fraction is the first unequal fraction and owns exactly the 180 row-`j=0`, level-`k=0` magnitude differences. | West and center have zero unequal bits; south has exactly 180 magnitude differences at that row/level; the ordered replay equals admitted `zpvo_nw` bitwise. | The first earlier unequal fraction owns the walk. If all fractions are exact, the first unequal ordered addition owns it. Preserve the failed prediction as REFUTED. |
| R111-P3 | The first south-fraction mismatch is an operand association in NEMO's southern halo, not division arithmetic. | At least one of `ff_f`, `e3f_0vor`, `r3f`, or `fe3mask` differs at the same 180 cells; recomputing from NEMO's recorded operands reproduces its recorded quotient bitwise. | If all operands are exact but the quotient differs, stop at division/source rounding. If the quotient cannot replay from its operands, reject the instrument. |
| R111-P4 | The west/center rank-seam association is not involved. | West and center fractions and operands are bit-exact, including the seam. | Any earlier seam-local operand mismatch stops the walk there. |
| R111-P5 | This round stops at the first unequal source-ordered fraction or operand. | No downstream product, accumulator, northern-fold, 68-cell, or later-substep statement is landed while this boundary remains open. | Advancing past an open earlier boundary is a process failure and must be retracted. |

## Measurement and landing bar

First run the committed round-110 launcher with `--admit-existing`; require its
unplanted PASS and every record plant to fire. Then extend the existing
round-109 model-side walk to compare the recorded fractions and operands in
compiled source order on executed levels and rank-complete owned cells. The
measurement must use production JIT on CPU with fp64 policy and x64 enabled,
and must print state and geometry dtypes.

The round may land a single statement only if the first mismatch is a
NEMO-cited shared implementation statement, the recorded NEMO inputs make that
statement bit-exact on every executed cell, no AT-BAR ORCA2 row leaves the bar,
the first-over-bar row does not move earlier, every moved row is registered,
and the full shared-card gate passes. Otherwise the round is HELD with the
first unequal statement and the next discriminator. No configuration choice
is authorized.

