# Preregistration — ORCA2 round 110 EEN fraction and southern-halo walk

Date: 2026-10-02. Base: `e78352254c58083ed39409cbde81725cd18fa204`.
Scope is instrumentation-only measurement on hierarchy rung 0. Every ocean
number is **independent** because rung 0 starts from NEMO's own from-rest
state. No model physics, card field, configuration value, carried state,
threshold, stabilizer, sea-ice selector, or `unmeasured_features` entry may
change.

## Admitted boundary

Round 109 admitted both rank-complete per-level streams after proving the base,
round-105, and round-108 builds have byte-identical kt=10 restarts. Its
source-ordered walk found the first non-bit boundary at NEMO's three-term
`zpvo_nw` assignment: 180 magnitude differences, all on global row `j=0`,
level `k=0`. The compiled rung-0 source evaluates the west, center, and south
fractions in that order
(`ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1241-1245`).
The admitted record contains only the completed sum, so no fraction or halo
operand may yet be named as owner.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R110-P1 | A write-only recorder for the three fractions and their operands is observationally passive. | Every rank's kt=1..10 restart shard is byte-identical to the admitted round-108 run; inherited round-105 and round-107 streams are byte-identical too. | Any moved restart or inherited stream rejects the instrument; reduce and reacquire before quoting a fraction. |
| R110-P2 | The west and center fractions are bit-exact; the south fraction is the first non-bit fraction and owns all 180 row-`j=0`, level-`k=0` magnitude differences. | West and center score 0 unequal bits, south scores exactly 180 magnitude differences at that one row and level, and replaying the recorded ordered sum reproduces admitted `zpvo_nw` bitwise. | The first earlier unequal fraction owns the walk. If the three fractions are exact but their ordered sum is not, stop at the sum's arithmetic association. Any other census refutes the predicted location/count and is reported without reinterpretation. |
| R110-P3 | The south-fraction difference is an operand-association difference in NEMO's southern halo, not division arithmetic. | At least one recorded south operand (`ff_f`, `e3f_0vor`, `r3f`, or `fe3mask`) is non-bit against legoESM's associated operand at the same 180 cells; recomputing the quotient from NEMO's recorded operands reproduces NEMO's recorded south fraction bitwise. | If all operands are exact but the quotient differs, division/source-rounding owns the boundary. If the recorded quotient does not replay from its recorded operands, the instrument is insufficient or perturbing and is rejected. |
| R110-P4 | The north/west cyclic association is not involved in this southern-row boundary. | West and center operand rows are bit-exact, including the rank seam. | Any seam-local difference stops the walk at the first source-ordered operand and refutes the south-only attribution. |
| R110-P5 | Downstream product and accumulator signed zeros remain unowned until `zpvo_nw` is closed. | The round ends at the first fraction/operand mismatch or with `zpvo_nw` bit-exact; no downstream signed-zero claim is landed from the same measurement. | Interpreting the product/addition rows while an earlier fraction is non-bit is a process failure and must be retracted. |

## Acquisition and admission bar

The additions-only patch records, for every executed U level on both ranks:

1. the west, center, and south fractions in compiled source order;
2. for each fraction, `ff_f`, `e3f_0vor`, `r3f`, `fe3mask`, and the exact
   denominator used by the quotient;
3. the ordered west-plus-center partial sum and final three-term sum; and
4. `mbku`, so unwritten levels and NEMO's dummy level are rejected.

The writer uses a new absolute, pre-created, per-rank path and a
self-describing header. Its checker parses field names and dimensions from the
record, requires exact two-rank owned-domain coverage, rejects writes outside
`mbku`, replays every denominator, quotient, and ordered sum bitwise, compares
the final sum bytewise to the admitted round-108 `zpvo_nw`, and proves each
header/field/dimension/truncation/rank/bottom/arithmetic/restart control fires.
The launcher pins the source deck, binary inputs, patch, writer, checker, and
this preregistration by content rather than by a moving commit SHA.

If the operator-run record does not already exist, this round ends
`STOPPED_FOR_RECORD` with `ACQUISITION_NEEDED` pointing to the committed
launcher. No fraction magnitude is reported from preflight or offline
reconstruction alone.

