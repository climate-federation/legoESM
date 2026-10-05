# NEMO testcase Lane 4 — ORCA2 card round 45 preregistration

Date: 2026-09-27

Parent: `010aea25d7423c48d5f3e4fb7b6840be05949c39`

Status: **PREREGISTERED BEFORE ROUND-45 SCIENTIFIC SCORING.**

All scientific numbers are **given NEMO's entry**.  The six sea-ice
selectors and the card's `unmeasured_features` tuple remain frozen.

Round 44 made the stage-1 after-SBC Krhs exact and exposed 57,160 temperature
and 57,180 salinity cells in the immediately following QCO/RK combine.  This
round walks that single compiled assignment from its admitted Kbb, Krhs,
r3t(Kbb), r3t(Kmm), and r3t(Kaa) operands.  No upstream owner is inferred from
the endpoint residual.

## Executing compiled statement

For stage 1, the executing QCO branch assigns Kaa as the sum of the
thickness-weighted Kbb tracer and the stage-interval-scaled,
thickness-weighted Krhs, divided by the Kaa stretch at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:670-681`.
The record writer immediately below that assignment stores both tracer pairs
and all three r3t operands at `stprk3_stg.f90:687-696`.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R45-P1 | The landed round-44 instrument reproduces before any arm. | After-SBC T/S are 0 unequal; stage-1 is 57,160 T and 57,180 S unequal with maxima `7.105427357601002e-15` and `1.4210854715202004e-14`. | Any count or maximum moves; stop for instrument drift. |
| R45-P2 | The admitted RK record contains every operand needed for the compiled assignment and the scorer binds. | Exact schema/digest, finite fp64 operands, expected stage/time-level indices, and a one-ULP active-cell plant fires once. | Missing/non-finite operand, schema/time-level drift, or inert plant; stop for record/instrument repair. |
| R45-P3 | Literal source-ordered replay of the compiled assignment closes Kaa. | T and S are both 0 unequal on the record-backed support. | Either tracer remains non-bit; name the first subexpression and hold. |
| R45-P4 | The first non-bit production subexpression is an arithmetic association within the QCO/RK assignment, not an operand mismatch. | Recorded operands replay exactly, while replacing them one at a time by production operands leaves the replay exact until a source-ordered arithmetic boundary moves. | Any production operand is already non-bit against its recorded counterpart; name that operand as first and hold. |
| R45-P5 | A single-statement source-order repair closes the stage-1 tracer row without collateral movement. | Production stage-1 T/S become 0 unequal; no ORCA2 AT-BAR row leaves the bar; first-over-bar is not earlier; every moved row is registered; GYRE is byte-identical. | Any stage-1 residual, unsafe ORCA2 movement, or GYRE movement; do not land. |

The score domain is the existing support-safe rank-0 wet interior.  The gate
must expose the ordered subexpressions, refuse an unexpected card selector or
record shape, print CPU/fp64/libm, and include a non-vacuous one-ULP plant.

## Choices

ASKED: continue at the exact-Krhs-to-stage-1 QCO/RK combine and land only a
single compiled-statement transcription if the complete ORCA2 and GYRE gates
pass.

UNASKED: none.  No configuration, carried state, selector, sea-ice field,
stabilizer, score domain, or scientific threshold changes.
