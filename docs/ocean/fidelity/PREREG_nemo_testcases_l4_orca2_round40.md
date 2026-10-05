# NEMO testcase Lane 4 — ORCA2 card round 40 preregistration

Date: 2026-09-26

Parent: `f07500248a8d326e785a6b78044dfdb9c78ceed3`

Status: **PREREGISTERED BEFORE ROUND-40 SCIENTIFIC SCORING.**

Round 39 proved that the small ranked slow-forcing residual is not a
bit-exact thickness/RHS product hidden by reduction order.  This round splits
the completed three-dimensional momentum RHS at the executing stage-1 operator
boundaries before changing any operand.

All numbers are **given NEMO's entry**.  The six sea-ice selectors and the
card's `unmeasured_features` tuple remain frozen.

## Compiled scope

The executing `stp_2D` order is pressure gradient, lateral diffusion,
vorticity, kinetic-energy gradient, then vertical advection.  The admitted
ranked record carries only the completed RHS, so a new WRITE-only per-rank
stream is required.  It must record the accumulator after every one of those
five calls and must reproduce the old completed-RHS stream before any operator
comparison is admissible.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R40-P1 | Round 39 reproduces exactly at the current parent. | Same product, partial-sum, completed-sum, and reciprocal-arm counts and maxima. | Any movement; stop for instrument drift. |
| R40-P2 | Pressure gradient is the first non-bit momentum operator on the 64 disputed rank-1 U rows; the rest-state LDF, VOR, KEG, and ZAD calls do not move those rows. | Post-HPG is unequal and post-HPG through post-ZAD are bit-identical to one another on all 1,920 layer values. | HPG is exact, or any later operator first changes a disputed value. |
| R40-P3 | The new stream is observational only and closes against the admitted completed RHS. | Post-ZAD equals the round-20 ranked `uu(Krhs)` and `vv(Krhs)` bit-for-bit on both ranks. | Any unequal value, changed restart identity, or changed old stream; reject the acquisition. |
| R40-P4 | No model statement lands unless one isolated operator substitution reaches zero unequal cells under the existing ranked boundary. | One operator-only arm closes 0/64 and all trajectory gates pass. | Any nonzero residual or multi-operator requirement; hold. |

The acquisition must use a new target name, one file per rank, exact header,
byte-count, completion, restart-identity, and old-stream calibration checks.
A swapped-rank header plant and a one-ULP scored-boundary plant must fire.
Failed predictions remain REFUTED.

## Choices

ASKED: split the completed three-dimensional RHS by the compiled momentum
operators before any fix.

UNASKED: none.
