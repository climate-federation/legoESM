# Preregistration — ORCA2 round 127 northern EEN accumulator/scale pair

Date: 2026-10-03. Base: `0547735fa0ff84943d348fb9bfbde3cf691e7cca`.
Every hierarchy-rung-0 number is **independent** because rung 0 starts from
NEMO's own from-rest state. Shipped rung-7 numbers are **given NEMO's recorded
entry** under Decision 52. No configuration, forcing, initial state, carried
state, stabilizer, sea-ice selector, or `unmeasured_features` entry may change.

## Admitted boundary and compiled statements

Rounds 121--126 made every recorded northern NW/NE `zpvo` fraction, live
thickness, and frozen-mask operand bit-exact. Round 119 had already shown that
ordinary IEEE zero addition closes the SW/SE V recurrences when their inputs
are exact. Round 106 measured the still-open northern pair: replacing only the
accumulator or only the final scale leaves 66 `ffv_nw` and 67 `ffv_ne`
magnitude differences, while replacing both closes the final coefficients.

The executed rung-0 source accumulates northwest and northeast V coefficients
with ordinary binary64 addition
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1339-1348`), then
forms their scales from northern `e2u(ji-1,jj+1)` and `e2u(ji,jj+1)` and
multiplies the accumulator
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1360-1367`).
The ordinary U-grid T-pivot north-fold association supplying those `e2u`
operands is the compiled `lbc_nfd` U branch
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/lbcnfd.f90:653-681`).

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R127-P1 | The admitted round-120 record and inherited round-105/118 streams remain passive and complete. | Both ranks cover the domain exactly once; record hashes, inherited streams, and terminal restarts retain their admitted identities. | Refuse all numerical interpretation and stop for a new record only if the existing one is incomplete or perturbed. |
| R127-P2 | With the round-126 operand associations applied, NW and NE recurrence terms are bit-exact before accumulation. | Every recorded NW/NE term becomes 0/0 bit/magnitude unequal. | Stop at the first surviving operand or product; do not interpret accumulation or scale. |
| R127-P3 | NEMO's ordinary IEEE zero-addition recurrence makes all four V accumulators bit-exact once their terms are exact. | NW/NE before/after streams and terminal accumulators become 0 unequal; SW/SE remain 0 unequal under the same rule. | Retain the failed prediction and name the first unequal recurrence level; no production change. |
| R127-P4 | The compiled U-grid north-fold association makes `scl_v_nw` and `scl_v_ne` bit-exact without moving the other six scales. | All eight recorded scales become 0 unequal. | Stop at the first surviving scale operand/association. |
| R127-P5 | The complete recurrence-plus-scale pair makes all eight final EEN coefficients bit-exact. | Every final coefficient is 0 bit unequal; recurrence-only and scale-only controls retain respectively 66/67 magnitude differences for NW/NE. | Any surviving final bit rejects a landing; either half becoming exact alone refutes the registered cancelling-pair framing. |
| R127-P6 | Resolved execution scope remains literal-EEN cards only. | ORCA2, both DINO recipes, VORTEX, and VORTEX_VEC execute; GYRE, LOCK_EXCHANGE, and OVERFLOW do not. | Any scope movement refuses the arm. |
| R127-P7 | If P1--P6 confirm, the paired source transcription loses no exact ORCA2 ladder row and does not worsen any executing-card gate. | Both 200-row ORCA2 ladders satisfy their frozen registries; every moved row is registered; GYRE, DINO, tanks, generic cards, citations, and tests satisfy the standing gates. | Hold with the first exact-row loss, earlier first debt, unregistered row, or executing-card regression named. |

## Measurement and landing bar

The measurement must reuse the admitted round-105 accumulator/scales,
round-118 per-level recurrence, round-120 fraction record, and round-98 final
coefficients. It runs production JIT on CPU under fp64/x64/libm, refuses a
dirty or wrongly stamped tree, and carries independent oracle-bit,
candidate-bit, recurrence-only, scale-only, wrong-fold-row, and scope-route
controls. No new NEMO run is predicted necessary.

The pair may land only together and only if every source-ordered operand,
recurrence, scale, and final coefficient is bit-exact; each half demonstrably
fails alone; both ORCA2 ten-step ladders retain their exact-row/first-debt
predicates; the complete shared-card gates pass; and the citation plant fires.
