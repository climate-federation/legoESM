# ORCA2 round 55 preregistration — OVERFLOW ENS signed-zero operand walk

Date frozen: 2026-09-27  
Base: `39fedb35cbc4158338750449fb4a5d67f07102c6`  
ORCA2 claim label: **given NEMO's entry** (Decision 52; the held QCO arm)  
OVERFLOW claim label: **given NEMO's recorded operands**

## Frozen question and scope

The operator completed round 53's additions-only acquisition under producer
commit `39fedb35cbc4158338750449fb4a5d67f07102c6`.  The record admission reports
`exact=30/33`, three inherited streams changed only in admitted passive bytes,
and all consumed fields equal.  This round admits the new self-describing ENS
record, locates round 52's first active-u signed-zero cell, and walks the
compiled `zwz`, `zuav`, product, and accumulator statement in source order.

The measurement may not change the card, selector, carried state, threshold,
mask, sea-ice tuple, or NEMO arithmetic.  A model statement is eligible only
if the acquired operands identify it exactly and the held source-ordered QCO
statement plus that one statement passes every shared-card gate.  Otherwise
the round is HELD with the first refuting row named.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R55-P1 | The operator record is admissible against its round-50 stage-2 parent and producer stamp. | The committed round-53 gate returns `AT_BAR`; the header is kt=3, stage=2, `Kmm`/shape/origin/fp width equal the parent, and the payload and stamp plants refuse. | Any admission or provenance failure: `STOPPED_FOR_RECORD`; make no scientific claim. |
| R55-P2 | Round 52's first unequal active-u cell is the compiled IEEE chain `-0.0 + +0.0 -> +0.0`. | At legoESM-local `[1,31,0]`, `rhs_before_u` is negative zero, `product_u` is positive zero, and `rhs_after_u` is positive zero, with `rhs_after_u` bit-equal to round 50's recorded `after_vor_u`. | Any different sign or nonzero arithmetic value: **REFUTED**; report the first differing operand and continue from it in source order. |
| R55-P3 | The same chain owns all 16,135 round-52 active-u bit differences. | The active-u census has exactly 16,135 cells with before negative zero, product positive zero, after positive zero; the acquired `rhs_before_u` and `rhs_after_u` are bit-equal to round 50's `after_hpg_u` and `after_vor_u` respectively on all active cells. | Any count or endpoint mismatch: **REFUTED**; register the unmatched population and do not generalize from the first cell. |
| R55-P4 | The operand-walk instrument is non-vacuous. | Flipping one recorded `product_u` sign at an affected active cell makes the chain checker refuse and changes exactly the registered statement result; an endpoint plant also refuses. | A plant stays green or perturbs a zero outside the scored domain: reject the instrument and quote no number. |
| R55-P5 | If P2-P4 confirm, source-materializing the existing vorticity accumulator addition is the single downstream partner needed by the held QCO statement. | With only the compiled vorticity addition association plus round 48's source-ordered QCO hunk, the five OVERFLOW kt6-10 U regressions disappear; the ORCA2 ladder loses no AT-BAR row and the full GYRE Decision 43/45/55/59, DINO, tank, generic-card, citation, and push gates pass. | Any earlier first-over-bar row, AT-BAR loss, unregistered moved row, or shared-card refusal: revert the candidate and finish **HELD**, naming the failing row. |

Failed predictions remain in the receipt as **REFUTED**.  Post-hoc statistics
are labelled post-hoc.  No configuration choice or sea-ice change is allowed.

## Required verification

1. Re-run the committed record admission and both existing plants.
2. Add one committed source-order analysis gate that consumes the admitted
   self-describing record and round-50 endpoints; run its sign and endpoint
   plants.
3. Before a model edit, review the measured claim against the compiled running
   `dynvor.f90` branch.  If a candidate is eligible, apply the smallest shared
   statement and restore round 48's held QCO hunk without changing selectors.
4. For any model edit, run the complete standing GYRE gate, ORCA2 ladder,
   DINO/tank/generic-card gates, default citation re-anchor gate, push gate,
   focused tests, the 170-test shared-card battery, and one
   `tests/ocean/fidelity -n 12` battery.
5. Run a separate read-only `codex exec` diff review and retain its verdict, or
   the mandated unavailable wording.

## Frozen OPEN if held

Continue from the first refuted ENS operand or accumulator statement before
retrying the QCO pair.  After this downstream pair closes, return to ORCA2's
whole-card kt=1 stage-1 T owner, Decision 52's independent-start trajectory,
and the round-20 slow-forcing walk.  Sea ice remains out of scope at
`STOP_SELECTOR_GAP`.
