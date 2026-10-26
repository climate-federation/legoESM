# ORCA2 round 57 preregistration — OVERFLOW stage-2 UP3 statement walk

Date frozen: 2026-09-27  
Base: `561483aeed99f3a678a879d31856929f41789afa`  
ORCA2 claim label: **given NEMO's entry** (Decision 52; no ORCA2 trajectory
measurement is planned)  
OVERFLOW claim label: **given NEMO's recorded operands**

## Frozen question and scope

Round 56 established that kt=3 stage-2 momentum advection changes 651 of
16,900 active U cells, then stopped for the missing UP3 internals.  The
operator has now produced the committed round-56 self-describing record.  This
round admits that record and walks the executed UP3 program in source order:
recorded transport and velocity operands, horizontal curvature, sign-selected
curvature, reconstructed face value, face flux, horizontal RHS addition,
vertical curvature/selection/flux, and vertical RHS addition.  The first
non-bit statement ends the walk.

No model, card, selector, carried state, score mask, threshold, stabiliser, or
sea-ice field may change.  The held source-ordered QCO statement is not restored
or paired in this round.  The ORCA2 card's six sea-ice selectors and
`unmeasured_features` tuple remain untouched at `STOP_SELECTOR_GAP`.

## Compiled branch read before prediction

The acquired executable calls flux-form advection with explicit `zFu/zFv/zFw`
at
`OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/stprk3_stg.f90:350-363`.
The compiled UP3 routine records the supplied operands at
`OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:142-144`, forms the
horizontal velocity curvatures at `:157-166`, selects the T-point curvature by
the sign of the advected-velocity pair at `:182-192`, forms the T-point flux at
`:194-195`, and accumulates its horizontal divergence at `:224-234`.  The
vertical curvature, selector, face flux, and RHS addition execute at `:334-364`,
with the bottom-level addition at `:369-383`.

The current legoESM operator chooses the same NEMO velocity-pair selector, but
its `_up3_reconstruct` evaluates expanded `(-far + 5*adv + 2*next)/6` branches
before multiplying the averaged transport.  NEMO instead materializes the
second difference and evaluates `pair - (1/3)*selected_curvature` before the
transport product.  These formulas are algebraically equal but not guaranteed
bit-identical.

## Existing implementation and instrument

Repository search found and will extend the round-56 header-derived parser and
the round-55 source-order/bit-pattern gate pattern.  It also found the
production `_up3_reconstruct`; the round does not add a second model operator.
The new gate consumes the existing record read-only and calls the production
reconstruction on NEMO's exact recorded inputs.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R57-P1 | The acquired round-56 record is admissible without changing its schema or parent endpoints. | Header-derived admission is `AT_BAR`; producer stamp, payload digest, inherited-state admission, and both existing plants retain their recorded passes. | Any provenance, schema, endpoint, or inherited-state failure: stop and reconcile; quote no statement number. |
| R57-P2 | The recorded U inputs and NEMO source-order curvature and selector statements replay bit-for-bit on all active owned cells. | Kbb/Kmm/transport shapes and fp64 width are exact; recomputed `curv_uu`, velocity pair, and selected curvature have zero unequal active cells. | Any earlier unequal cell: **REFUTED**; name that earlier statement and stop before reconstruction. |
| R57-P3 | The first non-bit statement is legoESM's expanded same-direction U face reconstruction, before the face-flux product. | NEMO's `0.5*(pair-(1/3)*selected_curvature)` and production `_up3_reconstruct` differ on at least one active owned face, while every P2 row is exact; report count, maximum and first location. | Zero reconstruction differences: **REFUTED**; continue in the same frozen order through face flux, horizontal RHS, vertical curvature/selection/flux, and vertical RHS, then name the earliest non-bit row or report the endpoint contradiction. |
| R57-P4 | The comparison is non-vacuous. | Advancing one baseline-equal active reconstruction value by one fp64 representable step adds exactly one refusal and exits nonzero. | No equal target or no added refusal: reject the instrument and quote no R57-P3 number. |
| R57-P5 | No scientific statement lands in this measurement round. | Final `packages/` diff is empty; disposition is `HELD` with the measured first statement registered for the next one-statement landing attempt. | A package change exists: remove it; a landing requires a separately preregistered ORCA2/OVERFLOW/GYRE/DINO/tank/generic gate. |

Failed predictions remain in the receipt as **REFUTED**.  Any additional
diagnostic is labelled post-hoc.  The score domain is the fixed OVERFLOW active-U
mask; V remains `UNMEASURED_NO_ACTIVE_FACE` on this one-row card.

## Required verification

1. Re-run round-56 admission and derive every field name, shape, payload length,
   and physical EOF from the record header.
2. Require CPU fp64/libm and a clean, exact analysis commit before scoring.
3. Score each source-order row bitwise and stop interpretation at the first
   unequal row; run the active reconstruction plant.
4. Run focused tests, Ruff/compile checks, the default and receipt citation
   gates with a firing shifted-line plant, the 170-test shared-card battery, and
   `tests/ocean/fidelity -n 12` once, with registered failures classified in
   isolation.
5. Run separate read-only `codex exec` review and quote its verdict, or retain
   the mandated unavailable wording if the sandbox cannot initialize it.

## Frozen OPEN

If the expanded reconstruction is first non-bit, the next round preregisters
one NEMO-source-ordered statement and runs the full shared landing gate before
retrying the QCO pair.  If another row is first, that row owns the same next
step.  After the step-level walk closes, return to Decision 52's independent
ORCA2 start and month-scale ranking.  Sea ice remains out of scope at
`STOP_SELECTOR_GAP`.

ASKED: admit the acquired record and walk the next compiled statement.  
UNASKED: none.
