# ORCA2 round 293 preregistration — fold-image V-depth operand split

Date: 2026-10-16

Round 292 proved that the first valued producer of the independent OMT-4
kt=1 stage-1 carried external-mode V target is the substep-1 live V-face
depth on NEMO's fold-image row: 13/8,589 active rank-slab faces differ,
maximum 719.0 m, while the interior is bit-exact.  This round splits that
depth statement into its recorded and card-carried operands.  It extends the
already-validated round-245 source replay; it does not add a second formula.

The compiled ORCA2 record build evaluates
`zhvp2_e = hv_0 + 0.5*r1_e1e2v*(e1e2t(j)*zsshp2_e(j) +
e1e2t(j+1)*zsshp2_e(j+1))*ssvmask` at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:509-520`.
The source-order split is frozen as `hv_0`, `r1_e1e2v`, the local T-cell
area and sea level, the northern T-cell area and sea level, then `ssvmask`.
The oracle operands come from the admitted round-222 substep/advmean streams
and the card's raw NEMO geometry bundle.  The candidate operands come from
the current production D114 route and the passive round-292 trace.

Every number is **independent OMT-4, kt=1 stage 1, external substep 1**.
No given-entry result is mixed into the table.  The held literal face-depth
arm is replayed offline only; no model, card, configuration, carried state,
or NEMO record changes are authorised.

## Frozen predictions and falsifiers

1. **R293-P1 — admission, precision, passivity.**  The inherited records
   admit, the card resolves vector-form momentum with the literal fold unit
   off, CPU production JIT and fp64/libm print, and every traced completed
   state is array-identical to the ordinary state.  Any miss refuses all
   scientific output.
2. **R293-P2 — replay calibration.**  The round-245 source replay, evaluated
   with the exact operands used by production, reproduces round 292's
   candidate substep-1 live V depth bit-for-bit on the complete rank slab.
   Any unequal cell makes the instrument invalid.
3. **R293-P3 — static operands.**  D114's raw reference `hv_0`,
   `r1_e1e2v`, and both T-cell areas are bit-exact against NEMO on the
   measured slab.  A non-bit row refutes the prediction and becomes the
   first operand in compiled order.
4. **R293-P4 — cancelling partner.**  The compact surface `ssvmask` is the
   first non-bit operand on the fold-image row.  Substituting NEMO's raw V
   surface mask alone does not close the 13-cell depth boundary because it
   exposes the northern T-point operand; substituting the mask together with
   NEMO's T-pivot north-neighbour association closes the live V depth to
   0 unequal cells.  A different first operand, single-arm closure, or a
   nonzero paired residual refutes the corresponding clause.
5. **R293-P5 — locality and causality.**  Every effective substitution is
   confined to the three-row fold band, and the paired depth closure also
   closes the immediate `zhV=e1v*va_e*zhvp2_e` consumer at substep 1.  Any
   valued interior movement or surviving consumer residual refutes the
   attribution.
6. **R293-P6 — non-vacuity.**  Admission/passivity, source order, replay
   calibration, first-operand, single-arm, pair-closure, locality,
   consumer-closure, and one-bit plants must each refuse.  A green plant
   invalidates the measurement.

The outcome is measurement-only and **HELD**.  The face-depth statement and
its cancelling partner remain inside the private complete fold unit; no
partial landing is allowed.  Decision 125's completed-stage top-V exchange
and the resolved stage mask remain off.  Decision 126 stays pending and the
shipped runoff setting is unchanged.  Sea ice and the shipped card's
`unmeasured_features` tuple are unchanged.

Independent review: operator, as required for this lane.

## Instrument clarification — 2026-10-16, before measurement

Prediction R293-P2's phrase "used by production" means the operands used by
the production-shaped **held literal-fold arm**, not the shipped selector-OFF
trajectory measured in round 292.  The immutable round-292 result supplies the
upstream 13-cell boundary; the reused round-245 executable/replay pair
calibrates the held arm at that same substep.  Three controlled executions are
frozen here: held arm with current compact masks, held arm with raw V surface
mask, and held arm with raw V surface mask plus the T-pivot north-neighbour
association.  This clarification changes no prediction or threshold.

## Instrument correction — 2026-10-16, before accepted measurement

The combined three-arm executable produced no report; the first isolated arm
was then killed during full-step compilation, also before emitting a report.
Round 292 already proves that substep-1 midpoint SSH and V are array-exact,
and the named boundary is purely the face-depth formula.  The accepted
instrument therefore evaluates the existing production face-depth function
and the independent round-245 source replay offline on NEMO's admitted
substep-1 SSH, with the card's instantiated geometry operands.  The two
implementations must be bit-identical and must reproduce round 292's frozen
13-cell / 719 m boundary before any split is read.  The three arms are then
offline one-variable substitutions on that calibrated statement.  This keeps
the frozen operands, order, predictions, and falsifiers; it removes only the
irrelevant full-step recompilation.

Prediction R293-P5 is still tested literally.  If the immediate transport was
already exact because substep-1 midpoint V is zero, its claimed closure is
**REFUTED as a null consumer**, not credited to the operand pair.
