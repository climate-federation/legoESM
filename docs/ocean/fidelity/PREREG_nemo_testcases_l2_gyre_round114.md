# Preregistration — NEMO testcase L2 GYRE round 114

Date: 2026-09-18

This document is frozen before any Round-114 scientific measurement.  The
production baseline is the unchanged Round-113 tip: kt2 T/S/U/V maximum error
`1.4210854715202004e-14` / `2.1316282072803006e-14` /
`2.7377110452773967e-12` / `3.2849219221489645e-12`, kt3 T/S maximum error
`8.600419718618468e-7` / `6.979441735666114e-8`, and day-30 T rms
`6.890484901489568e-5 K`.  These are comparison anchors, not new
measurements.

## P1 — admitted inputs and compiled U statement

Round 114 extends the existing Round-67/111/113 production-step instrument;
it does not create a second harness.  It will admit and stamp the existing
Round-111 FCT writer record, the Round-46 kt2-stage record, and the Round-77
copy of the admitted barotropic-mean record before comparing any arrays.
Missing or mismatched target/stamp/commit metadata is a hard refusal rather
than a zero-fill or inferred record.

The active compiled GYRE statement is
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:281-296`.
For the time-split/free-surface arm it forms `zub` from final `un_adv`, the
Kmm inverse U depth, and carried `uu_b(Kmm)`, then writes `zFu` in the literal
factor order

`e2u * (e3u_3d * (1 + r3u(Kmm) * umask)) * (uu(Kmm) + zub * umask)`.

The factor walk will compare the values actually consumed by that production
statement.  It will not claim that a recomputation of an operand is the
operand itself.

## P2 — split the transport family at its recorded consumer

The admitted Round-111 record supplies U, V, and W separately.  Four complete
production-jitted kt2 steps will be scored from the same NEMO step entry:
baseline, U-only substitution, V-only substitution, and W-only substitution.
Production eager and isolated-JIT rows will be labelled separately; neither
can stand in for production JIT.  For every arm the direct `adv_up1_T/S`
boundary and local kt3 T/S state errors are registered.

Prediction: each single-family arm changes the production-JIT `adv_up1_T`
row, no single arm makes both `adv_up1` tracer rows BIT, and U-only reduces
the local kt3 T maximum error by less than 50%.  The triplet result from Round
113 remains the closure control: T `adv_up1` maximum error falls from
`6.545655547452618e-11` to about `1.63e-16`, and local kt3 T falls from about
`8.6004205e-7` to about `5.7637973e-8`.  Falsifiers are an inert single-family
arm, either tracer becoming BIT under one arm, a U-only kt3 T reduction of at
least 50%, or failure to reproduce the triplet endpoints.  Every falsifier is
retained in the receipt.

## P3 — compiled-order U factor walk

The U walk is ordered exactly as the compiled assignment, with separate rows
for:

1. static `e2u`;
2. live Kmm U-face thickness (`e3u_3d`, including its `r3u(Kmm)` factor);
3. `uu(Kmm)`;
4. the `zub` operands final `un_adv`, Kmm inverse depth, and `uu_b(Kmm)`;
5. `zub`, corrected U, and final `zFu`.

Round-46 direct fields and the Round-77 final post-LBC barotropic mean are
used where present.  If a required operand is not directly recorded, its row
is `UNMEASURED`; it is not manufactured from a downstream result.  The first
directly recorded non-BIT factor owns this walk.

Prediction: `e2u` is BIT.  The first non-BIT factor is the live Kmm U-face
thickness, inherited from the non-BIT half-step free surface already exposed
by Round 113.  Its maximum thickness error is predicted to be of the same
order as the Kmm tracer-thickness error (`2.4726e-8 m`), while `uu(Kmm)` and
the final `zFu` are also non-BIT.  Substituting only the recorded U-face
thickness through the production statement is predicted to improve both the
direct `zFu` and `adv_up1_T` errors but reduce local kt3 T by less than 10%.
Falsifiers are non-BIT `e2u`, BIT live thickness, a different earlier factor,
or a thickness-only kt3 T reduction of at least 10%.

The production plant changes one finite, nonzero recorded element of the
first non-BIT factor by exactly one ULP before the production statement.  It
must change the corresponding `zFu` row and the consumed `adv_up1` result;
the plant exits nonzero and prints `STATUS PLANT-FIRED`.  A plant which does
not survive production JIT refuses the instrument.

## P4 — landing and blast-radius rules

This is predicted to be a diagnostic round: an inherited non-BIT input is not
a source correction and will not be landed.  If the walk instead exposes an
implementable same-statement correction with a production-JIT local proof, a
committed preregistration addendum will freeze its numeric ladder/month
prediction before that candidate is measured.

Any candidate then uses Decision 43 without relaxation: day-30 T rms must
decrease against a measured same-base arm; first-over-bar must not move
earlier; no kt=1 AT-BAR row may leave the bar; every moved row, including a
worsening row, must be in the registry; and every recipe-derived executing
card is measured.  The predicted no-physics outcome keeps all baseline
ladder and month numbers above bit-identical.

DINO shares the transport implementation, but a diagnostic oracle
substitution is not a production statement and therefore cannot move DINO.
If production code changes, DINO must be measured.  ORCA2 remains
`UNMEASURED`: it needs an admitted stage-3 record plus its certified
trajectory before any transfer claim.
