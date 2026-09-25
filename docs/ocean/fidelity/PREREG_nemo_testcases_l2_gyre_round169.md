# Preregistration — round 169, live e3w(Kmm) sensitivity

Committed before measuring any Round-169 arm.  Round 168 made NEMO's
already-formed tracer coefficient bit-exact at the production solve boundary
and left `1.241263037891706e-03` K day-240 T3D RMS.  This round changes one
additional consumed operand: the live `e3w(Kmm)` divisor.

## Compiled statement and admitted record

The selected temperature branch has already formed `zwt`.  It then computes
the lower and upper coefficients by dividing `zwt` by live `e3w(Kmm)` at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:468-469`, before the
diagonal reads `e3t(Kaa)` at `:470`.  The resulting coefficients enter the LU,
forward, and backward recurrences at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:527-582`.

The admitted Round-125 record supplies `zwt_mix`, `e3w_Kmm`, the remaining
matrix operands, and every output for steps 1081--1440.  Its literal rebuild
must remain bit-exact at every recorded boundary.  The Round-132 daily restart
supplies the exact NEMO step-1080 entry.  The free arm and the direct complete-K
arm must reproduce Round 168 exactly before the new sensitivity is interpreted.

## One-variable production arms

Every arm advances the same NEMO day-180 entry through 360 production-jitted
steps with identical forcing and fp64/libm policy.

1. `free`: no intervention.
2. `complete_K`: replace only the formed tracer coefficient with recorded
   NEMO `zwt_mix`, reproducing Round 168.
3. `complete_K_e3w_identity`: apply the same complete-K intervention and feed
   the solve its own live `e3w(Kmm)` through the new private seam.  This must be
   byte-identical to `complete_K` through step 1440.
4. `complete_K_e3w`: apply the complete-K intervention and replace only the
   tracer solve's live divisor with recorded NEMO `e3w_Kmm`.  Geometry used by
   every upstream operator, the cell thickness `e3t(Kaa)`, content RHS,
   coefficient, viscosity, closure state, forcing, and carried state remain
   the arm's own.

The seam extends the existing private vertical-solve tuple; omitted tuples and
the existing two- and three-array forms are unchanged.  It is not a public
configuration or a production physics branch.

## Frozen predictions and falsifiers

1. `free` reproduces Round 168's developed day-240 T RMS
   `1.584259320940647e-02` K, and `complete_K` reproduces
   `1.241263037891706e-03` K.  Either mismatch stops the round.
2. `complete_K_e3w_identity` is byte-identical to `complete_K` at every step.
   Any moved state byte refutes the seam.
3. On step 1081, `complete_K_e3w` makes both `zwt_mix` and `e3w(Kmm)` bit-exact
   on all 17,400 active interfaces without moving an upstream tracer boundary.
   Any unequal active interface or upstream movement refutes the arm.
4. The live-divisor mismatch does **not** carry the remaining magnitude: define
   "carries" as removing at least half of Round 168's remaining
   `1.241263037891706e-03` K day-240 T RMS.  CONFIRM if the new arm removes less
   than `6.20631518945853e-04` K relative to `complete_K`; REFUTE otherwise.
   A refutation promotes the live-thickness producer.  Confirmation exonerates
   this operand by magnitude and promotes the next source-ordered operands,
   `e3t(Kaa)` and the content RHS, for ranking.
5. Advancing one positive recorded `e3w_Kmm` value by one binary64 ULP at step
   1081 changes at least one registered matrix cell and the day-240 temperature,
   prints `STATUS PLANT-FIRED`, and exits nonzero.  An inert or success-printing
   plant invalidates the measurement.

This is a developed-state sensitivity, not a landing candidate.  No production
physics, configuration, carried state, immutable before arm, or pending user
decision changes.  No NEMO acquisition is authorized.

## Control re-preregistration after the frozen ULP prediction failed

The first plant was run only after the scientific arms.  Its frozen prediction
5 is **REFUTED** and remains part of the record: advancing one recorded
`e3w_Kmm` value by one binary64 ULP on step 1081 moved two registered matrix
cells but zero day-240 temperature cells.  The gate printed
`STATUS PLANT-BLIND` and exited 2, so the scientific endpoint is withheld until
a consumed control reaches it.

Before running another plant, freeze this replacement control.  On step 1081
only, multiply every finite positive active recorded `e3w_Kmm` interface by
the exactly representable factor `1 + 2**-20`; all later steps use the
unmodified record.  This is a synthetic violation, not a scientific arm.  It
must move at least one registered matrix coefficient and at least one wet
day-240 temperature cell, print `STATUS PLANT-FIRED`, and exit 1.  Any zero
count, success marker, or exit 0 invalidates the instrument.  The ordinary
baseline inside that plant run must still reproduce the already-recorded
`complete_K_e3w` arm; no normal-arm arithmetic is changed.
