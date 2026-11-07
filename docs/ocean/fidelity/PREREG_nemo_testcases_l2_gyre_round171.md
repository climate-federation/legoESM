# Preregistration — round 171, e3t(Kaa) versus tracer-content sensitivity

Committed before measuring any Round-171 arm.  Round 169 made NEMO's
already-formed temperature coefficient and live `e3w(Kmm)` divisor exact at
the production solve boundary and left `1.241262968697578e-03` K day-240 T3D
RMS.  This round ranks the next two compiled-order solve inputs: the live
`e3t(Kaa)` matrix weight and the already-formed temperature content RHS.

## Compiled statements and admitted record

The record build's selected adaptive-implicit branch consumes `e3t(Kaa)` in
the matrix diagonal after forming the two `e3w(Kmm)`-weighted off-diagonals at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:468-474`.
NEMO then forms the temperature content RHS from the before tracer/content and
the now-level accumulated tendency/content at `trazdf.f90:549-567`.  The same
matrix and content enter the ordered backward recurrence at
`trazdf.f90:577-582`.

The admitted Round-125 record supplies `zwt_mix`, `e3w_Kmm`, `e3t_Kaa`,
`rhs_T`, every rebuilt matrix boundary, and every output for steps 1081--1440.
Its literal rebuild must remain bit-exact at every recorded boundary.  The
Round-132 daily restart supplies the exact NEMO step-1080 entry.  The free,
complete-K, and complete-K-plus-e3w arms must reproduce Round 169 exactly
before either new sensitivity is interpreted.

## One-variable production arms

Every arm advances the same admitted NEMO day-180 entry through 360
production-jitted steps with identical forcing and fp64/libm policy.

1. `free`: no intervention.
2. `complete_K_e3w`: replace only the formed temperature coefficient and its
   tracer `e3w(Kmm)` divisor with their recorded NEMO values, reproducing
   Round 169.
3. `complete_K_e3w_e3t_identity`: feed the preceding arm its own live tracer
   cell thickness through the extended private solve seam.
4. `complete_K_e3w_e3t`: replace only the tracer matrix's `e3t(Kaa)` with the
   recorded NEMO `e3t_Kaa`; its content RHS and every upstream geometry use the
   arm's own values.
5. `complete_K_e3w_content_identity`: feed the base arm its own formed
   temperature content through the seam.
6. `complete_K_e3w_content`: replace only the temperature content consumed by
   the solve with recorded NEMO `rhs_T`; salinity content, matrix `e3t(Kaa)`,
   and every upstream field stay the arm's own.

The private seam extends the existing diagnostic tuple only.  Omitted tuples
and its two-, three-, and four-array forms are unchanged.  This is not a public
configuration, a production-physics arm, or carried state.

## Frozen predictions and falsifiers

1. `free` reproduces Round 169's `1.584259320940647e-02` K day-240 T RMS and
   `complete_K_e3w` reproduces `1.241262968697578e-03` K.  Either mismatch
   stops the round.
2. Each identity arm is byte-identical to `complete_K_e3w` at every step.
   Any moved state byte refutes that seam and withholds its scientific arm.
3. At step 1081, the directed e3t arm consumes NEMO `e3t_Kaa` bit-for-bit and
   the directed content arm consumes NEMO `rhs_T` bit-for-bit, while every
   registered upstream tracer boundary remains unchanged.  Any unequal
   targeted wet cell or upstream movement refutes that arm.
4. Neither operand alone carries half the Round-169 remainder: CONFIRM for an
   arm if it removes less than `6.20631484348789e-04` K from
   `complete_K_e3w`; REFUTE for any arm meeting or exceeding that threshold.
5. The live `e3t(Kaa)` arm removes more day-240 T RMS than the content arm.
   CONFIRM only if its reduction is strictly larger; equality or the reverse
   ordering REFUTES the prediction.  The larger measured reduction names the
   producer to walk next; if both are below the `2e-10` K run-to-run floor,
   both are exonerated and the first subsequent non-bit matrix/solve boundary
   is named instead.
6. For each directed input, multiplying every finite positive targeted wet
   value at step 1081 only by exactly `1 + 2**-20` changes at least one
   registered consumed boundary and at least one wet day-240 temperature
   cell, prints `STATUS PLANT-FIRED`, and exits nonzero.  A zero count, success
   marker, or exit zero invalidates that operand's measurement.

This is a magnitude-ranking diagnostic, not a landing candidate.  No
production physics, configuration, carried state, immutable before arm, or
pending decision changes.  No NEMO acquisition is authorized.
