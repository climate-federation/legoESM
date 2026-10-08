# Preregistration — round 167, developed vertical day-240 sensitivity

Committed before measuring any Round-167 arm.  This round does not revisit
the held shear route or the `rn2` rounding floor.  It ranks the next two
already-recorded vertical-diffusion boundaries by their effect over the
developed day-180-to-day-240 interval: NEMO's heat diffusivity `avt`, which is
the output of TKE plus EVD, and NEMO's complete temperature matrix coefficient
`avt + ah_wslp2`, which is the value consumed by the implicit tracer solve.

## Compiled program and admitted records

The executed closure dispatches `zdf_tke`, copies `avt_k` into `avt`, and then
applies EVD in
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfphy.f90:334-359`.
The TKE closure converts post-sweep energy and mixing lengths into `avt_k` at
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:681-712`.
The temperature solve then forms `zwt = avt + ah_wslp2` at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:418-444`, builds the
tridiagonal matrix at `:445-480`, and executes its three recurrences at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:527-582`.
The active EVD replacement is the literal conditional assignment at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfevd.f90:102-109`.

The admitted Round-125 record supplies those values at every step 1081--1440.
The Round-132 daily restart supplies the exact day-180 entry.  Before any arm
is interpreted, the existing record admission and source-literal matrix
rebuild must remain BIT for every recorded boundary.  The unmodified
production arm must also reproduce Round 164's step-1081 rows: vertical-
diffusion temperature RMS `2.1834094362625713e-05` K and heat-diffusivity
`5,721 / 17,400` unequal with maximum `1.594045423436441e-09` m2/s.
Failure of either calibration stops the round; no sensitivity number is then
citable.

## One-variable production arms

All arms start from the same admitted NEMO day-180 state and run the same 360
production-jitted steps, forcing, fp64/libm policy, and output score.  The
ordinary arm changes nothing.  At each implicit temperature solve:

1. `heat_K`: replace only the effective heat profile before the isoneutral
   addition with that step's recorded NEMO `avt`; retain legoESM's viscosity,
   isoneutral coefficient, content, geometry, matrix arithmetic, and TKE
   prognostic update.
2. `effective_K`: replace only that same heat-profile operand by
   `NEMO(zwt_mix) - legoESM(ah_wslp2)`, so the already-existing production
   addition produces NEMO's complete `avt + ah_wslp2` coefficient bit for bit;
   retain every other solve operand and the TKE update.

The existing private effective-coefficient seam is used at its production
solve boundary.  Omitted override and an identity override made from the
ordinary arm's own coefficient must return bit-identical trajectories.  A
one-ULP change to one nonzero recorded `avt` interface must move the heat arm's
matrix and day-240 temperature and must exit nonzero with
`STATUS PLANT-FIRED`.

## Frozen predictions and falsifiers

1. The ordinary and identity arms are BIT, and all three arms have identical
   upstream process boundaries on their first step.  REFUTED by any moved
   upstream row or any ordinary/identity state byte.
2. The `heat_K` arm makes the first-step heat coefficient BIT but does not make
   the complete matrix coefficient BIT because the isoneutral operand remains
   legoESM's.  REFUTED if heat remains non-bit or complete effective K becomes
   BIT.
3. The `effective_K` arm makes the first-step complete matrix coefficient BIT.
   REFUTED if any active interface differs.
4. Ranked by reduction of the day-240 T3D RMS from the ordinary developed arm,
   `effective_K` is larger than `heat_K`.  The TKE/heat closure is exonerated
   as the magnitude owner if it is not the winning arm.  REFUTED if `heat_K`
   removes at least as much as `effective_K`.
5. Historical day-240 controls are reproduced from committed artifacts, not
   rerun or silently transferred: daily TKE-family reset removed
   `1.2105103859601576e-04` K (`0.736018%`) in Round 134, while the admitted
   vertical-diffusion process carry over days 180--240 is
   `2.4168271578053416e-02` K from Rounds 125--126.  If their provenance or
   metric cannot be reconciled, they are omitted from the ranking rather than
   compared.

This is a sensitivity ranking, not permission to rewrite a downstream solve.
If `heat_K` wins, the next round returns to its first non-bit compiled TKE
statement.  If `effective_K` wins, the TKE closure is exonerated by magnitude
and the OPEN item moves to the first non-bit implicit-solve operand.  No
physics, configuration, carried state, or pending decision 53/54/57/58 is
changed in this measurement round.  No candidate proceeds to the
Decision-43/45/55 landing gate unless a single compiled statement is both
named and source-exact.

The immutable production headline remains Round 163's arm: kt2 T/S/U/V
AT-BAR, first-over-bar kt3, day-30 T RMS
`6.572574374770603e-05` K, day-240 `1.644836070117868e-02` K, and day-360
`1.122566001855131e-02` K.
