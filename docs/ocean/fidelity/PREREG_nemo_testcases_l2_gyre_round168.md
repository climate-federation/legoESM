# Preregistration — round 168, direct complete-coefficient sensitivity

Committed before measuring any Round-168 arm.  Round 167 established that
replacing only NEMO's developed heat diffusivity `avt` removes
`1.315561886821479e-02` K (`83.03955478957635%`) of the day-240 temperature
RMS, but its complete-coefficient arm was invalidated by subtraction/addition
rounding.  This round closes that missing comparator by replacing the already-
formed temperature coefficient directly after the isoneutral addition.

## Compiled program and controlled boundary

The compiled temperature arm selects `avt`, takes the active standard
isoneutral branch, and writes `zwt = avt + ah_wslp2` at
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:418-428`.
It then consumes that exact `zwt` in the lower, diagonal, and upper matrix
writes at `:465-481`; the LU, forward, and backward recurrences are at
`:527-582`.  The closure-to-heat boundary is separately established by
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfphy.f90:334-359` and
`GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdftke.f90:681-692`.

The existing production test seam is extended, not duplicated: its ordinary
two-array form still replaces pre-add heat diffusivity and viscosity, while a
private third array replaces only the already-formed effective temperature
coefficient immediately before the production matrix call.  No public API,
configuration field, carried state, or production default changes.

## One-variable production arms

Every arm starts from the admitted NEMO step-1080 restart and advances steps
1081--1440 through `LatLonCGridOceanModel.step -> self._step_jitted` with the
same forcing, fp64/libm policy, and recorded NEMO matrix stream:

1. `free`: no intervention.
2. `identity_postadd`: replace the formed effective coefficient by the free
   arm's own formed value; every trajectory byte must remain identical.
3. `heat_K`: the already-certified Round-167 arm, replacing only pre-add heat
   diffusivity with recorded NEMO `avt`; its day-240 result must reproduce
   `2.686974341191680e-03` K.
4. `complete_K`: retain legoESM's heat closure, viscosity, isoneutral
   computation, content, geometry, and every other operand, then replace only
   the formed effective temperature coefficient with recorded NEMO `zwt_mix`.

The admitted Round-125 record must first rebuild all NEMO matrix fields bit for
bit.  The free arm must reproduce Round 167's first-step heat/effective rows
and developed day-240 RMS `1.584259320940647e-02` K.  Failure stops the round.

## Frozen predictions and falsifiers

1. `free` and `identity_postadd` are byte-identical through step 1440.
   REFUTED by any state byte moving.
2. `complete_K` makes the step-1081 formed coefficient bit-exact on all 17,400
   active interfaces and changes no upstream tracer-process row.  REFUTED by
   any unequal active interface or moved upstream row.
3. `complete_K` removes more day-240 T3D RMS than `heat_K`, because it closes
   the heat coefficient plus the still-free isoneutral contribution at the
   exact compiled consumption boundary.  CONFIRM if its removed RMS exceeds
   `1.315561886821479e-02` K; REFUTE otherwise.  This is a sensitivity ranking,
   not a causal statement about which upstream family is wrong.
4. A one-ULP change to one positive recorded `zwt_mix` interface on step 1081
   moves a registered matrix cell and the day-240 temperature, prints
   `STATUS PLANT-FIRED`, and exits nonzero.  Any inert or success-printing plant
   invalidates the instrument.

If prediction 3 is confirmed, the first non-bit implicit-solve operand is
named from the calibrated Round-125 compiled-order record, but no downstream
replacement lands: the direct complete coefficient is an attribution arm, not
a NEMO statement.  If prediction 3 is refuted, the heat/TKE family remains the
magnitude owner and the next OPEN item is its first non-floor post-sweep
statement; the known `rn2` compiled-rounding floor is not revisited.

No NEMO acquisition or user decision is authorized.  The immutable production
headline remains Round 163: kt2 T/S/U/V AT-BAR, first-over-bar kt3, day-30 T
RMS `6.572574374770603e-05` K, day-240 `1.644836070117868e-02` K, and day-360
`1.122566001855131e-02` K.
