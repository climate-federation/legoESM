# GYRE high-resolution round 21 lateral-diffusion preregistration

Date: 2026-10-20. This document is committed before the developed-state
Smagorinsky coefficient acquisition is run or its substitution rows are
scored.

## Fixed comparison

Scalar-math NEMO, vector-math NEMO, and legoESM start from the same admitted
day-180 `r4_smag` restart, SHA-256
`a8e33a3d22e4483bac777839f6cdc9bdf8c2e348c3a1d638adeafa114fff1c12`,
and execute step 4321. The card, 3,600-s timestep, masks, restart bridge,
forcing clock, CPU fp64/libm policy, and round-16 four-times twin discriminator
are unchanged.

Round 20's acquired cumulative records are first rescored in NEMO's executed
stage-1 order. The compiled scalar writer calls `ldf_dyn` before `stp_2D` at
`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/stprk3.f90:181-190`; `stp_2D` then
updates the shared accumulator in the order HPG, LDF, VOR, KEG, ZAD at
`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/stp2d.f90:159-190`. legoESM's
ordinary traced step remains the passivity control. Its existing private
Round-119 association arm supplies only the cumulative comparison in that
same order; its returned state is not called passive and is never a run
candidate.

The first cumulative boundary whose maximum residual clears four times the
same-field scalar/vector NEMO residual selects the factor walk. Exactness is
still zero unequal fp64 words. A boundary inside a nonzero envelope is not
called exact or discharged.

## Coefficient discriminator

The acquired input is NEMO's post-`ldf_dyn` `ahmt`/`ahmf` pair at step 4321,
immediately before `stp_2D`. This is the square-root coefficient consumed by
both bilaplacian passes: the executed mode-32 formation and conversion are at
`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/ldfdyn.f90:462-534`; the selected
rotated two-pass operator is at
`GYRE_OMIP_L2_P3_R20RHSS/BLD/ppsrc/nemo/dynldf_lev.f90:249-334`.

The scorer substitutes one recorded input set at a time into the existing
production operator:

1. scalar-NEMO `ahmt`, with legoESM's live `ahmf`;
2. scalar-NEMO `ahmf`, with legoESM's live `ahmt`;
3. the scalar-NEMO pair.

A single coefficient field owns the boundary only if that one-field arm makes
both post-LDF U and V bit-exact while the upstream post-HPG U and V remain
bit-exact. If neither single field does so but the pair does, the result is a
coupled coefficient composition debt, not a single-factor owner. If the pair
does not do so, the coefficient is exonerated and the next acquisition must
record the two operator passes. A smaller norm never names an owner.

## Admission and controls

Both scalar and vector acquisitions use one rank and the exact same common
restart. The write-disabled and write-enabled terminal restarts must be
byte-identical. The disabled run must emit no round-21 record. Each enabled
record must have the pinned binary64 header, native `126 x 86 x 31` extents,
finite active words, and exactly one `ahmt` plus one `ahmf` array. Both
providers are manifest-verified before scoring.

The inherited output one-ULP plant remains binding for the selected cumulative
row. The coefficient substitution is not eligible to name an owner until a
one-ULP change in an active substituted coefficient is also shown to move the
post-LDF row; that control is scored after acquisition, because the active
coefficient support comes from the acquired array.

## Confirm / refute

- **Confirm one-factor coefficient owner:** exactly one of the `ahmt` or
  `ahmf` arms gives zero unequal post-LDF U and V words, with zero unequal
  post-HPG words, and its active one-ULP plant fires.
- **Confirm coupled coefficient debt:** neither single arm is exact, the pair
  is exact, and the plant fires.
- **Refute coefficient ownership:** the pair leaves any post-LDF word unequal;
  request the two-pass operator record and name no physics owner.

No physics, card, deck, threshold, or long-run configuration may change in
this measurement.
