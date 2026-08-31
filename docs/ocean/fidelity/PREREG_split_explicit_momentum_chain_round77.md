# Preregistration: Redi MSC live-W-thickness peel, round 77

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 76 stops the ordered tracer-tail walk at Redi temperature: all 9,920
registered wet columns are red at the accumulating `1e-12` bar (maximum
normalized error `0.11658880365945572`).  Salinity is scored only as a
co-located diagnostic and is likewise red; temperature remains the ordered
owner.  Its official receipt is frozen at SHA-256
`45e4f8afda737b41e457668fe1ab7cc28ded09d3f7be06fabdd15e9804936a76`.

The first source-declared approximation in the production Redi operator is
the W-point thickness used by the Method of Stabilizing Correction (MSC).
NEMO builds `akz` with live `e3w(:,:,jk,Kmm)^2`
(`traldf_iso.F90:314-332`) and divides the explicit A33 flux by the same live
`e3w(:,:,jk+1,Kmm)` (`traldf_iso_scheme.h90:126-129`).  legoESM instead
constructs `e3w_ab = 0.5*(e3t(k-1)+e3t(k))` in
`gm_redi_latlon_cgrid.py:2484-2491`.  The exact live Kmm W thickness is already
carried into this production dispatch and was independently certified in
round 71.

## Frozen factorial

Replay both captured production Redi calls, changing only the MSC thickness:

1. `legacy_t_average`: the unmodified production result;
2. `live_kmm_e3w`: substitute the carried exact Kmm W thickness in both the
   `akz` construction and the explicit-A33 `1/e3w` post-factor.

All tracer values, slopes, masks, metrics, Redi coefficients, `dt`, and GM
bolus inputs remain production values.  Compare each arm with the unchanged
stage-22/stage-23 NEMO RHS difference over the exact 9,920-column census.
Temperature is adjudicated first, salinity second, at the accumulating
`1e-12` bar.  The exact arm must execute the same two T-before/S-before calls;
identity must pass, while sign reversal, meridional roll, and a one-wet-point
perturbation must remain red.

## Ownership bars

- `REDI_MSC_E3W_OWNED`: the live-Kmm arm is AT-BAR for temperature and
  salinity, and removes at least 99% of each legacy arm's maximum normalized
  error.  Build the faithful selector only in this case.
- `REDI_MSC_E3W_MAJORITY`: both tracers improve, neither worsens, and the
  temperature maximum error removal is at least 50% but the exact arm remains
  red.  Record a bounded majority contribution and peel the first remaining
  Redi flux operand in NEMO order.
- `REDI_MSC_E3W_REFUTED`: temperature removal is below 50%, either tracer
  worsens, or a control fails.  Do not change the production default; proceed
  to the ordered Redi flux ladder (`zfu`, `zfv`, `zfw` skew terms, explicit
  A33, then divergence).

The optional replay operand must default to the historical construction so
non-DINO and preexisting call sites remain byte-identical.  Any eventual
production selector is faithful only on the two NEMO DINO cards and is pinned
to the historical mode everywhere else; tests require a red-capable legacy
arm rather than a self-comparison.
