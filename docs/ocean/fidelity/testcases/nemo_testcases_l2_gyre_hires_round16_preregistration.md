# GYRE higher-resolution round 16 term-twin preregistration

Registered before the scalar-math online budget exists.  This extends the
round-12 and round-14 native-surface-V budget; it does not change their field,
region, window, term definitions, or `1e-12 m s-1` pointwise closure bar.

## Claim and arithmetic envelope

The comparison remains days 30 through 180 inclusive in the western-boundary
wet-span bin.  The vector-math NEMO run remains the reference.  A new
scalar-math NEMO run must use the identical `r4_smag` namelist, writer,
one-rank cadence, and passive same-binary control; its only allowed source
difference is `arch-conda-scalarmath.fcm` in place of `arch-conda.fcm`.

For each source-aligned term, the candidate numerator is
`abs(mean(legoESM - NEMO_vector))` over all registered days and cells.  The
arithmetic-envelope numerator is
`abs(mean(NEMO_scalar - NEMO_vector))` over the same words.  Their ratio is
reported.  A candidate clears the envelope only at a ratio of at least four,
the lane's already registered state-twin discriminator.  A zero twin numerator
is reported explicitly rather than divided by zero.

The preregistered raw vertical-mixing and external-mode rows are also printed,
but the executed sources show that they do not have the same intermediate
boundary in the two models.  NEMO assigns the temporary barotropic removal to
vertical mixing and its restoration to the external correction, while legoESM
re-adds the mean before its recorded vertical boundary.  Therefore the
source-aligned ranking combines exactly those two rows as
`implicit_zdf_plus_external_mode`; it does not alter closure or relabel a
remainder.

## Confirm, refute, and next test

- `CONFIRM_CARRIER_ABOVE_ARITHMETIC_FLOOR`: the first source-aligned candidate
  has candidate/twin ratio at least four.
- `REFUTE_CARRIER_ABOVE_ARITHMETIC_FLOOR`: that ratio is below four.
- `UNMEASURED`: either provider, mask, days, provenance, passivity, closure, or
  scalar/vector source registry refuses.

Clearing this envelope names a carrier, not a defect owner.  The smallest
owner test is one common developed entry state with the stage-3 candidate term
captured by legoESM and scalar/vector NEMO, followed by a one-variable
substitution of NEMO's already-formed operand into legoESM.  A shared-path fix
is warranted only if that substitution removes the excess without moving an
untouched row.
