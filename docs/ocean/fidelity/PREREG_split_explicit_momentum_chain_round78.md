# Preregistration: ordered Redi flux ladder, round 78

Date: 2026-08-30. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 77 substitutes NEMO's live `e3w(:,:,Kmm)` consistently into MSC and
removes 99.381% of the Redi-temperature maximum normalized error and 99.465%
of salinity's, but both remain red in all 9,920 columns.  Its frozen receipt
SHA-256 is
`dcc0cff4c63b30024794ad25b023b65223fe87137e5052b6d7dc478066973d14`;
the registered `REDI_MSC_E3W_MAJORITY` rule forbids promoting the production
selector until the residual is owned.

The next measurement follows `traldf_iso_scheme.h90` source order.  A held
one-rank deterministic writer records, separately for temperature and
salinity, the full-halo flux arrays after each level's production calculation:

1. horizontal U flux `zfu` (`traldf_iso_scheme.h90:55-70`);
2. horizontal V flux `zfv` (`traldf_iso_scheme.h90:72-90`);
3. total below-cell vertical flux `zfw_kp1`, containing the skew A31/A32 and
   explicit MSC A33 pieces (`traldf_iso_scheme.h90:104-129`).

The twin replays the exact-live-e3w round-77 arm and returns those same three
production arrays before divergence.  Score temperature first in the order
above, then salinity only through the same first failure, on the exact wet-face
or wet-interface population.  Flux operands use the established pointwise
`1e-15` bar; the already-frozen stage-22/23 tendency remains at the
accumulating `1e-12` column bar.  Identity must pass; one wet-point
perturbation, the appropriate zonal/meridional roll, and sign reversal must
fail.  The OFF/ON run bracket requires every pre-existing stream byte-exact
and exactly six new full-halo `(35,203,56)` streams.

Disposition is the first red operand: `REDI_DIVERGED_ZFU_T`, then
`REDI_DIVERGED_ZFV_T`, then `REDI_DIVERGED_ZFW_T`, followed by the salinity
analogues.  Only if all six flux operands and both total tendencies are AT-BAR
may round 77's live-e3w selector be promoted and the walk advance to tracer
ZDF.  A failure in `zfw` triggers a registered A31/A32-versus-A33 split; a
failure only after all fluxes pass triggers a divergence/volume-factor
association peel.  No downstream tracer-tail receipt is promoted across the
first red row.

The writer claims units `9450--9455`; the build block must prove zero owners
over the complete source tree before applying the patch and exactly one owner
per unit afterward.  Buffers are explicitly zero-initialized and only the
cited interior domain is populated, so their halo contract is deterministic.
