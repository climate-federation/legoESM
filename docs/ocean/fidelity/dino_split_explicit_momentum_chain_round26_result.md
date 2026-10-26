# Split-explicit momentum chain: round 26 instrument result

Date: 2026-08-29. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

The retained 211/223 bracket scores cleanly after the committed A2D loader
correction. The binding artifact is
`/tmp/dino_split_explicit_momentum_chain_round26_bottom_update.json`, SHA-256
`16a03c0aca5cc1f4ea36c22b144e7ea2ddc339c69777abffddd059e65ffcc8cd`.
Disposition is
`NEMO_BOTTOM_UPDATE_IDENTITIES_AT_BAR_PRODUCTION_REPLAY_HELD`.

The writer was not defective. All twelve new streams are full-halo
`DIMENSION(jpi,jpj)` products with 90,944 bytes. The two failing pre-existing
forcing streams are genuine 82,784-byte interiors because NEMO declares
`REAL(wp), DIMENSION(A2D(0)) :: zu_frc, zv_frc` at
`dynspg_ts.F90:168`. The scorer now asserts and records that distinction.

NEMO's executed explicit-bottom commit at `dynspg_ts.F90:700-705` is bit-exact
under the source-ordered reconstruction: normalized RMS and maximum error are
zero, with `0/9758` U and `0/9868` V mismatches. The active vector update at
`:719-732` is also bit-exact for both components. Live reciprocal depths are
AT BAR: U/V normalized RMS `8.5783e-17/8.6341e-17` and maxima
`1.9523e-16/3.9158e-16`. Identity, exact-bracket, and four-step-nextafter
controls all fire.

These receipts validate the held NEMO instrument; they do not substitute for
the SHA-pinned production replay after the round-25-authorized literal EEN
coefficient builder. Row 1.4 and all later ordered rows remain blocked at that
production boundary.
