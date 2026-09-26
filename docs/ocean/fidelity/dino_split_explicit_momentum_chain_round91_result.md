# Split-explicit momentum chain — round 91 result

Date: 2026-08-30. Session: `01a04e34-d1fb-73e0-b25a-177641f0a246`.

The registered existing-dump stage substitution localized the vertical Redi
temperature skew residual to the pair of W-position slopes computed with the
Naa geometry. The joint arm removed `0.9999999570878265` of the maximum error:
`9462/9920`, `7.735372278997023e-8` became `86/9920`,
`3.319416367691509e-15`. The `wslpj` source field is at bar; `wslpi` has one
column at `1.1791547803655264e-15`. The remaining failure is therefore a
strict association-class residue, not an unowned physical operand. The two
single substitutions and their interaction are retained in the receipt.

Receipt: `/tmp/dino_split_explicit_momentum_chain_round91.json`, SHA-256
`f4d68e6802a48c7c11f52648035bd25be4524bfa7e6214db1cdc20f4630ad63a`;
disposition `REDI_ZFW_T_SKEW_NEEDS_EXISTING_ROW30_LADDER`. The disposition is
kept verbatim because its frozen exact-pass ownership clause did not fire; the
quantitative receipt supports the narrower majority ownership above.

NEMO computes the limited W-position triads in `ldfslp.F90:472-530` using live
`Kmm` depths and thicknesses. The consumer then uses `wslpi/wslpj` directly in
`traldf_iso_scheme.h90:109-125`. Production must therefore carry the
later-computed W pair into this consumer while retaining the already-certified
U/V slopes and the source association.
